//! Per-panel ownership of distinct effective cache requests, including their successful writes.
use crate::cache::{JUDGE_CACHE_KIND, grade_from_cache_value, grade_to_cache_value};
use crate::panel::{build_judge_request, grade_request};
use crate::request_identity::request_fingerprint;
use crate::{Grade, JudgeError, PanelJudge, Result};
use futures::{StreamExt, stream::FuturesUnordered};
use gw_providers::{ChatRequest, Provider};
use gw_storage::Store;
use std::{
    collections::HashMap,
    sync::atomic::{AtomicBool, Ordering},
};

/// Caller-owned failure classification, in increasing precedence when a panel drains.
/// The engine supplies its existing classification and seals shared dispatch at this boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PanelFailure {
    /// Secondary cancellation; never hide a substantive error with this result.
    Cancelled,
    /// A record's malformed output or other local content failure.
    Record,
    /// Real admission denial; preserve the panel's unfinished work for resume.
    Halt,
    /// Storage or systemic failure; outranks a content error discovered earlier.
    Fatal,
}

struct Prepared<'a> {
    judge: &'a PanelJudge,
    request: ChatRequest,
    fingerprint: String,
    effective_contract: crate::EffectiveJudgeContract,
    positions: Vec<usize>,
}
fn prepare<'a>(
    judges: &'a [PanelJudge],
    rubric: &str,
    candidate: &str,
) -> Result<Vec<Prepared<'a>>> {
    if judges.is_empty() {
        return Err(JudgeError::EmptyPanel(
            "grade_panel_cached requires at least one judge".into(),
        ));
    }
    let mut keys: HashMap<(String, String), usize> = HashMap::new();
    let mut unique: Vec<Prepared<'_>> = Vec::new();
    for (position, judge) in judges.iter().enumerate() {
        let request = build_judge_request(judge, rubric, candidate);
        let effective_contract = crate::EffectiveJudgeContract::json_score(&request)?;
        let fingerprint = request_fingerprint(&request, judge.rubric_id.as_deref())?;
        let key = (judge.slug.clone(), fingerprint.clone());
        if let Some(&index) = keys.get(&key) {
            unique[index].positions.push(position);
        } else {
            keys.insert(key, unique.len());
            unique.push(Prepared {
                judge,
                request,
                fingerprint,
                effective_contract,
                positions: vec![position],
            });
        }
    }
    Ok(unique)
}

/// Grade a cached panel concurrently, with at most one operation per distinct effective cache key.
/// Every original position is retained, in panel order. Equal keys share one grade and its actual
/// paid origin; different sampling requests remain independent even when their model slug matches.
/// Cache hits make no requests. A successful miss finishes interpretation and its cache write.
/// Repeated positions are retained for collection/audit; [`crate::HybridGrader`] rejects them as
/// duplicate evidence. Validate admission configurations with [`crate::validate_judge_panel`]
/// before collecting a panel intended for consensus.
///
/// `on_error` classifies each observed error immediately and may seal the caller's shared dispatch.
/// No later logical miss starts after an error. Every already-started operation is drained, including
/// opaque provider futures waiting on RPM/backoff/admission. The panel neither detaches nor aborts
/// those futures. The first error in the highest returned [`PanelFailure`] class wins.
///
/// The configured panel cardinality is a per-panel logical bound, not an endpoint-wide HTTP limit.
/// Physical admission can impose stricter serialization. No cross-panel single-flight is claimed.
///
/// # Errors
/// Returns an empty-panel, request-identity, cache, or judge error after all owned work settles.
/// The caller must await this operation to completion to retain that drain guarantee.
pub async fn grade_panel_cached<P: Provider + ?Sized>(
    store: &Store,
    provider: &P,
    judges: &[PanelJudge],
    rubric: &str,
    candidate_render: &str,
    content_hash: &str,
    on_error: impl Fn(&JudgeError) -> PanelFailure + Send,
) -> Result<Vec<Grade>> {
    let unique = prepare(judges, rubric, candidate_render).inspect_err(|error| {
        on_error(error);
    })?;
    let mut ordered = vec![None; judges.len()];
    let mut misses = Vec::new();
    for prepared in unique {
        let cached = store
            .cache_get(
                content_hash,
                JUDGE_CACHE_KIND,
                &prepared.judge.slug,
                Some(&prepared.fingerprint),
            )
            .await
            .map_err(JudgeError::from)
            .inspect_err(|error| {
                on_error(error);
            })?;
        if let Some(cached) = cached {
            let grade = grade_from_cache_value(&cached, prepared.effective_contract);
            for position in prepared.positions {
                ordered[position] = Some(grade.clone());
            }
        } else {
            misses.push(prepared);
        }
    }
    let stopped = AtomicBool::new(false);
    let mut pending = FuturesUnordered::new();
    for prepared in misses {
        let stopped = &stopped;
        pending.push(async move {
            if stopped.load(Ordering::Acquire) {
                return (prepared.positions, None);
            }
            let result = async {
                let grade = grade_request(provider, prepared.judge, prepared.request).await?;
                store
                    .cache_put(
                        content_hash,
                        JUDGE_CACHE_KIND,
                        &prepared.judge.slug,
                        Some(&prepared.fingerprint),
                        &grade_to_cache_value(&grade),
                    )
                    .await?;
                Ok(grade)
            }
            .await;
            (prepared.positions, Some(result))
        });
    }
    let mut failure: Option<(PanelFailure, JudgeError)> = None;
    while let Some((positions, result)) = pending.next().await {
        match result {
            Some(Ok(grade)) => {
                for position in positions {
                    ordered[position] = Some(grade.clone());
                }
            }
            Some(Err(error)) => {
                stopped.store(true, Ordering::Release);
                let class = on_error(&error);
                if failure
                    .as_ref()
                    .is_none_or(|(previous, _)| class > *previous)
                {
                    failure = Some((class, error));
                }
            }
            None => {} // This logical miss was never started; no provider/cache future was dropped.
        }
    }
    if let Some((_, error)) = failure {
        return Err(error);
    }
    ordered
        .into_iter()
        .map(|grade| {
            grade.ok_or_else(|| {
                JudgeError::Invariant("cached panel omitted an original position".into())
            })
        })
        .collect()
}

#[cfg(test)]
#[path = "cached_panel_tests.rs"]
mod tests;
