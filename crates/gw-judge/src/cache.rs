//! The never-re-spend cache wrapper for judge/verify model calls (A2; ARCHITECTURE §5, DATA-SCHEMA
//! §5.2/§6.2).
//!
//! Every judge call is content-hash-cached so re-runs and crash-restarts never re-spend tokens.
//! The cache is checked BEFORE spending and written AFTER. The key is the storage cache key
//! `(content_hash, kind, model, rubric_id)` (`gw_storage::Store::cache_get` / `cache_put`), with the
//! judge **`temperature` folded in** as `temperature_bits = f64::to_bits(temperature)`:
//!
//! - **`content_hash`** — the candidate's `record_hash` (`gw_storage::record_hash`).
//! - **`kind`** — [`JUDGE_CACHE_KIND`] (`"judge"`), distinguishing judge calls from teacher/verify.
//! - **`model`** — the judge slug.
//! - **`rubric_id`** — the folded key [`folded_rubric_key`]: `"<rubric>#t<temperature_bits>"`. The
//!   storage layer's cache key is a fixed 4-tuple, so `temperature_bits` rides inside the
//!   `rubric_id` component rather than widening the schema. This is the load-bearing A2 change: a
//!   re-run at a DIFFERENT judge temperature gets a DIFFERENT key and so cannot silently reuse a
//!   stale verdict — the CACHE, not temp=0, is what guarantees exact replay (OpenRouter MoE routing
//!   is non-deterministic even at temp 0).
//!
//! [`grade_panel_cached`] is the production panel entry: for each judge it checks the cache, calls
//! the provider only on a miss, and writes the result back. A fake provider that asserts on a
//! second call for the same key proves the never-re-spend property (the unit test below).

use gw_providers::Provider;
use gw_storage::Store;
use serde_json::Value;

use crate::error::Result;
use crate::panel::{Grade, PanelJudge, grade_one};

/// The `kind` discriminant for judge calls in the storage cache (distinct from teacher generation
/// and the deterministic verify rail). The verifier rail is pure/local and is not cached here.
pub const JUDGE_CACHE_KIND: &str = "judge";

/// Fold the judge `temperature` into the `rubric_id` cache-key component as
/// `"<rubric>#t<temperature_bits>"`, where `temperature_bits = f64::to_bits(temperature)` (A2).
///
/// A `None` rubric becomes `"#t<bits>"` (empty rubric + the temperature tag), so two calls that
/// differ ONLY in temperature land on distinct keys. Using the raw `u64` bit pattern (not the
/// printed float) makes `-0.0` vs `0.0` and any rounding-equal temperatures hash distinctly and
/// reproducibly.
#[must_use]
pub fn folded_rubric_key(rubric_id: Option<&str>, temperature: f64) -> String {
    let bits = temperature.to_bits();
    match rubric_id {
        Some(r) => format!("{r}#t{bits}"),
        None => format!("#t{bits}"),
    }
}

/// Serialize a [`Grade`] to the JSON value stored in the cache (and re-hydrated on a hit). The grade
/// is round-tripped through its audit-bearing fields; the cached value is the source of truth on a
/// hit, so no provider call is made.
fn grade_to_cache_value(grade: &Grade) -> Value {
    serde_json::json!({
        "judge_model": grade.judge_model,
        "score": grade.score,
        "verdict": verdict_token(grade.verdict),
        "confidence": grade.confidence,
        "meta_prediction": grade.meta_prediction,
        "dimensions": grade.dimensions,
        "rationale": grade.rationale,
        "raw": grade.raw,
        "temperature": grade.temperature,
        "top_p": grade.top_p,
        "seed": grade.seed,
        "rubric_id": grade.rubric_id,
    })
}

/// Stable token for a per-grade verdict in the cached value.
fn verdict_token(v: crate::decision::Verdict) -> &'static str {
    use crate::decision::Verdict;
    match v {
        Verdict::Accept => "accept",
        Verdict::Revise => "revise",
        Verdict::Reject => "reject",
        Verdict::Uncertain => "uncertain",
    }
}

/// Re-hydrate a [`Grade`] from a cached JSON value. Missing/garbled fields degrade gracefully (an
/// unreadable verdict becomes `Uncertain`), so a partially-written legacy cache row never crashes a
/// re-run.
fn grade_from_cache_value(value: &Value) -> Grade {
    use crate::decision::Verdict;
    let verdict = match value.get("verdict").and_then(Value::as_str) {
        Some("accept") => Verdict::Accept,
        Some("revise") => Verdict::Revise,
        Some("reject") => Verdict::Reject,
        _ => Verdict::Uncertain,
    };
    Grade {
        judge_model: value
            .get("judge_model")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        score: value.get("score").and_then(Value::as_f64).unwrap_or(0.0),
        verdict,
        confidence: value
            .get("confidence")
            .and_then(Value::as_f64)
            .unwrap_or(0.0),
        meta_prediction: value.get("meta_prediction").and_then(Value::as_f64),
        dimensions: value
            .get("dimensions")
            .and_then(|d| serde_json::from_value(d.clone()).ok()),
        rationale: value
            .get("rationale")
            .and_then(Value::as_str)
            .map(str::to_string),
        raw: value.get("raw").cloned().unwrap_or(Value::Null),
        temperature: value
            .get("temperature")
            .and_then(Value::as_f64)
            .unwrap_or(0.0),
        top_p: value.get("top_p").and_then(Value::as_f64),
        seed: value.get("seed").and_then(Value::as_i64),
        rubric_id: value
            .get("rubric_id")
            .and_then(Value::as_str)
            .map(str::to_string),
    }
}

/// Grade ONE judge with the never-re-spend cache: check the cache for `(content_hash, "judge",
/// slug, folded_rubric_key(rubric_id, temperature))`; on a HIT re-hydrate the grade WITHOUT calling
/// the provider; on a MISS call the judge once, write the result, and return it.
///
/// # Errors
/// - [`JudgeError::Storage`](crate::JudgeError::Storage) on a cache read/write fault.
/// - [`JudgeError::Provider`](crate::JudgeError::Provider) / `JudgeParse` from the underlying
///   [`grade_one`] on a miss.
pub async fn grade_one_cached<P: Provider + ?Sized>(
    store: &Store,
    provider: &P,
    judge: &PanelJudge,
    rubric: &str,
    candidate_render: &str,
    content_hash: &str,
) -> Result<Grade> {
    let folded = folded_rubric_key(judge.rubric_id.as_deref(), judge.sampling.temperature);
    if let Some(cached) = store
        .cache_get(content_hash, JUDGE_CACHE_KIND, &judge.slug, Some(&folded))
        .await?
    {
        return Ok(grade_from_cache_value(&cached));
    }
    let grade = grade_one(provider, judge, rubric, candidate_render).await?;
    store
        .cache_put(
            content_hash,
            JUDGE_CACHE_KIND,
            &judge.slug,
            Some(&folded),
            &grade_to_cache_value(&grade),
        )
        .await?;
    Ok(grade)
}

/// Grade the whole panel with the never-re-spend cache — the production panel entry. Each judge is
/// graded through [`grade_one_cached`] (cache hit ⇒ no provider call). Returns
/// [`JudgeError::EmptyPanel`](crate::JudgeError::EmptyPanel) for an empty panel.
///
/// Calls run sequentially here (not `try_join_all`) because they share the `&Store`; the per-judge
/// work is dominated by the provider round-trip, which the cache elides on a re-run anyway. The
/// blind-sealed property holds regardless of ordering: no judge prompt contains another's output.
///
/// # Errors
/// Propagates the first [`JudgeError`](crate::JudgeError) from cache or a judge call.
pub async fn grade_panel_cached<P: Provider + ?Sized>(
    store: &Store,
    provider: &P,
    judges: &[PanelJudge],
    rubric: &str,
    candidate_render: &str,
    content_hash: &str,
) -> Result<Vec<Grade>> {
    if judges.is_empty() {
        return Err(crate::error::JudgeError::EmptyPanel(
            "grade_panel_cached requires at least one judge".into(),
        ));
    }
    let mut grades = Vec::with_capacity(judges.len());
    for judge in judges {
        grades.push(
            grade_one_cached(
                store,
                provider,
                judge,
                rubric,
                candidate_render,
                content_hash,
            )
            .await?,
        );
    }
    Ok(grades)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::Verdict;
    use crate::panel::{JudgeSampling, JudgeScoring};
    use gw_providers::{ChatRequest, DeltaStream, ProviderError, StreamChatFuture, StreamDelta};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A provider that PANICS if called more than `max_calls` times — proves a cache hit skips it.
    struct CountingProvider {
        calls: AtomicUsize,
        max_calls: usize,
        body: String,
    }

    impl CountingProvider {
        fn new(max_calls: usize, body: &str) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                max_calls,
                body: body.to_string(),
            }
        }
    }

    impl Provider for CountingProvider {
        fn stream_chat(&self, _req: ChatRequest) -> StreamChatFuture<'_> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            assert!(
                n <= self.max_calls,
                "provider called {n} times; cache should have elided calls beyond {}",
                self.max_calls
            );
            let body = self.body.clone();
            Box::pin(async move {
                let stream: DeltaStream =
                    Box::pin(futures::stream::iter(vec![
                        Ok::<StreamDelta, ProviderError>(StreamDelta {
                            content: Some(body),
                            finish_reason: Some("stop".into()),
                            ..Default::default()
                        }),
                    ]));
                Ok(stream)
            })
        }
    }

    #[test]
    fn temperature_folds_into_the_rubric_key() {
        let a = folded_rubric_key(Some("math"), 0.0);
        let b = folded_rubric_key(Some("math"), 0.7);
        assert_ne!(a, b, "different temperatures must produce different keys");
        assert!(a.starts_with("math#t"));
        // None rubric still carries the temperature tag.
        assert!(folded_rubric_key(None, 0.0).starts_with("#t"));
        // The bit pattern, not the printed float, is used.
        assert!(a.contains(&0.0f64.to_bits().to_string()));
    }

    #[test]
    fn grade_round_trips_through_cache_value() {
        let g = Grade {
            judge_model: "m".into(),
            score: 0.82,
            verdict: Verdict::Accept,
            confidence: 0.5,
            meta_prediction: Some(0.7),
            dimensions: None,
            rationale: Some("ok".into()),
            raw: serde_json::json!({"x":1}),
            temperature: 0.3,
            top_p: Some(0.9),
            seed: Some(42),
            rubric_id: Some("math".into()),
        };
        let v = grade_to_cache_value(&g);
        let back = grade_from_cache_value(&v);
        assert_eq!(back.score, g.score);
        assert_eq!(back.verdict, g.verdict);
        assert_eq!(back.temperature, g.temperature);
        assert_eq!(back.seed, g.seed);
    }

    #[tokio::test]
    async fn cache_hit_skips_the_provider() {
        let store = Store::open_in_memory().await.unwrap();
        // max_calls = 1: a second call for the SAME key would panic.
        let provider = CountingProvider::new(1, "{\"score\":0.9,\"verdict\":\"accept\"}");
        let judge = PanelJudge::new("z-ai/glm-5.2", "glm")
            .with_rubric("math")
            .with_scoring(JudgeScoring::IntegerLikert);

        let first = grade_one_cached(&store, &provider, &judge, "rubric", "trace", "hash-1")
            .await
            .unwrap();
        assert_eq!(first.verdict, Verdict::Accept);

        // Second call, same key → cache hit → provider NOT called again (max_calls=1 holds).
        let second = grade_one_cached(&store, &provider, &judge, "rubric", "trace", "hash-1")
            .await
            .unwrap();
        assert_eq!(second.verdict, Verdict::Accept);
        assert!((second.score - 0.9).abs() < 1e-12);
    }

    #[tokio::test]
    async fn different_temperature_misses_the_cache() {
        let store = Store::open_in_memory().await.unwrap();
        // max_calls = 2: each distinct temperature is its own key, so both spend (no false hit).
        let provider = CountingProvider::new(2, "{\"score\":0.6,\"verdict\":\"revise\"}");

        let cold = PanelJudge::new("m", "fam")
            .with_rubric("r")
            .with_sampling(JudgeSampling {
                temperature: 0.0,
                ..JudgeSampling::default()
            });
        let warm = PanelJudge::new("m", "fam")
            .with_rubric("r")
            .with_sampling(JudgeSampling {
                temperature: 0.7,
                ..JudgeSampling::default()
            });

        grade_one_cached(&store, &provider, &cold, "rubric", "trace", "h")
            .await
            .unwrap();
        // Different temperature_bits → different key → a real second spend (does not hit cold's row).
        grade_one_cached(&store, &provider, &warm, "rubric", "trace", "h")
            .await
            .unwrap();
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn cached_panel_replays_without_respend() {
        let store = Store::open_in_memory().await.unwrap();
        let provider = CountingProvider::new(2, "{\"score\":0.8,\"verdict\":\"accept\"}");
        let judges = vec![
            PanelJudge::new("a", "fa").with_rubric("r"),
            PanelJudge::new("b", "fb").with_rubric("r"),
        ];
        let first = grade_panel_cached(&store, &provider, &judges, "rub", "trace", "hp")
            .await
            .unwrap();
        assert_eq!(first.len(), 2);
        // Re-run: both judges hit the cache; max_calls=2 means a third spend would panic.
        let second = grade_panel_cached(&store, &provider, &judges, "rub", "trace", "hp")
            .await
            .unwrap();
        assert_eq!(second.len(), 2);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn unparseable_grade_is_not_cached_and_re_spends_on_retry() {
        // V2: an unparseable judge body errors (not frozen). A retry must RE-CALL the provider,
        // so a transient garble does not permanently poison the cache with a degenerate grade.
        let store = Store::open_in_memory().await.unwrap();
        // A high max_calls so the second call does not panic — we assert the count directly.
        let provider = CountingProvider::new(5, "this is not json");
        let judge = PanelJudge::new("m", "fam").with_rubric("r");

        let e1 = grade_one_cached(&store, &provider, &judge, "rub", "trace", "h")
            .await
            .unwrap_err();
        assert!(matches!(e1, crate::error::JudgeError::JudgeParse(_)));
        // Retry: the provider is called AGAIN (the bad result was not cached).
        let e2 = grade_one_cached(&store, &provider, &judge, "rub", "trace", "h")
            .await
            .unwrap_err();
        assert!(matches!(e2, crate::error::JudgeError::JudgeParse(_)));
        assert_eq!(
            provider.calls.load(Ordering::SeqCst),
            2,
            "an unparseable grade must NOT be cached — the provider is re-called on retry"
        );
    }
}
