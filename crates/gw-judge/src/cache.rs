//! The never-re-spend cache wrapper for judge/verify model calls (A2; ARCHITECTURE §5, DATA-SCHEMA
//! §5.2/§6.2).
//!
//! Repeated judge requests under the same interpretation contract reuse their cached grade.
//! The cache is checked BEFORE spending and written AFTER. The key is the storage cache key
//! `(content_hash, kind, model, rubric_id)` (`gw_storage::Store::cache_get` / `cache_put`), with the
//! a versioned request fingerprint in the `rubric_id` slot:
//!
//! - **`content_hash`** — the candidate's `record_hash` (`gw_storage::record_hash`).
//! - **`kind`** — [`JUDGE_CACHE_KIND`] (`"judge"`), distinguishing judge calls from teacher/verify.
//! - **`model`** — the judge slug.
//! - **`rubric_id`** — `judge-request-v2:<BLAKE3>` of the built request and interpretation contract.
//!   This includes the actual prompt/rubric/candidate bytes, model, effective sampling/reasoning
//!   budgets, optional rubric identity, float sampling bits, JSON scoring method, and interpretation
//!   version. The same built request is dispatched on a miss. Changed prompts or interpretation
//!   cannot reuse a grade even if the caller supplies a stale content hash. Different raw token
//!   caps that clamp to the same effective request still reuse the grade.
//!
//! Legacy folded rubric entries lack this evidence and are neither read nor rewritten/deleted.
//! Run ids, total spending budgets, downstream admission thresholds, calibration, and judge family
//! do not affect an individual judge request. Endpoint/routing defaults applied inside the provider
//! and resolved model revisions are not captured here; this key does not establish their identity.
//!
//! [`crate::grade_panel_cached`] is the production panel entry: for each judge it checks the cache, calls
//! the provider only on a miss, and writes the result back. A fake provider that asserts on a
//! second call for the same key proves the never-re-spend property (the unit test below).

use gw_providers::Provider;
use gw_storage::Store;
use serde_json::Value;

use crate::error::Result;
use crate::panel::{Grade, PanelJudge, build_judge_request, grade_request};
use crate::request_identity::request_fingerprint;

/// The `kind` discriminant for judge calls in the storage cache (distinct from teacher generation
/// and the deterministic verify rail). The verifier rail is pure/local and is not cached here.
pub const JUDGE_CACHE_KIND: &str = "judge";

/// Serialize a [`Grade`] to the JSON value stored in the cache (and re-hydrated on a hit). The grade
/// is round-tripped through its audit-bearing fields; the cached value is the source of truth on a
/// hit, so no provider call is made.
pub(crate) fn grade_to_cache_value(grade: &Grade) -> Value {
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
/// unreadable verdict becomes `Uncertain`), so malformed cache values do not crash a re-run. Live
/// evidence comes from the exact request whose current cache key matched, not from cached labels;
/// the consensus boundary rejects inconsistent grade metadata.
pub(crate) fn grade_from_cache_value(
    value: &Value,
    effective_contract: crate::EffectiveJudgeContract,
) -> Grade {
    use crate::decision::Verdict;
    let verdict = match value.get("verdict").and_then(Value::as_str) {
        Some("accept") => Verdict::Accept,
        Some("revise") => Verdict::Revise,
        Some("reject") => Verdict::Reject,
        _ => Verdict::Uncertain,
    };
    let mut raw = value
        .get("raw")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    if let Some(object) = raw.as_object_mut() {
        object.entry("attempt_origin").or_insert(Value::Null);
    }
    Grade {
        effective_contract: Some(effective_contract),
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
        raw,
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
/// slug, request_fingerprint)`; on a HIT re-hydrate the grade WITHOUT calling the provider;
/// on a MISS dispatch the fingerprinted request once, write the result, and return it.
///
/// # Errors
/// - [`JudgeError::Storage`](crate::JudgeError::Storage) on a cache read/write fault.
/// - [`JudgeError::Provider`](crate::JudgeError::Provider) / `JudgeParse` from the underlying
///   judge call on a miss.
/// - [`JudgeError::Invariant`](crate::JudgeError::Invariant) if request identity cannot serialize.
pub async fn grade_one_cached<P: Provider + ?Sized>(
    store: &Store,
    provider: &P,
    judge: &PanelJudge,
    rubric: &str,
    candidate_render: &str,
    content_hash: &str,
) -> Result<Grade> {
    let request = build_judge_request(judge, rubric, candidate_render);
    let effective_contract = crate::EffectiveJudgeContract::json_score(&request)?;
    let fingerprint = request_fingerprint(&request, judge.rubric_id.as_deref())?;
    if let Some(cached) = store
        .cache_get(
            content_hash,
            JUDGE_CACHE_KIND,
            &judge.slug,
            Some(&fingerprint),
        )
        .await?
    {
        return Ok(grade_from_cache_value(&cached, effective_contract));
    }
    let grade = grade_request(provider, judge, request).await?;
    store
        .cache_put(
            content_hash,
            JUDGE_CACHE_KIND,
            &judge.slug,
            Some(&fingerprint),
            &grade_to_cache_value(&grade),
        )
        .await?;
    Ok(grade)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::Verdict;
    use crate::panel::JudgeSampling;
    use crate::{PanelFailure, grade_panel_cached};
    use gw_providers::{ChatRequest, DeltaStream, ProviderError, StreamChatFuture, StreamDelta};
    use gw_schema::ReasoningEffort;
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
    fn grade_round_trips_through_cache_value() {
        let g = Grade {
            effective_contract: None,
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
        let contract = crate::EffectiveJudgeContract::json_score(&build_judge_request(
            &PanelJudge::new("m", "family").with_sampling(JudgeSampling {
                temperature: 0.3,
                top_p: Some(0.9),
                seed: Some(42),
            }),
            "rubric",
            "candidate",
        ))
        .unwrap();
        let back = grade_from_cache_value(&v, contract.clone());
        assert_eq!(back.effective_contract, Some(contract));
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
        let judge = PanelJudge::new("z-ai/glm-5.2", "glm").with_rubric("math");

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
    async fn different_top_p_misses_the_cache() {
        let store = Store::open_in_memory().await.unwrap();
        let provider = CountingProvider::new(2, "{\"score\":0.6,\"verdict\":\"revise\"}");

        let narrow = PanelJudge::new("m", "fam")
            .with_rubric("r")
            .with_sampling(JudgeSampling {
                top_p: Some(0.8),
                ..JudgeSampling::default()
            });
        let wide = PanelJudge::new("m", "fam")
            .with_rubric("r")
            .with_sampling(JudgeSampling {
                top_p: Some(0.9),
                ..JudgeSampling::default()
            });

        grade_one_cached(&store, &provider, &narrow, "rubric", "trace", "h")
            .await
            .unwrap();
        grade_one_cached(&store, &provider, &wide, "rubric", "trace", "h")
            .await
            .unwrap();
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn different_seed_misses_the_cache() {
        let store = Store::open_in_memory().await.unwrap();
        let provider = CountingProvider::new(2, "{\"score\":0.6,\"verdict\":\"revise\"}");

        let seed_one = PanelJudge::new("m", "fam")
            .with_rubric("r")
            .with_sampling(JudgeSampling {
                seed: Some(1),
                ..JudgeSampling::default()
            });
        let seed_two = PanelJudge::new("m", "fam")
            .with_rubric("r")
            .with_sampling(JudgeSampling {
                seed: Some(2),
                ..JudgeSampling::default()
            });

        grade_one_cached(&store, &provider, &seed_one, "rubric", "trace", "h")
            .await
            .unwrap();
        grade_one_cached(&store, &provider, &seed_two, "rubric", "trace", "h")
            .await
            .unwrap();
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn different_max_tokens_misses_the_cache() {
        let store = Store::open_in_memory().await.unwrap();
        let provider = CountingProvider::new(2, "{\"score\":0.6,\"verdict\":\"revise\"}");

        let smaller = PanelJudge::new("m", "fam")
            .with_rubric("r")
            .with_max_tokens(3_600);
        let larger = PanelJudge::new("m", "fam")
            .with_rubric("r")
            .with_max_tokens(4_000);

        grade_one_cached(&store, &provider, &smaller, "rubric", "trace", "h")
            .await
            .unwrap();
        grade_one_cached(&store, &provider, &larger, "rubric", "trace", "h")
            .await
            .unwrap();
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn different_reasoning_budget_misses_the_cache() {
        let store = Store::open_in_memory().await.unwrap();
        let provider = CountingProvider::new(2, "{\"score\":0.6,\"verdict\":\"revise\"}");

        let shallow = PanelJudge::new("m", "fam")
            .with_rubric("r")
            .with_reasoning_max_tokens(1_000);
        let deeper = PanelJudge::new("m", "fam")
            .with_rubric("r")
            .with_reasoning_max_tokens(2_000);

        grade_one_cached(&store, &provider, &shallow, "rubric", "trace", "h")
            .await
            .unwrap();
        grade_one_cached(&store, &provider, &deeper, "rubric", "trace", "h")
            .await
            .unwrap();
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn reasoning_effort_misses_reasoning_max_tokens_cache() {
        let store = Store::open_in_memory().await.unwrap();
        let provider = CountingProvider::new(2, "{\"score\":0.6,\"verdict\":\"revise\"}");

        let effort = PanelJudge::new("m", "fam")
            .with_rubric("r")
            .with_max_tokens(4_000)
            .with_reasoning_effort(ReasoningEffort::Xhigh);
        let budget = PanelJudge::new("m", "fam")
            .with_rubric("r")
            .with_max_tokens(4_000)
            .with_reasoning_max_tokens(2_000);

        grade_one_cached(&store, &provider, &effort, "rubric", "trace", "h")
            .await
            .unwrap();
        grade_one_cached(&store, &provider, &budget, "rubric", "trace", "h")
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
        let first = grade_panel_cached(&store, &provider, &judges, "rub", "trace", "hp", |_| {
            PanelFailure::Record
        })
        .await
        .unwrap();
        assert_eq!(first.len(), 2);
        // Re-run: both judges hit the cache; max_calls=2 means a third spend would panic.
        let second = grade_panel_cached(&store, &provider, &judges, "rub", "trace", "hp", |_| {
            PanelFailure::Record
        })
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
