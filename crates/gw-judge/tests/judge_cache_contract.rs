//! Cached grades belong to the actual request and interpretation contract, not just a caller hash.

use std::sync::Mutex;

use gw_judge::{JUDGE_CACHE_KIND, PanelJudge, build_judge_request, grade_one_cached};
use gw_providers::{ChatRequest, DeltaStream, Provider, StreamChatFuture, StreamDelta};
use gw_storage::Store;

#[derive(Default)]
struct RecordingProvider {
    requests: Mutex<Vec<ChatRequest>>,
}

impl Provider for RecordingProvider {
    fn stream_chat(&self, request: ChatRequest) -> StreamChatFuture<'_> {
        self.requests.lock().unwrap().push(request);
        Box::pin(async {
            let stream: DeltaStream = Box::pin(futures::stream::iter([Ok(StreamDelta {
                content: Some(r#"{"score":0.94,"verdict":"accept","rationale":"fresh"}"#.into()),
                finish_reason: Some("stop".into()),
                ..Default::default()
            })]));
            Ok(stream)
        })
    }
}

#[tokio::test]
async fn changed_rubric_text_or_whitespace_misses_with_the_same_rubric_id() {
    let store = Store::open_in_memory().await.unwrap();
    let provider = RecordingProvider::default();
    let judge = PanelJudge::new("judge", "family").with_rubric("same-id");
    for rubric in ["rubric", "rubric ", " rubric", "different rubric"] {
        grade_one_cached(&store, &provider, &judge, rubric, "trace", "same-hash")
            .await
            .unwrap();
    }
    assert_eq!(provider.requests.lock().unwrap().len(), 4);
}

#[tokio::test]
async fn changed_candidate_misses_even_with_a_stale_content_hash() {
    let store = Store::open_in_memory().await.unwrap();
    let provider = RecordingProvider::default();
    let judge = PanelJudge::new("judge", "family");
    for trace in ["first trace", "different trace", "different trace "] {
        grade_one_cached(
            &store,
            &provider,
            &judge,
            "rubric",
            trace,
            "stale-or-fabricated-hash",
        )
        .await
        .unwrap();
    }
    assert_eq!(provider.requests.lock().unwrap().len(), 3);
}

#[tokio::test]
async fn absent_and_empty_rubric_ids_keep_distinct_audit_identity() {
    let store = Store::open_in_memory().await.unwrap();
    let provider = RecordingProvider::default();
    let absent = PanelJudge::new("judge", "family");
    let empty = absent.clone().with_rubric("");
    let first = grade_one_cached(&store, &provider, &absent, "rubric", "trace", "h")
        .await
        .unwrap();
    let second = grade_one_cached(&store, &provider, &empty, "rubric", "trace", "h")
        .await
        .unwrap();
    assert_eq!(provider.requests.lock().unwrap().len(), 2);
    assert_eq!(first.rubric_id, None);
    assert_eq!(second.rubric_id.as_deref(), Some(""));
}

#[tokio::test]
async fn legacy_folded_cache_row_is_ignored_and_preserved() {
    let store = Store::open_in_memory().await.unwrap();
    let provider = RecordingProvider::default();
    let judge = PanelJudge::new("judge", "family").with_rubric("r");
    // Hand-authored legacy identity for this judge's default sampling and reasoning budget.
    let legacy_key = "r#t0#pnone#snone#mt3500#rmt2000";
    let legacy_grade = serde_json::json!({
        "judge_model": "judge", "score": 0.123, "verdict": "reject",
        "raw": {"scoring_used": "g_eval_logprob", "response": "historical audit"}
    });
    store
        .cache_put(
            "h",
            JUDGE_CACHE_KIND,
            "judge",
            Some(legacy_key),
            &legacy_grade,
        )
        .await
        .unwrap();

    let fresh = grade_one_cached(&store, &provider, &judge, "rubric", "trace", "h")
        .await
        .unwrap();
    assert_eq!(
        provider.requests.lock().unwrap().len(),
        1,
        "legacy keys cannot prove request identity"
    );
    assert_eq!(fresh.score, 0.94);
    assert_eq!(
        store
            .cache_get("h", JUDGE_CACHE_KIND, "judge", Some(legacy_key))
            .await
            .unwrap(),
        Some(legacy_grade)
    );
}

#[tokio::test]
async fn same_effective_request_reuses_the_grade_despite_raw_cap_or_family_changes() {
    let store = Store::open_in_memory().await.unwrap();
    let provider = RecordingProvider::default();
    let first = PanelJudge::new("judge", "family-a").with_max_tokens(1);
    let second = PanelJudge::new("judge", "family-b").with_max_tokens(3_500);
    let first_grade = grade_one_cached(&store, &provider, &first, "rubric", "trace", "h")
        .await
        .unwrap();
    let second_grade = grade_one_cached(&store, &provider, &second, "rubric", "trace", "h")
        .await
        .unwrap();
    assert_eq!(provider.requests.lock().unwrap().len(), 1);
    assert_eq!(
        provider.requests.lock().unwrap()[0],
        build_judge_request(&first, "rubric", "trace")
    );
    assert_eq!(first_grade.raw["scoring_used"], "json_score");
    assert_eq!(first_grade.raw["interpretation_version"], 1);
    assert_eq!(first_grade.raw.to_string(), second_grade.raw.to_string());
    assert_eq!(first_grade.to_vote(), second_grade.to_vote());
}
