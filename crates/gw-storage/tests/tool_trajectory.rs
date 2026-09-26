//! Persistence + resume evidence for tool-result identity (INVARIANT i).
//!
//! A tool trajectory is stored as a whole `TrainingRecord` envelope in `records.record_json`, so
//! identity only holds end-to-end if the link, the null-content distinction and the retained raw
//! argument text all survive `put` → `get`, a lifecycle advance (the crash/restart path), and a
//! `scan` — and if the content-hash dedup key actually distinguishes two trajectories that differ
//! only in which call each result answers.
//!
//! This lives in its own integration file so it cannot collide with the shared `store.rs` suite.

use gw_schema::{
    Content, FunctionCall, Lifecycle, LifecycleState, Message, Provenance, Role, ToolCall,
    TrainingRecord, Verdict,
};
use gw_storage::{RecordFilter, Store};

/// One `read_file` call with an object payload (no normalization, so no retained raw text).
fn call(id: &str, start_line: u32) -> ToolCall {
    ToolCall {
        id: Some(id.into()),
        function: FunctionCall {
            name: "read_file".into(),
            arguments: serde_json::json!({ "filepath": "toy.py", "start_line": start_line }),
            raw_arguments: None,
        },
    }
}

/// A tool trajectory with two SAME-NAME calls, one of them string-encoded on the wire, and the two
/// results arriving in reversed order.
fn tool_trajectory() -> Vec<Message> {
    vec![
        Message {
            role: Role::User,
            content: Content::Text("Read two regions.".into()),
            reasoning: None,
            reasoning_details: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
        Message {
            role: Role::Assistant,
            content: Content::Null,
            reasoning: Some("Two regions; the ids distinguish them.".into()),
            reasoning_details: None,
            tool_calls: Some(vec![
                call("read-a", 1),
                ToolCall {
                    id: Some("read-b".into()),
                    function: FunctionCall {
                        name: "read_file".into(),
                        arguments: serde_json::json!({
                            "filepath": "toy.py", "start_line": 20, "end_line": 22
                        }),
                        // The provider sent this string-encoded; the wire text is retained.
                        raw_arguments: Some(
                            "{\"filepath\": \"toy.py\", \"start_line\": 20, \"end_line\": 22}"
                                .into(),
                        ),
                    },
                },
            ]),
            tool_call_id: None,
            name: None,
        },
        Message {
            role: Role::Tool,
            content: Content::Text("{\"status\": \"error\"}".into()),
            reasoning: None,
            reasoning_details: None,
            tool_calls: None,
            tool_call_id: Some("read-b".into()),
            name: Some("read_file".into()),
        },
        Message {
            role: Role::Tool,
            content: Content::Text(
                "{\"status\": \"ok\", \"content\": \"def toy():\\n    return \\\"λ\\\"\\n\"}"
                    .into(),
            ),
            reasoning: None,
            reasoning_details: None,
            tool_calls: None,
            tool_call_id: Some("read-a".into()),
            name: Some("read_file".into()),
        },
        Message {
            role: Role::Assistant,
            content: Content::Text("The second read failed; no repair is claimed.".into()),
            reasoning: Some("Region 20 was unreadable; stop.".into()),
            reasoning_details: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
    ]
}

fn tool_record(record_id: &str, run_id: &str) -> TrainingRecord {
    TrainingRecord {
        record_id: record_id.into(),
        schema_version: semver::Version::new(1, 0, 0),
        dataset_version: None,
        training_area: "swe-agents".into(),
        tags: vec!["tool-use".into()],
        messages: tool_trajectory(),
        tools: None,
        provenance: Provenance {
            run_id: run_id.into(),
            parent_ids: vec![],
            teacher: gw_schema::TeacherRef {
                provider: "openrouter".into(),
                slug: "z-ai/glm-5.2".into(),
                served_by: Some("Parasail".into()),
                model_card_revision: None,
            },
            user_synth_model: None,
            user_turn_kind: None,
            in_scope_safe: Some(true),
            judge_models: vec![],
            harness_version: "0.1.0".into(),
            git_commit: None,
        },
        generation: Default::default(),
        verification_contract: None,
        execution_evidence: None,
        verification: Default::default(),
        judging: gw_schema::Judging {
            verdict: Some(Verdict::Admit),
            aggregate: Some(0.9),
            ..Default::default()
        },
        reasoning_quality: None,
        lifecycle: Lifecycle::default(),
        hashes: Default::default(),
        cost: Default::default(),
    }
}

async fn seeded_store() -> Store {
    let store = Store::open_in_memory().await.unwrap();
    store
        .create_run("run-t", "{\"budget\":25}", Some(25.0))
        .await
        .unwrap();
    store
}

/// The stored envelope must keep the result links, the null content and the retained raw argument
/// text — identity is not a property of an in-memory struct.
#[tokio::test]
async fn put_get_preserves_result_links_and_null_content() {
    let store = seeded_store().await;
    let rec = tool_record("rec-t", "run-t");
    store.put(&rec).await.unwrap();
    let got = store.get("rec-t").await.unwrap();

    assert_eq!(
        got.messages, rec.messages,
        "messages must round-trip byte-identically"
    );
    let calls = &got.messages[1];
    assert_eq!(
        calls.content,
        Content::Null,
        "null content survives storage"
    );
    let declared = calls.tool_calls.as_ref().unwrap();
    assert_eq!(declared[0].id.as_deref(), Some("read-a"));
    assert_eq!(declared[1].id.as_deref(), Some("read-b"));
    assert_eq!(
        declared[1].function.raw_arguments.as_deref(),
        Some("{\"filepath\": \"toy.py\", \"start_line\": 20, \"end_line\": 22}"),
        "the retained raw wire text must survive storage"
    );
    // Reversed results, each with its own link.
    assert_eq!(got.messages[2].tool_call_id.as_deref(), Some("read-b"));
    assert_eq!(got.messages[3].tool_call_id.as_deref(), Some("read-a"));
    // The result body is the provider's own JSON *text*: unicode and its escaped newline survive
    // storage verbatim (the whole-message equality above is the primary check; this pins the
    // specific characters so a lossy re-encoding would be named).
    assert!(
        matches!(&got.messages[3].content, Content::Text(t) if t.contains('λ') && t.contains("\\n"))
    );
}

/// The crash/restart path: `advance_lifecycle` rewrites the whole envelope, so it must not drop or
/// reorder the tool fields while it does.
#[tokio::test]
async fn advance_lifecycle_preserves_identity_across_resume() {
    let store = seeded_store().await;
    store.put(&tool_record("rec-t", "run-t")).await.unwrap();

    store
        .advance_lifecycle("rec-t", LifecycleState::AssistantGenerated, None)
        .await
        .unwrap();
    store
        .advance_lifecycle("rec-t", LifecycleState::Judged, None)
        .await
        .unwrap();

    let got = store.get("rec-t").await.unwrap();
    assert_eq!(got.lifecycle.state, LifecycleState::Judged);
    assert_eq!(got.lifecycle.attempts, 2);
    assert_eq!(
        got.messages,
        tool_trajectory(),
        "identity survives a lifecycle rewrite"
    );

    // A resume cursor recorded alongside the state still resolves, and the record is intact.
    let cursor = serde_json::json!({"seed_offset": 128});
    store
        .checkpoint("run-t", 0, "judged", &cursor)
        .await
        .unwrap();
    let point = store.resume_cursor("run-t", 0).await.unwrap().unwrap();
    assert_eq!(point.state, "judged");
    assert_eq!(point.cursor, cursor);
    assert_eq!(
        store.get("rec-t").await.unwrap().messages[3]
            .tool_call_id
            .as_deref(),
        Some("read-a"),
        "a resumed run reads back the same result link"
    );

    let history = store.lifecycle_history("rec-t").await.unwrap();
    assert_eq!(history.len(), 2);
}

/// A filtered scan goes through the same JSON decode, so the links must be there too.
#[tokio::test]
async fn scan_preserves_result_links() {
    let store = seeded_store().await;
    store.put(&tool_record("rec-a", "run-t")).await.unwrap();
    store.put(&tool_record("rec-b", "run-t")).await.unwrap();
    let all = store
        .scan(&RecordFilter::new().run_id("run-t"))
        .await
        .unwrap();
    assert_eq!(all.len(), 2);
    for rec in &all {
        assert_eq!(rec.messages[2].tool_call_id.as_deref(), Some("read-b"));
        assert_eq!(rec.messages[3].tool_call_id.as_deref(), Some("read-a"));
    }
}

/// The dedup key must treat the result link as CONTENT. Two trajectories whose messages are
/// identical except for which call each result answers are different trajectories — if they
/// collided, the "never re-spend" cache would hand back the wrong record.
#[test]
fn record_hash_distinguishes_which_call_each_result_answers() {
    let a = tool_record("a", "run-t");
    let mut b = a.clone();
    let (first, second) = (
        b.messages[2].tool_call_id.take(),
        b.messages[3].tool_call_id.take(),
    );
    b.messages[2].tool_call_id = second;
    b.messages[3].tool_call_id = first;
    assert_ne!(
        gw_storage::record_hash(&a).unwrap(),
        gw_storage::record_hash(&b).unwrap(),
        "a swapped result link must change record_hash"
    );

    // And dropping a link is likewise a content change.
    let mut c = a.clone();
    c.messages[2].tool_call_id = None;
    assert_ne!(
        gw_storage::record_hash(&a).unwrap(),
        gw_storage::record_hash(&c).unwrap()
    );
}

/// Null content and empty text are different observations, so they cannot share a dedup key either.
#[test]
fn record_hash_distinguishes_null_content_from_empty_text() {
    let a = tool_record("a", "run-t");
    let mut b = a.clone();
    b.messages[1].content = Content::Text(String::new());
    assert_ne!(
        gw_storage::record_hash(&a).unwrap(),
        gw_storage::record_hash(&b).unwrap(),
        "a null body and an empty body must not collide"
    );
    // The retained raw argument text is source evidence, so it is content too.
    let mut c = a.clone();
    c.messages[1].tool_calls.as_mut().unwrap()[1]
        .function
        .raw_arguments = None;
    assert_ne!(
        gw_storage::record_hash(&a).unwrap(),
        gw_storage::record_hash(&c).unwrap()
    );
}

/// The link is part of a record's content, not of its provenance: re-running the same trajectory
/// under a different run / teacher / verdict must still hash equal (the "never re-spend" contract).
#[test]
fn record_hash_still_ignores_non_content_fields() {
    let a = tool_record("a", "run-t");
    let mut b = tool_record("b", "run-t");
    b.provenance.run_id = "run-other".into();
    b.provenance.git_commit = Some("deadbeef".into());
    b.judging.verdict = Some(Verdict::Reject);
    b.lifecycle.state = LifecycleState::Error;
    assert_eq!(
        gw_storage::record_hash(&a).unwrap(),
        gw_storage::record_hash(&b).unwrap()
    );
}
