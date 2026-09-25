//! Integration tests for `gw-storage`: migrations, idempotent upsert, transactional lifecycle
//! advance, scan filters, the content-hash cache, parquet export round-trip, and resume cursors.
//!
//! All in-process: in-memory SQLite + in-memory parquet buffers, no network, no temp files.

use std::sync::Arc;

use arrow::array::{Array, StringArray};
use gw_schema::{
    Content, Generation, Hashes, JudgeVote, Judging, Lifecycle, LifecycleState, Message,
    Provenance, ReasoningDetail, ReasoningEffort, TeacherRef, TrainingRecord, TrlFormat, Verdict,
};
use gw_storage::{RecordFilter, ResumePoint, RunStatus, Store};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// Build a minimal-but-valid record for `run_id` with the given id and verdict/aggregate.
fn record(
    record_id: &str,
    run_id: &str,
    verdict: Option<Verdict>,
    agg: Option<f64>,
) -> TrainingRecord {
    let mut judging = Judging {
        verdict,
        aggregate: agg,
        ..Default::default()
    };
    if let Some(a) = agg {
        judging.panel.push(JudgeVote {
            judge_model: "judge-x".into(),
            rubric_id: Some("rubric-1".into()),
            temperature: Some(0.0),
            top_p: None,
            seed: None,
            score: a,
            dimensions: None,
            rationale: None,
            raw_response: None,
        });
    }
    TrainingRecord {
        record_id: record_id.into(),
        schema_version: semver::Version::new(1, 0, 0),
        dataset_version: None,
        training_area: "rust-async".into(),
        tags: vec!["tokio".into()],
        messages: vec![
            Message {
                role: gw_schema::Role::User,
                content: Content::Text("What is 12*8?".into()),
                reasoning: None,
                reasoning_details: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            Message {
                role: gw_schema::Role::Assistant,
                content: Content::Text("96".into()),
                reasoning: Some("12*8 = 96".into()),
                reasoning_details: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ],
        tools: None,
        provenance: Provenance {
            run_id: run_id.into(),
            parent_ids: vec![],
            teacher: TeacherRef {
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
        generation: Generation {
            reasoning_effort: Some(ReasoningEffort::Xhigh),
            ..Default::default()
        },
        verification_contract: None,
        verification: Default::default(),
        judging,
        reasoning_quality: None,
        lifecycle: Lifecycle::default(),
        hashes: Hashes::default(),
        cost: Default::default(),
    }
}

async fn seeded_store() -> Store {
    let store = Store::open_in_memory().await.unwrap();
    store
        .create_run("run-1", "{\"budget\":25}", Some(25.0))
        .await
        .unwrap();
    store
}

#[tokio::test]
async fn put_is_idempotent_upsert() {
    let store = seeded_store().await;
    let rec = record("rec-1", "run-1", Some(Verdict::Admit), Some(0.9));
    store.put(&rec).await.unwrap();
    // Re-put the same record_id (re-processed seed) — must upsert, not duplicate.
    store.put(&rec).await.unwrap();
    let all = store.scan(&RecordFilter::new()).await.unwrap();
    assert_eq!(all.len(), 1, "re-put must not duplicate a record");
    let got = store.get("rec-1").await.unwrap();
    // put() populates the envelope's hashes (PRE-MERGE 1); everything else round-trips intact.
    let mut expected = rec.clone();
    expected.hashes.record_hash = gw_storage::record_hash(&rec).unwrap();
    expected.hashes.prompt_hash = gw_storage::prompt_hash(&rec.messages).unwrap();
    expected.hashes.completion_hash = gw_storage::completion_hash(&rec.messages).unwrap();
    assert_eq!(got, expected);
}

#[tokio::test]
async fn get_missing_is_not_found() {
    let store = seeded_store().await;
    let err = store.get("nope").await.unwrap_err();
    assert!(matches!(err, gw_storage::StorageError::NotFound(_)));
}

#[tokio::test]
async fn advance_lifecycle_updates_state_and_appends_history() {
    let store = seeded_store().await;
    store
        .put(&record("rec-1", "run-1", None, None))
        .await
        .unwrap();
    store
        .advance_lifecycle("rec-1", LifecycleState::Verified, Some("verifier ok"))
        .await
        .unwrap();
    store
        .advance_lifecycle("rec-1", LifecycleState::Admitted, None)
        .await
        .unwrap();

    // State column reflects the latest transition.
    let scanned = store
        .scan(&RecordFilter::new().lifecycle_state(LifecycleState::Admitted))
        .await
        .unwrap();
    assert_eq!(scanned.len(), 1);

    // History is event-sourced: two rows, in order, with details preserved.
    let hist = store.lifecycle_history("rec-1").await.unwrap();
    assert_eq!(hist.len(), 2);
    assert_eq!(hist[0].0, "verified");
    assert_eq!(hist[0].2.as_deref(), Some("verifier ok"));
    assert_eq!(hist[1].0, "admitted");
    assert_eq!(hist[1].2, None);

    // The stored ENVELOPE is kept in sync with the column: get() reflects the new state and its
    // own lifecycle.history grew alongside the lifecycle_history table.
    let got = store.get("rec-1").await.unwrap();
    assert_eq!(got.lifecycle.state, LifecycleState::Admitted);
    assert_eq!(got.lifecycle.history.len(), 2);
    assert_eq!(got.lifecycle.history[0].state, LifecycleState::Verified);
    assert_eq!(got.lifecycle.history[1].state, LifecycleState::Admitted);
    assert_eq!(got.lifecycle.attempts, 2);

    // A scan filtered by the new state returns an envelope that AGREES with the filter.
    assert_eq!(scanned[0].lifecycle.state, LifecycleState::Admitted);
}

#[tokio::test]
async fn advance_lifecycle_to_error_records_detail_in_envelope() {
    let store = seeded_store().await;
    store
        .put(&record("rec-e", "run-1", None, None))
        .await
        .unwrap();
    store
        .advance_lifecycle("rec-e", LifecycleState::Error, Some("teacher 500"))
        .await
        .unwrap();
    let got = store.get("rec-e").await.unwrap();
    assert_eq!(got.lifecycle.state, LifecycleState::Error);
    assert_eq!(got.lifecycle.error.as_deref(), Some("teacher 500"));
}

#[tokio::test]
async fn advance_lifecycle_missing_record_is_not_found() {
    let store = seeded_store().await;
    let err = store
        .advance_lifecycle("ghost", LifecycleState::Verified, None)
        .await
        .unwrap_err();
    assert!(matches!(err, gw_storage::StorageError::NotFound(_)));
    // The failed advance left no orphan history row.
    assert!(store.lifecycle_history("ghost").await.unwrap().is_empty());
}

#[tokio::test]
async fn scan_filters_by_verdict_and_min_aggregate() {
    let store = seeded_store().await;
    store
        .put(&record(
            "admit-hi",
            "run-1",
            Some(Verdict::Admit),
            Some(0.95),
        ))
        .await
        .unwrap();
    store
        .put(&record(
            "admit-lo",
            "run-1",
            Some(Verdict::Admit),
            Some(0.70),
        ))
        .await
        .unwrap();
    store
        .put(&record(
            "reject",
            "run-1",
            Some(Verdict::Reject),
            Some(0.20),
        ))
        .await
        .unwrap();

    let admits = store
        .scan(&RecordFilter::new().verdict(Verdict::Admit))
        .await
        .unwrap();
    assert_eq!(admits.len(), 2);

    let high = store
        .scan(&RecordFilter::new().min_judge_aggregate(0.8))
        .await
        .unwrap();
    assert_eq!(high.len(), 1);
    assert_eq!(high[0].record_id, "admit-hi");

    let admit_and_high = store
        .scan(
            &RecordFilter::new()
                .verdict(Verdict::Admit)
                .min_judge_aggregate(0.9),
        )
        .await
        .unwrap();
    assert_eq!(admit_and_high.len(), 1);
    assert_eq!(admit_and_high[0].record_id, "admit-hi");
}

#[tokio::test]
async fn min_aggregate_excludes_null_aggregate_records() {
    // E: an Admit record whose judging.aggregate is None (NULL column) must be EXCLUDED by
    // min_judge_aggregate — locks in the NULL-safe `judge_aggregate >= ?` contract.
    let store = seeded_store().await;
    store
        .put(&record("admit-null", "run-1", Some(Verdict::Admit), None))
        .await
        .unwrap();
    store
        .put(&record(
            "admit-scored",
            "run-1",
            Some(Verdict::Admit),
            Some(0.5),
        ))
        .await
        .unwrap();

    let filtered = store
        .scan(&RecordFilter::new().min_judge_aggregate(0.0))
        .await
        .unwrap();
    // Only the scored record qualifies; the NULL-aggregate one is excluded even at floor 0.0.
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].record_id, "admit-scored");
}

#[tokio::test]
async fn scan_filters_by_run() {
    let store = seeded_store().await;
    store.create_run("run-2", "{}", None).await.unwrap();
    store.put(&record("a", "run-1", None, None)).await.unwrap();
    store.put(&record("b", "run-2", None, None)).await.unwrap();
    let r1 = store
        .scan(&RecordFilter::new().run_id("run-1"))
        .await
        .unwrap();
    assert_eq!(r1.len(), 1);
    assert_eq!(r1[0].record_id, "a");
}

#[tokio::test]
async fn cache_get_put_round_trip() {
    let store = seeded_store().await;
    let value = serde_json::json!({"content": "cached teacher output", "tokens": 42});
    assert!(
        store
            .cache_get("hash-abc", "teacher", "z-ai/glm-5.2", None)
            .await
            .unwrap()
            .is_none()
    );
    store
        .cache_put("hash-abc", "teacher", "z-ai/glm-5.2", None, &value)
        .await
        .unwrap();
    let got = store
        .cache_get("hash-abc", "teacher", "z-ai/glm-5.2", None)
        .await
        .unwrap();
    assert_eq!(got, Some(value));

    // Distinct rubric_id is a distinct cache key.
    assert!(
        store
            .cache_get("hash-abc", "judge", "z-ai/glm-5.2", Some("rubric-7"))
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn record_hash_is_content_only_allowlist() {
    // The record_hash is a POSITIVE content-only projection: NO non-content field may move it.
    let base = record("id-A", "run-1", Some(Verdict::Admit), Some(0.9));
    let h0 = gw_storage::record_hash(&base).unwrap();

    // Clone and mutate every non-content field the denylist used to leak.
    let mut m = base.clone();
    m.record_id = "id-B".into();
    m.dataset_version = Some(semver::Version::new(0, 1, 0));
    m.provenance.run_id = "run-999".into();
    m.provenance.teacher.served_by = Some("Wafer".into());
    m.provenance.harness_version = "9.9.9".into();
    m.provenance.git_commit = Some("deadbeef".into());
    if let Some(vote) = m.judging.panel.first_mut() {
        vote.raw_response = Some("a totally different raw judge response".into());
        vote.seed = Some(12345);
        vote.temperature = Some(0.7);
    }
    m.verification.checks.push(gw_schema::Check {
        name: "rust_compiles".into(),
        kind: gw_schema::CheckKind::Compile,
        passed: true,
        score: None,
        detail: Some("compiled in 1.2s".into()),
    });
    m.verification.all_passed = true;
    m.cost.usd = 1.23;
    m.cost.reasoning_tokens = 999;
    m.generation.completion_index = Some(3);
    m.generation.sibling_group_id = Some("sib-7".into());
    m.lifecycle.state = LifecycleState::Exported;
    m.lifecycle.attempts = 5;
    m.lifecycle.history.push(gw_schema::StateTransition {
        state: LifecycleState::Verified,
        at: "2026-06-21T00:00:00Z".into(),
        attempt: 1,
    });
    m.hashes.record_hash = "stale".into();

    assert_eq!(
        gw_storage::record_hash(&m).unwrap(),
        h0,
        "no non-content field may change the record hash"
    );

    // A record advancing through its lifecycle keeps a STABLE hash (regression for the old leak).
    let store = seeded_store().await;
    store.put(&base).await.unwrap();
    let before = store.get("id-A").await.unwrap().hashes.record_hash.clone();
    store
        .advance_lifecycle("id-A", LifecycleState::Verified, None)
        .await
        .unwrap();
    let after = store.get("id-A").await.unwrap().hashes.record_hash.clone();
    assert_eq!(
        before, after,
        "record_hash must not change as lifecycle advances"
    );
    assert_eq!(before, h0);

    // Content DOES move the hash: a different answer, and (separately) different reasoning.
    let mut diff_content = base.clone();
    if let Some(msg) = diff_content.messages.last_mut() {
        msg.content = Content::Text("97".into());
    }
    assert_ne!(gw_storage::record_hash(&diff_content).unwrap(), h0);

    let mut diff_reasoning = base.clone();
    if let Some(msg) = diff_reasoning.messages.last_mut() {
        msg.reasoning = Some("an entirely different chain of thought".into());
    }
    assert_ne!(
        gw_storage::record_hash(&diff_reasoning).unwrap(),
        h0,
        "the flat reasoning text is content — different CoT must change record_hash"
    );

    // completion_hash still ignores reasoning (same answer, different CoT → same completion hash).
    assert_eq!(
        gw_storage::completion_hash(&base.messages).unwrap(),
        gw_storage::completion_hash(&diff_reasoning.messages).unwrap(),
    );
}

#[tokio::test]
async fn record_hash_distinguishes_reasoning_details_text() {
    // reasoning_details is VERBATIM CoT content: a record carrying its whole CoT in
    // reasoning_details[].text with flat reasoning=None must NOT collide with a different one.
    let mut a = record("rd-a", "run-1", Some(Verdict::Admit), Some(0.9));
    if let Some(m) = a.messages.last_mut() {
        m.reasoning = None;
        m.reasoning_details = Some(vec![ReasoningDetail::Text {
            text: "first structured chain of thought".into(),
            signature: None,
            id: Some("id-1".into()),
            format: Some("anthropic-claude-v1".into()),
            index: 0,
        }]);
    }
    let mut b = a.clone();
    if let Some(m) = b.messages.last_mut() {
        m.reasoning_details = Some(vec![ReasoningDetail::Text {
            text: "an ENTIRELY DIFFERENT structured chain of thought".into(),
            signature: None,
            id: Some("id-1".into()),
            format: Some("anthropic-claude-v1".into()),
            index: 0,
        }]);
    }
    assert_ne!(
        gw_storage::record_hash(&a).unwrap(),
        gw_storage::record_hash(&b).unwrap(),
        "different reasoning_details[].text must change record_hash"
    );

    // But the volatile per-detail id/index/signature/format must NOT change the hash.
    let mut c = a.clone();
    if let Some(m) = c.messages.last_mut() {
        m.reasoning_details = Some(vec![ReasoningDetail::Text {
            text: "first structured chain of thought".into(),
            signature: Some("sig-xyz".into()),
            id: Some("id-99".into()),
            format: Some("openai-o1".into()),
            index: 7,
        }]);
    }
    assert_eq!(
        gw_storage::record_hash(&a).unwrap(),
        gw_storage::record_hash(&c).unwrap(),
        "volatile reasoning_details ids/index/signature/format must not move the hash"
    );
}

#[tokio::test]
async fn record_hash_distinguishes_message_name() {
    // name = speaker / tool name; it is content. Distinct names must not collide.
    let a = record("nm-a", "run-1", Some(Verdict::Admit), Some(0.9));
    let mut b = a.clone();
    if let Some(m) = b.messages.last_mut() {
        m.name = Some("alice".into());
    }
    assert_ne!(
        gw_storage::record_hash(&a).unwrap(),
        gw_storage::record_hash(&b).unwrap(),
        "a differing message name must change record_hash"
    );
}

#[tokio::test]
async fn put_recomputes_and_overwrites_wrong_caller_hash() {
    // B: the store is authoritative — a deliberately WRONG non-empty caller hash is overwritten
    // by the content hash in BOTH the stored envelope and the indexed column.
    let store = seeded_store().await;
    let mut rec = record("rec-w", "run-1", Some(Verdict::Admit), Some(0.9));
    rec.hashes.record_hash = "deadbeefwronghash".into();
    rec.hashes.prompt_hash = "alsowrong".into();
    rec.hashes.completion_hash = "stillwrong".into();
    let truth = gw_storage::record_hash(&rec).unwrap();
    assert_ne!(truth, "deadbeefwronghash");

    store.put(&rec).await.unwrap();
    let got = store.get("rec-w").await.unwrap();
    assert_eq!(
        got.hashes.record_hash, truth,
        "envelope hash must be recomputed"
    );
    assert_ne!(got.hashes.record_hash, "deadbeefwronghash");

    let col: (String,) = sqlx_query_record_hash(&store, "rec-w").await;
    assert_eq!(col.0, truth, "indexed column must be recomputed too");
}

#[tokio::test]
async fn put_populates_hashes_in_envelope_and_columns() {
    // PRE-MERGE 1: a record put with EMPTY Hashes comes back with populated hashes that equal
    // the indexed projection columns.
    let store = seeded_store().await;
    let rec = record("rec-h", "run-1", Some(Verdict::Admit), Some(0.9));
    assert!(rec.hashes.record_hash.is_empty());
    store.put(&rec).await.unwrap();

    let got = store.get("rec-h").await.unwrap();
    assert!(
        !got.hashes.record_hash.is_empty(),
        "envelope record_hash must be populated"
    );
    assert!(!got.hashes.prompt_hash.is_empty());
    assert!(!got.hashes.completion_hash.is_empty());
    assert_eq!(
        got.hashes.record_hash,
        gw_storage::record_hash(&rec).unwrap()
    );

    // The envelope's record_hash equals the indexed column.
    let col: (String,) = sqlx_query_record_hash(&store, "rec-h").await;
    assert_eq!(col.0, got.hashes.record_hash);
}

/// Read the indexed `record_hash` column directly (round-trip cross-check for the test above).
async fn sqlx_query_record_hash(store: &Store, id: &str) -> (String,) {
    use sqlx::Row;
    let row = sqlx::query("SELECT record_hash FROM records WHERE record_id = ?1")
        .bind(id)
        .fetch_one(store.raw_pool())
        .await
        .unwrap();
    (row.get::<String, _>("record_hash"),)
}

#[tokio::test]
async fn parquet_export_round_trips() {
    let recs = vec![
        record("admit-1", "run-1", Some(Verdict::Admit), Some(0.95)),
        record("admit-2", "run-1", Some(Verdict::Admit), Some(0.88)),
        record("reject-1", "run-1", Some(Verdict::Reject), Some(0.10)),
    ];
    let (bytes, manifest) = gw_storage::export_parquet_bytes(
        &recs,
        TrlFormat::ChatML,
        gw_schema::CotPolicy::Supervised,
    )
    .await
    .unwrap();

    assert_eq!(manifest.n_records, 3);
    assert_eq!(manifest.n_admitted, 2, "only admitted records are written");
    assert_eq!(manifest.target, TrlFormat::ChatML);
    assert_eq!(manifest.cot_policy, gw_schema::CotPolicy::Supervised);
    assert_eq!(
        manifest.column_schema_version,
        gw_schema::ExportSchemaVersion::CURRENT,
        "the manifest names the shard's column contract"
    );
    assert!(!manifest.build_inputs_hash.is_empty());

    // Read the parquet bytes back and assert row count + columns.
    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::from(bytes))
        .unwrap()
        .build()
        .unwrap();
    let mut total = 0usize;
    let mut saw_columns = false;
    for batch in reader {
        let batch = batch.unwrap();
        total += batch.num_rows();
        let schema = batch.schema();
        assert!(schema.field_with_name("record_id").is_ok());
        assert!(schema.field_with_name("record_hash").is_ok());
        assert!(schema.field_with_name("messages_json").is_ok());
        assert!(
            schema.field_with_name("reasoning_json").is_err(),
            "the parallel reasoning column is gone: it could only agree with messages_json by index"
        );
        assert!(schema.field_with_name("judge_aggregate").is_ok());
        // The record_id column should hold the two admitted ids.
        let ids = batch
            .column_by_name("record_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let values: Vec<&str> = (0..ids.len()).map(|i| ids.value(i)).collect();
        assert!(values.contains(&"admit-1"));
        assert!(values.contains(&"admit-2"));
        assert!(!values.contains(&"reject-1"));

        // PRE-MERGE 2: the input records were built with EMPTY hashes; the export must still
        // write a content-derived record_hash per row (compute-if-empty fallback), never "".
        let hashes = batch
            .column_by_name("record_hash")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..hashes.len() {
            assert!(
                !hashes.value(i).is_empty(),
                "exported record_hash must be non-empty"
            );
        }
        saw_columns = true;
    }
    assert_eq!(total, 2);
    assert!(saw_columns);
}

#[tokio::test]
async fn export_content_hash_is_order_independent() {
    let a = record("x", "run-1", Some(Verdict::Admit), Some(0.9));
    let b = record("y", "run-1", Some(Verdict::Admit), Some(0.8));
    // Give them stable record hashes so the shard hash is meaningful.
    let mut a = a;
    let mut b = b;
    a.hashes.record_hash = gw_storage::record_hash(&a).unwrap();
    b.hashes.record_hash = gw_storage::record_hash(&b).unwrap();

    let (_, m1) = gw_storage::export_parquet_bytes(
        &[a.clone(), b.clone()],
        TrlFormat::ChatML,
        gw_schema::CotPolicy::Supervised,
    )
    .await
    .unwrap();
    let (_, m2) = gw_storage::export_parquet_bytes(
        &[b, a],
        TrlFormat::ChatML,
        gw_schema::CotPolicy::Supervised,
    )
    .await
    .unwrap();
    assert_eq!(m1.build_inputs_hash, m2.build_inputs_hash);
}

#[tokio::test]
async fn resume_cursor_round_trips() {
    let store = seeded_store().await;
    // No checkpoint yet → None.
    assert!(store.resume_cursor("run-1", 0).await.unwrap().is_none());

    let cursor = serde_json::json!({"seed_offset": 4096});
    store
        .checkpoint("run-1", 0, "assistant_generated", &cursor)
        .await
        .unwrap();
    let got = store.resume_cursor("run-1", 0).await.unwrap();
    assert_eq!(
        got,
        Some(ResumePoint {
            state: "assistant_generated".into(),
            cursor: cursor.clone(),
        })
    );

    // Re-checkpointing the same shard advances it (upsert, not a second row).
    let cursor2 = serde_json::json!({"seed_offset": 8192});
    store
        .checkpoint("run-1", 0, "judged", &cursor2)
        .await
        .unwrap();
    let got2 = store.resume_cursor("run-1", 0).await.unwrap().unwrap();
    assert_eq!(got2.state, "judged");
    assert_eq!(got2.cursor, cursor2);
}

#[tokio::test]
async fn run_status_lifecycle() {
    let store = seeded_store().await;
    assert_eq!(
        store.run_status("run-1").await.unwrap().as_deref(),
        Some("running")
    );
    store
        .set_run_status("run-1", RunStatus::Completed)
        .await
        .unwrap();
    assert_eq!(
        store.run_status("run-1").await.unwrap().as_deref(),
        Some("completed")
    );
}

#[tokio::test]
async fn foreign_key_enforced_on_record_put() {
    let store = Store::open_in_memory().await.unwrap();
    // No run created → putting a record that references a missing run violates the FK.
    let err = store
        .put(&record("orphan", "missing-run", None, None))
        .await
        .unwrap_err();
    assert!(matches!(err, gw_storage::StorageError::Sqlx(_)));
}

#[tokio::test]
async fn scan_stream_yields_records() {
    use futures::StreamExt;
    let store = seeded_store().await;
    for i in 0..3 {
        store
            .put(&record(&format!("rec-{i}"), "run-1", None, None))
            .await
            .unwrap();
    }
    let stream = store.scan_stream(&RecordFilter::new());
    let collected: Vec<_> = stream.collect().await;
    assert_eq!(collected.len(), 3);
    assert!(collected.iter().all(|r| r.is_ok()));
}

/// Keep an `Arc<StringArray>` import path exercised so the dev-dep arrow array surface is used.
#[test]
fn arrow_dev_dep_present() {
    let arr = Arc::new(StringArray::from(vec!["a", "b"]));
    assert_eq!(arr.len(), 2);
}
