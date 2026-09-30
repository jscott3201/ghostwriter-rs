//! Publication is the only operation that can acknowledge records as exported.

mod common;

use common::*;
use gw_engine::{Engine, EngineEvent, EventSink, ExportSpec, RunReport, load_cursor};
use gw_schema::{CotPolicy, LifecycleState, TrlFormat};
use gw_storage::{RecordFilter, Store};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio_util::sync::CancellationToken;

fn path(label: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "gw-publication-{}-{}-{label}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

fn spec(dst: PathBuf) -> ExportSpec {
    ExportSpec {
        dst,
        target: TrlFormat::ChatML,
        cot: CotPolicy::Masked,
        dataset_version: None,
    }
}

fn engine(store: &Store, sink: EventSink, accept: bool) -> Engine {
    let teacher = Arc::new(ScriptedTeacher::new(vec![good_cot(0.01)], 1));
    let body = judge_body(
        if accept { 0.95 } else { 0.1 },
        if accept { "accept" } else { "reject" },
    );
    let judge = Arc::new(ScriptedJudge::new(vec![&body]));
    Engine::new(
        clients(store.clone(), teacher, judge, 25.0, sink),
        area_k1(one_judge(), lenient_thresholds()),
        1,
    )
}

#[tokio::test]
async fn no_output_leaves_formatted_readiness_and_zero_exported() {
    let store = Store::open_in_memory().await.unwrap();
    let report = engine(&store, EventSink::disconnected(), true)
        .run("ready", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();
    assert!(report.completed);
    assert_eq!(report.admitted, 1);
    assert_eq!(
        report.exported, 0,
        "readiness is not an artifact acknowledgment"
    );
    let records = store
        .scan(&RecordFilter::new().run_id("ready"))
        .await
        .unwrap();
    assert_eq!(records[0].lifecycle.state, LifecycleState::Formatted);
}

async fn resume_without_output(store: &Store, run_id: &str) -> RunReport {
    let records = store
        .scan(&RecordFilter::new().run_id(run_id))
        .await
        .unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].lifecycle.state, LifecycleState::Exported);
    assert_eq!(load_cursor(store, run_id, 0).await.unwrap().next_offset, 1);
    let history = store
        .lifecycle_history(&records[0].record_id)
        .await
        .unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(vec![], 0));
    let judge = Arc::new(ScriptedJudge::new(vec![]));
    let resumed = Engine::new(
        clients(
            store.clone(),
            teacher.clone(),
            judge.clone(),
            25.0,
            EventSink::disconnected(),
        ),
        area_k1(one_judge(), lenient_thresholds()),
        1,
    )
    .run(run_id, &one_item_source(), CancellationToken::new())
    .await
    .unwrap();
    assert!(resumed.completed);
    assert_eq!(resumed.admitted, 1);
    assert_eq!(teacher.call_count(), 0);
    assert_eq!(judge.call_count(), 0);
    assert_eq!(store.get(&records[0].record_id).await.unwrap(), records[0]);
    assert_eq!(
        store
            .lifecycle_history(&records[0].record_id)
            .await
            .unwrap(),
        history
    );
    resumed
}

#[tokio::test]
async fn resume_without_output_does_not_count_legacy_exported_readiness() {
    let store = Store::open_in_memory().await.unwrap();
    engine(&store, EventSink::disconnected(), true)
        .run("legacy", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();
    let record = store
        .scan(&RecordFilter::new().run_id("legacy"))
        .await
        .unwrap()
        .remove(0);
    // Old engines advanced readiness to Exported without publishing an artifact.
    store
        .advance_lifecycle(
            &record.record_id,
            LifecycleState::Exported,
            Some("legacy readiness-only export"),
        )
        .await
        .unwrap();
    let (receipts,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM export_receipts")
        .fetch_one(store.raw_pool())
        .await
        .unwrap();
    assert_eq!(receipts, 0);

    let report = resume_without_output(&store, "legacy").await;
    assert_eq!(report.exported, 0, "this invocation published no artifact");
}

#[tokio::test]
async fn resume_without_output_does_not_recount_previously_acknowledged_records() {
    let store = Store::open_in_memory().await.unwrap();
    let dst = path("previously-acknowledged.parquet");
    let first = engine(&store, EventSink::disconnected(), true)
        .with_export(spec(dst.clone()))
        .run("acknowledged", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(first.exported, 1);
    let (acknowledged,): (i64,) =
        sqlx::query_as("SELECT COUNT(*) FROM export_receipts WHERE state = 'acknowledged'")
            .fetch_one(store.raw_pool())
            .await
            .unwrap();
    assert_eq!(acknowledged, 1);
    let bytes = std::fs::read(&dst).unwrap();

    let report = resume_without_output(&store, "acknowledged").await;
    let after = std::fs::read(&dst).unwrap();
    std::fs::remove_file(dst).unwrap();
    assert_eq!(after, bytes);
    assert_eq!(
        report.exported, 0,
        "the prior publication is not this run's export"
    );
}

#[tokio::test]
async fn configured_empty_export_replaces_old_destination() {
    let dst = path("empty.parquet");
    std::fs::write(&dst, b"stale artifact").unwrap();
    let store = Store::open_in_memory().await.unwrap();
    let (sink, mut events) = EventSink::subscribe();
    let report = engine(&store, sink, false)
        .with_export(spec(dst.clone()))
        .run("empty", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();
    let bytes = std::fs::read(&dst).unwrap();
    std::fs::remove_file(dst).unwrap();
    assert!(report.completed);
    assert_eq!(report.admitted, 0);
    assert_eq!(
        &bytes[..4],
        b"PAR1",
        "an empty result must replace stale output"
    );
    let mut published = false;
    while let Ok(event) = events.try_recv() {
        if let EngineEvent::ShardExported { manifest, .. } = event {
            assert_eq!(manifest.n_records, 1);
            assert_eq!(manifest.n_admitted, 0);
            published = true;
        }
    }
    assert!(published);
}

#[tokio::test]
async fn failed_export_returns_error_and_failed_run_without_exported_records() {
    let dst = path("missing").join("out.parquet");
    let store = Store::open_in_memory().await.unwrap();
    let (sink, mut events) = EventSink::subscribe();
    let result = engine(&store, sink, true)
        .with_export(spec(dst))
        .run("failure", &one_item_source(), CancellationToken::new())
        .await;
    assert!(result.is_err(), "publication failure must reach the caller");
    assert_eq!(
        store.run_status("failure").await.unwrap().as_deref(),
        Some("failed")
    );
    let records = store
        .scan(&RecordFilter::new().run_id("failure"))
        .await
        .unwrap();
    assert_eq!(records[0].lifecycle.state, LifecycleState::Formatted);
    let mut failed = false;
    while let Ok(event) = events.try_recv() {
        match event {
            EngineEvent::ShardExportFailed { .. } => failed = true,
            EngineEvent::ShardExported { .. }
            | EngineEvent::RunFinished {
                completed: true, ..
            } => panic!("false success event"),
            _ => {}
        }
    }
    assert!(failed);
}

#[tokio::test]
async fn acknowledgment_failure_leaves_running_until_failure_and_replay_never_respends() {
    let dst = path("ack-retry.parquet");
    let store = Store::open_in_memory().await.unwrap();
    sqlx::query("CREATE TRIGGER fail_publication_ack BEFORE UPDATE ON export_receipts WHEN NEW.state = 'acknowledged' BEGIN \
        SELECT CASE WHEN (SELECT status FROM runs WHERE run_id = 'retry') != 'running' THEN RAISE(FAIL, 'premature run completion') END; \
        SELECT RAISE(FAIL, 'publication acknowledgment primary failure'); END")
        .execute(store.raw_pool()).await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(vec![good_cot(0.01)], 1));
    let judge = Arc::new(ScriptedJudge::new(vec![&judge_body(0.95, "accept")]));
    let (sink, mut events) = EventSink::subscribe();
    let engine = Engine::new(
        clients(store.clone(), teacher.clone(), judge.clone(), 25.0, sink),
        area_k1(one_judge(), lenient_thresholds()),
        1,
    )
    .with_export(spec(dst.clone()));
    let error = engine
        .run("retry", &one_item_source(), CancellationToken::new())
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("publication acknowledgment primary failure")
    );
    assert_eq!(
        store.run_status("retry").await.unwrap().as_deref(),
        Some("failed")
    );
    let gw_storage::ArtifactVerification::Verified(artifact) =
        gw_storage::verify_artifact(&dst).unwrap()
    else {
        panic!("artifact must be valid before acknowledgment");
    };
    let records = store
        .scan(&RecordFilter::new().run_id("retry"))
        .await
        .unwrap();
    assert_eq!(records[0].lifecycle.state, LifecycleState::Formatted);
    while let Ok(event) = events.try_recv() {
        assert!(!matches!(
            event,
            EngineEvent::ShardExported { .. }
                | EngineEvent::RunFinished {
                    completed: true,
                    ..
                }
                | EngineEvent::StateAdvanced {
                    to: LifecycleState::Exported,
                    ..
                }
        ));
    }
    sqlx::query("DROP TRIGGER fail_publication_ack")
        .execute(store.raw_pool())
        .await
        .unwrap();
    let bytes = std::fs::read(&dst).unwrap();
    let report = engine
        .run("retry", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();
    assert!(report.completed);
    assert_eq!(report.exported, 1);
    assert_eq!(teacher.call_count(), 1);
    assert_eq!(judge.call_count(), 1);
    assert_eq!(std::fs::read(&dst).unwrap(), bytes);
    assert_eq!(
        store.run_status("retry").await.unwrap().as_deref(),
        Some("completed")
    );
    let export_history: Vec<_> = store
        .lifecycle_history(&records[0].record_id)
        .await
        .unwrap()
        .into_iter()
        .filter(|entry| entry.0 == "exported")
        .collect();
    assert_eq!(export_history.len(), 1);
    assert_eq!(
        export_history[0].2.as_deref(),
        Some(format!("artifact:{}", artifact.artifact_id).as_str())
    );
    let mut published = false;
    while let Ok(event) = events.try_recv() {
        if matches!(event, EngineEvent::ShardExported { .. }) {
            published = true;
        }
        if matches!(
            event,
            EngineEvent::RunFinished {
                completed: true,
                ..
            }
        ) {
            assert!(published);
        }
    }
    assert!(published);
    std::fs::remove_file(dst).unwrap();
}

#[tokio::test]
async fn publication_primary_error_survives_failed_status_finalizer() {
    let store = Store::open_in_memory().await.unwrap();
    sqlx::query("CREATE TRIGGER fail_failed_status BEFORE UPDATE ON runs WHEN NEW.status = 'failed' BEGIN SELECT RAISE(FAIL, 'secondary status failure'); END")
        .execute(store.raw_pool()).await.unwrap();
    let (sink, mut events) = EventSink::subscribe();
    let missing = path("nonexistent").join("output.parquet");
    let error = engine(&store, sink, true)
        .with_export(spec(missing))
        .run("primary", &one_item_source(), CancellationToken::new())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("io error"));
    assert!(!error.to_string().contains("secondary"));
    let mut terminal = 0;
    while let Ok(event) = events.try_recv() {
        if let EngineEvent::RunFinished { completed, .. } = event {
            assert!(!completed);
            terminal += 1;
        }
    }
    assert_eq!(terminal, 1);
}
