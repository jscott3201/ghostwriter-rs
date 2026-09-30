//! The actual verification step cannot commit its envelope independently of its transition.
mod common;
use common::*;
use gw_engine::{EventSink, step};
use gw_schema::TrainingRecord;
use gw_storage::Store;
use std::sync::Arc;

#[tokio::test]
async fn verification_history_failure_rolls_back_the_complete_envelope() {
    let store = Store::open_in_memory().await.unwrap();
    store
        .insert_historical_run("run", "{}", None)
        .await
        .unwrap();
    let mut initial: TrainingRecord = serde_json::from_value(serde_json::json!({
        "record_id":"record", "schema_version":"1.0.0", "training_area":"fixture",
        "messages":[{"role":"user","content":"2+3"},{"role":"assistant","content":"5","reasoning":"addition"}],
        "provenance":{"run_id":"run","teacher":{"provider":"fixture","slug":"fixture"},"harness_version":"fixture"},
        "generation":{}, "lifecycle":{"state":"assistant_generated"}
    })).unwrap();
    initial.verification_contract = Some(good_candidate("2+3").contract);
    store.replace_record_for_import(&initial).await.unwrap();
    let before = store.get("record").await.unwrap();
    sqlx::query("CREATE TRIGGER reject_history BEFORE INSERT ON lifecycle_history BEGIN SELECT RAISE(ABORT, 'injected history failure'); END")
        .execute(store.raw_pool()).await.unwrap();
    let clients = clients(
        store.clone(),
        Arc::new(ScriptedTeacher::new(vec![], 0)),
        Arc::new(ScriptedJudge::new(vec![])),
        EventSink::disconnected(),
    );
    let result = step(
        before.clone(),
        &clients,
        &area_k1(one_judge(), lenient_thresholds()),
    )
    .await;
    assert!(result.is_err());
    assert_eq!(store.get("record").await.unwrap(), before);
    assert!(store.lifecycle_history("record").await.unwrap().is_empty());
}

#[tokio::test]
async fn committed_verification_retry_after_publication_returns_current_without_event() {
    use gw_engine::{EngineEvent, drive};
    use gw_schema::{CotPolicy, ExportOptions, ExportScope, LifecycleState, TrlFormat};
    use tokio_util::sync::CancellationToken;
    let store = Store::open_in_memory().await.unwrap();
    store
        .insert_historical_run("run", "{}", None)
        .await
        .unwrap();
    let mut initial:TrainingRecord=serde_json::from_value(serde_json::json!({
        "record_id":"record","schema_version":"1.0.0","training_area":"math",
        "messages":[{"role":"user","content":"2+3"},{"role":"assistant","content":"5","reasoning":"addition","reasoning_details":[{"type":"reasoning.text","text":"addition","index":0}]}],
        "provenance":{"run_id":"run","teacher":{"provider":"fixture","slug":"fixture"},"harness_version":"fixture"},
        "generation":{},"lifecycle":{"state":"assistant_generated"}
    })).unwrap();
    initial.verification_contract = Some(good_candidate("2+3").contract);
    initial.cost.reasoning_tokens = 10;
    let original = store.insert_record(&initial).await.unwrap().record;
    let (sink, mut events) = EventSink::subscribe();
    let body = judge_body(0.95, "accept");
    let clients = clients(
        store.clone(),
        Arc::new(ScriptedTeacher::new(vec![], 0)),
        Arc::new(ScriptedJudge::new(vec![&body])),
        sink,
    );
    let area = area_k1(one_judge(), lenient_thresholds());
    let verified = step(original.clone(), &clients, &area).await.unwrap();
    let ready = drive(verified, &clients, &area, &CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(ready.lifecycle.state, LifecycleState::Formatted);
    let dst = std::env::temp_dir().join(format!(
        "gw-retry-publication-{}.parquet",
        std::process::id()
    ));
    store
        .publish_export(
            ExportOptions {
                target: TrlFormat::ChatML,
                cot_policy: CotPolicy::Masked,
                dataset_version: None,
                scope: ExportScope::Run {
                    run_id: "run".into(),
                },
            },
            &dst,
            gw_storage::ExportPurpose::Engine,
        )
        .await
        .unwrap();
    let published = store.get("record").await.unwrap();
    assert_eq!(published.lifecycle.state, LifecycleState::Exported);
    let history = store.lifecycle_history("record").await.unwrap();
    while events.try_recv().is_ok() {}
    assert_eq!(step(original, &clients, &area).await.unwrap(), published);
    assert_eq!(store.lifecycle_history("record").await.unwrap(), history);
    while let Ok(event) = events.try_recv() {
        assert!(!matches!(event, EngineEvent::StateAdvanced { .. }));
    }
    std::fs::remove_file(dst).unwrap();
}
