//! Replay semantics are pinned before launch state or model work changes.
mod common;
use common::*;
use gw_engine::{Engine, EventSink};
use gw_storage::Store;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn semantic_drift_is_rejected_before_cancelled_replay_mutates_the_run() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(vec![], 0));
    let judge = Arc::new(ScriptedJudge::new(vec![]));
    let clients = clients(
        store.clone(),
        teacher.clone(),
        judge.clone(),
        EventSink::disconnected(),
    );
    let area = area_k1(one_judge(), lenient_thresholds());
    let cancel = CancellationToken::new();
    cancel.cancel();
    Engine::new(clients.clone(), area.clone(), 1)
        .run("pinned", &one_item_source(), cancel.clone())
        .await
        .unwrap();
    let before: (String, String, String) =
        sqlx::query_as("SELECT config_json, created_at, status FROM runs WHERE run_id='pinned'")
            .fetch_one(store.raw_pool())
            .await
            .unwrap();
    let mut changed = area;
    changed.teacher_slug = "different-teacher".into();
    let replay = Engine::new(clients, changed, 1)
        .run("pinned", &one_item_source(), cancel)
        .await;
    assert!(
        replay.is_err(),
        "a cancelled or cache-only relaunch still requires compatible meaning"
    );
    let after: (String, String, String) =
        sqlx::query_as("SELECT config_json, created_at, status FROM runs WHERE run_id='pinned'")
            .fetch_one(store.raw_pool())
            .await
            .unwrap();
    assert_eq!(before, after);
    assert_eq!(teacher.call_count(), 0);
    assert_eq!(judge.call_count(), 0);
}

#[tokio::test]
async fn compatible_partial_replay_reuses_persisted_work_and_preserves_original_bytes() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(vec![], 2));
    let judge = Arc::new(ScriptedJudge::new(vec![
        &judge_body(0.95, "accept"),
        &judge_body(0.95, "accept"),
    ]));
    let mut clients = clients(
        store.clone(),
        teacher.clone(),
        judge.clone(),
        EventSink::disconnected(),
    );
    let area = area_k1(one_judge(), lenient_thresholds());
    let source = gw_engine::InMemorySeedSource::new(
        vec![
            good_candidate("first question"),
            good_candidate("second question"),
        ],
        1,
    );
    sqlx::query("CREATE TRIGGER stop_before_checkpoint BEFORE INSERT ON checkpoints BEGIN SELECT RAISE(FAIL,'interrupted-checkpoint'); END").execute(store.raw_pool()).await.unwrap();
    let error = Engine::new(clients.clone(), area.clone(), 1)
        .run("partial", &source, CancellationToken::new())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("interrupted-checkpoint"));
    assert_eq!(teacher.call_count(), 1);
    assert_eq!(judge.call_count(), 1);
    sqlx::query("DROP TRIGGER stop_before_checkpoint")
        .execute(store.raw_pool())
        .await
        .unwrap();
    let original: (String, String) =
        sqlx::query_as("SELECT config_json,created_at FROM runs WHERE run_id='partial'")
            .fetch_one(store.raw_pool())
            .await
            .unwrap();
    let pretty = serde_json::to_string_pretty(
        &serde_json::from_str::<serde_json::Value>(&original.0).unwrap(),
    )
    .unwrap();
    sqlx::query("UPDATE runs SET config_json=? WHERE run_id='partial'")
        .bind(&pretty)
        .execute(store.raw_pool())
        .await
        .unwrap();
    clients.policy = gw_schema::AccountingPolicy::FiniteUsd { limit_usd: 20.0 };
    let engine = Engine::new(clients, area, 4);
    let prepared = engine.prepare(&source).unwrap();
    let report = engine
        .run_prepared(
            "partial",
            prepared,
            gw_storage::RunMode::Replay,
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert!(report.completed);
    assert_eq!(report.admitted, 2);
    assert_eq!(teacher.call_count(), 2);
    assert_eq!(judge.call_count(), 2);
    let after: (String, String) =
        sqlx::query_as("SELECT config_json,created_at FROM runs WHERE run_id='partial'")
            .fetch_one(store.raw_pool())
            .await
            .unwrap();
    assert_eq!(after, (pretty, original.1));
    assert_eq!(store.model_launches("partial").await.unwrap().len(), 2);
}
