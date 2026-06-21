//! Regression tests for the run partition manifest guard.

mod common;

use std::sync::Arc;

use common::*;
use gw_engine::{Engine, EngineError, EventSink, InMemorySeedSource};
use gw_schema::LifecycleState;
use gw_storage::{RecordFilter, RunStatus, Store};
use tokio_util::sync::CancellationToken;

fn source(prompts: &[&str], shard_count: usize) -> InMemorySeedSource {
    InMemorySeedSource::new(
        prompts
            .iter()
            .map(|prompt| good_candidate(prompt))
            .collect(),
        shard_count,
    )
}

async fn run_status(store: &Store, run_id: &str) -> Option<String> {
    store.run_status(run_id).await.unwrap()
}

async fn record_count(store: &Store, run_id: &str) -> usize {
    store
        .scan(&RecordFilter::new().run_id(run_id))
        .await
        .unwrap()
        .len()
}

fn assert_partition_invariant(err: EngineError) {
    match err {
        EngineError::Invariant(msg) => {
            assert!(msg.contains("run partition mismatch"), "{msg}");
        }
        other => panic!("expected partition invariant, got {other:?}"),
    }
}

#[tokio::test]
async fn relaunch_with_different_shard_count_hard_errors_before_spend_or_status_reset() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(
        vec![
            good_cot(0.01),
            good_cot(0.01),
            good_cot(0.01),
            good_cot(0.01),
        ],
        4,
    ));
    let judge = Arc::new(ScriptedJudge::new(vec![
        &judge_body(0.95, "accept"),
        &judge_body(0.95, "accept"),
        &judge_body(0.95, "accept"),
        &judge_body(0.95, "accept"),
    ]));
    let engine = Engine::new(
        clients(
            store.clone(),
            teacher.clone(),
            judge.clone(),
            25.0,
            EventSink::disconnected(),
        ),
        area_k1(one_judge(), lenient_thresholds()),
        2,
    );
    let prompts = ["q0", "q1", "q2", "q3"];
    engine
        .run("run-shards", &source(&prompts, 2), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(teacher.call_count(), 4);
    assert_eq!(
        run_status(&store, "run-shards").await.as_deref(),
        Some(RunStatus::Completed.as_str())
    );

    let engine2 = Engine::new(
        clients(
            store.clone(),
            teacher.clone(),
            judge,
            25.0,
            EventSink::disconnected(),
        ),
        area_k1(one_judge(), lenient_thresholds()),
        2,
    );
    // ≥3 prompts required so %2 vs %3 re-buckets at least one prompt to a fresh shard —
    // otherwise the no-spend/no-orphan assertions don't discriminate.
    let err = engine2
        .run("run-shards", &source(&prompts, 3), CancellationToken::new())
        .await
        .unwrap_err();
    assert_partition_invariant(err);
    assert_eq!(teacher.call_count(), 4, "mismatch must not spend");
    assert_eq!(record_count(&store, "run-shards").await, 4);
    assert_eq!(
        run_status(&store, "run-shards").await.as_deref(),
        Some(RunStatus::Completed.as_str()),
        "mismatch must not reset status"
    );
}

#[tokio::test]
async fn relaunch_with_reordered_prompts_hard_errors_before_spend() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(
        vec![
            good_cot(0.01),
            good_cot(0.01),
            good_cot(0.01),
            good_cot(0.01),
        ],
        4,
    ));
    let judge = Arc::new(ScriptedJudge::new(vec![
        &judge_body(0.95, "accept"),
        &judge_body(0.95, "accept"),
        &judge_body(0.95, "accept"),
        &judge_body(0.95, "accept"),
    ]));
    let engine = Engine::new(
        clients(
            store.clone(),
            teacher.clone(),
            judge.clone(),
            25.0,
            EventSink::disconnected(),
        ),
        area_k1(one_judge(), lenient_thresholds()),
        2,
    );
    let prompts = ["q0", "q1", "q2", "q3"];
    engine
        .run(
            "run-prompts",
            &source(&prompts, 2),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    let engine2 = Engine::new(
        clients(
            store.clone(),
            teacher.clone(),
            judge,
            25.0,
            EventSink::disconnected(),
        ),
        area_k1(one_judge(), lenient_thresholds()),
        2,
    );
    let err = engine2
        .run(
            "run-prompts",
            &source(&["q0", "q2", "q1", "q3"], 2),
            CancellationToken::new(),
        )
        .await
        .unwrap_err();
    assert_partition_invariant(err);
    assert_eq!(teacher.call_count(), 4);
    assert_eq!(record_count(&store, "run-prompts").await, 4);
}

#[tokio::test]
async fn matching_resume_keeps_happy_path_and_zero_shards_match_one() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(
        vec![good_cot(0.01), good_cot(0.01)],
        2,
    ));
    let judge = Arc::new(ScriptedJudge::new(vec![
        &judge_body(0.95, "accept"),
        &judge_body(0.95, "accept"),
    ]));
    let engine = Engine::new(
        clients(
            store.clone(),
            teacher.clone(),
            judge.clone(),
            25.0,
            EventSink::disconnected(),
        ),
        area_k1(one_judge(), lenient_thresholds()),
        2,
    );
    let prompts = ["q0", "q1"];
    engine
        .run("run-match", &source(&prompts, 0), CancellationToken::new())
        .await
        .unwrap();

    let engine2 = Engine::new(
        clients(
            store.clone(),
            teacher.clone(),
            judge,
            25.0,
            EventSink::disconnected(),
        ),
        area_k1(one_judge(), lenient_thresholds()),
        2,
    );
    let report = engine2
        .run("run-match", &source(&prompts, 1), CancellationToken::new())
        .await
        .unwrap();
    assert!(report.completed);
    assert_eq!(report.exported, 2);
    assert_eq!(teacher.call_count(), 2, "matching resume must not re-spend");

    let exported = store
        .scan(
            &RecordFilter::new()
                .run_id("run-match")
                .lifecycle_state(LifecycleState::Exported),
        )
        .await
        .unwrap();
    assert_eq!(
        exported.len(),
        2,
        "matching resume must not duplicate records"
    );
}
