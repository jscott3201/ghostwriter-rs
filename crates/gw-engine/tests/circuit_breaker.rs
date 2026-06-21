//! F2 — systemic-fault fail-fast + the run-level circuit-breaker. A SYSTEMIC provider fault must NOT be
//! laundered into N per-record `Error`s with a "completed" run. HERMETIC: fakes + `Store::open_in_memory`.

mod common;

use std::sync::Arc;

use common::*;
use gw_engine::{Engine, EventSink, InMemorySeedSource};
use gw_schema::LifecycleState;
use gw_storage::{RecordFilter, Store};
use tokio_util::sync::CancellationToken;

/// F2 (H-B): a SYSTEMIC provider fault — an invalid/revoked key surfacing as a non-retryable HTTP 401 on
/// EVERY call — must ABORT the run fail-fast, NOT be classified record-level and park every seed item at
/// `Error` one-by-one (which would report a "completed" run hiding an operator misconfiguration). With 3
/// items the run aborts on the FIRST item's auth fault: the teacher is called exactly once and no record
/// is parked.
#[tokio::test]
async fn systemic_auth_fault_aborts_run_fail_fast() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(AlwaysFailingTeacher::new(401)); // invalid/revoked key → 401 every call
    let judge = Arc::new(ScriptedJudge::new(vec![]));
    let cl = clients(
        store.clone(),
        teacher.clone(),
        judge,
        25.0,
        EventSink::disconnected(),
    );
    let area = area_k1(one_judge(), lenient_thresholds());
    let engine = Engine::new(cl, area, 1);

    let source = InMemorySeedSource::new(
        vec![
            good_candidate("q0"),
            good_candidate("q1"),
            good_candidate("q2"),
        ],
        1,
    );
    let result = engine
        .run("run-auth", &source, CancellationToken::new())
        .await;

    assert!(
        result.is_err(),
        "a systemic auth fault must ABORT the run, not complete with N parked records"
    );
    assert_eq!(
        teacher.call_count(),
        1,
        "fail-fast: the run aborts on the FIRST item's auth fault, not after churning every item"
    );
    // No seed item was parked at Error — the fault was fatal, not laundered into per-record errors.
    let errored = store
        .scan(
            &RecordFilter::new()
                .run_id("run-auth")
                .lifecycle_state(LifecycleState::Error),
        )
        .await
        .unwrap();
    assert!(
        errored.is_empty(),
        "a systemic fault is NOT parked per-record (it aborts before any park)"
    );
}

/// F2 circuit-breaker: a PERSISTENT record-level fault (HTTP 404 every call — non-retryable AND
/// non-systemic, so the error taxonomy parks it per-record rather than aborting) must NOT churn the
/// whole seed space. The run-level breaker trips after `CIRCUIT_BREAKER_PARKS` (8) consecutive
/// all-errored items with zero successes and aborts the run — the load-bearing backstop for any systemic
/// fault the taxonomy itself does not classify as fatal.
#[tokio::test]
async fn circuit_breaker_trips_on_persistent_record_level_fault() {
    let store = Store::open_in_memory().await.unwrap();
    // 404 is non-retryable and NON-auth → record-level → parked per item (never classified infra), so
    // only the run-level circuit-breaker can stop the churn.
    let teacher = Arc::new(AlwaysFailingTeacher::new(404));
    let judge = Arc::new(ScriptedJudge::new(vec![]));
    let cl = clients(
        store.clone(),
        teacher.clone(),
        judge,
        25.0,
        EventSink::disconnected(),
    );
    let area = area_k1(one_judge(), lenient_thresholds());
    let engine = Engine::new(cl, area, 1);

    // 20 items in one shard, processed sequentially — but the breaker trips after 8 consecutive parks.
    let candidates: Vec<_> = (0..20).map(|i| good_candidate(&format!("q{i}"))).collect();
    let source = InMemorySeedSource::new(candidates, 1);
    let result = engine
        .run("run-cb", &source, CancellationToken::new())
        .await;

    assert!(
        result.is_err(),
        "the circuit-breaker aborts a systemically-broken run"
    );
    // Exactly 8 items were attempted before the breaker tripped (CIRCUIT_BREAKER_PARKS = 8); the run does
    // NOT churn all 20 items into `Error`.
    assert_eq!(
        teacher.call_count(),
        8,
        "the breaker trips after 8 consecutive all-errored items, not after churning every item"
    );
}
