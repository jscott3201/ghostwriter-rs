//! Fault-isolation regressions for the second remediation fold (the "remediation trap" cases a passing
//! suite missed): F1 — a record-level fault MID-DRIVE (the judge rail faulting a sibling already at
//! `Verified`) must TERMINALIZE the faulting record at `Error`, never silently strand it at a forward
//! state; F2 — a systemic judge outage must still TRIP the circuit-breaker (a stranded non-`Error`
//! record must not falsely disarm it); X1/X2 — a record-level fault in the bounded REVISE retry parks
//! the faulting RETRY (not the `Revising` original) and does not falsely arm the breaker. HERMETIC.

mod common;

use std::sync::Arc;

use common::*;
use gw_engine::{Engine, EngineEvent, EventSink, InMemorySeedSource};
use gw_schema::LifecycleState;
use gw_storage::{RecordFilter, Store};
use tokio_util::sync::CancellationToken;

/// F1 (blocker regression): a sibling generates + verifies (reaching `Verified`), then the JUDGE rail
/// faults record-level. The no-clobber guard must STILL advance the faulting record to `Error` — pre-fix
/// it refused to park a record at a forward state and left it stranded at `Verified` forever (uncounted,
/// teacher spend lost).
#[tokio::test]
async fn judge_rail_fault_terminalizes_record_at_error_not_stranded() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(vec![good_cot(0.01)], 1));
    // The judge faults on its only call → the sibling, already at Verified, faults in the judge rail.
    let judge = Arc::new(FailingJudge::new(0, &judge_body(0.95, "accept")));
    let (sink, mut rx) = EventSink::subscribe();
    let cl = clients(store.clone(), teacher, judge, 25.0, sink);
    let area = area_k1(one_judge(), lenient_thresholds());
    let engine = Engine::new(cl, area, 1);

    let report = engine
        .run("run-jf", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(
        report.errored, 1,
        "the judge-rail fault TERMINALIZES the record at Error (not stranded at Verified)"
    );
    assert_eq!(report.exported, 0);

    let all = store
        .scan(&RecordFilter::new().run_id("run-jf"))
        .await
        .unwrap();
    assert_eq!(all.len(), 1);
    assert_eq!(
        all[0].lifecycle.state,
        LifecycleState::Error,
        "the faulting record is at Error, NOT stranded at Verified (the F-1 regression)"
    );
    assert!(
        all[0].lifecycle.error.is_some(),
        "the Error carries the fault detail"
    );

    let mut saw_errored = false;
    while let Ok(ev) = rx.try_recv() {
        if let EngineEvent::RecordErrored { .. } = ev {
            saw_errored = true;
        }
    }
    assert!(saw_errored, "a RecordErrored event is emitted");
}

/// F2 regression: a SYSTEMIC judge outage (the judge provider down on every call) must trip the
/// circuit-breaker and abort. Every sibling generates then faults in the judge rail; with F-1 fixed each
/// reaches `Error` → `AllErrored`, so the breaker arms and trips after 8 consecutive items. Pre-fix the
/// stranded-at-`Verified` siblings reported `Decided`, falsely disarming the breaker so the run churned
/// the whole seed space.
#[tokio::test]
async fn systemic_judge_outage_trips_circuit_breaker() {
    let store = Store::open_in_memory().await.unwrap();
    // Empty-script teacher streams a good CoT on every call; the JUDGE is down on every call.
    let teacher = Arc::new(ScriptedTeacher::new(vec![], 100));
    let judge = Arc::new(FailingJudge::new(0, &judge_body(0.95, "accept")));
    let cl = clients(
        store.clone(),
        teacher.clone(),
        judge,
        100.0,
        EventSink::disconnected(),
    );
    let area = area_k1(one_judge(), lenient_thresholds());
    let engine = Engine::new(cl, area, 1);

    // 12 items in one shard — the breaker must trip after 8 consecutive all-errored items.
    let candidates: Vec<_> = (0..12).map(|i| good_candidate(&format!("q{i}"))).collect();
    let source = InMemorySeedSource::new(candidates, 1);
    let result = engine
        .run("run-jo", &source, CancellationToken::new())
        .await;

    assert!(
        result.is_err(),
        "a systemic judge outage trips the circuit-breaker and aborts the run"
    );
    assert_eq!(
        teacher.call_count(),
        8,
        "the breaker trips after 8 consecutive all-errored items — it is NOT falsely disarmed by a \
         stranded non-Error sibling, and does NOT churn all 12 items"
    );
}

/// X1/X2 regression: the bounded REVISE retry faults record-level. The faulting RETRY (attempt 1) must
/// be parked at `Error` (the fault honestly recorded, not swallowed), the `Revising` original left intact
/// as its audit row, and the run must COMPLETE — a single decided-then-revise-faulting item must not
/// falsely arm the circuit-breaker. Pre-fix the un-attributed revise fault mis-targeted `c0` (a
/// no-clobber no-op → fault swallowed, `report.errored == 0`) and counted the item `AllErrored`.
#[tokio::test]
async fn revise_retry_fault_parks_the_retry_and_keeps_run_completing() {
    let store = Store::open_in_memory().await.unwrap();
    // call 1 = the original (good → judged into the Revise band → Revising); call 2 = the bounded retry
    // (FAULTS with a record-level Decode).
    let teacher = Arc::new(FailingTeacher::new(2, 0.01));
    // 0.65 ∈ [reject_below 0.50, accept 0.80) → the Revise band → the original reconciles to Revising.
    let judge = Arc::new(ScriptedJudge::new(vec![&judge_body(0.65, "revise")]));
    let cl = clients(
        store.clone(),
        teacher.clone(),
        judge,
        25.0,
        EventSink::disconnected(),
    );
    let area = area_k1(one_judge(), lenient_thresholds());
    let engine = Engine::new(cl, area, 1);

    let report = engine
        .run("run-rvf", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(
        report.errored, 1,
        "the faulting RETRY is parked at Error (the fault is recorded, not silently swallowed)"
    );
    assert_eq!(
        report.revising, 1,
        "the original stays at Revising as its audit row — never clobbered"
    );
    assert!(
        report.completed,
        "a failed revise on a DECIDED group does not falsely arm the circuit-breaker (X2)"
    );
    assert_eq!(
        teacher.call_count(),
        2,
        "the original + exactly one bounded retry"
    );

    let all = store
        .scan(&RecordFilter::new().run_id("run-rvf"))
        .await
        .unwrap();
    let errored: Vec<_> = all
        .iter()
        .filter(|r| r.lifecycle.state == LifecycleState::Error)
        .collect();
    assert_eq!(errored.len(), 1);
    assert!(
        errored[0].record_id.contains("-a1-"),
        "the Error is attributed to the attempt-1 RETRY, not the attempt-0 original"
    );
}
