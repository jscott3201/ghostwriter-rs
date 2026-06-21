//! E1 — best-of-k crash-resume must NOT double-admit. HERMETIC: fakes + `Store::open_in_memory`.
//!
//! Simulates the crash window where the best sibling reached `Exported` but the shard cursor had not
//! committed and the runner-up was still at `Judged`. On resume `run_group` re-runs over the same
//! group; the E1 guard must recognize the established winner and NEVER elect a second one.

mod common;

use std::sync::Arc;

use common::*;
use gw_engine::{EventSink, InMemorySeedSource, SeedSource, run_group};
use gw_schema::LifecycleState;
use gw_storage::{RecordFilter, Store};

/// A k=2 group is driven once (sibling0 admitted→Exported, sibling1 retained→Rejected). We then force
/// the precise crash window — reset sibling1 back to `Judged` — and re-run the group. The E1 guard must
/// keep sibling0 as the sole winner and drive sibling1 to Rejected, never admitting a SECOND sibling.
#[tokio::test]
async fn best_of_k_resume_does_not_double_admit() {
    let store = Store::open_in_memory().await.unwrap();
    store.create_run("run-e1", "{}", Some(25.0)).await.unwrap();

    // Two siblings with distinct answers (distinct record_hash → independent grades), sibling0 the
    // higher score (0.95) so it is the winner; sibling1 (0.90) the retained runner-up.
    let teacher = Arc::new(ScriptedTeacher::new(
        vec![answer_cot("96", 0.01), answer_cot("97", 0.01)],
        2,
    ));
    let judge = Arc::new(ScriptedJudge::new(vec![
        &judge_body(0.95, "accept"),
        &judge_body(0.90, "accept"),
    ]));
    let cl = clients(
        store.clone(),
        teacher,
        judge,
        25.0,
        EventSink::disconnected(),
    );
    let area = area_k(one_judge(), lenient_thresholds(), 2);

    // The one seed item for shard 0.
    let source = InMemorySeedSource::new(vec![good_candidate("What is 12*8?")], 1);
    let item = source.items_for_shard(0).remove(0);

    // First pass: drives the group. Exactly one admit (sibling0 → Exported), one retained (sibling1).
    let out1 = run_group("run-e1", 0, &item, &cl, &area).await.unwrap();
    let winner1 = out1
        .best
        .clone()
        .expect("a winner is admitted on the first pass");
    let all1 = store
        .scan(&RecordFilter::new().run_id("run-e1"))
        .await
        .unwrap();
    assert_eq!(all1.len(), 2);
    let exported1 = all1
        .iter()
        .filter(|r| r.lifecycle.state == LifecycleState::Exported)
        .count();
    assert_eq!(
        exported1, 1,
        "exactly one sibling exported on the first pass"
    );

    // Identify the runner-up (the non-winner sibling) and force it back to `Judged` — the exact state a
    // crash would leave it in if the process died after admitting the winner but before retaining it.
    let runner_up_id = all1
        .iter()
        .find(|r| r.record_id != winner1)
        .map(|r| r.record_id.clone())
        .unwrap();
    store
        .advance_lifecycle(&runner_up_id, LifecycleState::Judged, None)
        .await
        .unwrap();
    assert_eq!(
        store.get(&runner_up_id).await.unwrap().lifecycle.state,
        LifecycleState::Judged,
        "runner-up forced back to Judged (the crash window)"
    );

    // RESUME: re-run the group. The teacher must NOT be called again (siblings already persisted;
    // ScriptedTeacher max_calls=2 would panic on a 3rd call). The E1 guard recognizes sibling0 is
    // already Exported → it is the established winner; sibling1 is driven to Rejected, NOT admitted.
    let out2 = run_group("run-e1", 0, &item, &cl, &area).await.unwrap();
    assert_eq!(
        out2.best.as_deref(),
        Some(winner1.as_str()),
        "resume returns the SAME established winner, never re-elects"
    );

    let all2 = store
        .scan(&RecordFilter::new().run_id("run-e1"))
        .await
        .unwrap();
    let exported2 = all2
        .iter()
        .filter(|r| {
            matches!(
                r.lifecycle.state,
                LifecycleState::Exported | LifecycleState::Admitted | LifecycleState::Formatted
            )
        })
        .count();
    assert_eq!(
        exported2, 1,
        "after resume STILL exactly one admitted sibling — no double-admit"
    );
    // The runner-up ended retained at Rejected (driven there by the E1 guard).
    assert_eq!(
        store.get(&runner_up_id).await.unwrap().lifecycle.state,
        LifecycleState::Rejected,
        "the runner-up is retained at Rejected on resume, never admitted"
    );
}

/// F1 (H-C regression): in a k>1 group, a record-level fault on a LATER sibling must NEVER clobber the
/// healthy earlier sibling. With k=2: c0 generates + drives to `Judged`, then c1's generation faults
/// (record-level). The faulting c1 is parked at `Error` (correctly attributed to ITS id); c0 is admitted
/// and driven to `Exported` — never overwritten. (Pre-fix `park_item_errored` hardcoded `c0` and the
/// `is_terminal` guard did not cover `Judged`, so the healthy winner was clobbered to `Error`.)
#[tokio::test]
async fn later_sibling_fault_does_not_clobber_healthy_sibling() {
    let store = Store::open_in_memory().await.unwrap();
    store.create_run("run-f1", "{}", Some(25.0)).await.unwrap();
    // call 1 = c0 (good → Judged → admitted), call 2 = c1 (FAULT → record-level → parked at Error).
    let teacher = Arc::new(FailingTeacher::new(2, 0.01));
    let judge = Arc::new(ScriptedJudge::new(vec![&judge_body(0.95, "accept")]));
    let cl = clients(
        store.clone(),
        teacher,
        judge,
        25.0,
        EventSink::disconnected(),
    );
    let area = area_k(one_judge(), lenient_thresholds(), 2);
    let source = InMemorySeedSource::new(vec![good_candidate("What is 12*8?")], 1);
    let item = source.items_for_shard(0).remove(0);

    let out = run_group("run-f1", 0, &item, &cl, &area).await.unwrap();
    let winner = out.best.expect("the healthy survivor c0 is admitted");

    let all = store
        .scan(&RecordFilter::new().run_id("run-f1"))
        .await
        .unwrap();
    assert_eq!(
        all.len(),
        2,
        "both siblings persisted (admitted winner + parked fault)"
    );

    // The healthy winner is at an ADMITTED state — NOT clobbered to Error.
    let win = store.get(&winner).await.unwrap();
    assert!(
        matches!(
            win.lifecycle.state,
            LifecycleState::Exported | LifecycleState::Admitted | LifecycleState::Formatted
        ),
        "the healthy survivor is admitted, not clobbered to Error (got {:?})",
        win.lifecycle.state
    );

    // EXACTLY the faulting sibling is at Error — and it is NOT the winner (correct attribution).
    let errored: Vec<_> = all
        .iter()
        .filter(|r| r.lifecycle.state == LifecycleState::Error)
        .collect();
    assert_eq!(
        errored.len(),
        1,
        "exactly the faulting sibling is parked at Error"
    );
    assert_ne!(
        errored[0].record_id, winner,
        "the Error is attributed to the FAULTING sibling, never the healthy winner"
    );
}

/// F3 (H-A): on resume the established-winner guard must DRIVE the winner forward, not strand it. We
/// force the winner back to `Admitted` (the crash window between admission and export); on resume the
/// guard must drive it to `Exported`, not return it undriven (which left `report.exported` short and the
/// record stuck short of the dataset's terminal-good state).
#[tokio::test]
async fn resume_drives_established_winner_to_exported() {
    let store = Store::open_in_memory().await.unwrap();
    store.create_run("run-f3", "{}", Some(25.0)).await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(
        vec![answer_cot("96", 0.01), answer_cot("97", 0.01)],
        2,
    ));
    let judge = Arc::new(ScriptedJudge::new(vec![
        &judge_body(0.95, "accept"),
        &judge_body(0.90, "accept"),
    ]));
    let cl = clients(
        store.clone(),
        teacher,
        judge,
        25.0,
        EventSink::disconnected(),
    );
    let area = area_k(one_judge(), lenient_thresholds(), 2);
    let source = InMemorySeedSource::new(vec![good_candidate("What is 12*8?")], 1);
    let item = source.items_for_shard(0).remove(0);

    let out1 = run_group("run-f3", 0, &item, &cl, &area).await.unwrap();
    let winner = out1.best.expect("a winner is admitted on the first pass");

    // Force the winner back to `Admitted` — the crash window AFTER admission but BEFORE export.
    store
        .advance_lifecycle(&winner, LifecycleState::Admitted, None)
        .await
        .unwrap();
    assert_eq!(
        store.get(&winner).await.unwrap().lifecycle.state,
        LifecycleState::Admitted
    );

    // RESUME: the F3 guard recognizes the established winner and DRIVES it forward to Exported (the
    // teacher is NOT re-spent — ScriptedTeacher max_calls=2 would panic on a 3rd call).
    let out2 = run_group("run-f3", 0, &item, &cl, &area).await.unwrap();
    assert_eq!(
        out2.best.as_deref(),
        Some(winner.as_str()),
        "resume returns the SAME established winner"
    );
    assert_eq!(
        store.get(&winner).await.unwrap().lifecycle.state,
        LifecycleState::Exported,
        "F3: the established winner is driven to Exported on resume, not stranded at Admitted"
    );
}
