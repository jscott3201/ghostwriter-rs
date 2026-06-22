//! E1 — best-of-k crash-resume must NOT double-admit. HERMETIC: fakes + `Store::open_in_memory`.
//!
//! Simulates the crash window where the best sibling reached `Exported` but the shard cursor had not
//! committed and the runner-up was still at `Judged`. On resume `run_group` re-runs over the same
//! group; the E1 guard must recognize the established winner and NEVER elect a second one.

mod common;

use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::*;
use gw_engine::{EventSink, InMemorySeedSource, RunControl, SeedSource, run_group};
use gw_providers::{
    ChatRequest, DeltaStream, Provider, ProviderError, StreamChatFuture, StreamDelta,
};
use gw_schema::{BudgetBreach, LifecycleState};
use gw_storage::{RecordFilter, Store};
use tokio_util::sync::CancellationToken;

fn no_cancel() -> CancellationToken {
    CancellationToken::new()
}

fn control(cancel: &CancellationToken) -> RunControl<'_> {
    RunControl::new(cancel, BudgetBreach::Drain)
}

struct SeedBarrierFaultTeacher {
    barrier: Arc<tokio::sync::Barrier>,
    fail_seeds: BTreeSet<i64>,
    calls: AtomicUsize,
}

impl SeedBarrierFaultTeacher {
    fn new(n: usize, fail_seeds: impl IntoIterator<Item = i64>) -> Self {
        Self {
            barrier: Arc::new(tokio::sync::Barrier::new(n)),
            fail_seeds: fail_seeds.into_iter().collect(),
            calls: AtomicUsize::new(0),
        }
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl Provider for SeedBarrierFaultTeacher {
    fn stream_chat(&self, req: ChatRequest) -> StreamChatFuture<'_> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let barrier = Arc::clone(&self.barrier);
        let fail = req.seed.is_some_and(|seed| self.fail_seeds.contains(&seed));
        Box::pin(async move {
            barrier.wait().await;
            if fail {
                Err(ProviderError::Decode(
                    "injected concurrent teacher fault".into(),
                ))
            } else {
                let items = answer_cot("96", 0.01)
                    .into_iter()
                    .map(Ok::<StreamDelta, ProviderError>);
                let stream: DeltaStream =
                    Box::pin(futures::stream::iter(items.collect::<Vec<_>>()));
                Ok(stream)
            }
        })
    }
}

struct SeedBarrierStatusTeacher {
    barrier: Arc<tokio::sync::Barrier>,
    fatal_seed: i64,
    status: u16,
    calls: AtomicUsize,
}

impl SeedBarrierStatusTeacher {
    fn new(n: usize, fatal_seed: i64, status: u16) -> Self {
        Self {
            barrier: Arc::new(tokio::sync::Barrier::new(n)),
            fatal_seed,
            status,
            calls: AtomicUsize::new(0),
        }
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl Provider for SeedBarrierStatusTeacher {
    fn stream_chat(&self, req: ChatRequest) -> StreamChatFuture<'_> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let barrier = Arc::clone(&self.barrier);
        let seed = req.seed.unwrap_or_default();
        let fatal = seed == self.fatal_seed;
        let status = self.status;
        Box::pin(async move {
            barrier.wait().await;
            if fatal {
                return Err(ProviderError::from_status(status, None));
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
            let answer = format!("answer-{seed}");
            let items = answer_cot(&answer, 0.01)
                .into_iter()
                .map(Ok::<StreamDelta, ProviderError>);
            let stream: DeltaStream = Box::pin(futures::stream::iter(items.collect::<Vec<_>>()));
            Ok(stream)
        })
    }
}

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
    let cancel = no_cancel();
    let out1 = run_group("run-e1", 0, &item, &cl, &area, control(&cancel))
        .await
        .unwrap();
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
    let out2 = run_group("run-e1", 0, &item, &cl, &area, control(&cancel))
        .await
        .unwrap();
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

    let cancel = no_cancel();
    let out = run_group("run-f1", 0, &item, &cl, &area, control(&cancel))
        .await
        .unwrap();
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_sibling_faults_park_only_faulting_records() {
    let store = Store::open_in_memory().await.unwrap();
    store
        .create_run("run-f1-concurrent", "{}", Some(25.0))
        .await
        .unwrap();

    let source = InMemorySeedSource::new(vec![good_candidate("What is 12*8?")], 1);
    let item = source.items_for_shard(0).remove(0);
    let teacher = Arc::new(SeedBarrierFaultTeacher::new(
        3,
        [item.seed.wrapping_add(1), item.seed.wrapping_add(2)],
    ));
    let judge = Arc::new(ScriptedJudge::new(vec![&judge_body(0.95, "accept")]));
    let cl = clients(
        store.clone(),
        teacher.clone(),
        judge,
        25.0,
        EventSink::disconnected(),
    );
    let area = area_k(one_judge(), lenient_thresholds(), 3);
    let cancel = no_cancel();

    let out = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        run_group("run-f1-concurrent", 0, &item, &cl, &area, control(&cancel)),
    )
    .await
    .expect("all siblings should fan out concurrently")
    .unwrap();
    let winner = out.best.expect("the healthy survivor c0 is admitted");

    assert_eq!(teacher.call_count(), 3);
    let all = store
        .scan(&RecordFilter::new().run_id("run-f1-concurrent"))
        .await
        .unwrap();
    assert_eq!(all.len(), 3);

    let win = store.get(&winner).await.unwrap();
    assert!(
        matches!(
            win.lifecycle.state,
            LifecycleState::Exported | LifecycleState::Admitted | LifecycleState::Formatted
        ),
        "the healthy survivor is admitted, got {:?}",
        win.lifecycle.state
    );

    let mut errored_ids: Vec<_> = all
        .iter()
        .filter(|r| r.lifecycle.state == LifecycleState::Error)
        .map(|r| r.record_id.clone())
        .collect();
    errored_ids.sort_unstable();
    let expected = vec![
        format!("run-f1-concurrent-s0-seed{}-a0-c1", item.seed),
        format!("run-f1-concurrent-s0-seed{}-a0-c2", item.seed),
    ];
    assert_eq!(
        errored_ids, expected,
        "only the two faulting siblings are parked"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn systemic_fatal_waits_for_sibling_fanout_to_settle() {
    let store = Store::open_in_memory().await.unwrap();
    store
        .create_run("run-f1-systemic", "{}", Some(25.0))
        .await
        .unwrap();

    let source = InMemorySeedSource::new(vec![good_candidate("What is 12*8?")], 1);
    let item = source.items_for_shard(0).remove(0);
    let teacher = Arc::new(SeedBarrierStatusTeacher::new(
        3,
        item.seed.wrapping_add(1),
        401,
    ));
    let judge = Arc::new(ScriptedJudge::new(vec![
        &judge_body(0.95, "accept"),
        &judge_body(0.90, "accept"),
    ]));
    let cl = clients(
        store.clone(),
        teacher.clone(),
        judge,
        25.0,
        EventSink::disconnected(),
    );
    let budget = cl.budget.clone();
    let area = area_k(one_judge(), lenient_thresholds(), 3);
    let cancel = no_cancel();

    let err = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        run_group("run-f1-systemic", 0, &item, &cl, &area, control(&cancel)),
    )
    .await
    .expect("all siblings should fan out concurrently")
    .expect_err("the systemic 401 should surface after sibling join");

    let expected_fatal = format!("run-f1-systemic-s0-seed{}-a0-c1", item.seed);
    assert_eq!(
        err.attributed_record(),
        Some(expected_fatal.as_str()),
        "the fatal is attributed to the 401 sibling"
    );
    assert_eq!(teacher.call_count(), 3);
    assert!(
        (budget.spent() - 0.02).abs() < 1e-12,
        "healthy siblings are allowed to settle and charge before the fatal returns"
    );

    let all = store
        .scan(&RecordFilter::new().run_id("run-f1-systemic"))
        .await
        .unwrap();
    assert_eq!(
        all.len(),
        2,
        "the two healthy siblings persisted before the systemic fatal surfaced"
    );
    assert!(
        all.iter()
            .all(|rec| rec.lifecycle.state == LifecycleState::Judged),
        "healthy siblings stop at Judged because the group returns the fatal before finalization"
    );
    let mut ids: Vec<_> = all.iter().map(|rec| rec.record_id.clone()).collect();
    ids.sort_unstable();
    assert_eq!(
        ids,
        vec![
            format!("run-f1-systemic-s0-seed{}-a0-c0", item.seed),
            format!("run-f1-systemic-s0-seed{}-a0-c2", item.seed),
        ]
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

    let cancel = no_cancel();
    let out1 = run_group("run-f3", 0, &item, &cl, &area, control(&cancel))
        .await
        .unwrap();
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
    let out2 = run_group("run-f3", 0, &item, &cl, &area, control(&cancel))
        .await
        .unwrap();
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
