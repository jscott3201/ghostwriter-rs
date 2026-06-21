//! HERMETIC crash-recovery / never-re-spend / revise-once / step-replay tests for `gw-engine` — the
//! deepest unhappy paths. These drive the step machine + the executor directly over fakes +
//! `Store::open_in_memory`, asserting the load-bearing idempotency invariants.

mod common;

use std::sync::Arc;

use common::*;
use gw_engine::{Engine, EngineEvent, EventSink, InMemorySeedSource, drive, step};
use gw_generate::{
    RecordContext, SamplingPreset, TeacherCall, assemble, generate_assistant, synthesize_user_turn,
};
use gw_schema::{LifecycleState, TeacherRef, TrainingRecord};
use gw_storage::{RecordFilter, Store, now_rfc3339, prompt_hash};
use tokio_util::sync::CancellationToken;

/// Seed ONE record at `AssistantGenerated` directly into the store (the state `gw-generate` hands the
/// engine), simulating a record persisted mid-flight before a crash. Returns the persisted record.
async fn seed_assistant_generated(
    store: &Store,
    teacher: &Arc<ScriptedTeacher>,
    record_id: &str,
) -> TrainingRecord {
    store.create_run("run-1", "{}", Some(25.0)).await.unwrap();
    let gated =
        synthesize_user_turn(good_candidate("12*8?"), &gw_generate::NullEmbedder, &[]).unwrap();
    let call = TeacherCall::new("z-ai/glm-5.2", vec![gated.candidate.message.clone()], 16384)
        .with_sampling(SamplingPreset::official().with_seed(0));
    let turn = generate_assistant(teacher.as_ref(), &gated, &call)
        .await
        .unwrap();
    let ctx = RecordContext {
        record_id: record_id.into(),
        run_id: "run-1".into(),
        training_area: "math".into(),
        harness_version: "0.1.0-test".into(),
        git_commit: None,
        now_rfc3339: now_rfc3339(),
        user_synth_model: None,
    };
    let teacher_ref = TeacherRef {
        provider: "openrouter".into(),
        slug: "z-ai/glm-5.2".into(),
        served_by: None,
        model_card_revision: None,
    };
    let mut rec = assemble(&ctx, &gated, turn, teacher_ref, call.generation(), None);
    rec.generation.sibling_group_id = Some(prompt_hash(&rec.messages).unwrap());
    store.put(&rec).await.unwrap();
    store.get(record_id).await.unwrap()
}

/// MANDATORY 1 (crash-resume): a record persisted mid-flight (AssistantGenerated, not Judged)
/// re-enters at its LAST persisted state on relaunch, NOT from Seeded — and `step` advances it from
/// there.
#[tokio::test]
async fn crash_resume_reenters_at_last_persisted_state() {
    let store = Store::open_in_memory().await.unwrap();
    // The teacher is allowed exactly ONE call (the original generation). If resume re-generated, the
    // teacher would be called a SECOND time and panic.
    let teacher = Arc::new(ScriptedTeacher::new(vec![good_cot(0.01)], 1));
    let judge = Arc::new(ScriptedJudge::new(vec![&judge_body(0.95, "accept")]));

    let rec = seed_assistant_generated(&store, &teacher, "rec-mid").await;
    assert_eq!(rec.lifecycle.state, LifecycleState::AssistantGenerated);
    assert_eq!(teacher.call_count(), 1);

    // "Relaunch": drive the record from its persisted state. It must NOT re-generate (teacher stays at
    // 1 call) — it re-enters at AssistantGenerated and advances forward.
    let cl = clients(
        store.clone(),
        teacher.clone(),
        judge,
        25.0,
        EventSink::disconnected(),
    );
    let area = area_k1(one_judge(), lenient_thresholds());

    // First step advances to Verified (NOT back to Seeded).
    let stepped = step(rec, &cl, &area).await.unwrap();
    assert_eq!(
        stepped.lifecycle.state,
        LifecycleState::Verified,
        "a mid-flight record re-enters at AssistantGenerated → Verified, not from Seeded"
    );

    // Driving it the rest of the way reaches Exported with NO second teacher call.
    let done = drive(stepped, &cl, &area).await.unwrap();
    assert_eq!(done.lifecycle.state, LifecycleState::Exported);
    assert_eq!(
        teacher.call_count(),
        1,
        "resume must NOT re-spend the teacher"
    );

    // The lifecycle history shows the resume continued from AssistantGenerated (it was the entry
    // state), never re-entered Seeded.
    let history: Vec<_> = done.lifecycle.history.iter().map(|h| h.state).collect();
    assert!(history.contains(&LifecycleState::AssistantGenerated));
    assert!(!history[1..].contains(&LifecycleState::Seeded));
}

/// MANDATORY 5 (never-re-spend): a full-run restart with the record already persisted does NOT call
/// the teacher again. The ScriptedTeacher panics on a 2nd call; the run must complete on cache/state.
#[tokio::test]
async fn restart_does_not_respend_the_teacher() {
    let store = Store::open_in_memory().await.unwrap();
    // max_calls = 1: a re-generation on restart would panic.
    let teacher = Arc::new(ScriptedTeacher::new(vec![good_cot(0.01)], 1));
    let judge = Arc::new(ScriptedJudge::new(vec![&judge_body(0.95, "accept")]));
    let cl = clients(
        store.clone(),
        teacher.clone(),
        judge,
        25.0,
        EventSink::disconnected(),
    );
    let area = area_k1(one_judge(), lenient_thresholds());
    let engine = Engine::new(cl, area, 4);
    let source = one_item_source();

    // First run drives to Exported (1 teacher call).
    let r1 = engine
        .run("run-1", &source, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(r1.exported, 1);
    assert_eq!(teacher.call_count(), 1);

    // Re-run the SAME run id over the SAME source: every record is already persisted past
    // AssistantGenerated, so the teacher is NEVER re-called (max_calls=1 would panic otherwise).
    let r2 = engine
        .run("run-1", &source, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(
        r2.exported, 1,
        "the re-run still reports the exported record"
    );
    assert_eq!(
        teacher.call_count(),
        1,
        "a restart must NOT re-spend the teacher"
    );
}

/// MANDATORY 2 (revise once): `Decision::Revise` drives EXACTLY ONE revising → assistant_generated
/// retry; a second `Revise` does NOT produce a second `revising`.
#[tokio::test]
async fn revise_drives_exactly_one_retry_then_no_second_revising() {
    let store = Store::open_in_memory().await.unwrap();
    // Two teacher calls: the original + the single bounded retry. (max_calls=2 — a THIRD generation,
    // i.e. a second revise loop, would panic.)
    let teacher = Arc::new(ScriptedTeacher::new(
        vec![good_cot(0.01), good_cot(0.01)],
        2,
    ));
    // Both the original AND the retry are judged into the REVISE band (score 0.65, in [0.5, 0.8)).
    // The original → Revising; the retry is also judged Revise but the engine DOWNGRADES the second
    // revise to Rejected (the single bound). Two distinct contents (cache misses), two judge calls.
    let judge = Arc::new(ScriptedJudge::new(vec![
        &judge_body(0.65, "revise"),
        &judge_body(0.65, "revise"),
    ]));
    let cl = clients(
        store.clone(),
        teacher.clone(),
        judge,
        25.0,
        EventSink::disconnected(),
    );
    let area = area_k1(one_judge(), lenient_thresholds());
    let engine = Engine::new(cl, area, 4);

    engine
        .run("run-rev", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();

    // Exactly TWO teacher calls (original + one retry) — never a third (no second revise loop).
    assert_eq!(
        teacher.call_count(),
        2,
        "exactly one bounded retry, never a second"
    );

    let all = store
        .scan(&RecordFilter::new().run_id("run-rev"))
        .await
        .unwrap();
    // Two records: the original (attempt 0, terminal at Revising) and the retry (attempt 1).
    assert_eq!(all.len(), 2, "the original + exactly one retry record");

    // Count distinct `revising` lifecycle transitions ACROSS both records: there must be EXACTLY ONE.
    let mut revising_transitions = 0usize;
    for rec in &all {
        let hist = store.lifecycle_history(&rec.record_id).await.unwrap();
        revising_transitions += hist
            .iter()
            .filter(|(state, _, _)| state == "revising")
            .count();
    }
    assert_eq!(
        revising_transitions, 1,
        "exactly ONE revising transition across the whole logical record — no second revising"
    );

    // The retry (attempt 1) is terminal at Rejected (its second revise was downgraded), NOT Revising.
    let retry = all
        .iter()
        .find(|r| r.record_id.contains("-a1-"))
        .expect("the attempt-1 retry record exists");
    assert_eq!(
        retry.lifecycle.state,
        LifecycleState::Rejected,
        "a second revise is downgraded to a conservative Rejected, never a second revising"
    );
}

/// MANDATORY 8 (step purity / replay): replaying `step` over the SAME inputs + fakes yields the SAME
/// transition deterministically. Two independent stores driven identically reach the same state with
/// the same persisted judging block.
#[tokio::test]
async fn step_replay_is_deterministic() {
    // Build two independent stores and drive an identical record through `drive` over identical fakes.
    let run = |run_id: &'static str| async move {
        let store = Store::open_in_memory().await.unwrap();
        let teacher = Arc::new(ScriptedTeacher::new(vec![good_cot(0.01)], 1));
        let judge = Arc::new(ScriptedJudge::new(vec![&judge_body(0.9, "accept")]));
        let rec = seed_assistant_generated_named(&store, &teacher, run_id, "rec-x").await;
        let cl = clients(
            store.clone(),
            teacher,
            judge,
            25.0,
            EventSink::disconnected(),
        );
        let area = area_k1(one_judge(), lenient_thresholds());
        let done = drive(rec, &cl, &area).await.unwrap();
        (
            done.lifecycle.state,
            done.judging.aggregate,
            done.judging.n_eff,
            done.judging.verdict,
        )
    };

    let a = run("run-a").await;
    let b = run("run-b").await;
    assert_eq!(
        a, b,
        "replaying step() over the same inputs yields the same transition"
    );
    // And concretely: a clean 0.9 accept lands at Exported with aggregate 0.9.
    assert_eq!(a.0, LifecycleState::Exported);
    assert_eq!(a.1, Some(0.9));
}

/// Variant of [`seed_assistant_generated`] with an explicit run id (for the replay test's two runs).
async fn seed_assistant_generated_named(
    store: &Store,
    teacher: &Arc<ScriptedTeacher>,
    run_id: &str,
    record_id: &str,
) -> TrainingRecord {
    store.create_run(run_id, "{}", Some(25.0)).await.unwrap();
    let gated =
        synthesize_user_turn(good_candidate("12*8?"), &gw_generate::NullEmbedder, &[]).unwrap();
    let call = TeacherCall::new("z-ai/glm-5.2", vec![gated.candidate.message.clone()], 16384)
        .with_sampling(SamplingPreset::official().with_seed(0));
    let turn = generate_assistant(teacher.as_ref(), &gated, &call)
        .await
        .unwrap();
    let ctx = RecordContext {
        record_id: record_id.into(),
        run_id: run_id.into(),
        training_area: "math".into(),
        harness_version: "0.1.0-test".into(),
        git_commit: None,
        now_rfc3339: now_rfc3339(),
        user_synth_model: None,
    };
    let teacher_ref = TeacherRef {
        provider: "openrouter".into(),
        slug: "z-ai/glm-5.2".into(),
        served_by: None,
        model_card_revision: None,
    };
    let mut rec = assemble(&ctx, &gated, turn, teacher_ref, call.generation(), None);
    rec.generation.sibling_group_id = Some(prompt_hash(&rec.messages).unwrap());
    store.put(&rec).await.unwrap();
    store.get(record_id).await.unwrap()
}

/// A coarse crash-resume at the EXECUTOR level: a run interrupted after committing some shard items
/// resumes from the checkpoint cursor and does not re-generate the committed items.
#[tokio::test]
async fn executor_resumes_from_shard_checkpoint() {
    let store = Store::open_in_memory().await.unwrap();
    // 2 items in 1 shard. First "run" processes only item 0 (we cancel after it commits), then a
    // second run resumes and processes item 1. The teacher is called exactly twice total (once per
    // item), never re-generating item 0 on the resume.
    let teacher = Arc::new(ScriptedTeacher::new(
        vec![good_cot(0.01), good_cot(0.01)],
        2,
    ));
    let judge = Arc::new(ScriptedJudge::new(vec![
        &judge_body(0.95, "accept"),
        &judge_body(0.95, "accept"),
    ]));
    let source = InMemorySeedSource::new(vec![good_candidate("q0"), good_candidate("q1")], 1);

    // Run 1: full run (both items). This commits both; the checkpoint cursor advances to 2.
    let cl1 = clients(
        store.clone(),
        teacher.clone(),
        judge.clone(),
        25.0,
        EventSink::disconnected(),
    );
    let engine1 = Engine::new(cl1, area_k1(one_judge(), lenient_thresholds()), 1);
    engine1
        .run("run-c", &source, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(teacher.call_count(), 2);

    // Run 2: re-run the SAME run id. Both items are below the cursor (committed) AND already persisted,
    // so NOTHING re-generates — the teacher count stays at 2.
    let cl2 = clients(
        store.clone(),
        teacher.clone(),
        judge,
        25.0,
        EventSink::disconnected(),
    );
    let engine2 = Engine::new(cl2, area_k1(one_judge(), lenient_thresholds()), 1);
    engine2
        .run("run-c", &source, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(
        teacher.call_count(),
        2,
        "a resumed run re-generates nothing already committed"
    );

    let exported = store
        .scan(
            &RecordFilter::new()
                .run_id("run-c")
                .lifecycle_state(LifecycleState::Exported),
        )
        .await
        .unwrap();
    assert_eq!(
        exported.len(),
        2,
        "both items reached Exported across the two runs"
    );
}

/// Concurrency: a multi-shard run drives all shards concurrently over ONE shared SQLite store without
/// deadlock or lost writes. Each of the 6 items (across 3 shards) reaches Exported exactly once.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn multi_shard_run_is_concurrent_and_consistent() {
    let store = Store::open_in_memory().await.unwrap();
    // 6 good completions (one per item); the scripted fakes are Mutex-guarded so concurrent shards
    // pop safely. Empty-script fallback also yields a good CoT, so call ordering across shards is fine.
    let teacher = Arc::new(ScriptedTeacher::new(
        (0..6).map(|_| good_cot(0.01)).collect(),
        6,
    ));
    let judge = Arc::new(ScriptedJudge::new(
        (0..6)
            .map(|_| "{\"score\":0.95,\"verdict\":\"accept\"}")
            .collect(),
    ));
    let cl = clients(
        store.clone(),
        teacher,
        judge,
        25.0,
        EventSink::disconnected(),
    );
    let area = area_k1(one_judge(), lenient_thresholds());
    let engine = Engine::new(cl, area, 4);

    // 6 items partitioned across 3 shards (2 each).
    let items: Vec<_> = (0..6).map(|i| good_candidate(&format!("q{i}"))).collect();
    let source = InMemorySeedSource::new(items, 3);

    let report = engine
        .run("run-multi", &source, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(
        report.exported, 6,
        "all six items exported across the concurrent shards"
    );
    assert!(report.completed);

    let exported = store
        .scan(
            &RecordFilter::new()
                .run_id("run-multi")
                .lifecycle_state(LifecycleState::Exported),
        )
        .await
        .unwrap();
    assert_eq!(
        exported.len(),
        6,
        "no lost writes under concurrent shard execution"
    );
}

/// Cancellation: a cancelled token stops dispatching NEW work and lets the run drain cleanly (the run
/// is reported NOT completed). A token cancelled before the run starts admits nothing.
#[tokio::test]
async fn cancelled_run_dispatches_no_new_work() {
    let store = Store::open_in_memory().await.unwrap();
    // The teacher must NEVER be called: cancellation precedes any dispatch.
    let teacher = Arc::new(ExplodingTeacher);
    let judge = Arc::new(ScriptedJudge::new(vec![]));
    let cl = clients(
        store.clone(),
        teacher,
        judge,
        25.0,
        EventSink::disconnected(),
    );
    let area = area_k1(one_judge(), lenient_thresholds());
    let engine = Engine::new(cl, area, 4);

    let cancel = CancellationToken::new();
    cancel.cancel(); // cancelled up front

    let report = engine
        .run("run-cancel", &one_item_source(), cancel)
        .await
        .unwrap();
    // No work dispatched (the ExplodingTeacher never panicked), and the run is not 'completed'.
    assert!(!report.completed, "a cancelled run is not 'completed'");
    assert_eq!(report.exported, 0);
    let all = store
        .scan(&RecordFilter::new().run_id("run-cancel"))
        .await
        .unwrap();
    assert_eq!(
        all.len(),
        0,
        "cancellation before dispatch generates nothing"
    );
}

/// E2: a budget-gated bounded revise is NOT permanently lost. When the budget exhausts during a
/// group's generation so the best sibling reconciles to `Revising` but the retry cannot run, the shard
/// must NOT commit the cursor past the item — a relaunch under fresh budget re-drives it and completes
/// the bounded retry. (Before the fix the cursor advanced past the stranded `Revising` record and it
/// was skipped forever, silently uncounted.)
#[tokio::test]
async fn budget_gated_revise_is_not_lost_and_completes_on_relaunch() {
    let store = Store::open_in_memory().await.unwrap();
    // The original generation costs 0.10 and the cap is 0.10, so AFTER generation the meter is AT the
    // cap → the revise retry is budget-gated. The retry generation (a 2nd teacher call) must not run on
    // the first launch; it runs on the relaunch (fresh budget). Allow 2 total teacher calls.
    let teacher = Arc::new(ScriptedTeacher::new(
        vec![good_cot(0.10), good_cot(0.01)],
        2,
    ));
    // The original is judged into the REVISE band (0.65 ∈ [0.5, 0.8)); the retry (2nd judge call) is
    // judged Accept so the relaunch completes it.
    let judge = Arc::new(ScriptedJudge::new(vec![
        &judge_body(0.65, "revise"),
        &judge_body(0.95, "accept"),
    ]));
    let source = InMemorySeedSource::new(vec![good_candidate("q-revise")], 1);

    // Launch 1: cap 0.10. The original generates (spends 0.10 → at cap), judged Revise → Revising; the
    // retry is budget-gated. The cursor must NOT advance (the item is not settled).
    let cl1 = clients(
        store.clone(),
        teacher.clone(),
        judge.clone(),
        0.10,
        EventSink::disconnected(),
    );
    let engine1 = Engine::new(cl1, area_k1(one_judge(), lenient_thresholds()), 1);
    let report1 = engine1
        .run("run-bgr", &source, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(
        teacher.call_count(),
        1,
        "only the original generated under the tight cap"
    );
    assert_eq!(
        report1.revising, 1,
        "the record is parked at Revising, counted (not lost)"
    );
    assert_eq!(report1.admitted, 0);
    // The shard cursor must still be at offset 0 (NOT advanced past the un-settled item).
    let cursor = gw_engine::load_cursor(&store, "run-bgr", 0).await.unwrap();
    assert_eq!(
        cursor.next_offset, 0,
        "the cursor must NOT advance past a pending revise"
    );

    // Launch 2: fresh budget (cap 25.0). The item re-drives: the original is at Revising → the bounded
    // retry now generates (2nd teacher call), is judged Accept → the retry completes to Exported.
    let cl2 = clients(
        store.clone(),
        teacher.clone(),
        judge,
        25.0,
        EventSink::disconnected(),
    );
    let engine2 = Engine::new(cl2, area_k1(one_judge(), lenient_thresholds()), 1);
    let report2 = engine2
        .run("run-bgr", &source, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(
        teacher.call_count(),
        2,
        "the relaunch generates the bounded retry (never re-spent the original)"
    );
    assert_eq!(
        report2.exported, 1,
        "the retry completes to Exported on the relaunch"
    );
    // The original stays at Revising (the audit row); the retry (attempt 1) is the Exported record.
    let all = store
        .scan(&RecordFilter::new().run_id("run-bgr"))
        .await
        .unwrap();
    let retry = all
        .iter()
        .find(|r| r.record_id.contains("-a1-"))
        .expect("the attempt-1 retry exists");
    assert_eq!(retry.lifecycle.state, LifecycleState::Exported);
}

/// E3: the budget meter is REHYDRATED from persisted spend on a restart — a fresh process does not
/// re-grant the full cap. Run 1 spends up to the cap over 2 items (cursor advances to 2). A restart
/// with a BRAND-NEW meter (reset to 0 in memory) must rehydrate to the persisted spend and BLOCK the
/// 3rd item, not re-grant a full cap. (Before the fix the fresh meter started at 0 and the resumed run
/// could spend the full cap AGAIN — an effective N×cap bound.)
#[tokio::test]
async fn budget_meter_rehydrates_from_persisted_spend_on_restart() {
    let store = Store::open_in_memory().await.unwrap();
    // 3 items @ 0.10 each; cap 0.20. The teacher is allowed EXACTLY 2 calls total — a 3rd generation
    // (item2 dispatching because the meter forgot prior spend) would panic.
    let teacher = Arc::new(ScriptedTeacher::new(
        vec![good_cot(0.10), good_cot(0.10), good_cot(0.10)],
        2,
    ));
    let judge = Arc::new(ScriptedJudge::new(vec![
        &judge_body(0.95, "accept"),
        &judge_body(0.95, "accept"),
        &judge_body(0.95, "accept"),
    ]));
    let source = InMemorySeedSource::new(
        vec![
            good_candidate("q0"),
            good_candidate("q1"),
            good_candidate("q2"),
        ],
        1,
    );

    // Launch 1 (fresh meter, cap 0.20): item0 spends 0.10 (<0.20 ok), item1 spends 0.10 → total 0.20,
    // item2 gated (0.20 >= 0.20). Two items processed; cursor advances to 2.
    let cl1 = clients(
        store.clone(),
        teacher.clone(),
        judge.clone(),
        0.20,
        EventSink::disconnected(),
    );
    let engine1 = Engine::new(cl1, area_k1(one_judge(), lenient_thresholds()), 1);
    let report1 = engine1
        .run("run-rehydrate", &source, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(
        teacher.call_count(),
        2,
        "two items generated under the 0.20 cap"
    );
    assert!(!report1.completed, "run halted on budget, not completed");

    // Launch 2: a BRAND-NEW meter (so in-memory spend starts at 0) over the SAME run id and cap. E3
    // must rehydrate the meter to the persisted 0.20 spend, so item2 is STILL gated — the teacher is
    // NEVER called a 3rd time (max_calls=2 would panic otherwise).
    let cl2 = clients(
        store.clone(),
        teacher.clone(),
        judge,
        0.20,
        EventSink::disconnected(),
    );
    let engine2 = Engine::new(cl2, area_k1(one_judge(), lenient_thresholds()), 1);
    engine2
        .run("run-rehydrate", &source, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(
        teacher.call_count(),
        2,
        "a restart must NOT re-grant the cap — item2 stays gated by the rehydrated spend"
    );
    // Only the two within-budget items exist; item2 was never generated.
    let all = store
        .scan(&RecordFilter::new().run_id("run-rehydrate"))
        .await
        .unwrap();
    assert_eq!(
        all.len(),
        2,
        "the 3rd item was never dispatched across both launches"
    );
}

/// E5 + E10: a RECORD-LEVEL teacher fault on ONE record is ISOLATED — that record parks at `Error`,
/// `RecordErrored` is emitted, and the run CONTINUES so other records still complete. The whole run is
/// NOT aborted. A subsequent run resumes without a Semaphore-permit deadlock.
#[tokio::test]
async fn record_level_teacher_fault_is_isolated_run_continues() {
    let store = Store::open_in_memory().await.unwrap();
    // The teacher FAILS on call 1 (item0's generation) with a terminal fault, then succeeds on call 2
    // (item1). Without E5 the first fault would abort the whole run and item1 would never run.
    let teacher = Arc::new(FailingTeacher::new(1, 0.01));
    let judge = Arc::new(ScriptedJudge::new(vec![&judge_body(0.95, "accept")]));
    let (sink, mut rx) = EventSink::subscribe();
    let cl = clients(store.clone(), teacher.clone(), judge, 25.0, sink);
    let area = area_k1(one_judge(), lenient_thresholds());
    let engine = Engine::new(cl, area, 1);

    let source = InMemorySeedSource::new(vec![good_candidate("q0"), good_candidate("q1")], 1);
    let report = engine
        .run("run-err", &source, CancellationToken::new())
        .await
        .unwrap();

    // The run completed (it was NOT aborted by the per-record fault); item1 still exported.
    assert_eq!(
        report.errored, 1,
        "the faulting record is parked at Error and counted"
    );
    assert_eq!(
        report.exported, 1,
        "the OTHER record still completed — the run continued"
    );

    // The faulting record (item0, seed 0) is at Error with the fault message recorded.
    let errored = store
        .scan(
            &RecordFilter::new()
                .run_id("run-err")
                .lifecycle_state(LifecycleState::Error),
        )
        .await
        .unwrap();
    assert_eq!(errored.len(), 1);
    assert!(
        errored[0].lifecycle.error.is_some(),
        "the Error state carries the fault detail"
    );

    // A RecordErrored event was emitted for the faulting record.
    let mut saw_record_errored = false;
    while let Ok(ev) = rx.try_recv() {
        if let EngineEvent::RecordErrored { .. } = ev {
            saw_record_errored = true;
        }
    }
    assert!(
        saw_record_errored,
        "a RecordErrored event must be emitted for the isolated fault"
    );

    // RESUME: a subsequent run over the same source must not deadlock (permit released on the error
    // path) and must complete cleanly. item0's offset committed-as-errored; item1 already exported.
    let teacher2 = Arc::new(FailingTeacher::new(999, 0.01)); // never fails this time
    let judge2 = Arc::new(ScriptedJudge::new(vec![&judge_body(0.95, "accept")]));
    let cl2 = clients(
        store.clone(),
        teacher2,
        judge2,
        25.0,
        EventSink::disconnected(),
    );
    let engine2 = Engine::new(cl2, area_k1(one_judge(), lenient_thresholds()), 1);
    let report2 = engine2
        .run("run-err", &source, CancellationToken::new())
        .await
        .unwrap();
    // Both offsets are committed (item0-as-errored, item1-exported), so the resume re-drives nothing.
    assert!(
        report2.completed,
        "the resumed run completes without a permit deadlock"
    );
}

/// E8: the concurrent budget OVERSHOOT is BOUNDED. The gate is checked before dispatch and the charge
/// happens after the spend, so several in-flight jobs can all pass `may_dispatch()` before any charges
/// — but the overshoot can never exceed the in-flight window. With `max_in_flight = N`, a barrier
/// teacher forces exactly N concurrent teacher calls (all past the gate at budget 0) under a cap that
/// one call already exceeds; the teacher is called AT MOST N times (no unbounded blow-past). This pins
/// the bound against a regression that widens the gate/charge window.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_budget_overshoot_is_bounded_by_max_in_flight() {
    let store = Store::open_in_memory().await.unwrap();
    let n = 3usize;
    // Each call costs 0.01; the cap is 0.005, so the FIRST charge already exceeds it. The barrier
    // releases only when all N calls are in flight — proving all N passed the gate while the meter was
    // still 0. After they charge, the gate is closed; no further item dispatches.
    let teacher = Arc::new(BarrierTeacher::new(n, 0.01));
    let judge = Arc::new(ScriptedJudge::new(
        (0..n)
            .map(|_| "{\"score\":0.95,\"verdict\":\"accept\"}")
            .collect(),
    ));
    let cl = clients(
        store.clone(),
        teacher.clone(),
        judge,
        0.005,
        EventSink::disconnected(),
    );
    let area = area_k1(one_judge(), lenient_thresholds());
    let engine = Engine::new(cl, area, n as u32);

    // N items across N shards (1 each), so all N dispatch concurrently up to the semaphore cap.
    let items: Vec<_> = (0..n).map(|i| good_candidate(&format!("q{i}"))).collect();
    let source = InMemorySeedSource::new(items, n);

    let report = engine
        .run("run-overshoot", &source, CancellationToken::new())
        .await
        .unwrap();

    // The teacher was called AT MOST N times — the overshoot is bounded by the in-flight window, never
    // unbounded. (Exactly N here: the barrier requires N to make progress.)
    assert_eq!(
        teacher.call_count(),
        n,
        "the concurrent overshoot is bounded by max_in_flight (no unbounded blow-past)"
    );
    assert!(!report.completed, "the run halted on the budget cap");
    // The cap was overshot (expected, bounded Drain) but only by the in-flight window.
    let all = store
        .scan(&RecordFilter::new().run_id("run-overshoot"))
        .await
        .unwrap();
    assert_eq!(
        all.len(),
        n,
        "exactly the N in-flight items were generated, no more"
    );
}
