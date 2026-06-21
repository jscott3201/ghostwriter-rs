//! HERMETIC end-to-end pipeline tests for `gw-engine`: the happy path, the best-of-k retain
//! invariant, the verdict→lifecycle mapping (incl. `NeedsReview` leaving the pipeline), the budget
//! cutoff, and the R-prior never-identity guard. Fakes + `Store::open_in_memory`, no network.

mod common;

use std::sync::Arc;

use common::*;
use gw_engine::{Engine, EngineEvent, EventSink, InMemorySeedSource, correlation_prior};
use gw_schema::{LifecycleState, Verdict};
use gw_storage::{RecordFilter, Store};
use tokio_util::sync::CancellationToken;

/// A clean single-record run drives all the way to `Exported` and counts as admitted.
#[tokio::test]
async fn happy_path_drives_to_exported() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(vec![good_cot(0.01)], 1));
    let judge = Arc::new(ScriptedJudge::new(vec![&judge_body(0.95, "accept")]));
    let (sink, mut rx) = EventSink::subscribe();
    let cl = clients(store.clone(), teacher, judge, 25.0, sink);
    let area = area_k1(one_judge(), lenient_thresholds());
    let engine = Engine::new(cl, area, 4);

    let report = engine
        .run("run-1", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(report.exported, 1, "the clean record exports");
    assert_eq!(report.admitted, 1);
    assert_eq!(report.rejected, 0);
    assert!(report.completed);

    // The record reached Exported in storage.
    let exported = store
        .scan(
            &RecordFilter::new()
                .run_id("run-1")
                .lifecycle_state(LifecycleState::Exported),
        )
        .await
        .unwrap();
    assert_eq!(exported.len(), 1);

    // The event stream carried a RunFinished{completed:true}.
    let mut saw_finish = false;
    while let Ok(ev) = rx.try_recv() {
        if let EngineEvent::RunFinished { completed, .. } = ev {
            assert!(completed);
            saw_finish = true;
        }
    }
    assert!(saw_finish, "a RunFinished event must be emitted");
}

/// MANDATORY 7 (verdict→lifecycle): a below-threshold panel REJECTS; the record is not exported and
/// not counted admitted.
#[tokio::test]
async fn reject_maps_to_rejected_not_exported() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(vec![good_cot(0.01)], 1));
    // A low score → reject band (below reject_below=0.5).
    let judge = Arc::new(ScriptedJudge::new(vec![&judge_body(0.2, "reject")]));
    let cl = clients(
        store.clone(),
        teacher,
        judge,
        25.0,
        EventSink::disconnected(),
    );
    let area = area_k1(one_judge(), lenient_thresholds());
    let engine = Engine::new(cl, area, 4);

    let report = engine
        .run("run-r", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(report.rejected, 1);
    assert_eq!(report.admitted, 0);
    assert_eq!(report.exported, 0);

    // The persisted record carries verdict=reject and is at Rejected.
    let rejected = store
        .scan(&RecordFilter::new().run_id("run-r").verdict(Verdict::Reject))
        .await
        .unwrap();
    assert_eq!(rejected.len(), 1);
    assert_eq!(rejected[0].lifecycle.state, LifecycleState::Rejected);
}

/// MANDATORY 7 (NeedsReview leaves the pipeline): a correlated 3-judge panel under the DEFAULT n_eff
/// floor ESCALATES → `NeedsReview`. The record is NOT exported and NOT counted admitted.
#[tokio::test]
async fn escalate_maps_to_needs_review_and_leaves_pipeline() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(vec![good_cot(0.01)], 1));
    // Three high-scoring judges; with the cold-start rho=0.7 prior and the DEFAULT min_n_eff=1.5, a
    // 3-judge panel has n_eff≈1.25 < 1.5 → escalate (the correlation guard, NOT identity R).
    let judge = Arc::new(ScriptedJudge::new(vec![
        &judge_body(0.95, "accept"),
        &judge_body(0.95, "accept"),
        &judge_body(0.95, "accept"),
    ]));
    let cl = clients(
        store.clone(),
        teacher,
        judge,
        25.0,
        EventSink::disconnected(),
    );
    // DEFAULT thresholds (min_n_eff=1.5) — the escalation floor.
    let area = area_k1(three_judges(), Default::default());
    let engine = Engine::new(cl, area, 4);

    let report = engine
        .run("run-e", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(report.needs_review, 1, "the correlated panel must escalate");
    assert_eq!(report.admitted, 0, "NeedsReview is NOT counted admitted");
    assert_eq!(report.exported, 0, "NeedsReview is NOT exported");

    let parked = store
        .scan(
            &RecordFilter::new()
                .run_id("run-e")
                .lifecycle_state(LifecycleState::NeedsReview),
        )
        .await
        .unwrap();
    assert_eq!(parked.len(), 1);
    assert_eq!(parked[0].judging.verdict, Some(Verdict::NeedsReview));
}

/// MANDATORY 3 (best-of-k): a k=3 fan-out admits the highest-aggregate sibling (verifier gate
/// passing) AND retains the rejected siblings — none dropped.
#[tokio::test]
async fn best_of_k_admits_best_and_retains_rejected_siblings() {
    let store = Store::open_in_memory().await.unwrap();
    // 3 siblings, each a good CoT with a DISTINCT answer (distinct record_hash → distinct judge cache
    // key, so the siblings get independent grades while sharing one prompt_hash group id). Sibling 0:
    // 0.95 accept (best), sibling 1: 0.85 accept, sibling 2: 0.2 reject.
    let teacher = Arc::new(ScriptedTeacher::new(
        vec![
            answer_cot("96", 0.01),
            answer_cot("97", 0.01),
            answer_cot("98", 0.01),
        ],
        3,
    ));
    // One judge body per distinct content (cache miss per distinct answer): 0.95, 0.85, 0.2.
    let judge = Arc::new(ScriptedJudge::new(vec![
        &judge_body(0.95, "accept"),
        &judge_body(0.85, "accept"),
        &judge_body(0.2, "reject"),
    ]));
    let cl = clients(
        store.clone(),
        teacher.clone(),
        judge,
        25.0,
        EventSink::disconnected(),
    );
    let area = area_k(one_judge(), lenient_thresholds(), 3);
    let engine = Engine::new(cl, area, 4);

    let report = engine
        .run("run-k", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();

    // 3 teacher calls (one per sibling), all three siblings persisted (none dropped).
    assert_eq!(teacher.call_count(), 3);
    let all = store
        .scan(&RecordFilter::new().run_id("run-k"))
        .await
        .unwrap();
    assert_eq!(all.len(), 3, "all three siblings retained — none dropped");

    // Exactly one admitted (the best, sibling 0 @ 0.95); at least one rejected sibling retained.
    assert!(report.admitted >= 1, "the best sibling is admitted");
    assert!(
        report.rejected >= 1,
        "a rejected sibling is retained, not dropped"
    );

    // The admitted sibling has the HIGHEST aggregate among admitted siblings.
    let admitted: Vec<_> = all
        .iter()
        .filter(|r| {
            matches!(
                r.lifecycle.state,
                LifecycleState::Admitted | LifecycleState::Formatted | LifecycleState::Exported
            )
        })
        .collect();
    let best_agg = admitted
        .iter()
        .filter_map(|r| r.judging.aggregate)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        (best_agg - 0.95).abs() < 1e-6,
        "the admitted sibling is the highest-aggregate one (0.95), got {best_agg}"
    );

    // All siblings share one sibling_group_id == prompt_hash.
    let group_ids: std::collections::BTreeSet<_> = all
        .iter()
        .filter_map(|r| r.generation.sibling_group_id.clone())
        .collect();
    assert_eq!(
        group_ids.len(),
        1,
        "siblings share one group id (== prompt_hash)"
    );
}

/// MANDATORY 3 (group-level admission): when TWO siblings would BOTH individually pass the admit
/// threshold, ONLY the highest-aggregate one is admitted — the other is RETAINED as `Rejected`, never
/// independently admitted/exported. This guards the group-level-admission invariant (a per-sibling
/// drive would wrongly admit both).
#[tokio::test]
async fn best_of_k_admits_only_one_when_two_would_pass() {
    let store = Store::open_in_memory().await.unwrap();
    // Two siblings, both clearly above the 0.80 accept threshold (0.95 and 0.90). Distinct answers →
    // distinct content → independent grades.
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
    let engine = Engine::new(cl, area, 4);

    let report = engine
        .run("run-k2", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();

    // EXACTLY one admitted (the 0.95 sibling); the 0.90 sibling — which on its own would pass the 0.80
    // threshold — is RETAINED as Rejected, NOT admitted/exported.
    assert_eq!(report.admitted, 1, "exactly ONE sibling admitted, not both");
    assert_eq!(report.exported, 1);
    assert_eq!(
        report.rejected, 1,
        "the runner-up is retained, not admitted"
    );

    let all = store
        .scan(&RecordFilter::new().run_id("run-k2"))
        .await
        .unwrap();
    assert_eq!(all.len(), 2, "both siblings retained — none dropped");
    // The admitted one is the 0.95 sibling; the retained one's verdict is Reject.
    let admitted: Vec<_> = all
        .iter()
        .filter(|r| {
            matches!(
                r.lifecycle.state,
                LifecycleState::Exported | LifecycleState::Admitted
            )
        })
        .collect();
    assert_eq!(admitted.len(), 1);
    assert_eq!(admitted[0].judging.aggregate, Some(0.95));
}

/// E9: a best-of-k aggregate TIE admits exactly ONE sibling deterministically — the lowest
/// `completion_index` (the selection uses strict `>` so the first candidate keeps the tie). The other
/// is retained at `Rejected`, never dropped, never a second admit.
#[tokio::test]
async fn best_of_k_aggregate_tie_admits_lowest_index_only() {
    let store = Store::open_in_memory().await.unwrap();
    // Two siblings with distinct answers (independent grades) but the SAME judge score → SAME aggregate.
    let teacher = Arc::new(ScriptedTeacher::new(
        vec![answer_cot("96", 0.01), answer_cot("97", 0.01)],
        2,
    ));
    let judge = Arc::new(ScriptedJudge::new(vec![
        &judge_body(0.90, "accept"),
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
    let engine = Engine::new(cl, area, 4);

    let report = engine
        .run("run-tie", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(report.admitted, 1, "a tie admits exactly ONE sibling");
    assert_eq!(
        report.rejected, 1,
        "the tied runner-up is retained, not dropped"
    );

    let all = store
        .scan(&RecordFilter::new().run_id("run-tie"))
        .await
        .unwrap();
    assert_eq!(all.len(), 2, "both siblings retained — none dropped");
    // The admitted sibling is completion_index 0 (the deterministic tie-break).
    let admitted: Vec<_> = all
        .iter()
        .filter(|r| {
            matches!(
                r.lifecycle.state,
                LifecycleState::Exported | LifecycleState::Admitted | LifecycleState::Formatted
            )
        })
        .collect();
    assert_eq!(admitted.len(), 1);
    assert_eq!(
        admitted[0].generation.completion_index,
        Some(0),
        "the tie is broken toward the lowest completion_index"
    );
}

/// MANDATORY 6 (budget cutoff): once `cap_usd` is reached, no new teacher work is dispatched.
#[tokio::test]
async fn budget_cutoff_stops_new_teacher_work() {
    let store = Store::open_in_memory().await.unwrap();
    // The first teacher call costs 0.10 and the cap is 0.10, so AFTER the first record the meter is at
    // 0.10 >= cap → the gate closes and the SECOND record's generation is never dispatched. The
    // ScriptedTeacher's max_calls=1 asserts the second call would panic if the gate failed to close.
    let teacher = Arc::new(ScriptedTeacher::new(
        vec![good_cot(0.10), good_cot(0.10)],
        1,
    ));
    let judge = Arc::new(ScriptedJudge::new(vec![
        &judge_body(0.95, "accept"),
        &judge_body(0.95, "accept"),
    ]));
    let cl = clients(
        store.clone(),
        teacher.clone(),
        judge,
        0.10,
        EventSink::disconnected(),
    );
    let area = area_k1(one_judge(), lenient_thresholds());
    let engine = Engine::new(cl, area, 1); // serial: deterministic budget ordering

    // Two seed items in one shard.
    let source = InMemorySeedSource::new(vec![good_candidate("q1"), good_candidate("q2")], 1);
    let report = engine
        .run("run-b", &source, CancellationToken::new())
        .await
        .unwrap();

    // Only ONE teacher call was dispatched: the first record's 0.10 spend reached the 0.15 cap (>=),
    // so the second record's generation was gated (ScriptedTeacher max_calls=1 would panic on a 2nd).
    assert_eq!(
        teacher.call_count(),
        1,
        "budget cap gates the second teacher call"
    );
    // The run is NOT marked completed (it halted on budget).
    assert!(!report.completed, "a budget-halted run is not 'completed'");
    // Exactly one record exists (the second was never generated).
    let all = store
        .scan(&RecordFilter::new().run_id("run-b"))
        .await
        .unwrap();
    assert_eq!(all.len(), 1);
}

/// MANDATORY 4 (R-prior): the engine builds a NON-IDENTITY `uniform_offdiagonal(k, ~0.7)` for a k>1
/// panel — asserted directly via the public `correlation_prior` the step machine uses.
#[tokio::test]
async fn r_prior_is_non_identity_for_multi_judge_panel() {
    // The engine's own constructor: k>1 with the cold-start prior is NON-identity.
    let r = correlation_prior(3, gw_engine::DEFAULT_CORRELATION_RHO).unwrap();
    assert!(
        !r.is_identity(),
        "k>1 panel must NOT be graded with identity R"
    );
    assert_eq!(r.dim(), 3);

    // And a degenerate rho that WOULD make it identity fails loud (the engine never silently degrades).
    assert!(correlation_prior(3, 0.0).is_err());
}

/// A multi-judge run actually grades with the non-identity prior end-to-end (it reaches a decision,
/// not a panic, and the persisted n_eff reflects the correlation discount, not the identity k).
#[tokio::test]
async fn multi_judge_run_uses_correlation_discounted_n_eff() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(vec![good_cot(0.01)], 1));
    let judge = Arc::new(ScriptedJudge::new(vec![
        &judge_body(0.9, "accept"),
        &judge_body(0.9, "accept"),
        &judge_body(0.9, "accept"),
    ]));
    let cl = clients(
        store.clone(),
        teacher,
        judge,
        25.0,
        EventSink::disconnected(),
    );
    // Lenient floor so it admits, but the n_eff is still the CORRELATION-discounted value.
    let area = area_k1(three_judges(), lenient_thresholds());
    let engine = Engine::new(cl, area, 4);

    engine
        .run("run-m", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();

    let all = store
        .scan(&RecordFilter::new().run_id("run-m"))
        .await
        .unwrap();
    let n_eff = all[0].judging.n_eff.expect("n_eff persisted");
    // With identity R, n_eff would be ~3. With rho=0.7 it is ~1.25 — proving non-identity R was used.
    assert!(
        n_eff < 2.0,
        "n_eff={n_eff} must reflect the correlation discount (identity R would give ~3)"
    );
}
