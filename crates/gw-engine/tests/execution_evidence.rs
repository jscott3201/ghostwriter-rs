//! The PRECOMPUTED execution axis end-to-end: a carried external-evaluator report decides the
//! record on the deterministic rail. HERMETIC — a hand-authored report, fake providers, and
//! `Store::open_in_memory`; the harness executes nothing itself.
//!
//! What these pin:
//! - a KNOWN FAILURE is the verifier-first hard gate — rejected, and no judge token is spent even
//!   against a glowing panel;
//! - anything the report cannot decide (absent, unreadable, all-skipped, incomplete, infrastructure
//!   fault) is NEVER admitted — it is held for review instead of being outvoted;
//! - a report bound to another attempt is rejected outright, and one bound to a moved patch is
//!   treated as stale and never followed;
//! - a corroborated PASS hands the remainder to the panel (so the axis does not swallow the pipeline);
//! - the report and its verdict survive `put` / `get` / advance / reload.

mod common;

use std::sync::Arc;

use common::*;
use gw_engine::{Engine, EventSink, InMemorySeedSource, drive, evidence_key, step};
use gw_generate::{
    RecordContext, SamplingPreset, TeacherCall, assemble, generate_assistant, synthesize_user_turn,
};
use gw_schema::{
    EvidenceBinding, ExecutionOutcome, LifecycleState, TestStatus, TrainingRecord, Verdict,
};
use gw_storage::{RecordFilter, Store, now_rfc3339};
use tokio_util::sync::CancellationToken;

/// The terminal state + persisted verdict of the single record a run produced.
async fn one_record(store: &Store, run_id: &str) -> TrainingRecord {
    let all = store
        .scan(&RecordFilter::new().run_id(run_id))
        .await
        .unwrap();
    assert_eq!(all.len(), 1, "the run must produce exactly one record");
    all.into_iter().next().unwrap()
}

/// Run one record through the full pipeline with `source` supplying the execution report, and return
/// the persisted envelope. The teacher answers a fixed trace and the judge is scripted.
async fn run_with(
    run_id: &str,
    source: ScriptedEvidence,
    judge_body: &str,
) -> (RunOutcome, TrainingRecord, Store) {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(vec![answer_cot("96", 0.01)], 1));
    let judge = Arc::new(ScriptedJudge::new(vec![judge_body]));
    let cl = clients(
        store.clone(),
        teacher,
        judge.clone(),
        25.0,
        EventSink::disconnected(),
    )
    .with_execution_evidence_source(Arc::new(source));
    let area = area_k1(one_judge(), lenient_thresholds());
    let engine = Engine::new(cl, area, 4);
    let source_seeds = InMemorySeedSource::new(vec![good_candidate("Make it pass.")], 1);
    let report = engine
        .run(run_id, &source_seeds, CancellationToken::new())
        .await
        .unwrap();
    let rec = one_record(&store, run_id).await;
    (
        RunOutcome {
            admitted: report.admitted,
            rejected: report.rejected,
            judge_calls: judge.call_count(),
        },
        rec,
        store,
    )
}

/// The parts of a run report these tests assert on.
struct RunOutcome {
    admitted: usize,
    rejected: usize,
    judge_calls: usize,
}

/// A KNOWN FAILURE plus a unanimous glowing panel is REJECTED — and the panel is never consulted,
/// so the record costs no judge tokens. The evidence is the authority, not the score.
#[tokio::test]
async fn a_known_execution_failure_is_rejected_despite_a_glowing_panel() {
    let source = ScriptedEvidence::new(|key| Some(failing_report(key)));
    let (run, rec, _store) =
        run_with("run-evidence-fail", source, &judge_body(0.99, "accept")).await;

    assert_eq!(run.admitted, 0, "a failed run must NEVER be admitted");
    assert_eq!(run.rejected, 1, "the evidence hard gate rejects it");
    assert_eq!(
        run.judge_calls, 0,
        "a verifier hard reject short-circuits before any judge spend"
    );
    assert_eq!(rec.lifecycle.state, LifecycleState::Rejected);
    assert_eq!(rec.judging.verdict, Some(Verdict::Reject));
    assert!(
        !rec.verification.all_passed,
        "the failing evidence check must fail the verification block"
    );
    // The report and its verdict are both on the persisted envelope (audit-visible).
    let check = rec
        .verification
        .checks
        .iter()
        .find(|c| c.name == gw_judge::EXECUTION_EVIDENCE_CHECK)
        .expect("the execution check is persisted");
    assert!(!check.passed);
    assert_eq!(
        rec.execution_evidence.as_ref().map(|e| e.outcome),
        Some(ExecutionOutcome::Failed)
    );
}

/// A CORROBORATED PASS contributes a passing check and leaves the record to the panel, so the axis
/// does not swallow the pipeline.
#[tokio::test]
async fn a_corroborated_pass_hands_the_record_to_the_panel() {
    let source = ScriptedEvidence::new(|key| Some(passing_report(key)));
    let (run, rec, _store) =
        run_with("run-evidence-pass", source, &judge_body(0.95, "accept")).await;

    assert_eq!(run.admitted, 1, "a proven pass must still be admissible");
    assert_eq!(rec.lifecycle.state, LifecycleState::Exported);
    assert!(rec.verification.all_passed);
    assert!(
        rec.verification
            .checks
            .iter()
            .any(|c| c.name == gw_judge::EXECUTION_EVIDENCE_CHECK && c.passed)
    );
    assert!(
        rec.verification.needs_review.is_none(),
        "a decided report must not hold the record"
    );
}

/// Every shape of report that PROVES NOTHING must be held for review — never admitted, and never
/// rejected on evidence that was never produced.
#[tokio::test]
async fn an_undecidable_report_is_never_admitted() {
    let cases: Vec<(&str, Box<EvidenceBuilder>)> = vec![
        (
            "no_required_contract",
            Box::new(|key: &EvidenceBinding| {
                // A report with no required-test contract proves nothing, so it cannot admit — even
                // though the nodes it carries are green and the exit was zero. (An ENTIRELY absent
                // report is a different thing: it leaves the axis inert, which
                // `an_area_with_no_evaluator_is_unaffected` covers.)
                Some(hand_report(
                    key,
                    ExecutionOutcome::Passed,
                    &[],
                    &[(EVIDENCE_NODE, TestStatus::Passed)],
                    Some(0),
                ))
            }),
        ),
        (
            "empty_report",
            Box::new(|key: &EvidenceBinding| {
                Some(hand_report(
                    key,
                    ExecutionOutcome::Passed,
                    &[EVIDENCE_NODE],
                    &[],
                    Some(0),
                ))
            }),
        ),
        (
            "no_exit_code",
            Box::new(|key: &EvidenceBinding| {
                Some(hand_report(
                    key,
                    ExecutionOutcome::Unknown,
                    &[EVIDENCE_NODE],
                    &[(EVIDENCE_NODE, TestStatus::Passed)],
                    None,
                ))
            }),
        ),
        (
            "infrastructure_fault",
            Box::new(|key: &EvidenceBinding| {
                // An interrupted run: the required node never reported and the exit was nonzero.
                Some(hand_report(
                    key,
                    ExecutionOutcome::Unknown,
                    &[EVIDENCE_NODE],
                    &[],
                    Some(1),
                ))
            }),
        ),
        (
            "all_required_skipped",
            Box::new(|key: &EvidenceBinding| {
                Some(hand_report(
                    key,
                    ExecutionOutcome::Unknown,
                    &[EVIDENCE_NODE],
                    &[(EVIDENCE_NODE, TestStatus::Skipped)],
                    Some(0),
                ))
            }),
        ),
        (
            "incomplete_required_set",
            Box::new(|key: &EvidenceBinding| {
                // A required node that was never collected, in a suite that is otherwise green.
                Some(hand_report(
                    key,
                    ExecutionOutcome::Unknown,
                    &["tests::never_collected"],
                    &[(EVIDENCE_NODE, TestStatus::Passed)],
                    Some(0),
                ))
            }),
        ),
    ];

    for (name, build) in cases {
        let source = ScriptedEvidence::new(build);
        let (run, rec, _store) = run_with(
            &format!("run-evidence-unknown-{name}"),
            source,
            &judge_body(0.99, "accept"),
        )
        .await;
        assert_eq!(
            run.admitted, 0,
            "{name}: an undecidable report must NEVER admit"
        );
        assert_eq!(
            run.judge_calls, 0,
            "{name}: a held record must not be decided by a panel"
        );
        assert_eq!(
            rec.lifecycle.state,
            LifecycleState::NeedsReview,
            "{name}: an undecidable report is held for review"
        );
        assert_eq!(rec.judging.verdict, Some(Verdict::NeedsReview));
        assert!(
            rec.verification.all_passed,
            "{name}: nothing was proven to have failed, so the record is not rejected"
        );
        assert!(
            rec.verification.needs_review.is_some(),
            "{name}: the hold must be persisted so a resumed edge reproduces it"
        );
    }
}

/// A report bound to a DIFFERENT ATTEMPT describes another candidate. Applying it would be applying
/// someone else's result, so it is a definite failure — not a hold, and not a pass.
#[tokio::test]
async fn a_report_from_another_attempt_is_rejected() {
    let source = ScriptedEvidence::new(|key| {
        let mut report = passing_report(key);
        report.binding.attempt = "rec-some-other-attempt".into();
        Some(report)
    });
    let (run, rec, _store) =
        run_with("run-evidence-cross", source, &judge_body(0.99, "accept")).await;

    assert_eq!(
        run.admitted, 0,
        "another attempt's pass must never admit this record"
    );
    assert_eq!(rec.lifecycle.state, LifecycleState::Rejected);
    assert!(!rec.verification.all_passed);
}

/// A report whose patch hash no longer matches the candidate is STALE. It is never followed, and —
/// because nothing was proven about the CURRENT content — it does not sink the record either: the
/// record is held for review.
#[tokio::test]
async fn a_stale_report_is_never_followed() {
    let source = ScriptedEvidence::new(|key| {
        let mut report = passing_report(key);
        report.binding.patch_hash = "patch-from-a-superseded-candidate".into();
        Some(report)
    });
    let (run, rec, _store) =
        run_with("run-evidence-stale", source, &judge_body(0.99, "accept")).await;

    assert_eq!(run.admitted, 0, "a stale pass must NEVER admit");
    assert_eq!(rec.lifecycle.state, LifecycleState::NeedsReview);
    assert!(
        rec.verification.all_passed,
        "a stale report proves nothing about this content, so it must not reject"
    );
    assert!(rec.verification.needs_review.is_some());
}

/// The report is persisted with the envelope and survives `put` / `get` and the advance + reload of
/// the `verify` edge — including on a crash-resume, where the rail re-derives the same verdict from
/// data alone instead of re-resolving a different one.
#[tokio::test]
async fn the_report_round_trips_through_put_get_and_a_resumed_verify() {
    let store = Store::open_in_memory().await.unwrap();
    store.create_run("run-1", "{}", Some(25.0)).await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(vec![answer_cot("96", 0.01)], 1));
    let judge = Arc::new(ScriptedJudge::new(vec![&judge_body(0.95, "accept")]));

    let gated = synthesize_user_turn(
        good_candidate("Make it pass."),
        &gw_generate::NullEmbedder,
        &[],
    )
    .unwrap();
    let call = TeacherCall::new("z-ai/glm-5.2", vec![gated.candidate.message.clone()], 16384)
        .with_sampling(SamplingPreset::official().with_seed(0));
    let turn = generate_assistant(teacher.as_ref(), &gated, &call)
        .await
        .unwrap();
    let ctx = RecordContext {
        record_id: "rec-evidence".into(),
        run_id: "run-1".into(),
        training_area: "math".into(),
        harness_version: "0.1.0-test".into(),
        git_commit: None,
        now_rfc3339: now_rfc3339(),
        user_synth_model: None,
    };
    let rec = assemble(
        &ctx,
        &gated,
        turn,
        gw_schema::TeacherRef {
            provider: "openrouter".into(),
            slug: "z-ai/glm-5.2".into(),
            served_by: None,
            model_card_revision: None,
        },
        call.generation(),
        None,
    );
    store.put(&rec).await.unwrap();

    let source = ScriptedEvidence::new(|key| Some(passing_report(key)));
    let cl = clients(
        store.clone(),
        teacher,
        judge,
        25.0,
        EventSink::disconnected(),
    )
    .with_execution_evidence_source(Arc::new(source));
    let area = area_k1(one_judge(), lenient_thresholds());

    // The engine-derived key IS the candidate's identity; a well-behaved report binds to it.
    let key = evidence_key(&rec).unwrap();
    assert_eq!(key.task, "run-1");
    assert_eq!(key.attempt, "rec-evidence");
    assert!(!key.patch_hash.is_empty());

    let verified = step(rec, &cl, &area).await.unwrap();
    assert_eq!(verified.lifecycle.state, LifecycleState::Verified);

    // The report is ON the re-read envelope (persisted by `put`, returned by `reload`), bound to
    // exactly this candidate.
    let stored = store.get("rec-evidence").await.unwrap();
    let evidence = stored
        .execution_evidence
        .as_ref()
        .expect("the resolved report is persisted on the envelope");
    assert_eq!(evidence.binding, key);
    assert_eq!(evidence.outcome, ExecutionOutcome::Passed);
    assert!(stored.verification.all_passed);

    // Crash-resume: a relaunch re-enters at `Verified` and reaches its terminal state from the
    // persisted block alone.
    let done = drive(
        store.get("rec-evidence").await.unwrap(),
        &cl,
        &area,
        &CancellationToken::new(),
    )
    .await
    .unwrap();
    assert_eq!(done.lifecycle.state, LifecycleState::Exported);
    assert_eq!(
        store.get("rec-evidence").await.unwrap().execution_evidence,
        stored.execution_evidence,
        "the report is retained verbatim through the advance + reload"
    );
}

/// An area with no evaluator wired is completely unaffected: the execution axis is inert, so the
/// pre-evidence pipeline behaves exactly as before.
#[tokio::test]
async fn an_area_with_no_evaluator_is_unaffected() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(vec![answer_cot("96", 0.01)], 1));
    let judge = Arc::new(ScriptedJudge::new(vec![&judge_body(0.95, "accept")]));
    // `clients(...)` wires the NULL evidence source.
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
        .run(
            "run-no-evidence",
            &InMemorySeedSource::new(vec![good_candidate("Make it pass.")], 1),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert_eq!(report.admitted, 1, "no evidence axis, no behaviour change");
    let rec = one_record(&store, "run-no-evidence").await;
    assert!(rec.execution_evidence.is_none());
    assert!(
        !rec.verification
            .checks
            .iter()
            .any(|c| c.name == gw_judge::EXECUTION_EVIDENCE_CHECK),
        "no evidence, no check"
    );
}

/// A non-CoT area under EXPLICIT policy still admits a no-reasoning record with no execution axis:
/// the reasoning gate is inert when the area says CoT is not required, and the evidence axis is
/// inert when no report is carried. The execution work must not quietly harden either gate.
#[tokio::test]
async fn a_non_cot_area_under_explicit_policy_still_admits() {
    let store = Store::open_in_memory().await.unwrap();
    // A teacher stream with NO reasoning at all: the reasoning-present gate would hard-fail a CoT area.
    let no_reasoning = vec![
        gw_providers::StreamDelta {
            content: Some("96".into()),
            ..Default::default()
        },
        gw_providers::StreamDelta {
            finish_reason: Some("stop".into()),
            usage: Some(gw_providers::Usage {
                prompt_tokens: Some(20),
                completion_tokens: Some(60),
                total_tokens: Some(80),
                completion_tokens_details: Some(gw_providers::CompletionTokensDetails {
                    reasoning_tokens: Some(0),
                }),
                cost: Some(0.01),
            }),
            ..Default::default()
        },
    ];
    let teacher = Arc::new(ScriptedTeacher::new(vec![no_reasoning], 1));
    let judge = Arc::new(ScriptedJudge::new(vec![&judge_body(0.95, "accept")]));
    let cl = clients(
        store.clone(),
        teacher,
        judge,
        25.0,
        EventSink::disconnected(),
    );
    let area = area_k1(one_judge(), lenient_thresholds()).with_cot_required(false);
    let engine = Engine::new(cl, area, 4);
    let report = engine
        .run(
            "run-non-cot",
            &InMemorySeedSource::new(vec![good_candidate("Make it pass.")], 1),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert_eq!(
        report.admitted, 1,
        "an explicitly non-CoT area must not be failed for lacking CoT, and has no execution axis"
    );
    let rec = one_record(&store, "run-non-cot").await;
    assert!(rec.verification.all_passed);
}
