//! Task policy regression tests through the actual engine and persisted records.
mod common;
use common::*;
use gw_engine::{Engine, EventSink, InMemorySeedSource};
use gw_storage::{RecordFilter, Store};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn authoritative_unknown_holds_without_judge_spend() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(
        vec![answer_cot("cannot parse this", 0.01)],
        1,
    ));
    let judge = Arc::new(ScriptedJudge::new(vec![&judge_body(0.99, "accept")]));
    let engine = Engine::new(
        clients(
            store.clone(),
            teacher,
            judge.clone(),
            EventSink::disconnected(),
        ),
        area_k1(one_judge(), lenient_thresholds()),
        1,
    );
    let mut candidate = numeric_candidate("Compute six times seven", "42");
    candidate.contract.answer_policy = Some(gw_schema::VerificationPolicy::Authoritative);
    let source = InMemorySeedSource::new(vec![candidate], 1);
    let report = engine
        .run("unknown", &source, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(report.needs_review, 1);
    assert_eq!(judge.call_count(), 0);
}

#[tokio::test]
async fn advisory_failure_remains_factually_failed() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(vec![answer_cot("41", 0.01)], 1));
    let judge = Arc::new(ScriptedJudge::new(vec![&judge_body(0.99, "accept")]));
    let engine = Engine::new(
        clients(
            store.clone(),
            teacher,
            judge.clone(),
            EventSink::disconnected(),
        ),
        area_k1(one_judge(), lenient_thresholds()),
        1,
    );
    let source =
        InMemorySeedSource::new(vec![numeric_candidate("Compute six times seven", "42")], 1);
    let report = engine
        .run("advisory", &source, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(report.admitted, 1);
    assert_eq!(judge.call_count(), 1);
    let records = store
        .scan(&RecordFilter::new().run_id("advisory"))
        .await
        .unwrap();
    let answer = records[0]
        .verification
        .checks
        .iter()
        .find(|check| check.name == "answer_match")
        .unwrap();
    assert!(
        !answer.passed,
        "advisory policy must not rewrite a factual failure into a pass"
    );
}

use gw_generate::UserTurnCandidate;
use gw_schema::{
    ExecutionOutcome, LifecycleState, Oracle, TestStatus, TrainingRecord, VerificationKind,
    VerificationOutcome as Fact, VerificationPolicy as Policy,
};

async fn run_case(
    candidate: UserTurnCandidate,
    answer: &str,
    evidence: Option<Arc<ScriptedEvidence>>,
    score: f64,
) -> (TrainingRecord, usize) {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(vec![answer_cot(answer, 0.01)], 1));
    let judge = Arc::new(ScriptedJudge::new(vec![&judge_body(
        score,
        if score > 0.5 { "accept" } else { "reject" },
    )]));
    let mut cl = clients(
        store.clone(),
        teacher,
        judge.clone(),
        EventSink::disconnected(),
    );
    if let Some(evidence) = evidence {
        cl = cl.with_execution_evidence_source(evidence);
    }
    let engine = Engine::new(cl, area_k1(one_judge(), lenient_thresholds()), 1);
    engine
        .run(
            "policy-case",
            &InMemorySeedSource::new(vec![candidate], 1),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    let rec = store
        .scan(&RecordFilter::new().run_id("policy-case"))
        .await
        .unwrap()
        .remove(0);
    (rec, judge.call_count())
}
fn execution_candidate(policy: Policy) -> UserTurnCandidate {
    let mut candidate = good_candidate("Implement the required behavior");
    candidate.contract.execution_policy = Some(policy);
    candidate.contract.required_tests = vec![EVIDENCE_NODE.into()];
    candidate
}
fn expected_gate(policy: Policy, outcome: Fact) -> (LifecycleState, usize) {
    match (policy, outcome) {
        (Policy::Authoritative, Fact::Fail) => (LifecycleState::Rejected, 0),
        (Policy::Authoritative, Fact::Unknown) => (LifecycleState::NeedsReview, 0),
        _ => (LifecycleState::Formatted, 1),
    }
}
#[tokio::test]
async fn answer_policy_by_factual_outcome_matrix() {
    for policy in [Policy::Absent, Policy::Advisory, Policy::Authoritative] {
        for (answer, outcome) in [
            ("42", Fact::Pass),
            ("41", Fact::Fail),
            ("unparseable", Fact::Unknown),
        ] {
            let mut candidate = numeric_candidate("Compute six times seven", "42");
            candidate.contract.answer_policy = Some(policy);
            let (rec, calls) = run_case(candidate, answer, None, 0.99).await;
            assert_eq!(
                (rec.lifecycle.state, calls),
                expected_gate(policy, outcome),
                "{policy:?}/{outcome:?}"
            );
            let axis = &rec.verification.interpretation.as_ref().unwrap().answer;
            assert_eq!(axis.policy, policy);
            assert_eq!(
                axis.observation.as_ref().map(|fact| fact.outcome),
                (policy != Policy::Absent).then_some(outcome)
            );
            if policy != Policy::Absent {
                assert_eq!(
                    rec.verification
                        .checks
                        .iter()
                        .find(|check| check.name == "answer_match")
                        .unwrap()
                        .passed,
                    outcome == Fact::Pass
                );
            }
        }
    }
}
#[tokio::test]
async fn execution_policy_by_factual_outcome_matrix() {
    for policy in [Policy::Absent, Policy::Advisory, Policy::Authoritative] {
        for outcome in [Fact::Pass, Fact::Fail, Fact::Unknown] {
            let evidence = Arc::new(ScriptedEvidence::new(move |key| match outcome {
                Fact::Pass => Some(passing_report(key)),
                Fact::Fail => Some(failing_report(key)),
                Fact::Unknown => None,
            }));
            let (rec, calls) = run_case(
                execution_candidate(policy),
                "42",
                Some(evidence.clone()),
                0.99,
            )
            .await;
            assert_eq!(
                (rec.lifecycle.state, calls),
                expected_gate(policy, outcome),
                "{policy:?}/{outcome:?}"
            );
            let axis = &rec.verification.interpretation.as_ref().unwrap().execution;
            assert_eq!(axis.policy, policy);
            assert_eq!(
                axis.observation.as_ref().map(|fact| fact.outcome),
                (policy != Policy::Absent).then_some(outcome)
            );
            assert_eq!(evidence.call_count(), usize::from(policy != Policy::Absent));
            if policy != Policy::Absent {
                assert_eq!(
                    rec.verification
                        .checks
                        .iter()
                        .find(|check| check.name == gw_judge::EXECUTION_EVIDENCE_CHECK)
                        .unwrap()
                        .passed,
                    outcome == Fact::Pass
                );
            }
        }
    }
}
#[tokio::test]
async fn passing_authoritative_facts_still_require_quality_panel() {
    let mut candidate = numeric_candidate("Compute six times seven", "42");
    candidate.contract.answer_policy = Some(Policy::Authoritative);
    candidate.contract.execution_policy = Some(Policy::Authoritative);
    candidate.contract.required_tests = vec![EVIDENCE_NODE.into()];
    let (rec, calls) = run_case(
        candidate,
        "42",
        Some(Arc::new(ScriptedEvidence::new(|key| {
            Some(passing_report(key))
        }))),
        0.01,
    )
    .await;
    assert_eq!((rec.lifecycle.state, calls), (LifecycleState::Rejected, 1));
    assert!(rec.verification.all_passed);
}
#[tokio::test]
async fn authoritative_failure_outranks_another_authoritative_unknown() {
    for answer_fails in [true, false] {
        let mut candidate = numeric_candidate("Compute six times seven", "42");
        candidate.contract.answer_policy = Some(Policy::Authoritative);
        candidate.contract.execution_policy = Some(Policy::Authoritative);
        candidate.contract.required_tests = vec![EVIDENCE_NODE.into()];
        let evidence = Arc::new(ScriptedEvidence::new(move |key| {
            if answer_fails {
                None
            } else {
                Some(failing_report(key))
            }
        }));
        let (rec, calls) = run_case(
            candidate,
            if answer_fails { "41" } else { "unknown" },
            Some(evidence),
            0.99,
        )
        .await;
        assert_eq!((rec.lifecycle.state, calls), (LifecycleState::Rejected, 0));
        assert!(!rec.verification.all_passed);
    }
}
#[tokio::test]
async fn both_absent_are_identifiably_judge_only() {
    let (rec, calls) = run_case(
        good_candidate("Write an explanation"),
        "unverifiable",
        None,
        0.99,
    )
    .await;
    assert_eq!((rec.lifecycle.state, calls), (LifecycleState::Formatted, 1));
    let facts = rec.verification.interpretation.unwrap();
    assert_eq!(facts.answer.policy, Policy::Absent);
    assert_eq!(facts.execution.policy, Policy::Absent);
    assert!(facts.answer.observation.is_none() && facts.execution.observation.is_none());
}
#[tokio::test]
async fn task_required_tests_cannot_be_reduced_by_report() {
    let mut candidate = execution_candidate(Policy::Authoritative);
    candidate
        .contract
        .required_tests
        .push("must-also-run".into());
    let (rec, calls) = run_case(
        candidate,
        "42",
        Some(Arc::new(ScriptedEvidence::new(|key| {
            Some(passing_report(key))
        }))),
        0.99,
    )
    .await;
    assert_eq!((rec.lifecycle.state, calls), (LifecycleState::Rejected, 0));
    // The converse: a task owns its requirements even when a report omits its own list.
    let evidence = Arc::new(ScriptedEvidence::new(|key| {
        let mut report = passing_report(key);
        report.required_tests.clear();
        Some(report)
    }));
    let (rec, calls) = run_case(
        execution_candidate(Policy::Authoritative),
        "42",
        Some(evidence),
        0.99,
    )
    .await;
    assert_eq!((rec.lifecycle.state, calls), (LifecycleState::Formatted, 1));
}
#[tokio::test]
async fn exact_required_ids_survive_round_trip_without_trimming() {
    let mut candidate = execution_candidate(Policy::Authoritative);
    candidate.contract.required_tests = vec![" test::id ".into()];
    let evidence = Arc::new(ScriptedEvidence::new(|key| {
        Some(hand_report(
            key,
            ExecutionOutcome::Passed,
            &[],
            &[("test::id", TestStatus::Passed)],
            Some(0),
        ))
    }));
    let (rec, _) = run_case(candidate, "42", Some(evidence), 0.99).await;
    assert_eq!(rec.lifecycle.state, LifecycleState::Rejected);
    assert_eq!(
        rec.verification_contract.unwrap().required_tests,
        [" test::id "]
    );
}
#[tokio::test]
async fn foreign_and_moved_reports_never_pass() {
    for foreign in [true, false] {
        let evidence = Arc::new(ScriptedEvidence::new(move |key| {
            let mut report = passing_report(key);
            if foreign {
                report.binding.attempt = "another-attempt".into();
            } else {
                report.binding.patch_hash = "previous-completion".into();
            }
            Some(report)
        }));
        let (rec, calls) = run_case(
            execution_candidate(Policy::Authoritative),
            "42",
            Some(evidence),
            0.99,
        )
        .await;
        assert_eq!(
            (rec.lifecycle.state, calls),
            (
                if foreign {
                    LifecycleState::Rejected
                } else {
                    LifecycleState::NeedsReview
                },
                0
            )
        );
        let fact = rec
            .verification
            .interpretation
            .unwrap()
            .execution
            .observation
            .unwrap();
        assert_eq!(
            fact.outcome,
            if foreign { Fact::Fail } else { Fact::Unknown }
        );
        if foreign {
            assert!(fact.reason.contains("evidence-contract failure"));
        }
    }
}
#[tokio::test]
async fn invalid_contract_anywhere_in_plan_precedes_all_teacher_dispatch() {
    let mut invalid = vec![];
    let mut candidate = good_candidate("bad answer policy");
    candidate.contract.answer_policy = Some(Policy::Authoritative);
    invalid.push(candidate);
    let mut candidate = numeric_candidate("bad oracle", "42");
    candidate.contract.oracle = Oracle::RefusalPolicy {
        policy_id: "p".into(),
    };
    invalid.push(candidate);
    let mut candidate = execution_candidate(Policy::Authoritative);
    candidate.contract.required_tests.clear();
    invalid.push(candidate);
    let mut candidate = execution_candidate(Policy::Authoritative);
    candidate.contract.required_tests = vec![" ".into()];
    invalid.push(candidate);
    let mut candidate = execution_candidate(Policy::Authoritative);
    candidate.contract.required_tests.push(EVIDENCE_NODE.into());
    invalid.push(candidate);
    let mut candidate = good_candidate("legacy missing declaration");
    candidate.contract.answer_policy = None;
    invalid.push(candidate);
    for candidate in invalid {
        let store = Store::open_in_memory().await.unwrap();
        let teacher = Arc::new(ScriptedTeacher::new(vec![], 0));
        let judge = Arc::new(ScriptedJudge::new(vec![]));
        let engine = Engine::new(
            clients(
                store.clone(),
                teacher.clone(),
                judge.clone(),
                EventSink::disconnected(),
            ),
            area_k1(one_judge(), lenient_thresholds()),
            2,
        );
        let source =
            InMemorySeedSource::new(vec![good_candidate("valid first task"), candidate], 2);
        assert!(
            engine
                .run("invalid-plan", &source, CancellationToken::new())
                .await
                .is_err()
        );
        assert_eq!(teacher.call_count(), 0);
        assert_eq!(judge.call_count(), 0);
        assert!(store.scan(&RecordFilter::new()).await.unwrap().is_empty());
    }
}
#[tokio::test]
async fn conservative_sql_schema_and_unavailable_sandbox_remain_unknown() {
    for kind in [
        VerificationKind::SqlResultMatch,
        VerificationKind::SchemaShape,
        VerificationKind::NumericMatch,
    ] {
        let mut candidate = numeric_candidate("Compare to the oracle", "expected");
        candidate.contract.answer_policy = Some(Policy::Authoritative);
        candidate.contract.kind = kind;
        if kind != VerificationKind::NumericMatch {
            candidate.contract.numeric = None;
        }
        if kind == VerificationKind::NumericMatch {
            candidate.contract.oracle = Oracle::SandboxExecution {
                tool_or_sql: "reference tool".into(),
                expected: None,
            };
        }
        let (rec, calls) = run_case(candidate, "different", None, 0.99).await;
        assert_eq!(
            (rec.lifecycle.state, calls),
            (LifecycleState::NeedsReview, 0)
        );
        assert_eq!(
            rec.verification
                .interpretation
                .unwrap()
                .answer
                .observation
                .unwrap()
                .outcome,
            Fact::Unknown
        );
    }
}
#[tokio::test]
async fn refusal_uses_the_declared_answer_policy() {
    for policy in [Policy::Advisory, Policy::Authoritative] {
        for (answer, outcome) in [
            ("I cannot help with that request.", Fact::Pass),
            ("Sure! Here are the steps.", Fact::Fail),
            ("The topic is complex.", Fact::Unknown),
        ] {
            let mut candidate = refusal_candidate("Decline this request");
            candidate.contract.answer_policy = Some(policy);
            let (rec, calls) = run_case(candidate, answer, None, 0.99).await;
            assert_eq!((rec.lifecycle.state, calls), expected_gate(policy, outcome));
        }
    }
}
