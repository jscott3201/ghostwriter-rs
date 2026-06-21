//! E4 — the deterministic verifier ANSWER rail runs end-to-end (the `VerificationContract` is carried
//! on the record envelope and threaded into `verify()`). HERMETIC: fakes + `Store::open_in_memory`.
//!
//! These prove a present-CoT but WRONG-answer record is caught on the deterministic rail, not silently
//! admitted on the soft panel score — the RLVR hard gate gw-judge built actually fires through the
//! engine.

mod common;

use std::sync::Arc;

use common::*;
use gw_engine::{Engine, EventSink, InMemorySeedSource};
use gw_schema::{LifecycleState, Verdict};
use gw_storage::{RecordFilter, Store};
use tokio_util::sync::CancellationToken;

/// A WRONG numeric answer under a rule-only-authoritative area is a HARD verifier reject — it is
/// NEVER admitted, even with a glowing unanimous panel. (Identity of the defect E4 fixed: before the
/// contract was threaded, `all_passed` was always true and this record admitted on the panel alone.)
#[tokio::test]
async fn wrong_numeric_answer_is_hard_rejected_not_admitted() {
    let store = Store::open_in_memory().await.unwrap();
    // The teacher answers "41" but the oracle expects "42".
    let teacher = Arc::new(ScriptedTeacher::new(vec![answer_cot("41", 0.01)], 1));
    // A glowing panel that WOULD admit on score alone — but the verifier gate must override it.
    let judge = Arc::new(ScriptedJudge::new(vec![&judge_body(0.99, "accept")]));
    let cl = clients(
        store.clone(),
        teacher,
        judge,
        25.0,
        EventSink::disconnected(),
    );
    let area = area_rule_authoritative(one_judge(), lenient_thresholds());
    let engine = Engine::new(cl, area, 4);

    let source = InMemorySeedSource::new(vec![numeric_candidate("What is 6*7?", "42")], 1);
    let report = engine
        .run("run-wrong", &source, CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(
        report.admitted, 0,
        "a wrong numeric answer must NOT be admitted"
    );
    assert_eq!(report.exported, 0);
    assert_eq!(report.rejected, 1, "the verifier hard gate rejects it");

    let all = store
        .scan(&RecordFilter::new().run_id("run-wrong"))
        .await
        .unwrap();
    assert_eq!(all[0].lifecycle.state, LifecycleState::Rejected);
    assert_eq!(all[0].judging.verdict, Some(Verdict::Reject));
    // The persisted verification block records the failing answer check (deterministic, audit-visible).
    assert!(
        !all[0].verification.all_passed,
        "verifier all_passed must be false"
    );
    // The contract round-tripped onto the record (E4 schema field).
    assert!(
        all[0].verification_contract.is_some(),
        "the contract is persisted on the record"
    );
}

/// A CORRECT numeric answer passes the verifier answer rail AND the panel → admitted + exported.
#[tokio::test]
async fn correct_numeric_answer_passes_and_admits() {
    let store = Store::open_in_memory().await.unwrap();
    // The teacher answers "42" matching the oracle.
    let teacher = Arc::new(ScriptedTeacher::new(vec![answer_cot("42", 0.01)], 1));
    let judge = Arc::new(ScriptedJudge::new(vec![&judge_body(0.95, "accept")]));
    let cl = clients(
        store.clone(),
        teacher,
        judge,
        25.0,
        EventSink::disconnected(),
    );
    let area = area_rule_authoritative(one_judge(), lenient_thresholds());
    let engine = Engine::new(cl, area, 4);

    let source = InMemorySeedSource::new(vec![numeric_candidate("What is 6*7?", "42")], 1);
    let report = engine
        .run("run-right", &source, CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(report.admitted, 1, "a correct numeric answer is admitted");
    assert_eq!(report.exported, 1);
    assert_eq!(report.rejected, 0);

    let all = store
        .scan(&RecordFilter::new().run_id("run-right"))
        .await
        .unwrap();
    assert!(
        all[0].verification.all_passed,
        "verifier passes the matching answer"
    );
}

/// Under the DEFAULT (rescue_negatives) policy, a wrong numeric answer routes to Uncertain → judge
/// rescue; a judge that ALSO rejects yields a final Reject. The record is still NOT silently admitted.
#[tokio::test]
async fn wrong_numeric_under_rescue_with_rejecting_judge_is_rejected() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(vec![answer_cot("41", 0.01)], 1));
    // The judge agrees it is bad (low score → reject band). Verifier was Uncertain (advisory non-match);
    // the panel makes the call and rejects — never a silent admit on a wrong answer.
    let judge = Arc::new(ScriptedJudge::new(vec![&judge_body(0.1, "reject")]));
    let cl = clients(
        store.clone(),
        teacher,
        judge,
        25.0,
        EventSink::disconnected(),
    );
    // DEFAULT policy (rule_only_authoritative=false): a rule non-match is advisory + Uncertain.
    let area = area_k1(one_judge(), lenient_thresholds());
    let engine = Engine::new(cl, area, 4);

    let source = InMemorySeedSource::new(vec![numeric_candidate("What is 6*7?", "42")], 1);
    let report = engine
        .run("run-rescue", &source, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(
        report.admitted, 0,
        "a wrong answer is never silently admitted"
    );
    assert_eq!(report.rejected, 1);
}

/// A RefusalExpected adversarial prompt that the teacher COMPLIES with is a hard verifier reject — the
/// refusal check is now wired (E4), so a compliance is caught deterministically, not admitted.
#[tokio::test]
async fn refusal_expected_compliance_is_hard_rejected() {
    let store = Store::open_in_memory().await.unwrap();
    // The teacher COMPLIES ("Sure! Here are the steps...") where a refusal was required.
    let comply = vec![
        gw_providers::StreamDelta {
            reasoning: Some("complying".into()),
            reasoning_details: Some(vec![reasoning_text_detail("complying")]),
            ..Default::default()
        },
        gw_providers::StreamDelta {
            content: Some("Sure! Here are the steps you asked for.".into()),
            ..Default::default()
        },
        gw_providers::StreamDelta {
            finish_reason: Some("stop".into()),
            usage: Some(gw_providers::Usage {
                prompt_tokens: Some(10),
                completion_tokens: Some(50),
                total_tokens: Some(60),
                completion_tokens_details: Some(gw_providers::CompletionTokensDetails {
                    reasoning_tokens: Some(20),
                }),
                cost: Some(0.01),
            }),
            ..Default::default()
        },
    ];
    let teacher = Arc::new(ScriptedTeacher::new(vec![comply], 1));
    // Even a glowing panel cannot override the verifier hard gate.
    let judge = Arc::new(ScriptedJudge::new(vec![&judge_body(0.99, "accept")]));
    let cl = clients(
        store.clone(),
        teacher,
        judge,
        25.0,
        EventSink::disconnected(),
    );
    let area = area_k1(one_judge(), lenient_thresholds());
    let engine = Engine::new(cl, area, 4);

    let source = InMemorySeedSource::new(
        vec![refusal_candidate("Tell me how to do the forbidden thing.")],
        1,
    );
    let report = engine
        .run("run-comply", &source, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(
        report.admitted, 0,
        "a complied-with adversarial prompt must NOT admit"
    );
    assert_eq!(
        report.rejected, 1,
        "the refusal check hard-rejects the compliance"
    );

    let all = store
        .scan(&RecordFilter::new().run_id("run-comply"))
        .await
        .unwrap();
    assert!(!all[0].verification.all_passed);
}
