//! Replay uses persisted policy/facts, and never adopts historical booleans as current evidence.
mod common;
use common::*;
use gw_engine::{
    Engine, EventSink, InMemorySeedSource, RunControl, SeedSource, record_id, revise_once,
    run_group, step,
};
use gw_generate::{
    RecordContext, SamplingPreset, TeacherCall, UserTurnCandidate, assemble, generate_assistant,
    synthesize_user_turn,
};
use gw_judge::SandboxOracle;
use gw_schema::{LifecycleState, Oracle, TrainingRecord, VerificationOutcome, VerificationPolicy};
use gw_storage::{RecordFilter, Store, now_rfc3339};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokio_util::sync::CancellationToken;

async fn generated(
    store: &Store,
    run: &str,
    id: &str,
    candidate: UserTurnCandidate,
) -> TrainingRecord {
    let gated = synthesize_user_turn(candidate, &gw_generate::NullEmbedder, &[])
        .await
        .unwrap();
    let call = TeacherCall::new("teacher", vec![gated.candidate.message.clone()], 16384)
        .with_sampling(SamplingPreset::official().with_seed(0));
    let teacher = ScriptedTeacher::new(vec![answer_cot("42", 0.01)], 1);
    let turn = generate_assistant(&teacher, &gated, &call).await.unwrap();
    let rec = assemble(
        &RecordContext {
            record_id: id.into(),
            run_id: run.into(),
            training_area: "math".into(),
            harness_version: "test".into(),
            git_commit: None,
            now_rfc3339: now_rfc3339(),
            user_synth_model: None,
        },
        &gated,
        turn,
        gw_schema::TeacherRef {
            provider: "openrouter".into(),
            slug: "teacher".into(),
            served_by: None,
            model_card_revision: None,
        },
        call.generation(),
        None,
    );
    store.put(&rec).await.unwrap();
    store.get(id).await.unwrap()
}
struct CountOracle(AtomicUsize);
impl SandboxOracle for CountOracle {
    fn semantic_declaration(&self) -> Option<gw_schema::SemanticDeclaration> {
        fixture_semantics("count-oracle")
    }
    fn execute(&self, _: &str) -> Result<String, String> {
        assert_eq!(
            self.0.fetch_add(1, Ordering::SeqCst),
            0,
            "verified resume must not rerun sandbox"
        );
        Ok("42".into())
    }
}
#[tokio::test]
async fn verified_resume_and_reconciliation_use_facts_without_oracle_or_evidence_lookup() {
    let store = Store::open_in_memory().await.unwrap();
    store
        .insert_historical_run("resume", "{}", None)
        .await
        .unwrap();
    let mut candidate = numeric_candidate("Compute the result", "42");
    candidate.contract.answer_policy = Some(VerificationPolicy::Authoritative);
    candidate.contract.oracle = Oracle::SandboxExecution {
        tool_or_sql: "reference".into(),
        expected: None,
    };
    candidate.contract.execution_policy = Some(VerificationPolicy::Authoritative);
    candidate.contract.required_tests = vec![EVIDENCE_NODE.into()];
    let rec = generated(&store, "resume", "record", candidate).await;
    let oracle = Arc::new(CountOracle(AtomicUsize::new(0)));
    let evidence = Arc::new(ScriptedEvidence::new(|key| Some(passing_report(key))));
    let judge = Arc::new(ScriptedJudge::new(vec![&judge_body(0.99, "accept")]));
    let mut cl = clients(
        store.clone(),
        Arc::new(ScriptedTeacher::new(vec![], 0)),
        judge.clone(),
        EventSink::disconnected(),
    )
    .with_execution_evidence_source(evidence.clone());
    cl.sandbox = oracle.clone();
    let area = area_k1(one_judge(), lenient_thresholds());
    let verified = step(rec, &cl, &area).await.unwrap();
    let facts = verified.verification.interpretation.clone();
    assert_eq!(verified.lifecycle.state, LifecycleState::Verified);
    // Reopen the persisted envelope at the exact crash boundary; no in-memory grade survives.
    let mut loaded = store.get("record").await.unwrap();
    loaded.verification.all_passed = false; // This historical projection is not the authority.
    let judged = step(loaded, &cl, &area).await.unwrap();
    let admitted = step(judged, &cl, &area).await.unwrap();
    assert_eq!(admitted.lifecycle.state, LifecycleState::Admitted);
    assert_eq!(admitted.verification.interpretation, facts);
    assert_eq!(oracle.0.load(Ordering::SeqCst), 1);
    assert_eq!(evidence.call_count(), 1);
    assert_eq!(judge.call_count(), 1);
}
#[tokio::test]
async fn invalid_retry_contract_precedes_teacher_dispatch() {
    let store = Store::open_in_memory().await.unwrap();
    store
        .insert_historical_run("retry", "{}", None)
        .await
        .unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(vec![], 0));
    let judge = Arc::new(ScriptedJudge::new(vec![]));
    let cl = clients(
        store.clone(),
        teacher.clone(),
        judge.clone(),
        EventSink::disconnected(),
    );
    let area = area_k1(one_judge(), lenient_thresholds());
    let mut rec = generated(
        &store,
        "retry",
        &record_id("retry", 0, 0, 0, 0),
        good_candidate("Explain the result"),
    )
    .await;
    rec = step(rec, &cl, &area).await.unwrap();
    rec.lifecycle.state = LifecycleState::Revising;
    let source = one_item_source();
    let mut seed = source.items_for_shard(0).remove(0);
    seed.candidate.contract.execution_policy = Some(VerificationPolicy::Authoritative);
    seed.candidate.contract.required_tests.clear();
    let cancel = CancellationToken::new();
    assert!(
        revise_once(
            "retry",
            0,
            &seed,
            &rec,
            &cl,
            &area,
            RunControl::new(&cancel)
        )
        .await
        .is_err()
    );
    assert_eq!(teacher.call_count(), 0);
    assert_eq!(judge.call_count(), 0);
    assert_eq!(store.scan(&RecordFilter::new()).await.unwrap().len(), 1);
}
#[tokio::test]
async fn legacy_or_unsupported_record_stops_entire_run_before_paid_work_and_preserves_bytes() {
    for variant in 0..4 {
        let store = Store::open_in_memory().await.unwrap();
        let teacher = Arc::new(ScriptedTeacher::new(vec![], 0));
        let judge = Arc::new(ScriptedJudge::new(vec![]));
        let cl = clients(
            store.clone(),
            teacher.clone(),
            judge.clone(),
            EventSink::disconnected(),
        );
        let area = area_k1(one_judge(), lenient_thresholds());
        let engine = Engine::new(cl.clone(), area.clone(), 2);
        let source = InMemorySeedSource::new(
            vec![
                good_candidate("new first task"),
                good_candidate("legacy second task"),
            ],
            2,
        );
        register_run(&store, &engine, "legacy", &source).await;
        let mut rec = generated(
            &store,
            "legacy",
            &record_id("legacy", 1, 1, 0, 0),
            good_candidate("legacy second task"),
        )
        .await;
        rec = step(rec, &cl, &area).await.unwrap();
        match variant {
            0 => rec.verification_contract = None,
            1 => rec.verification_contract.as_mut().unwrap().answer_policy = None,
            2 => rec.verification.interpretation = None,
            _ => rec.verification.interpretation.as_mut().unwrap().version += 1,
        }
        rec.verification.all_passed = true;
        store.put(&rec).await.unwrap();
        let before = store.get(&rec.record_id).await.unwrap();
        let bytes = serde_json::to_vec(&before).unwrap();
        let hash = gw_storage::record_hash(&before).unwrap();
        let accounting = store.accounting_snapshot("legacy").await.unwrap();
        assert!(
            engine
                .run("legacy", &source, CancellationToken::new())
                .await
                .is_err()
        );
        assert!(step(before.clone(), &cl, &area).await.is_err());
        assert_eq!(teacher.call_count(), 0);
        assert_eq!(judge.call_count(), 0);
        let after = store.get(&rec.record_id).await.unwrap();
        assert_eq!(serde_json::to_vec(&after).unwrap(), bytes);
        assert_eq!(gw_storage::record_hash(&after).unwrap(), hash);
        assert_eq!(
            store.accounting_snapshot("legacy").await.unwrap(),
            accounting
        );
        assert_eq!(store.scan(&RecordFilter::new()).await.unwrap().len(), 1);
    }
}
#[tokio::test]
async fn legacy_sibling_stops_direct_group_before_missing_sibling_generation() {
    let store = Store::open_in_memory().await.unwrap();
    store
        .insert_historical_run("group", "{}", None)
        .await
        .unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(vec![], 0));
    let judge = Arc::new(ScriptedJudge::new(vec![]));
    let cl = clients(
        store.clone(),
        teacher.clone(),
        judge.clone(),
        EventSink::disconnected(),
    );
    let mut rec = generated(
        &store,
        "group",
        &record_id("group", 0, 0, 0, 1),
        good_candidate("explain"),
    )
    .await;
    rec.lifecycle.state = LifecycleState::Verified;
    rec.verification.all_passed = true;
    store.put(&rec).await.unwrap();
    let seed = one_item_source().items_for_shard(0).remove(0);
    assert!(
        run_group(
            "group",
            0,
            &seed,
            &cl,
            &area_k(one_judge(), lenient_thresholds(), 2),
            RunControl::new(&CancellationToken::new())
        )
        .await
        .is_err()
    );
    assert_eq!(teacher.call_count(), 0);
    assert_eq!(judge.call_count(), 0);
}
#[tokio::test]
async fn judged_reconciliation_cannot_override_authoritative_facts() {
    let store = Store::open_in_memory().await.unwrap();
    store
        .insert_historical_run("reconcile", "{}", None)
        .await
        .unwrap();
    let judge = Arc::new(ScriptedJudge::new(vec![&judge_body(0.99, "accept")]));
    let cl = clients(
        store.clone(),
        Arc::new(ScriptedTeacher::new(vec![], 0)),
        judge.clone(),
        EventSink::disconnected(),
    );
    let area = area_k1(one_judge(), lenient_thresholds());
    let mut candidate = numeric_candidate("compute", "42");
    candidate.contract.answer_policy = Some(VerificationPolicy::Authoritative);
    let rec = generated(&store, "reconcile", "record", candidate).await;
    let verified = step(rec, &cl, &area).await.unwrap();
    let mut judged = step(verified, &cl, &area).await.unwrap();
    judged
        .verification
        .interpretation
        .as_mut()
        .unwrap()
        .answer
        .observation
        .as_mut()
        .unwrap()
        .outcome = VerificationOutcome::Unknown;
    let held = step(judged, &cl, &area).await.unwrap();
    assert_eq!(held.lifecycle.state, LifecycleState::NeedsReview);
    assert_eq!(judge.call_count(), 1);
}
