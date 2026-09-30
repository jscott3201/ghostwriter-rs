//! Resolved admission settings fail before generation or grading spends anything.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::*;
use gw_engine::{
    AreaConfig, Engine, EventSink, RunControl, SeedSource, decision_from_judging, run_group, step,
};
use gw_judge::{AreaThresholds, PanelJudge};
use gw_providers::{ChatRequest, DeltaStream, Provider, StreamChatFuture, StreamDelta};
use gw_schema::{AdmissionIntent, LifecycleState};
use gw_storage::Store;
use tokio_util::sync::CancellationToken;

async fn rejects_without_spending(area: AreaConfig) {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(vec![], 1));
    let judge = Arc::new(FailingJudge::new(usize::MAX, &judge_body(0.95, "accept")));
    let engine = Engine::new(
        clients(
            store,
            teacher.clone(),
            judge.clone(),
            EventSink::disconnected(),
        ),
        area,
        1,
    );
    let result = engine
        .run("preflight", &one_item_source(), CancellationToken::new())
        .await;
    assert_eq!(
        teacher.call_count(),
        0,
        "invalid admission must fail before generation"
    );
    assert_eq!(judge.call_count(), 0);
    assert!(result.is_err());
}

#[tokio::test]
async fn empty_generation_panel_fails_before_spending() {
    rejects_without_spending(AreaConfig::new("math", "teacher", vec![], "grade")).await;
}

#[tokio::test]
async fn default_thresholds_are_unattainable_before_spending() {
    rejects_without_spending(AreaConfig::new("math", "teacher", one_judge(), "grade")).await;
}

#[tokio::test]
async fn invalid_numeric_settings_fail_before_spending() {
    let mut area = area_k1(one_judge(), lenient_thresholds());
    area.thresholds.accept_threshold = f64::NAN;
    rejects_without_spending(area).await;
}

struct SubsetJudge;

impl Provider for SubsetJudge {
    fn semantic_declaration(&self) -> Option<gw_schema::SemanticDeclaration> {
        fixture_semantics("judge")
    }
    fn stream_chat(&self, request: ChatRequest) -> StreamChatFuture<'_> {
        let verdict = if matches!(request.model.as_str(), "judge-2" | "judge-3") {
            "uncertain"
        } else {
            "accept"
        };
        let body = judge_body(0.95, verdict);
        Box::pin(async move {
            let stream: DeltaStream = Box::pin(futures::stream::iter([Ok(StreamDelta {
                content: Some(body),
                finish_reason: Some("stop".into()),
                ..Default::default()
            })]));
            Ok(stream)
        })
    }
}

#[tokio::test]
async fn attainable_decisive_subset_is_admitted_by_engine_reconciliation() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(vec![], 1));
    let judges = (0..4)
        .map(|i| PanelJudge::new(format!("judge-{i}"), format!("family-{i}")))
        .collect();
    let thresholds = AreaThresholds {
        min_n_eff: 1.5,
        min_n_eff_ratio: 0.7,
        ..AreaThresholds::default()
    };
    let area = area_k1(judges, thresholds).with_correlation_rho(0.2);
    let engine = Engine::new(
        clients(
            store.clone(),
            teacher,
            Arc::new(SubsetJudge),
            EventSink::disconnected(),
        ),
        area,
        1,
    );
    let report = engine
        .run("subset", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(report.exported, 0);
    assert_eq!(report.admitted, 1);
    let rec = store
        .get(&gw_engine::record_id("subset", 0, 0, 0, 0))
        .await
        .unwrap();
    assert_eq!(rec.judging.decisive_count, Some(2));
    assert_eq!(rec.judging.verdict, Some(gw_schema::Verdict::Admit));
}

#[tokio::test]
async fn documented_assumed_prior_admits_one_of_three_candidates_with_two_judges() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(vec![], 3));
    let judges = (0..2)
        .map(|i| PanelJudge::new(format!("judge-{i}"), format!("family-{i}")))
        .collect();
    let thresholds = AreaThresholds {
        min_n_eff: 1.5,
        min_n_eff_ratio: 0.7,
        ..AreaThresholds::default()
    };
    let area = area_k(judges, thresholds, 3).with_correlation_rho(0.2);
    let engine = Engine::new(
        clients(
            store,
            teacher.clone(),
            Arc::new(SubsetJudge),
            EventSink::disconnected(),
        ),
        area,
        1,
    );
    let report = engine
        .run("documented", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(report.admitted, 1);
    assert_eq!(report.exported, 0);
    assert_eq!(report.admitted, 1);
    assert_eq!(report.rejected, 2);
    assert_eq!(teacher.call_count(), 3);
}

#[tokio::test]
async fn bounded_revision_inherits_review_only_intent() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(
        vec![answer_cot("96", 0.01), answer_cot("97", 0.01)],
        2,
    ));
    let judge = Arc::new(ScriptedJudge::new(vec![
        &judge_body(0.65, "revise"),
        &judge_body(0.95, "accept"),
    ]));
    let area = area_k1(one_judge(), lenient_thresholds())
        .with_admission_intent(AdmissionIntent::ReviewOnly);
    let engine = Engine::new(
        clients(
            store.clone(),
            teacher.clone(),
            judge,
            EventSink::disconnected(),
        ),
        area,
        1,
    );
    let report = engine
        .run("revision", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(report.revising, 1);
    assert_eq!(report.needs_review, 1);
    assert_eq!(report.admitted, 0);
    let retry = store
        .get(&gw_engine::record_id("revision", 0, 0, 1, 0))
        .await
        .unwrap();
    assert_eq!(retry.judging.admission_intent, AdmissionIntent::ReviewOnly);
    assert_eq!(retry.judging.verdict_reason.as_deref(), Some("review_only"));
    assert_eq!(teacher.call_count(), 2);
}

#[tokio::test]
async fn review_only_allows_unattainable_defaults_without_admission() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(vec![], 1));
    let judge = Arc::new(FailingJudge::new(usize::MAX, &judge_body(0.95, "accept")));
    let area = AreaConfig::new("math", "teacher", one_judge(), "grade")
        .with_admission_intent(AdmissionIntent::ReviewOnly);
    let engine = Engine::new(
        clients(store.clone(), teacher, judge, EventSink::disconnected()),
        area,
        1,
    );
    let report = engine
        .run(
            "review-default",
            &one_item_source(),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(report.needs_review, 1);
    assert_eq!(report.admitted, 0);
    let rec = store
        .get(&gw_engine::record_id("review-default", 0, 0, 0, 0))
        .await
        .unwrap();
    assert_eq!(rec.judging.admission_intent, AdmissionIntent::ReviewOnly);
}

#[tokio::test]
async fn replay_rejects_changing_review_only_to_automatic_admission() {
    let store = Store::open_in_memory().await.unwrap();
    let source = one_item_source();
    let teacher = Arc::new(ScriptedTeacher::new(vec![], 3));
    let judge = Arc::new(FailingJudge::new(usize::MAX, &judge_body(0.95, "accept")));
    let cl = clients(
        store.clone(),
        teacher.clone(),
        judge.clone(),
        EventSink::disconnected(),
    );
    let automatic = area_k(one_judge(), lenient_thresholds(), 3);
    let review = automatic
        .clone()
        .with_admission_intent(AdmissionIntent::ReviewOnly);
    register_run(
        &store,
        &Engine::new(cl.clone(), review.clone(), 1),
        "sticky",
        &source,
    )
    .await;
    let item = source.items_for_shard(0).remove(0);
    let cancel = CancellationToken::new();
    let outcome = run_group("sticky", 0, &item, &cl, &review, RunControl::new(&cancel))
        .await
        .unwrap();
    assert_eq!(outcome.siblings.len(), 3);
    assert!(outcome.best.is_none());
    for rec in outcome.siblings {
        assert_eq!(rec.lifecycle.state, LifecycleState::NeedsReview);
        assert_eq!(rec.judging.verdict_reason.as_deref(), Some("review_only"));
        assert_eq!(
            decision_from_judging(&rec, &automatic)
                .unwrap()
                .to_lifecycle(),
            LifecycleState::NeedsReview
        );
        // Recreate a crash after judging, before lifecycle reconciliation. Intent is already durable.
        store
            .advance_lifecycle(&rec.record_id, LifecycleState::Judged, None)
            .await
            .unwrap();
    }
    let judge_calls = judge.call_count();
    let replay = Engine::new(cl, automatic, 1)
        .run("sticky", &source, CancellationToken::new())
        .await
        .unwrap_err();
    assert!(replay.to_string().contains("manifest"));
    assert_eq!(teacher.call_count(), 3);
    assert_eq!(judge.call_count(), judge_calls);
}

struct CancelTeacher {
    cancel: CancellationToken,
    calls: AtomicUsize,
}

impl Provider for CancelTeacher {
    fn semantic_declaration(&self) -> Option<gw_schema::SemanticDeclaration> {
        fixture_semantics("teacher")
    }
    fn stream_chat(&self, _: ChatRequest) -> StreamChatFuture<'_> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.cancel.cancel();
        Box::pin(async {
            let stream: DeltaStream =
                Box::pin(futures::stream::iter(good_cot(0.01).into_iter().map(Ok)));
            Ok(stream)
        })
    }
}

#[tokio::test]
async fn review_only_intent_is_persisted_before_the_first_grade() {
    let store = Store::open_in_memory().await.unwrap();
    let cancel = CancellationToken::new();
    let teacher = Arc::new(CancelTeacher {
        cancel: cancel.clone(),
        calls: AtomicUsize::new(0),
    });
    let judge = Arc::new(FailingJudge::new(usize::MAX, &judge_body(0.95, "accept")));
    let cl = clients(
        store.clone(),
        teacher.clone(),
        judge.clone(),
        EventSink::disconnected(),
    );
    let automatic = area_k1(one_judge(), lenient_thresholds());
    let review = automatic
        .clone()
        .with_admission_intent(AdmissionIntent::ReviewOnly);
    let report = Engine::new(cl.clone(), review.clone(), 1)
        .run("generated", &one_item_source(), cancel)
        .await
        .unwrap();
    assert!(!report.completed);
    let rec = store
        .get(&gw_engine::record_id("generated", 0, 0, 0, 0))
        .await
        .unwrap();
    assert_eq!(rec.lifecycle.state, LifecycleState::AssistantGenerated);
    assert_eq!(rec.judging.admission_intent, AdmissionIntent::ReviewOnly);
    assert_eq!(judge.call_count(), 0);
    let replay = Engine::new(cl, review, 1)
        .run("generated", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(replay.needs_review, 1);
    assert_eq!(replay.admitted, 0);
    assert_eq!(teacher.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn direct_judge_step_preflights_but_verifier_reject_remains_authoritative() {
    let store = Store::open_in_memory().await.unwrap();
    let cancel = CancellationToken::new();
    let teacher = Arc::new(CancelTeacher {
        cancel: cancel.clone(),
        calls: AtomicUsize::new(0),
    });
    let judge = Arc::new(FailingJudge::new(usize::MAX, &judge_body(0.95, "accept")));
    let cl = clients(
        store.clone(),
        teacher,
        judge.clone(),
        EventSink::disconnected(),
    );
    Engine::new(cl.clone(), area_k1(one_judge(), lenient_thresholds()), 1)
        .run("direct", &one_item_source(), cancel)
        .await
        .unwrap();
    let rid = gw_engine::record_id("direct", 0, 0, 0, 0);
    let mut rec = store.get(&rid).await.unwrap();
    rec = step(rec, &cl, &area_k1(one_judge(), lenient_thresholds()))
        .await
        .unwrap();
    assert_eq!(rec.lifecycle.state, LifecycleState::Verified);
    let invalid = AreaConfig::new("math", "teacher", vec![], "grade");
    assert!(step(rec.clone(), &cl, &invalid).await.is_err());
    assert_eq!(judge.call_count(), 0);
    rec.verification
        .interpretation
        .as_mut()
        .unwrap()
        .reasoning
        .observation
        .as_mut()
        .unwrap()
        .outcome = gw_schema::VerificationOutcome::Fail;
    let judged = step(rec, &cl, &invalid).await.unwrap();
    assert_eq!(judged.judging.verdict, Some(gw_schema::Verdict::Reject));
    let rejected = step(judged, &cl, &invalid).await.unwrap();
    assert_eq!(rejected.lifecycle.state, LifecycleState::Rejected);
    assert_eq!(judge.call_count(), 0);
}
