//! End-to-end receipt ownership, interpretation, embedding failures and replay.
mod attempt_common;
mod common;
use attempt_common::{Response, Server};
use common::*;
use gw_engine::{Engine, EngineEvent, EventSink, InMemorySeedSource};
use gw_providers::{ChatRequest, Provider, StreamChatFuture};
use gw_schema::{
    AccountingCapability as Cap, AccountingHistory, AttemptPurpose as Purpose, AttemptRole as Role,
    LifecycleState, OutputInterpretation, ReportedCost, TransportOutcome,
};
use gw_storage::{RecordFilter, Store};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokio_util::sync::CancellationToken;

async fn normal_server() -> Server {
    Server::new(|path, request, _| {
        Response::ok(if path == "/embeddings" {
            attempt_common::embedding(Some(0.0))
        } else if request["model"].as_str().unwrap().starts_with("judge") {
            attempt_common::grade(&judge_body(0.95, "accept"), Some(0.2))
        } else {
            attempt_common::teacher("96", "stop", Some(0.1))
        })
    })
    .await
}
fn real_clients(store: &Store, server: &Server) -> gw_engine::Clients {
    let provider = Arc::new(server.provider());
    clients_with_embedder(
        store.clone(),
        provider.clone(),
        provider,
        Arc::new(server.embedder()),
    )
}

fn unfinished_source() -> InMemorySeedSource {
    InMemorySeedSource::new(
        vec![
            good_candidate("What is 12*8?"),
            good_candidate("Compute twelve times eight."),
        ],
        1,
    )
}
async fn interrupt_after_first_record(engine: &Engine, store: &Store, run: &str) {
    sqlx::query("CREATE TRIGGER interrupt_checkpoint BEFORE INSERT ON checkpoints BEGIN SELECT RAISE(FAIL, 'checkpoint interruption'); END").execute(store.raw_pool()).await.unwrap();
    let error = engine
        .run(run, &unfinished_source(), CancellationToken::new())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("checkpoint interruption"));
    sqlx::query("DROP TRIGGER interrupt_checkpoint")
        .execute(store.raw_pool())
        .await
        .unwrap();
}

#[tokio::test]
async fn real_teacher_judge_and_all_three_embedding_purposes_retain_context_and_usage() {
    let store = Store::open_in_memory().await.unwrap();
    let server = normal_server().await;
    let engine = Engine::new(
        real_clients(&store, &server),
        area_k1(one_judge(), lenient_thresholds()),
        1,
    );
    interrupt_after_first_record(&engine, &store, "all-lanes").await;
    let receipts = store.model_attempts("all-lanes").await.unwrap();
    assert_eq!(receipts.len(), 4);
    for purpose in [
        Purpose::Initial,
        Purpose::Grade,
        Purpose::CandidateQc,
        Purpose::AdmittedPrior,
    ] {
        let receipt = receipts
            .iter()
            .find(|r| r.intent.context.purpose == purpose)
            .unwrap();
        assert_eq!(receipt.intent.context.run_id, "all-lanes");
        assert_eq!(receipt.intent.context.shard, Some(0));
        assert_eq!(
            receipt.intent.context.record_id.as_deref(),
            Some("all-lanes-s0-seed0-a0-c0")
        );
        assert_eq!(
            receipt.transport.as_ref().unwrap().outcome,
            TransportOutcome::Complete
        );
        assert_eq!(receipt.interpretation, Some(OutputInterpretation::Accepted));
        assert!(receipt.metadata.total_tokens.is_some());
    }
    let embedding = receipts
        .iter()
        .find(|r| r.intent.context.role == Role::Embedding)
        .unwrap();
    assert_eq!(embedding.metadata.cost_usd, ReportedCost::Known(0.0));
    assert_eq!(embedding.intent.requested_model, "embedding-fixture");
    assert_eq!(
        embedding.metadata.model.as_deref(),
        Some("embedding-actual")
    );
    let launches = store.model_launches("all-lanes").await.unwrap();
    assert_eq!(launches[0].teacher, Cap::PhysicalAttemptsV1);
    assert_eq!(launches[0].judge, Cap::PhysicalAttemptsV1);
    assert_eq!(launches[0].embedding, Cap::PhysicalAttemptsV1);
    assert_eq!(launches[0].history, AccountingHistory::RecordedFromCreation);
    let first_launch = launches[0].launch_id.clone();
    engine
        .run("all-lanes", &unfinished_source(), CancellationToken::new())
        .await
        .unwrap();
    let after = store.model_attempts("all-lanes").await.unwrap();
    assert_eq!(
        after.len(),
        6,
        "resume rebuilds the prior and gates the still-new candidate"
    );
    let resumed = after
        .iter()
        .find(|r| r.intent.context.purpose == Purpose::ResumePrior)
        .unwrap();
    assert_ne!(resumed.intent.context.launch_id, first_launch);
    assert_eq!(resumed.intent.context.role, Role::Embedding);
    assert_eq!(resumed.metadata.cost_usd, ReportedCost::Known(0.0));
}

#[tokio::test]
async fn truncation_retry_has_separate_receipt_and_success_does_not_erase_failed_spend() {
    let store = Store::open_in_memory().await.unwrap();
    let server = Server::new(|_, request, _| {
        if request["model"].as_str().unwrap().starts_with("judge") {
            Response::ok(attempt_common::grade(
                &judge_body(0.95, "accept"),
                Some(0.05),
            ))
        } else if request["max_tokens"] == 16384 {
            Response::ok(attempt_common::teacher("", "length", Some(0.4)))
        } else {
            Response::ok(attempt_common::teacher("96", "stop", Some(0.1)))
        }
    })
    .await;
    let provider = Arc::new(server.provider());
    let clients = clients(
        store.clone(),
        provider.clone(),
        provider,
        EventSink::disconnected(),
    );
    Engine::new(clients, area_k1(one_judge(), lenient_thresholds()), 1)
        .run("truncated", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();
    let receipts = store.model_attempts("truncated").await.unwrap();
    assert_eq!(receipts.len(), 3);
    let first = receipts
        .iter()
        .find(|r| r.intent.context.purpose == Purpose::Initial)
        .unwrap();
    let retry = receipts
        .iter()
        .find(|r| r.intent.context.purpose == Purpose::TruncationRetry)
        .unwrap();
    assert_ne!(first.attempt_id, retry.attempt_id);
    assert_ne!(first.intent.request_digest, retry.intent.request_digest);
    assert_eq!(first.metadata.cost_usd, ReportedCost::Known(0.4));
    assert_eq!(first.interpretation, Some(OutputInterpretation::Truncated));
    assert_eq!(retry.interpretation, Some(OutputInterpretation::Accepted));
    let record = store
        .scan(&RecordFilter::new().run_id("truncated"))
        .await
        .unwrap()
        .pop()
        .unwrap();
    assert_eq!(
        record.cost.usd, 0.1,
        "record cost remains the successful-turn projection"
    );
}

#[tokio::test]
async fn revision_uses_its_intended_retry_record_and_separate_purpose() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher_calls = Arc::new(AtomicUsize::new(0));
    let calls = teacher_calls.clone();
    let server = Server::new(move |_, _, _| {
        let index = calls.fetch_add(1, Ordering::SeqCst);
        Response::ok(attempt_common::teacher(
            &format!("Answer {index}"),
            "stop",
            Some(0.1),
        ))
    })
    .await;
    let judge = Arc::new(ScriptedJudge::new(vec![
        &judge_body(0.65, "revise"),
        &judge_body(0.95, "accept"),
    ]));
    let clients = clients(
        store.clone(),
        Arc::new(server.provider()),
        judge,
        EventSink::disconnected(),
    );
    Engine::new(clients, area_k1(one_judge(), lenient_thresholds()), 1)
        .run("revision", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();
    let receipts = store.model_attempts("revision").await.unwrap();
    assert_eq!(receipts.len(), 2);
    let revised = receipts
        .iter()
        .find(|r| r.intent.context.purpose == Purpose::Revision)
        .unwrap();
    assert_eq!(
        revised.intent.context.record_id.as_deref(),
        Some("revision-s0-seed0-a1-c0")
    );
    assert_eq!(revised.interpretation, Some(OutputInterpretation::Accepted));
}

#[tokio::test]
async fn malformed_teacher_or_grade_preserves_observed_cost_and_transport_success() {
    for malformed_judge in [false, true] {
        let store = Store::open_in_memory().await.unwrap();
        let server = Server::new(move |_, request, _| {
            let judge = request["model"].as_str().unwrap().starts_with("judge");
            Response::ok(if judge {
                attempt_common::grade("not a grade", Some(0.2))
            } else {
                attempt_common::teacher(if malformed_judge { "96" } else { "" }, "stop", Some(0.1))
                    .replace(
                        "Work out the answer carefully.",
                        if malformed_judge {
                            "valid reasoning"
                        } else {
                            ""
                        },
                    )
            })
        })
        .await;
        let provider = Arc::new(server.provider());
        let clients = clients(
            store.clone(),
            provider.clone(),
            provider,
            EventSink::disconnected(),
        );
        let report = Engine::new(clients, area_k1(one_judge(), lenient_thresholds()), 1)
            .run("malformed", &one_item_source(), CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(report.errored, 1);
        let receipts = store.model_attempts("malformed").await.unwrap();
        let malformed = receipts
            .iter()
            .find(|r| {
                r.intent.context.role
                    == if malformed_judge {
                        Role::Judge
                    } else {
                        Role::Teacher
                    }
            })
            .unwrap();
        assert_eq!(
            malformed.interpretation,
            Some(OutputInterpretation::Invalid)
        );
        assert_eq!(
            malformed.transport.as_ref().unwrap().outcome,
            TransportOutcome::Complete
        );
        assert_eq!(
            malformed.metadata.cost_usd,
            ReportedCost::Known(if malformed_judge { 0.2 } else { 0.1 })
        );
    }
}

#[tokio::test]
async fn candidate_and_resume_prior_accounting_failures_are_fatal_and_send_nothing() {
    for purpose in ["candidate_qc", "resume_prior"] {
        let store = Store::open_in_memory().await.unwrap();
        let server = normal_server().await;
        let engine = Engine::new(
            real_clients(&store, &server),
            area_k1(one_judge(), lenient_thresholds()),
            1,
        );
        if purpose == "resume_prior" {
            interrupt_after_first_record(&engine, &store, "prior-error").await;
        }
        let before = server.requests.lock().unwrap().len();
        let trigger = if purpose == "candidate_qc" {
            "CREATE TRIGGER reject_embedding BEFORE INSERT ON model_attempts WHEN json_extract(NEW.receipt_json, '$.intent.context.purpose') = 'candidate_qc' BEGIN SELECT RAISE(FAIL, 'prior intent failed'); END"
        } else {
            "CREATE TRIGGER reject_embedding BEFORE INSERT ON model_attempts WHEN json_extract(NEW.receipt_json, '$.intent.context.purpose') = 'resume_prior' BEGIN SELECT RAISE(FAIL, 'prior intent failed'); END"
        };
        sqlx::query(trigger)
            .execute(store.raw_pool())
            .await
            .unwrap();
        let error = engine
            .run(
                "prior-error",
                &unfinished_source(),
                CancellationToken::new(),
            )
            .await
            .unwrap_err();
        assert!(!error.is_record_level());
        assert!(error.to_string().contains("accounting failure"));
        assert_eq!(server.requests.lock().unwrap().len(), before);
        assert_eq!(
            store.run_status("prior-error").await.unwrap().as_deref(),
            Some("failed")
        );
        if purpose == "resume_prior" {
            assert_eq!(
                store
                    .scan(&RecordFilter::new().run_id("prior-error"))
                    .await
                    .unwrap()[0]
                    .lifecycle
                    .state,
                LifecycleState::Formatted
            );
        }
    }
}

#[tokio::test]
async fn admitted_prior_settlement_failure_preserves_the_committed_admission_and_event() {
    let store = Store::open_in_memory().await.unwrap();
    sqlx::query("CREATE TRIGGER reject_prior_settlement BEFORE UPDATE ON model_attempts WHEN json_extract(NEW.receipt_json, '$.intent.context.purpose') = 'admitted_prior' AND json_extract(NEW.receipt_json, '$.transport') IS NOT NULL BEGIN SELECT RAISE(FAIL, 'prior settlement failed'); END").execute(store.raw_pool()).await.unwrap();
    let server = normal_server().await;
    let mut clients = real_clients(&store, &server);
    let (events, mut rx) = EventSink::subscribe();
    clients.events = events;
    let error = Engine::new(clients, area_k1(one_judge(), lenient_thresholds()), 1)
        .run(
            "admitted-error",
            &one_item_source(),
            CancellationToken::new(),
        )
        .await
        .unwrap_err();
    assert!(!error.is_record_level());
    assert!(error.to_string().contains("accounting failure"));
    let record = store
        .scan(&RecordFilter::new().run_id("admitted-error"))
        .await
        .unwrap()
        .pop()
        .unwrap();
    assert_eq!(record.lifecycle.state, LifecycleState::Admitted);
    let mut admission_event = false;
    while let Ok(event) = rx.try_recv() {
        admission_event |= matches!(
            event,
            EngineEvent::StateAdvanced {
                to: LifecycleState::Admitted,
                ..
            }
        );
    }
    assert!(
        admission_event,
        "committed admission must remain observable"
    );
    let receipts = store.model_attempts("admitted-error").await.unwrap();
    let failed = receipts
        .iter()
        .find(|r| r.intent.context.purpose == Purpose::AdmittedPrior)
        .unwrap();
    assert_eq!(failed.metadata.cost_usd, ReportedCost::Known(0.0));
    assert!(failed.transport.is_none());
}

struct UnknownProvider(Arc<dyn Provider>);
impl Provider for UnknownProvider {
    fn semantic_declaration(&self) -> Option<gw_schema::SemanticDeclaration> {
        self.0.semantic_declaration()
    }
    fn stream_chat(&self, req: ChatRequest) -> StreamChatFuture<'_> {
        self.0.stream_chat(req)
    }
}
#[tokio::test]
async fn replacing_a_public_client_is_reassessed_and_null_embedder_stays_request_free() {
    let store = Store::open_in_memory().await.unwrap();
    let pure = Arc::new(ScriptedTeacher::new(vec![], 0));
    let mut clients = clients(
        store.clone(),
        pure.clone(),
        pure.clone(),
        EventSink::disconnected(),
    );
    clients.teacher = Arc::new(UnknownProvider(pure));
    let engine = Engine::new(clients, area_k1(one_judge(), lenient_thresholds()), 1);
    engine
        .run(
            "empty-unknown",
            &InMemorySeedSource::new(vec![], 1),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    let coverage = store
        .model_launches("empty-unknown")
        .await
        .unwrap()
        .pop()
        .unwrap();
    assert_eq!(coverage.teacher, Cap::Unknown);
    assert_eq!(coverage.judge, Cap::NoModelRequests);
    assert_eq!(coverage.embedding, Cap::NoModelRequests);
    assert_eq!(coverage.history, AccountingHistory::RecordedFromCreation);
    assert!(
        store
            .model_attempts("empty-unknown")
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn a_later_judge_accounting_failure_preserves_completed_judges_in_the_cache() {
    let store = Store::open_in_memory().await.unwrap();
    sqlx::query("CREATE TRIGGER fail_second_judge BEFORE UPDATE ON model_attempts WHEN json_extract(NEW.receipt_json, '$.intent.requested_model') = 'judge-b' AND json_extract(NEW.receipt_json, '$.transport') IS NOT NULL BEGIN SELECT RAISE(FAIL, 'judge settlement failed'); END").execute(store.raw_pool()).await.unwrap();
    let server = normal_server().await;
    let provider = Arc::new(server.provider());
    let clients = clients(
        store.clone(),
        provider.clone(),
        provider,
        EventSink::disconnected(),
    );
    let engine = Engine::new(
        clients,
        area_k1(three_judges()[..2].to_vec(), lenient_thresholds()),
        1,
    );
    assert!(
        engine
            .run(
                "partial-panel",
                &one_item_source(),
                CancellationToken::new()
            )
            .await
            .is_err()
    );
    let record = store
        .scan(&RecordFilter::new().run_id("partial-panel"))
        .await
        .unwrap()
        .pop()
        .unwrap();
    assert_eq!(record.lifecycle.state, LifecycleState::Verified);
    sqlx::query("DROP TRIGGER fail_second_judge")
        .execute(store.raw_pool())
        .await
        .unwrap();
    assert_eq!(
        engine
            .run(
                "partial-panel",
                &one_item_source(),
                CancellationToken::new()
            )
            .await
            .unwrap()
            .admitted,
        1
    );
    {
        let requests = server.requests.lock().unwrap();
        let model_calls = |model: &str| {
            requests
                .iter()
                .filter(|(_, req)| req["model"] == model)
                .count()
        };
        assert_eq!(
            model_calls("judge-a"),
            1,
            "completed grade replay uses the partial cache"
        );
        assert_eq!(
            model_calls("judge-b"),
            2,
            "unfinished grade must be retried"
        );
        assert_eq!(
            model_calls("z-ai/glm-5.2"),
            1,
            "persisted teacher turn is reused"
        );
    }
    assert_eq!(
        store.model_attempts("partial-panel").await.unwrap().len(),
        4
    );
}

#[tokio::test]
async fn launch_coverage_persistence_failure_atomically_rolls_back_fresh_run() {
    let store = Store::open_in_memory().await.unwrap();
    sqlx::query("CREATE TRIGGER reject_launch BEFORE INSERT ON model_launches BEGIN SELECT RAISE(FAIL, 'launch failed'); END").execute(store.raw_pool()).await.unwrap();
    let server = normal_server().await;
    let engine = Engine::new(
        real_clients(&store, &server),
        area_k1(one_judge(), lenient_thresholds()),
        1,
    );
    let error = engine
        .run(
            "launch-failed",
            &one_item_source(),
            CancellationToken::new(),
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("launch failed"));
    assert!(server.requests.lock().unwrap().is_empty());
    assert_eq!(
        store.run_status("launch-failed").await.unwrap().as_deref(),
        None
    );
}

#[tokio::test]
async fn an_unknown_logical_wrapper_runs_with_explicit_incomplete_lane_coverage() {
    let store = Store::open_in_memory().await.unwrap();
    let server = normal_server().await;
    let provider = Arc::new(server.provider());
    let clients = clients(
        store.clone(),
        Arc::new(UnknownProvider(provider.clone())),
        provider,
        EventSink::disconnected(),
    );
    let report = Engine::new(clients, area_k1(one_judge(), lenient_thresholds()), 1)
        .run("unknown-live", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(report.admitted, 1);
    assert_eq!(server.requests.lock().unwrap().len(), 2);
    let receipts = store.model_attempts("unknown-live").await.unwrap();
    assert_eq!(
        receipts.len(),
        1,
        "a logical wrapper cannot certify hidden transmissions"
    );
    assert_eq!(receipts[0].intent.context.role, Role::Judge);
    let coverage = store
        .model_launches("unknown-live")
        .await
        .unwrap()
        .pop()
        .unwrap();
    assert_eq!(coverage.teacher, Cap::Unknown);
    assert_eq!(coverage.judge, Cap::PhysicalAttemptsV1);
}

struct CancelledEmbedding;
impl gw_generate::Embedder for CancelledEmbedding {
    fn semantic_declaration(&self) -> Option<gw_schema::SemanticDeclaration> {
        fixture_semantics("cancelled-embedding")
    }
    fn embed<'a>(&'a self, _: &'a str) -> gw_generate::EmbeddingFuture<'a> {
        Box::pin(async { Err(gw_providers::ProviderError::Cancelled) })
    }
}
#[tokio::test]
async fn replacement_embedding_is_rejected_before_resume_mutates_coverage() {
    let store = Store::open_in_memory().await.unwrap();
    let server = normal_server().await;
    let area = area_k1(one_judge(), lenient_thresholds());
    let original = Engine::new(real_clients(&store, &server), area.clone(), 1);
    interrupt_after_first_record(&original, &store, "cancelled-embedding").await;
    let before = server.requests.lock().unwrap().len();
    let mut clients = real_clients(&store, &server);
    clients.embedder = Arc::new(CancelledEmbedding);
    let before_snapshot = store
        .accounting_snapshot("cancelled-embedding")
        .await
        .unwrap();
    let error = Engine::new(clients, area, 1)
        .run(
            "cancelled-embedding",
            &unfinished_source(),
            CancellationToken::new(),
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("manifest"));
    assert_eq!(server.requests.lock().unwrap().len(), before);
    assert_eq!(
        store
            .accounting_snapshot("cancelled-embedding")
            .await
            .unwrap(),
        before_snapshot
    );
    assert_eq!(
        store
            .model_launches("cancelled-embedding")
            .await
            .unwrap()
            .len(),
        1
    );
}
