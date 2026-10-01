//! Production cached panels overlap distinct requests without losing ordered, durable results.
mod attempt_common;
mod common;
use attempt_common::{Response, Server};
use common::*;
use gw_engine::{Engine, EventSink};
use gw_providers::{
    CallObservation, ChatCompletionsProvider, ChatRequest, Provider, StreamChatFuture,
};
use gw_schema::{AccountingPolicy, LifecycleState};
use gw_storage::Store;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};
use tokio::sync::{Semaphore, mpsc};
use tokio_util::sync::CancellationToken;

/// The C operation starts, but its opaque provider future is held before accounted dispatch.
struct QueuedProvider {
    inner: ChatCompletionsProvider,
    entered: mpsc::UnboundedSender<String>,
    c_once: AtomicBool,
    c_release: Semaphore,
    c_returned: Semaphore,
}
impl Provider for QueuedProvider {
    fn semantic_declaration(&self) -> Option<gw_schema::SemanticDeclaration> {
        self.inner.semantic_declaration()
    }
    fn accounting_capability(&self) -> gw_schema::AccountingCapability {
        self.inner.accounting_capability()
    }
    fn stream_chat(&self, req: ChatRequest) -> StreamChatFuture<'_> {
        self.inner.stream_chat(req)
    }
    fn stream_chat_observed(
        &self,
        req: ChatRequest,
        observation: CallObservation,
    ) -> StreamChatFuture<'_> {
        Box::pin(async move {
            let held = req.model == "judge-c" && !self.c_once.swap(true, Ordering::SeqCst);
            self.entered.send(req.model.clone()).unwrap();
            if held {
                self.c_release.acquire().await.unwrap().forget();
            }
            let result = self.inner.stream_chat_observed(req, observation).await;
            if held {
                self.c_returned.add_permits(1);
            }
            result
        })
    }
}
fn queued(server: &Server) -> (Arc<QueuedProvider>, mpsc::UnboundedReceiver<String>) {
    let (entered, receiver) = mpsc::unbounded_channel();
    (
        Arc::new(QueuedProvider {
            inner: server.provider(),
            entered,
            c_once: AtomicBool::new(false),
            c_release: Semaphore::new(0),
            c_returned: Semaphore::new(0),
        }),
        receiver,
    )
}
async fn logical_calls(receiver: &mut mpsc::UnboundedReceiver<String>, expected: &[&str]) {
    let mut actual = vec![];
    for _ in expected {
        actual.push(
            tokio::time::timeout(Duration::from_secs(5), receiver.recv())
                .await
                .unwrap()
                .unwrap(),
        );
    }
    actual.sort();
    assert_eq!(actual, expected);
}
fn engine(
    store: &Store,
    provider: Arc<dyn Provider>,
    policy: AccountingPolicy,
    panel_size: usize,
) -> Engine {
    let mut clients = clients(
        store.clone(),
        Arc::new(ScriptedTeacher::new(vec![good_cot(0.0)], 1)),
        provider,
        EventSink::disconnected(),
    );
    clients.policy = policy;
    Engine::new(
        clients,
        area_k1(three_judges()[..panel_size].to_vec(), lenient_thresholds()),
        1,
    )
}
fn posts(server: &Server, model: &str) -> usize {
    server
        .requests
        .lock()
        .unwrap()
        .iter()
        .filter(|(_, request)| request["model"] == model)
        .count()
}
async fn cache_count(store: &Store, model: &str) -> i64 {
    sqlx::query_scalar("SELECT COUNT(*) FROM cache WHERE kind='judge' AND model=?")
        .bind(model)
        .fetch_one(store.raw_pool())
        .await
        .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn distinct_cached_panel_misses_reach_http_before_either_response_is_released() {
    let store = Store::open_in_memory().await.unwrap();
    let started = Arc::new(Semaphore::new(0));
    let release = Arc::new(Semaphore::new(0));
    let (entered, held) = (started.clone(), release.clone());
    let server = Server::new(move |_, request, _| {
        let score = if request["model"] == "judge-a" {
            0.95
        } else {
            0.85
        };
        Response::ok(attempt_common::grade(
            &judge_body(score, "accept"),
            Some(0.1),
        ))
        .held(0, entered.clone(), held.clone())
    })
    .await;
    let judges = three_judges()[..2].to_vec();
    let area = area_k1(judges.clone(), lenient_thresholds());
    let engine = Engine::new(
        clients(
            store.clone(),
            Arc::new(ScriptedTeacher::new(vec![good_cot(0.0)], 1)),
            Arc::new(server.provider()),
            EventSink::disconnected(),
        ),
        area.clone(),
        1,
    );
    let running = tokio::spawn(async move {
        engine
            .run("overlap", &one_item_source(), CancellationToken::new())
            .await
    });
    let overlap = tokio::time::timeout(Duration::from_secs(5), started.acquire_many(2)).await;
    let overlapped = match overlap {
        Ok(permit) => {
            permit.unwrap().forget();
            true
        }
        Err(_) => false,
    };
    // Release and join even on the red path, so the failed regression owns no leftover work.
    release.add_permits(3);
    let report = tokio::time::timeout(Duration::from_secs(5), running)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(
        overlapped,
        "both distinct misses must reach HTTP while both responses are held"
    );
    assert!(report.completed);
    assert_eq!(server.requests.lock().unwrap().len(), 2);
    let record = store.get("overlap-s0-seed0-a0-c0").await.unwrap();
    assert_eq!(
        record
            .judging
            .panel
            .iter()
            .map(|vote| vote.judge_model.as_str())
            .collect::<Vec<_>>(),
        ["judge-a", "judge-b"]
    );
    let raw: serde_json::Value =
        serde_json::from_str(record.judging.panel[0].raw_response.as_ref().unwrap()).unwrap();
    assert_eq!(raw["attempt_origin"]["run_id"], "overlap");
    assert!(raw["attempt_origin"]["attempt_id"].is_string());
    assert_eq!(store.model_attempts("overlap").await.unwrap().len(), 2);

    // Repeated positions remain supported at the low-level collection boundary. They reuse the
    // actual paid origins above, preserve ordering, and do not become another consensus vote.
    let mut audit_panel = judges;
    audit_panel.push(audit_panel[0].clone().with_max_tokens(1));
    let candidate = gw_format::render(
        &record.messages,
        gw_schema::TrlFormat::OpenAiMessages,
        gw_schema::CotPolicy::Supervised,
    )
    .unwrap();
    let collected = gw_judge::grade_panel_cached(
        &store,
        &ExplodingTeacher,
        &audit_panel,
        &area.rubric,
        &candidate,
        &gw_storage::record_hash(&record).unwrap(),
        |_| gw_judge::PanelFailure::Fatal,
    )
    .await
    .unwrap();
    assert_eq!(
        collected
            .iter()
            .map(|grade| grade.judge_model.as_str())
            .collect::<Vec<_>>(),
        ["judge-a", "judge-b", "judge-a"]
    );
    assert_eq!(collected[0], collected[2]);
    for (position, grade) in collected[..2].iter().enumerate() {
        assert_eq!(grade.to_vote(), record.judging.panel[position]);
    }
    assert_eq!(collected[0].raw["attempt_origin"], raw["attempt_origin"]);
    assert!(matches!(
        gw_judge::HybridGrader::new(area.thresholds).grade(
            None,
            &collected,
            &[],
            None,
            &gw_judge::CorrelationMatrix::uniform_offdiagonal(3, area.correlation_rho),
        ),
        Err(gw_judge::JudgeError::DuplicateJudgeEvidence {
            first: 0,
            duplicate: 2
        })
    ));
    assert_eq!(server.requests.lock().unwrap().len(), 2);
    assert_eq!(store.model_attempts("overlap").await.unwrap().len(), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fatal_provider_or_cache_error_seals_dispatch_before_held_siblings_drain() {
    for cache_failure in [false, true] {
        let store = Store::open_in_memory().await.unwrap();
        let started = Arc::new(Semaphore::new(0));
        let release_a = Arc::new(Semaphore::new(0));
        let release_b = Arc::new(Semaphore::new(0));
        let a_calls = AtomicUsize::new(0);
        let (entered, a, b) = (started.clone(), release_a.clone(), release_b.clone());
        let server = Server::new(move |_, request, _| {
            let response = Response::ok(attempt_common::grade(
                &judge_body(0.95, "accept"),
                Some(0.1),
            ));
            if request["model"] == "judge-a" && a_calls.fetch_add(1, Ordering::SeqCst) == 0 {
                let mut response = response.held(0, entered.clone(), a.clone());
                if !cache_failure {
                    response.status = 401;
                    response.body = "unauthorized fixture".into();
                }
                response
            } else if request["model"] == "judge-b" {
                response.held(0, entered.clone(), b.clone())
            } else {
                response
            }
        })
        .await;
        if cache_failure {
            sqlx::query("CREATE TRIGGER fail_judge_a_cache BEFORE INSERT ON cache WHEN NEW.model='judge-a' BEGIN SELECT RAISE(FAIL,'judge A cache failed'); END").execute(store.raw_pool()).await.unwrap();
        }
        let (provider, mut logical) = queued(&server);
        let engine = Arc::new(engine(
            &store,
            provider.clone(),
            AccountingPolicy::ObservationOnly,
            3,
        ));
        let cancel = CancellationToken::new();
        let (task_engine, task_cancel) = (engine.clone(), cancel.clone());
        let running = tokio::spawn(async move {
            task_engine
                .run("failure", &one_item_source(), task_cancel)
                .await
        });
        logical_calls(&mut logical, &["judge-a", "judge-b", "judge-c"]).await;
        attempt_common::signal(&started).await;
        attempt_common::signal(&started).await;
        release_a.add_permits(1);
        tokio::time::timeout(Duration::from_secs(5), cancel.cancelled())
            .await
            .expect("fatal classification must seal dispatch while B remains held");
        assert!(!running.is_finished());
        provider.c_release.add_permits(1);
        attempt_common::signal(&provider.c_returned).await;
        assert_eq!(
            posts(&server, "judge-c"),
            0,
            "the started opaque C future was drained without sending after the seal"
        );
        assert!(!running.is_finished(), "B's paid stream is still held");
        release_b.add_permits(1);
        let error = tokio::time::timeout(Duration::from_secs(5), running)
            .await
            .unwrap()
            .unwrap()
            .unwrap_err();
        if cache_failure {
            assert!(
                error.to_string().contains("judge A cache failed"),
                "{error}"
            );
        } else {
            assert!(error.to_string().contains("401"), "{error}");
        }
        assert!(
            !error.is_halt(),
            "C's secondary cancellation must not hide the fatal cause"
        );
        assert_eq!(cache_count(&store, "judge-b").await, 1);
        assert_eq!(cache_count(&store, "judge-a").await, 0);
        assert_eq!(server.requests.lock().unwrap().len(), 2);
        assert_eq!(
            store
                .get("failure-s0-seed0-a0-c0")
                .await
                .unwrap()
                .lifecycle
                .state,
            LifecycleState::Verified
        );
        assert!(store.resume_cursor("failure", 0).await.unwrap().is_none());
        let attempts = store.model_attempts("failure").await.unwrap();
        assert_eq!(attempts.len(), 2);
        assert!(attempts.iter().all(|attempt| attempt.transport.is_some()));
        assert_eq!(
            store
                .accounting_snapshot("failure")
                .await
                .unwrap()
                .unresolved_attempts,
            0
        );
        let original_b = attempts
            .iter()
            .find(|attempt| attempt.intent.requested_model == "judge-b")
            .unwrap();
        if cache_failure {
            sqlx::query("DROP TRIGGER fail_judge_a_cache")
                .execute(store.raw_pool())
                .await
                .unwrap();
        }
        let report = engine
            .run("failure", &one_item_source(), CancellationToken::new())
            .await
            .unwrap();
        assert!(report.completed);
        assert!(store.resume_cursor("failure", 0).await.unwrap().is_some());
        assert_eq!(
            (
                posts(&server, "judge-a"),
                posts(&server, "judge-b"),
                posts(&server, "judge-c")
            ),
            (2, 1, 1)
        );
        let record = store.get("failure-s0-seed0-a0-c0").await.unwrap();
        let raw: serde_json::Value =
            serde_json::from_str(record.judging.panel[1].raw_response.as_ref().unwrap()).unwrap();
        assert_eq!(raw["attempt_origin"]["attempt_id"], original_b.attempt_id);
        assert_eq!(
            store
                .accounting_snapshot("failure")
                .await
                .unwrap()
                .unresolved_attempts,
            0
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_logical_misses_obey_finite_serialization_and_replay_only_missing_work() {
    let store = Store::open_in_memory().await.unwrap();
    let started = Arc::new(Semaphore::new(0));
    let release = Arc::new(Semaphore::new(0));
    let (entered, held) = (started.clone(), release.clone());
    let server = Server::new(move |_, _, ordinal| {
        let response = Response::ok(attempt_common::grade(
            &judge_body(0.95, "accept"),
            Some(0.1),
        ));
        if ordinal == 0 {
            response.held(0, entered.clone(), held.clone())
        } else {
            response
        }
    })
    .await;
    let (provider, mut logical) = queued(&server);
    let first = engine(
        &store,
        provider.clone(),
        AccountingPolicy::FiniteUsd { limit_usd: 0.1 },
        2,
    );
    let running = tokio::spawn(async move {
        first
            .run("finite", &one_item_source(), CancellationToken::new())
            .await
    });
    logical_calls(&mut logical, &["judge-a", "judge-b"]).await;
    attempt_common::signal(&started).await;
    assert_eq!(
        server.requests.lock().unwrap().len(),
        1,
        "two started logical operations still admit only one physical attempt"
    );
    release.add_permits(1);
    let first = tokio::time::timeout(Duration::from_secs(5), running)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(!first.completed);
    assert_eq!(first.pending_items, 1);
    assert_eq!(first.errored, 0);
    assert_eq!(
        server.requests.lock().unwrap().len(),
        1,
        "known spend at threshold denies the queued physical attempt"
    );
    let first_model = server.requests.lock().unwrap()[0].1["model"]
        .as_str()
        .unwrap()
        .to_owned();
    assert!(store.resume_cursor("finite", 0).await.unwrap().is_none());
    assert_eq!(
        store
            .get("finite-s0-seed0-a0-c0")
            .await
            .unwrap()
            .lifecycle
            .state,
        LifecycleState::Verified
    );
    assert_eq!(cache_count(&store, &first_model).await, 1);
    assert_eq!(first.accounting.unwrap().unresolved_attempts, 0);
    let second = engine(
        &store,
        provider.clone(),
        AccountingPolicy::FiniteUsd { limit_usd: 1.0 },
        2,
    )
    .run("finite", &one_item_source(), CancellationToken::new())
    .await
    .unwrap();
    assert!(second.completed);
    assert_eq!(second.pending_items, 0);
    assert!(store.resume_cursor("finite", 0).await.unwrap().is_some());
    assert_eq!(
        (posts(&server, "judge-a"), posts(&server, "judge-b")),
        (1, 1)
    );
    assert!((second.accounting.unwrap().known_usd.unwrap() - 0.2).abs() < 1e-10);
    let consumer = engine(
        &store,
        provider,
        AccountingPolicy::FiniteUsd { limit_usd: 0.0 },
        2,
    )
    .run("consumer", &one_item_source(), CancellationToken::new())
    .await
    .unwrap();
    assert!(
        consumer.completed,
        "a fully cached panel needs no monetary admission"
    );
    let summary = consumer.accounting.unwrap();
    assert_eq!(summary.attempts, 0);
    assert_eq!(summary.known_usd, Some(0.0));
    assert_eq!(server.requests.lock().unwrap().len(), 2);
    let original = store.get("finite-s0-seed0-a0-c0").await.unwrap();
    let consumer = store.get("consumer-s0-seed0-a0-c0").await.unwrap();
    for (original, consumer) in original
        .judging
        .panel
        .iter()
        .zip(consumer.judging.panel.iter())
    {
        assert_eq!(
            original.raw_response, consumer.raw_response,
            "paid provenance follows the cached grade, not the consumer run"
        );
    }
}
