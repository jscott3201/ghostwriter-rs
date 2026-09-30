//! Monetary admission follows durable physical attempts, while local work remains runnable.
mod attempt_common;
mod common;
use attempt_common::{Response, Server};
use common::*;
use gw_engine::Engine;
use gw_schema::LifecycleState;
use gw_storage::Store;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn finite_spend_stops_judge_before_post_and_preserves_generated_record() {
    let store = Store::open_in_memory().await.unwrap();
    let server = Server::new(|_, request, _| {
        Response::ok(if request["model"].as_str().unwrap().starts_with("judge") {
            attempt_common::grade(&judge_body(0.95, "accept"), Some(0.2))
        } else {
            attempt_common::teacher("96", "stop", Some(0.1))
        })
    })
    .await;
    let provider = Arc::new(server.provider());
    let mut clients = clients(
        store.clone(),
        provider.clone(),
        provider,
        gw_engine::EventSink::disconnected(),
    );
    clients.policy = gw_schema::AccountingPolicy::FiniteUsd { limit_usd: 0.1 };
    let report = Engine::new(clients, area_k1(one_judge(), lenient_thresholds()), 1)
        .run("policy", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(
        server.requests.lock().unwrap().len(),
        1,
        "the next physical judge call must be denied"
    );
    assert!(!report.completed);
    assert!(store.resume_cursor("policy", 0).await.unwrap().is_none());
    assert_eq!(
        store
            .get("policy-s0-seed0-a0-c0")
            .await
            .unwrap()
            .lifecycle
            .state,
        LifecycleState::Verified
    );
}

fn finite(store: &Store, server: &Server, limit: f64) -> gw_engine::Clients {
    let provider = Arc::new(server.provider());
    let mut clients = clients(
        store.clone(),
        provider.clone(),
        provider,
        gw_engine::EventSink::disconnected(),
    );
    clients.policy = gw_schema::AccountingPolicy::FiniteUsd { limit_usd: limit };
    clients
}
async fn priced_server() -> Server {
    Server::new(|_, request, _| {
        Response::ok(if request["model"].as_str().unwrap().starts_with("judge") {
            attempt_common::grade(&judge_body(0.95, "accept"), Some(0.2))
        } else {
            attempt_common::teacher("96", "stop", Some(0.1))
        })
    })
    .await
}

#[tokio::test]
async fn threshold_crossing_on_final_response_still_completes_local_work_and_publication() {
    let store = Store::open_in_memory().await.unwrap();
    let server = priced_server().await;
    let path =
        std::env::temp_dir().join(format!("gw-policy-complete-{}.parquet", std::process::id()));
    let report = Engine::new(
        finite(&store, &server, 0.15),
        area_k1(one_judge(), lenient_thresholds()),
        1,
    )
    .with_export(gw_engine::ExportSpec {
        dst: path.clone(),
        target: gw_schema::TrlFormat::ChatML,
        cot: gw_schema::CotPolicy::Supervised,
        dataset_version: None,
    })
    .run("crossed", &one_item_source(), CancellationToken::new())
    .await
    .unwrap();
    assert!(report.completed);
    assert_eq!(report.exported, 1);
    assert!(report.accounting.unwrap().known_usd.unwrap() > 0.15);
    assert_eq!(server.requests.lock().unwrap().len(), 2);
    assert!(store.resume_cursor("crossed", 0).await.unwrap().is_some());
    std::fs::remove_file(path).unwrap();
}

#[tokio::test]
async fn partial_panel_resume_reuses_completed_grade_without_importing_or_erasing_spend() {
    let store = Store::open_in_memory().await.unwrap();
    let server = priced_server().await;
    let area = area_k1(three_judges()[..2].to_vec(), lenient_thresholds());
    let first = Engine::new(finite(&store, &server, 0.3), area.clone(), 1)
        .run("panel", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();
    assert!(!first.completed);
    assert_eq!(server.requests.lock().unwrap().len(), 2);
    assert_eq!(first.pending_items, 1);
    let second = Engine::new(finite(&store, &server, 1.0), area, 1)
        .run("panel", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();
    assert!(second.completed);
    let requests = server.requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    assert_eq!(
        requests
            .iter()
            .filter(|(_, req)| req["model"] == "judge-a")
            .count(),
        1
    );
    assert_eq!(
        requests
            .iter()
            .filter(|(_, req)| req["model"] == "judge-b")
            .count(),
        1
    );
    assert!((second.accounting.unwrap().known_usd.unwrap() - 0.5).abs() < 1e-10);
}

#[tokio::test]
async fn cache_only_grade_at_zero_threshold_retains_origin_without_consumer_spend() {
    let store = Store::open_in_memory().await.unwrap();
    let server = priced_server().await;
    for (run_id, policy) in [
        ("origin", gw_schema::AccountingPolicy::ObservationOnly),
        (
            "consumer",
            gw_schema::AccountingPolicy::FiniteUsd { limit_usd: 0.0 },
        ),
    ] {
        let mut clients = clients(
            store.clone(),
            Arc::new(ScriptedTeacher::new(vec![good_cot(0.0)], 1)),
            Arc::new(server.provider()),
            gw_engine::EventSink::disconnected(),
        );
        clients.policy = policy;
        let report = Engine::new(clients, area_k1(one_judge(), lenient_thresholds()), 1)
            .run(run_id, &one_item_source(), CancellationToken::new())
            .await
            .unwrap();
        assert!(report.completed);
    }
    assert_eq!(server.requests.lock().unwrap().len(), 1);
    assert_eq!(
        store
            .accounting_snapshot("consumer")
            .await
            .unwrap()
            .attempts,
        0
    );
    assert_eq!(
        store
            .accounting_snapshot("consumer")
            .await
            .unwrap()
            .known_usd,
        Some(0.0)
    );
    let original = store.model_attempts("origin").await.unwrap().pop().unwrap();
    let record = store.get("consumer-s0-seed0-a0-c0").await.unwrap();
    let raw: serde_json::Value =
        serde_json::from_str(record.judging.panel[0].raw_response.as_ref().unwrap()).unwrap();
    assert_eq!(raw["attempt_origin"]["run_id"], "origin");
    assert_eq!(raw["attempt_origin"]["attempt_id"], original.attempt_id);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finite_siblings_wait_for_live_settlement_then_deny_unknown_price_or_cancel() {
    use tokio::sync::Semaphore;
    for cancelled in [false, true] {
        let store = Store::open_in_memory().await.unwrap();
        let started = Arc::new(Semaphore::new(0));
        let release = Arc::new(Semaphore::new(0));
        let (s, r) = (started.clone(), release.clone());
        let server = Server::new(move |_, _, _| {
            Response::ok(attempt_common::teacher("96", "stop", None)).held(0, s.clone(), r.clone())
        })
        .await;
        let cancel = CancellationToken::new();
        let token = cancel.clone();
        let engine = Engine::new(
            finite(&store, &server, 5.0),
            area_k(one_judge(), lenient_thresholds(), 3),
            1,
        );
        let task = tokio::spawn(async move { engine.run("wait", &one_item_source(), token).await });
        attempt_common::signal(&started).await;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(
            server.requests.lock().unwrap().len(),
            1,
            "finite sends cannot overlap"
        );
        if cancelled {
            cancel.cancel();
        }
        release.add_permits(1);
        let report = tokio::time::timeout(std::time::Duration::from_secs(5), task)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(!report.completed);
        assert_eq!(server.requests.lock().unwrap().len(), 1);
        assert_eq!(report.pending_items, 1);
        let snapshot = report.accounting.unwrap();
        assert_eq!(snapshot.unresolved_attempts, 0);
        assert_eq!(snapshot.unknown_cost_attempts, 1);
        assert_eq!(snapshot.attempts, 1);
        if !cancelled {
            assert!(report.halted_reason.unwrap().contains("unknown cost"));
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn superseded_live_coordinator_drains_its_old_attempt_and_never_dispatches_again() {
    use gw_engine::SeedSource;
    use tokio::sync::Semaphore;
    let store = Store::open_in_memory().await.unwrap();
    let started = Arc::new(Semaphore::new(0));
    let release = Arc::new(Semaphore::new(0));
    let (s, r) = (started.clone(), release.clone());
    let server = Server::new(move |_, _, _| {
        Response::ok(attempt_common::teacher("96", "stop", Some(0.1))).held(0, s.clone(), r.clone())
    })
    .await;
    let engine = Engine::new(
        finite(&store, &server, 5.0),
        area_k(one_judge(), lenient_thresholds(), 2),
        1,
    );
    let task = tokio::spawn(async move {
        engine
            .run("epoch", &one_item_source(), CancellationToken::new())
            .await
    });
    attempt_common::signal(&started).await;
    store
        .register_accounting_launch(gw_storage::LaunchRequest {
            run_id: "epoch",
            config_json: "{}",
            shard_count: 1,
            prompts_hash: &one_item_source().prompts_hash().unwrap(),
            policy: &gw_schema::AccountingPolicy::ObservationOnly,
            teacher: gw_schema::AccountingCapability::PhysicalAttemptsV1,
            judge: gw_schema::AccountingCapability::PhysicalAttemptsV1,
            embedding: gw_schema::AccountingCapability::NoModelRequests,
        })
        .await
        .unwrap();
    release.add_permits(1);
    let report = tokio::time::timeout(std::time::Duration::from_secs(5), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(!report.completed);
    assert!(report.halted_reason.unwrap().contains("superseded"));
    assert_eq!(server.requests.lock().unwrap().len(), 1);
    let summary = report.accounting.unwrap();
    assert_eq!(summary.configured.unwrap().epoch, 1);
    assert_eq!(summary.effective.unwrap().epoch, 2);
    assert_eq!(summary.unresolved_attempts, 0);
}

#[tokio::test]
async fn an_unknown_embedding_price_prevents_the_next_teacher_send() {
    let store = Store::open_in_memory().await.unwrap();
    let server = Server::new(|_, _, _| Response::ok(attempt_common::embedding(None))).await;
    let mut clients = finite(&store, &server, 5.0);
    clients.embedder = Arc::new(server.embedder());
    let report = Engine::new(clients, area_k1(one_judge(), lenient_thresholds()), 1)
        .run("embedding", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();
    assert!(!report.completed);
    assert_eq!(server.requests.lock().unwrap().len(), 1);
    assert_eq!(server.requests.lock().unwrap()[0].0, "/embeddings");
    assert_eq!(report.accounting.unwrap().unknown_cost_attempts, 1);
}

#[tokio::test]
async fn direct_helpers_and_replaced_unknown_clients_cannot_bypass_registration() {
    use gw_engine::{RunControl, SeedSource};
    struct Unknown(Arc<dyn gw_providers::Provider>);
    impl gw_providers::Provider for Unknown {
        fn stream_chat(
            &self,
            request: gw_providers::ChatRequest,
        ) -> gw_providers::StreamChatFuture<'_> {
            self.0.stream_chat(request)
        }
    }
    let store = Store::open_in_memory().await.unwrap();
    let server = priced_server().await;
    store.create_run("direct", "{}", None).await.unwrap();
    let clients = finite(&store, &server, 5.0);
    let seed = one_item_source().items_for_shard(0).remove(0);
    let area = area_k1(one_judge(), lenient_thresholds());
    let cancelled = CancellationToken::new();
    let outcome = gw_engine::run_group(
        "direct",
        0,
        &seed,
        &clients,
        &area,
        RunControl::new(&cancelled),
    )
    .await
    .unwrap();
    assert!(outcome.interrupted);
    assert!(server.requests.lock().unwrap().is_empty());
    let mut replaced = finite(&store, &server, 5.0);
    replaced.teacher = Arc::new(Unknown(replaced.teacher.clone()));
    assert!(
        Engine::new(replaced, area, 1)
            .run("replaced", &one_item_source(), CancellationToken::new())
            .await
            .is_err()
    );
    assert!(server.requests.lock().unwrap().is_empty());
    assert!(store.run_status("replaced").await.unwrap().is_none());
}

#[tokio::test]
async fn dropping_the_owner_leaves_an_orphan_that_a_new_launch_denies_without_waiting() {
    use tokio::sync::Semaphore;
    let store = Store::open_in_memory().await.unwrap();
    let started = Arc::new(Semaphore::new(0));
    let release = Arc::new(Semaphore::new(0));
    let (s, r) = (started.clone(), release.clone());
    let server = Server::new(move |_, _, _| {
        Response::ok(attempt_common::teacher("96", "stop", Some(0.1))).held(0, s.clone(), r.clone())
    })
    .await;
    let engine = Engine::new(
        finite(&store, &server, 5.0),
        area_k1(one_judge(), lenient_thresholds()),
        1,
    );
    let running = engine.clone();
    let task = tokio::spawn(async move {
        running
            .run("orphan", &one_item_source(), CancellationToken::new())
            .await
    });
    attempt_common::signal(&started).await;
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    release.add_permits(1);
    let resumed = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        engine.run("orphan", &one_item_source(), CancellationToken::new()),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(!resumed.completed);
    assert!(resumed.halted_reason.unwrap().contains("not owned"));
    assert_eq!(resumed.accounting.unwrap().unresolved_attempts, 1);
    assert_eq!(server.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn dropped_accounting_events_cannot_change_authoritative_terminal_totals() {
    let store = Store::open_in_memory().await.unwrap();
    let server = priced_server().await;
    let (sink, _undrained) = gw_engine::EventSink::subscribe_with_capacity(1);
    let mut clients = finite(&store, &server, 5.0);
    clients.events = sink;
    let report = Engine::new(clients, area_k1(one_judge(), lenient_thresholds()), 1)
        .run("events", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();
    let mut actual = report.accounting.unwrap();
    actual.configured = None;
    assert_eq!(actual, store.accounting_snapshot("events").await.unwrap());
    assert_eq!(actual.attempts, 2);
    assert!((actual.known_usd.unwrap() - 0.3).abs() < 1e-10);
    assert_eq!(actual.total_tokens.known, Some(24));
}

#[tokio::test]
async fn truncation_retry_and_revision_each_need_fresh_physical_admission() {
    for revision in [false, true] {
        let store = Store::open_in_memory().await.unwrap();
        let server = Server::new(move |_, _, _| {
            Response::ok(attempt_common::teacher(
                if revision { "96" } else { "" },
                if revision { "stop" } else { "length" },
                Some(0.1),
            ))
        })
        .await;
        let mut clients = finite(&store, &server, 0.1);
        clients.judge = Arc::new(ScriptedJudge::new(vec![&judge_body(0.65, "revise")]));
        let report = Engine::new(clients, area_k1(one_judge(), lenient_thresholds()), 1)
            .run("retry", &one_item_source(), CancellationToken::new())
            .await
            .unwrap();
        assert!(!report.completed);
        assert_eq!(report.pending_items, 1);
        assert_eq!(report.errored, 0);
        assert_eq!(server.requests.lock().unwrap().len(), 1);
        assert_eq!(report.accounting.unwrap().attempts, 1);
        assert!(store.resume_cursor("retry", 0).await.unwrap().is_none());
        if revision {
            assert_eq!(report.revising, 1);
            assert_eq!(
                store
                    .get("retry-s0-seed0-a0-c0")
                    .await
                    .unwrap()
                    .lifecycle
                    .state,
                LifecycleState::Revising
            );
        }
    }
}
