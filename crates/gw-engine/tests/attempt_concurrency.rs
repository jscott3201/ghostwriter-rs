//! Gated HTTP fixtures prove observation and embedding do not serialize a shard's siblings.
mod attempt_common;
mod common;
use attempt_common::{Response, Server, signal};
use common::*;
use gw_engine::{Engine, EngineEvent, EventSink};
use gw_schema::{AttemptPurpose, AttemptRole, LifecycleState, ReportedCost};
use gw_storage::Store;
use std::{sync::Arc, time::Duration};
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn held_candidate_embedding_does_not_suspend_an_active_sibling_stream_or_bypass_its_own_qc() {
    let store = Store::open_in_memory().await.unwrap();
    let embedding_started = Arc::new(Semaphore::new(0));
    let embedding_release = Arc::new(Semaphore::new(0));
    let teacher_started = Arc::new(Semaphore::new(0));
    let teacher_release = Arc::new(Semaphore::new(0));
    let (es, er, ts, tr) = (
        embedding_started.clone(),
        embedding_release.clone(),
        teacher_started.clone(),
        teacher_release.clone(),
    );
    let server = Server::new(move |path, _, ordinal| {
        if path == "/embeddings" {
            let response = Response::ok(attempt_common::embedding(None));
            if ordinal == 1 {
                response.held(0, es.clone(), er.clone())
            } else {
                response
            }
        } else {
            let body = attempt_common::teacher("96", "stop", None);
            let prefix = body.find("data: [DONE]").unwrap();
            let response = Response::ok(body);
            if ordinal == 0 {
                response.held(prefix, ts.clone(), tr.clone())
            } else {
                response
            }
        }
    })
    .await;
    let mut clients = clients_with_embedder(
        store.clone(),
        Arc::new(server.provider()),
        Arc::new(ScriptedJudge::new(vec![&judge_body(0.95, "accept")])),
        Arc::new(server.embedder()),
    );
    let (events, mut rx) = EventSink::subscribe();
    clients.events = events;
    let engine = Engine::new(clients, area_k(one_judge(), lenient_thresholds(), 2), 1);
    let task = tokio::spawn(async move {
        engine
            .run("same-shard", &one_item_source(), CancellationToken::new())
            .await
    });
    signal(&teacher_started).await;
    signal(&embedding_started).await;
    let active = store.model_attempts("same-shard").await.unwrap();
    let held_record = active
        .iter()
        .find(|r| r.intent.context.purpose == AttemptPurpose::CandidateQc && r.transport.is_none())
        .unwrap()
        .intent
        .context
        .record_id
        .clone();
    assert!(active.iter().all(|r| r.intent.context.shard == Some(0)));
    assert!(
        !active
            .iter()
            .any(|r| r.intent.context.role == AttemptRole::Teacher
                && r.intent.context.record_id == held_record),
        "held candidate cannot bypass its own embedding gate"
    );

    teacher_release.add_permits(1);
    // The first sibling must finish its already-active stream and grading while the second
    // candidate is still waiting for its own embedding response in the SAME poll-driven group.
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if matches!(
                rx.recv().await,
                Some(EngineEvent::StateAdvanced {
                    to: LifecycleState::Judged,
                    ..
                })
            ) {
                break;
            }
        }
    })
    .await
    .expect("held embedding suspended another sibling's stream or grading");
    let progressing = store.model_attempts("same-shard").await.unwrap();
    assert_eq!(
        progressing
            .iter()
            .filter(|r| r.intent.context.role == AttemptRole::Teacher)
            .count(),
        1
    );
    assert!(
        progressing
            .iter()
            .any(|r| r.intent.context.role == AttemptRole::Teacher && r.transport.is_some())
    );
    assert!(
        progressing
            .iter()
            .any(|r| r.intent.context.record_id == held_record && r.transport.is_none())
    );
    embedding_release.add_permits(1);
    let report = tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(report.admitted, 1);
    assert_eq!(
        server
            .requests
            .lock()
            .unwrap()
            .iter()
            .filter(|(path, _)| path == "/chat/completions")
            .count(),
        2
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_priced_physical_requests_overlap_within_the_existing_fanout_bound() {
    let store = Store::open_in_memory().await.unwrap();
    let started = Arc::new(Semaphore::new(0));
    let release = Arc::new(Semaphore::new(0));
    let (s, r) = (started.clone(), release.clone());
    let server = Server::new(move |_, _, _| {
        let body = attempt_common::teacher("96", "stop", None);
        let prefix = body.find("data: [DONE]").unwrap();
        Response::ok(body).held(prefix, s.clone(), r.clone())
    })
    .await;
    let clients = clients(
        store.clone(),
        Arc::new(server.provider()),
        Arc::new(ScriptedJudge::new(vec![
            &judge_body(0.95, "accept"),
            &judge_body(0.95, "accept"),
        ])),
        EventSink::disconnected(),
    );
    let engine = Engine::new(clients, area_k(one_judge(), lenient_thresholds(), 2), 1);
    let task = tokio::spawn(async move {
        engine
            .run("overlap", &one_item_source(), CancellationToken::new())
            .await
    });
    signal(&started).await;
    signal(&started).await;
    let active = tokio::time::timeout(Duration::from_secs(2), store.model_attempts("overlap"))
        .await
        .expect("a stream retained the database lock")
        .unwrap();
    assert_eq!(active.len(), 2);
    assert!(
        active
            .iter()
            .all(|r| r.transport.is_none() && r.metadata.cost_usd == ReportedCost::Missing)
    );
    assert_eq!(
        server.requests.lock().unwrap().len(),
        2,
        "k=2 is the current sibling fanout bound"
    );
    release.add_permits(2);
    let report = tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(report.admitted, 1);
    let settled = store.model_attempts("overlap").await.unwrap();
    assert!(
        settled
            .iter()
            .all(|r| r.transport.is_some() && r.metadata.cost_usd == ReportedCost::Missing)
    );
}
