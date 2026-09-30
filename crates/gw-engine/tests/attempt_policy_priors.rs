//! Historical priors are rebuilt only for a fresh candidate that passed local framing checks.
mod attempt_common;
mod common;
use attempt_common::{Response, Server};
use common::*;
use gw_engine::{Engine, ExportSpec, InMemorySeedSource};
use gw_schema::{AccountingPolicy, AttemptPurpose, CotPolicy, TrlFormat};
use gw_storage::Store;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

async fn server() -> Server {
    Server::new(|path, request, _| {
        Response::ok(if path == "/embeddings" {
            attempt_common::embedding(Some(0.1))
        } else if request["model"].as_str().unwrap().starts_with("judge") {
            attempt_common::grade(&judge_body(0.95, "accept"), Some(0.2))
        } else {
            attempt_common::teacher("96", "stop", Some(0.1))
        })
    })
    .await
}
fn engine(store: &Store, server: &Server, limit: f64) -> Engine {
    let provider = Arc::new(server.provider());
    let mut clients = clients_with_embedder(
        store.clone(),
        provider.clone(),
        provider,
        Arc::new(server.embedder()),
    );
    clients.policy = AccountingPolicy::FiniteUsd { limit_usd: limit };
    Engine::new(clients, area_k1(one_judge(), lenient_thresholds()), 1)
}
async fn leave_first_formatted(store: &Store, engine: &Engine, source: &InMemorySeedSource) {
    sqlx::query("CREATE TRIGGER stop_checkpoint BEFORE INSERT ON checkpoints BEGIN SELECT RAISE(FAIL, 'checkpoint interrupted'); END").execute(store.raw_pool()).await.unwrap();
    let error = engine
        .run("priors", source, CancellationToken::new())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("checkpoint interrupted"));
    sqlx::query("DROP TRIGGER stop_checkpoint")
        .execute(store.raw_pool())
        .await
        .unwrap();
    assert_eq!(
        store
            .get("priors-s0-seed0-a0-c0")
            .await
            .unwrap()
            .lifecycle
            .state,
        gw_schema::LifecycleState::Formatted
    );
}

#[tokio::test]
async fn persisted_local_completion_publishes_above_threshold_without_rebuilding_embeddings() {
    let store = Store::open_in_memory().await.unwrap();
    let server = server().await;
    let engine = engine(&store, &server, 0.49);
    leave_first_formatted(&store, &engine, &one_item_source()).await;
    assert_eq!(server.requests.lock().unwrap().len(), 4);
    assert!(
        store
            .accounting_snapshot("priors")
            .await
            .unwrap()
            .known_usd
            .unwrap()
            > 0.49
    );
    let path = std::env::temp_dir().join(format!("gw-lazy-priors-{}.parquet", std::process::id()));
    let report = engine
        .with_export(ExportSpec {
            dst: path.clone(),
            target: TrlFormat::ChatML,
            cot: CotPolicy::Supervised,
            dataset_version: None,
        })
        .run("priors", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();
    assert!(report.completed);
    assert_eq!(report.exported, 1);
    assert_eq!(report.pending_items, 0);
    assert_eq!(
        server.requests.lock().unwrap().len(),
        4,
        "cache/local completion sends no new embedding or chat POST"
    );
    std::fs::remove_file(path).unwrap();
}

#[tokio::test]
async fn persisted_revision_retry_finishes_and_publishes_at_zero_without_new_requests() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let store = Store::open_in_memory().await.unwrap();
    let teachers = AtomicUsize::new(0);
    let judges = AtomicUsize::new(0);
    let server = Server::new(move |path, request, _| {
        Response::ok(if path == "/embeddings" {
            attempt_common::embedding(Some(0.1))
        } else if request["model"].as_str().unwrap().starts_with("judge") {
            let (score, verdict) = if judges.fetch_add(1, Ordering::SeqCst) == 0 {
                (0.65, "revise")
            } else {
                (0.95, "accept")
            };
            attempt_common::grade(&judge_body(score, verdict), Some(0.2))
        } else {
            let index = teachers.fetch_add(1, Ordering::SeqCst);
            attempt_common::teacher(&format!("Answer {index}"), "stop", Some(0.1))
        })
    })
    .await;
    sqlx::query("CREATE TRIGGER stop_checkpoint BEFORE INSERT ON checkpoints BEGIN SELECT RAISE(FAIL, 'checkpoint interrupted'); END")
        .execute(store.raw_pool()).await.unwrap();
    let failed = engine(&store, &server, 5.0)
        .run("priors", &one_item_source(), CancellationToken::new())
        .await
        .unwrap_err();
    assert!(failed.to_string().contains("checkpoint interrupted"));
    sqlx::query("DROP TRIGGER stop_checkpoint")
        .execute(store.raw_pool())
        .await
        .unwrap();
    for (id, state) in [
        ("priors-s0-seed0-a0-c0", gw_schema::LifecycleState::Revising),
        (
            "priors-s0-seed0-a1-c0",
            gw_schema::LifecycleState::Formatted,
        ),
    ] {
        assert_eq!(store.get(id).await.unwrap().lifecycle.state, state);
    }
    let before = server.requests.lock().unwrap().len();
    let path = std::env::temp_dir().join(format!("gw-retry-local-{}.parquet", std::process::id()));
    let report = engine(&store, &server, 0.0)
        .with_export(ExportSpec {
            dst: path.clone(),
            target: TrlFormat::ChatML,
            cot: CotPolicy::Supervised,
            dataset_version: None,
        })
        .run("priors", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();
    assert!(
        report.completed,
        "persisted retry requires only local completion: {report:?}"
    );
    assert_eq!(report.pending_items, 0);
    assert_eq!(report.exported, 1);
    assert_eq!(server.requests.lock().unwrap().len(), before);
    assert_eq!(
        store
            .get("priors-s0-seed0-a1-c0")
            .await
            .unwrap()
            .lifecycle
            .state,
        gw_schema::LifecycleState::Exported
    );
    std::fs::remove_file(path).unwrap();
}

#[tokio::test]
async fn genuinely_new_generation_rebuilds_priors_through_normal_admission() {
    let store = Store::open_in_memory().await.unwrap();
    let server = server().await;
    let source = InMemorySeedSource::new(
        vec![
            good_candidate("What is 12*8?"),
            good_candidate("Compute twelve times eight."),
        ],
        1,
    );
    let finite = engine(&store, &server, 0.49);
    leave_first_formatted(&store, &finite, &source).await;
    let denied = finite
        .run("priors", &source, CancellationToken::new())
        .await
        .unwrap();
    assert!(!denied.completed);
    assert_eq!(denied.pending_items, 1);
    assert_eq!(server.requests.lock().unwrap().len(), 4);
    assert!(denied.halted_reason.unwrap().contains("threshold"));
    let allowed = engine(&store, &server, 1.0)
        .run("priors", &source, CancellationToken::new())
        .await
        .unwrap();
    assert!(allowed.completed);
    assert_eq!(
        allowed.errored, 1,
        "the new candidate is a vector duplicate, so no teacher is needed"
    );
    let receipts = store.model_attempts("priors").await.unwrap();
    assert_eq!(
        receipts
            .iter()
            .filter(|r| r.intent.context.purpose == AttemptPurpose::ResumePrior)
            .count(),
        1
    );
    assert_eq!(
        server.requests.lock().unwrap().len(),
        6,
        "one resume prior and one candidate embedding were admitted"
    );
}

#[tokio::test]
async fn locally_rejected_framing_never_initializes_priors_or_hits_a_finite_gate() {
    let store = Store::open_in_memory().await.unwrap();
    let server = server().await;
    let source = InMemorySeedSource::new(
        vec![
            good_candidate("What is 12*8?"),
            good_candidate("A leaked <think> marker"),
        ],
        1,
    );
    let engine = engine(&store, &server, 0.49);
    leave_first_formatted(&store, &engine, &source).await;
    let report = engine
        .run("priors", &source, CancellationToken::new())
        .await
        .unwrap();
    assert!(report.completed);
    assert_eq!(report.errored, 1);
    assert_eq!(report.pending_items, 0);
    assert_eq!(server.requests.lock().unwrap().len(), 4);
}

#[tokio::test]
async fn competing_generation_callers_do_not_retry_prior_initialization_after_a_halt_or_fault() {
    for storage_fault in [false, true] {
        let store = Store::open_in_memory().await.unwrap();
        let server = server().await;
        let source = InMemorySeedSource::new(
            vec![
                good_candidate("What is 12*8?"),
                good_candidate("Compute twelve times eight."),
            ],
            1,
        );
        let build = |limit| {
            let provider = Arc::new(server.provider());
            let mut clients = clients_with_embedder(
                store.clone(),
                provider.clone(),
                provider,
                Arc::new(server.embedder()),
            );
            clients.policy = AccountingPolicy::FiniteUsd { limit_usd: limit };
            Engine::new(clients, area_k(one_judge(), lenient_thresholds(), 3), 1)
        };
        leave_first_formatted(&store, &build(5.0), &source).await;
        let before = server.requests.lock().unwrap().len();
        if storage_fault {
            sqlx::query("CREATE TRIGGER fail_resume_prior BEFORE INSERT ON model_attempts WHEN json_extract(NEW.receipt_json, '$.intent.context.purpose') = 'resume_prior' BEGIN SELECT RAISE(FAIL, 'resume prior fault'); END").execute(store.raw_pool()).await.unwrap();
        }
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            build(if storage_fault { 5.0 } else { 0.0 }).run(
                "priors",
                &source,
                CancellationToken::new(),
            ),
        )
        .await
        .unwrap();
        if storage_fault {
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("resume prior fault")
            );
        } else {
            let report = result.unwrap();
            assert!(!report.completed);
            assert_eq!(report.pending_items, 1);
        }
        assert_eq!(
            server.requests.lock().unwrap().len(),
            before,
            "waiting initializers cannot send after the first halt or fault"
        );
    }
}
