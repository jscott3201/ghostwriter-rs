//! Hermetic coverage for live and resumed embedding-prior deduplication.

mod common;

use std::sync::Arc;

use common::*;
use gw_engine::{Engine, InMemorySeedSource};
use gw_generate::Embedder;
use gw_providers::Provider;
use gw_schema::LifecycleState;
use gw_storage::{RecordFilter, Store};
use tokio_util::sync::CancellationToken;

struct SameVectorEmbedder;

impl Embedder for SameVectorEmbedder {
    fn embed(&self, _text: &str) -> Result<Vec<f32>, String> {
        Ok(vec![1.0, 0.0])
    }
}

fn two_items() -> InMemorySeedSource {
    InMemorySeedSource::new(
        vec![
            good_candidate("What is 12*8?"),
            good_candidate("Compute twelve times eight."),
        ],
        1,
    )
}

#[tokio::test]
async fn admitted_turn_is_appended_and_duplicate_parks_at_error() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher: Arc<dyn Provider> = Arc::new(ScriptedTeacher::new(vec![good_cot(0.01)], 1));
    let judge: Arc<dyn Provider> = Arc::new(ScriptedJudge::new(vec![&judge_body(0.95, "accept")]));
    let clients = clients_with_embedder(
        store.clone(),
        teacher,
        judge,
        Arc::new(SameVectorEmbedder),
        25.0,
    );
    let engine = Engine::new(clients, area_k1(one_judge(), lenient_thresholds()), 1);

    let report = engine
        .run("live-priors", &two_items(), CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(report.admitted, 1);
    assert_eq!(report.errored, 1);
    let errors = store
        .scan(
            &RecordFilter::new()
                .run_id("live-priors")
                .lifecycle_state(LifecycleState::Error),
        )
        .await
        .unwrap();
    assert_eq!(errors.len(), 1);
}

#[tokio::test]
async fn resumed_run_seeds_priors_before_gating_remaining_item() {
    let store = Store::open_in_memory().await.unwrap();
    let first_teacher: Arc<dyn Provider> = Arc::new(ScriptedTeacher::new(vec![good_cot(0.01)], 1));
    let first_judge: Arc<dyn Provider> =
        Arc::new(ScriptedJudge::new(vec![&judge_body(0.95, "accept")]));
    let first_clients = clients_with_embedder(
        store.clone(),
        first_teacher,
        first_judge,
        Arc::new(SameVectorEmbedder),
        0.01,
    );
    Engine::new(first_clients, area_k1(one_judge(), lenient_thresholds()), 1)
        .run("resume-priors", &two_items(), CancellationToken::new())
        .await
        .unwrap();

    let resume_teacher: Arc<dyn Provider> = Arc::new(ScriptedTeacher::new(vec![], 0));
    let resume_judge: Arc<dyn Provider> = Arc::new(ScriptedJudge::new(vec![]));
    let resume_clients = clients_with_embedder(
        store.clone(),
        resume_teacher,
        resume_judge,
        Arc::new(SameVectorEmbedder),
        25.0,
    );
    let report = Engine::new(
        resume_clients,
        area_k1(one_judge(), lenient_thresholds()),
        1,
    )
    .run("resume-priors", &two_items(), CancellationToken::new())
    .await
    .unwrap();

    assert_eq!(report.admitted, 1);
    assert_eq!(report.errored, 1);
}
