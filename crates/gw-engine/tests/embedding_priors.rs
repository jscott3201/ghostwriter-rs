//! Hermetic coverage for live and resumed embedding-prior deduplication.

mod common;

use std::sync::{Arc, Mutex};

use common::*;
use gw_engine::{Engine, InMemorySeedSource, SeedSource};
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

struct PanicEmbedder;

impl Embedder for PanicEmbedder {
    fn embed(&self, _text: &str) -> Result<Vec<f32>, String> {
        panic!("cancelled seeding must not call the embedder")
    }
}

struct ScriptedEmbedder {
    results: Mutex<Vec<Result<Vec<f32>, String>>>,
}

impl ScriptedEmbedder {
    fn new(results: Vec<Result<Vec<f32>, String>>) -> Self {
        Self {
            results: Mutex::new(results),
        }
    }
}

impl Embedder for ScriptedEmbedder {
    fn embed(&self, _text: &str) -> Result<Vec<f32>, String> {
        self.results.lock().unwrap().remove(0)
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

#[tokio::test]
async fn revise_retry_dedups_against_other_items_prior() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher: Arc<dyn Provider> = Arc::new(ScriptedTeacher::new(
        vec![good_cot(0.01), good_cot(0.01)],
        2,
    ));
    let judge: Arc<dyn Provider> = Arc::new(ScriptedJudge::new(vec![
        &judge_body(0.95, "accept"),
        &judge_body(0.65, "revise"),
    ]));
    // first gate=A, first append=A, second gate=B, retry gate=A => duplicate.
    let embedder = Arc::new(ScriptedEmbedder::new(vec![
        Ok(vec![1.0, 0.0]),
        Ok(vec![1.0, 0.0]),
        Ok(vec![0.0, 1.0]),
        Ok(vec![1.0, 0.0]),
    ]));
    let clients = clients_with_embedder(store.clone(), teacher, judge, embedder, 25.0);
    let report = Engine::new(clients, area_k1(one_judge(), lenient_thresholds()), 1)
        .run("revise-priors", &two_items(), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(report.admitted, 1);
    assert_eq!(report.errored, 1, "dedup failure parks the retry at Error");
}

#[tokio::test]
async fn append_and_resume_seed_embed_failures_do_not_abort_run() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher: Arc<dyn Provider> = Arc::new(ScriptedTeacher::new(vec![good_cot(0.01)], 1));
    let judge: Arc<dyn Provider> = Arc::new(ScriptedJudge::new(vec![&judge_body(0.95, "accept")]));
    let first_embedder = Arc::new(ScriptedEmbedder::new(vec![
        Ok(vec![1.0, 0.0]),
        Err("append probe".into()),
    ]));
    let source = InMemorySeedSource::new(vec![good_candidate("failure semantics")], 1);
    let first = clients_with_embedder(store.clone(), teacher, judge, first_embedder, 25.0);
    let report = Engine::new(first, area_k1(one_judge(), lenient_thresholds()), 1)
        .run("failure-priors", &source, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(report.admitted, 1);

    let resume_teacher: Arc<dyn Provider> = Arc::new(ScriptedTeacher::new(vec![], 0));
    let resume_judge: Arc<dyn Provider> = Arc::new(ScriptedJudge::new(vec![]));
    let failing_seed = Arc::new(ScriptedEmbedder::new(vec![Err("seed probe".into())]));
    let resume = clients_with_embedder(store, resume_teacher, resume_judge, failing_seed, 25.0);
    let resumed = Engine::new(resume, area_k1(one_judge(), lenient_thresholds()), 1)
        .run("failure-priors", &source, CancellationToken::new())
        .await
        .unwrap();
    assert!(resumed.completed);
    assert_eq!(resumed.admitted, 1);
}

#[tokio::test]
async fn resumed_all_duplicate_stream_trips_circuit_breaker() {
    let store = Store::open_in_memory().await.unwrap();
    let source = InMemorySeedSource::new(
        (0..9)
            .map(|index| good_candidate(&format!("duplicate candidate {index}")))
            .collect(),
        1,
    );
    let first_teacher: Arc<dyn Provider> = Arc::new(ScriptedTeacher::new(vec![good_cot(0.01)], 1));
    let first_judge: Arc<dyn Provider> =
        Arc::new(ScriptedJudge::new(vec![&judge_body(0.95, "accept")]));
    let first = clients_with_embedder(
        store.clone(),
        first_teacher,
        first_judge,
        Arc::new(SameVectorEmbedder),
        0.01,
    );
    Engine::new(first, area_k1(one_judge(), lenient_thresholds()), 1)
        .run("breaker-priors", &source, CancellationToken::new())
        .await
        .unwrap();

    let teacher: Arc<dyn Provider> = Arc::new(ScriptedTeacher::new(vec![], 0));
    let judge: Arc<dyn Provider> = Arc::new(ScriptedJudge::new(vec![]));
    let resume = clients_with_embedder(store, teacher, judge, Arc::new(SameVectorEmbedder), 25.0);
    let error = Engine::new(resume, area_k1(one_judge(), lenient_thresholds()), 1)
        .run("breaker-priors", &source, CancellationToken::new())
        .await
        .expect_err("eight all-duplicate items trip the breaker");
    assert!(
        error.to_string().contains("circuit breaker"),
        "got: {error}"
    );
}

#[tokio::test]
async fn resumed_item_does_not_dedup_against_its_own_admitted_vector() {
    let store = Store::open_in_memory().await.unwrap();
    let source = InMemorySeedSource::new(vec![good_candidate("same resumed item")], 1);

    // Mint a realistic admitted envelope in a donor run, then stage it as an earlier admission for
    // the target run without advancing the target shard cursor.
    let donor_teacher: Arc<dyn Provider> = Arc::new(ScriptedTeacher::new(vec![good_cot(0.01)], 1));
    let donor_judge: Arc<dyn Provider> =
        Arc::new(ScriptedJudge::new(vec![&judge_body(0.95, "accept")]));
    let donor = clients(
        store.clone(),
        donor_teacher,
        donor_judge,
        25.0,
        gw_engine::EventSink::disconnected(),
    );
    Engine::new(donor, area_k1(one_judge(), lenient_thresholds()), 1)
        .run("donor-own", &source, CancellationToken::new())
        .await
        .unwrap();
    let mut prior = store
        .scan(&RecordFilter::new().run_id("donor-own"))
        .await
        .unwrap()
        .remove(0);
    prior.record_id = "staged-own-prior".into();
    prior.provenance.run_id = "resume-own".into();
    store
        .validate_or_record_run_partition("resume-own", 1, &source.prompts_hash().unwrap())
        .await
        .unwrap();
    store
        .create_run("resume-own", "{}", Some(25.0))
        .await
        .unwrap();
    store.put(&prior).await.unwrap();

    let teacher = Arc::new(ScriptedTeacher::new(vec![good_cot(0.01)], 1));
    let teacher_probe = Arc::clone(&teacher);
    let judge: Arc<dyn Provider> = Arc::new(ScriptedJudge::new(vec![&judge_body(0.95, "accept")]));
    let live = clients_with_embedder(store, teacher, judge, Arc::new(SameVectorEmbedder), 25.0);
    Engine::new(live, area_k1(one_judge(), lenient_thresholds()), 1)
        .run("resume-own", &source, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(
        teacher_probe.call_count(),
        1,
        "same-item prior must be excluded so resumed generation proceeds"
    );
}

#[tokio::test]
async fn pre_cancelled_resume_stops_prior_seeding_before_embed() {
    let store = Store::open_in_memory().await.unwrap();
    let source = InMemorySeedSource::new(vec![good_candidate("cancel seeded item")], 1);
    let teacher: Arc<dyn Provider> = Arc::new(ScriptedTeacher::new(vec![good_cot(0.01)], 1));
    let judge: Arc<dyn Provider> = Arc::new(ScriptedJudge::new(vec![&judge_body(0.95, "accept")]));
    let donor = clients(
        store.clone(),
        teacher,
        judge,
        25.0,
        gw_engine::EventSink::disconnected(),
    );
    Engine::new(donor, area_k1(one_judge(), lenient_thresholds()), 1)
        .run("cancel-donor", &source, CancellationToken::new())
        .await
        .unwrap();
    let mut prior = store
        .scan(&RecordFilter::new().run_id("cancel-donor"))
        .await
        .unwrap()
        .remove(0);
    prior.record_id = "cancel-staged-prior".into();
    prior.provenance.run_id = "cancel-seeding".into();
    store
        .validate_or_record_run_partition("cancel-seeding", 1, &source.prompts_hash().unwrap())
        .await
        .unwrap();
    store
        .create_run("cancel-seeding", "{}", Some(25.0))
        .await
        .unwrap();
    store.put(&prior).await.unwrap();

    let no_teacher: Arc<dyn Provider> = Arc::new(ScriptedTeacher::new(vec![], 0));
    let no_judge: Arc<dyn Provider> = Arc::new(ScriptedJudge::new(vec![]));
    let clients = clients_with_embedder(store, no_teacher, no_judge, Arc::new(PanicEmbedder), 25.0);
    let cancel = CancellationToken::new();
    cancel.cancel();
    let report = Engine::new(clients, area_k1(one_judge(), lenient_thresholds()), 1)
        .run("cancel-seeding", &source, cancel)
        .await
        .unwrap();
    assert!(!report.completed);
}
