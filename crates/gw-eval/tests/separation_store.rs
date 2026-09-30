//! Exact outcome joins through a real in-memory SQLite store; no provider or file I/O.

mod common;

use common::*;
use gw_eval::outcomes::OutcomeReason;
use gw_eval::{OutcomeStatus, SeparationConfig, analyze_store};
use gw_schema::Content;
use gw_storage::{RecordFilter, Store};

#[tokio::test]
async fn store_round_trip_preserves_exact_bindings_and_distinct_candidate_contents() {
    let store = Store::open_in_memory().await.unwrap();
    store
        .insert_historical_run("run-1", "{}", None)
        .await
        .unwrap();
    let (records, evidence) = corpus(100, true);
    assert_eq!(
        evidence.corpus[0].prompt_hash,
        evidence.corpus[1].prompt_hash
    );
    assert_ne!(
        evidence.corpus[0].record_hash,
        evidence.corpus[1].record_hash
    );
    for record in &records {
        store.put(record).await.unwrap();
    }
    let report = analyze_store(
        &store,
        &RecordFilter::new().run_id("run-1"),
        &SeparationConfig::default(),
        Some(&evidence),
    )
    .await
    .unwrap();
    assert!(report.passed());
    assert_eq!(report.diagnostics.n_allpass, 100);
    assert_eq!(report.outcome_evaluation.coverage.matched_records, 200);
    assert_eq!(report.outcome_evaluation.coverage.evaluated_prompts, 100);
    assert_eq!(report.outcome_evaluation.statistics.unwrap().mean_gap, 0.5);
}

#[tokio::test]
async fn run_filter_cannot_silently_drop_declared_members() {
    let store = Store::open_in_memory().await.unwrap();
    store
        .insert_historical_run("run-1", "{}", None)
        .await
        .unwrap();
    store
        .insert_historical_run("run-2", "{}", None)
        .await
        .unwrap();
    let (records, evidence) = corpus(100, true);
    for record in &records {
        store.put(record).await.unwrap();
    }
    let mut other_run = candidate(999, 0, Some(1.0));
    other_run.provenance.run_id = "run-2".into();
    store.put(&other_run).await.unwrap();
    let all = analyze_store(
        &store,
        &RecordFilter::new(),
        &SeparationConfig::default(),
        Some(&evidence),
    )
    .await
    .unwrap();
    let one = analyze_store(
        &store,
        &RecordFilter::new().run_id("run-1"),
        &SeparationConfig::default(),
        Some(&evidence),
    )
    .await
    .unwrap();
    assert!(all.passed() && one.passed());
    assert_eq!(
        all.outcome_evaluation.statistics,
        one.outcome_evaluation.statistics
    );
    assert_eq!(all.outcome_evaluation.coverage.scanned_records, 201);
    assert_eq!(one.outcome_evaluation.coverage.scanned_records, 200);
    let excluded = analyze_store(
        &store,
        &RecordFilter::new().run_id("run-2"),
        &SeparationConfig::default(),
        Some(&evidence),
    )
    .await
    .unwrap();
    assert_eq!(
        excluded.outcome_evaluation.status,
        OutcomeStatus::InvalidEvidence
    );
    assert_eq!(excluded.outcome_evaluation.coverage.matched_records, 0);
    assert!(
        excluded
            .outcome_evaluation
            .reasons
            .contains(&OutcomeReason::MissingRecord {
                record_id: "p0-c0".into()
            })
    );
}

#[tokio::test]
async fn an_updated_candidate_invalidates_the_frozen_content_binding() {
    let store = Store::open_in_memory().await.unwrap();
    store
        .insert_historical_run("run-1", "{}", None)
        .await
        .unwrap();
    let (mut records, evidence) = corpus(2, true);
    for record in &records {
        store.put(record).await.unwrap();
    }
    records[0].messages[1].content = Content::Text("changed after outcome collection".into());
    store.put(&records[0]).await.unwrap();
    let report = analyze_store(
        &store,
        &RecordFilter::new(),
        &SeparationConfig::default(),
        Some(&evidence),
    )
    .await
    .unwrap();
    assert_eq!(
        report.outcome_evaluation.status,
        OutcomeStatus::InvalidEvidence
    );
    assert!(
        report
            .outcome_evaluation
            .reasons
            .contains(&OutcomeReason::IdentityMismatch {
                record_id: "p0-c0".into(),
                field: "record_hash".into()
            })
    );
}

#[tokio::test]
async fn later_unlisted_records_do_not_enter_the_frozen_control_population() {
    let store = Store::open_in_memory().await.unwrap();
    store
        .insert_historical_run("run-1", "{}", None)
        .await
        .unwrap();
    let (records, evidence) = corpus(1, true);
    for record in &records {
        store.put(record).await.unwrap();
    }
    // A later candidate for the same prompt has the highest judge score, but is outside the
    // explicitly declared frozen corpus. Its outcome is neither invented nor added to the pool.
    store.put(&candidate(0, 2, Some(1.0))).await.unwrap();
    let report = analyze_store(
        &store,
        &RecordFilter::new(),
        &SeparationConfig::default(),
        Some(&evidence),
    )
    .await
    .unwrap();
    assert_eq!(report.outcome_evaluation.coverage.scanned_records, 3);
    assert_eq!(report.outcome_evaluation.coverage.declared_records, 2);
    assert_eq!(report.outcome_evaluation.coverage.known_outcomes, 2);
    assert_eq!(report.outcome_evaluation.statistics.unwrap().mean_gap, 0.5);
}
