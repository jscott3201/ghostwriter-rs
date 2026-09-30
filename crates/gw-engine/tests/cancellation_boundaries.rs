//! Cancellation while a storage lookup is pending must not admit a new teacher transition.

mod common;

use std::sync::Arc;

use common::*;
use gw_engine::{EventSink, RunControl, SeedSource, revise_once, run_group};
use gw_schema::{BudgetBreach, LifecycleState};
use gw_storage::Store;
use tokio_util::sync::CancellationToken;

struct CancelOnEmbed(CancellationToken);

impl gw_generate::Embedder for CancelOnEmbed {
    fn embed(&self, _: &str) -> std::result::Result<Vec<f32>, String> {
        self.0.cancel();
        Ok(Vec::new())
    }
}

#[tokio::test]
async fn cancellation_during_generation_preparation_stops_initial_dispatch() {
    let store = Store::open_in_memory().await.unwrap();
    store.create_run("prepare", "{}", None).await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(vec![], 1));
    let cancel = CancellationToken::new();
    let cl = clients_with_embedder(
        store,
        teacher.clone(),
        Arc::new(ScriptedJudge::new(vec![])),
        Arc::new(CancelOnEmbed(cancel.clone())),
        25.0,
    );
    let area = area_k1(one_judge(), lenient_thresholds());
    let seed = one_item_source().items_for_shard(0).remove(0);
    let group = run_group(
        "prepare",
        0,
        &seed,
        &cl,
        &area,
        RunControl::new(&cancel, BudgetBreach::Drain),
    )
    .await
    .unwrap();
    assert!(group.interrupted);
    assert!(group.siblings.is_empty());
    assert_eq!(teacher.call_count(), 0);
}

#[tokio::test]
async fn cancellation_during_missing_sibling_lookup_stops_generation() {
    let store = Store::open_in_memory().await.unwrap();
    store.create_run("lookup", "{}", None).await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(vec![], 1));
    let cl = clients(
        store.clone(),
        teacher.clone(),
        Arc::new(ScriptedJudge::new(vec![])),
        25.0,
        EventSink::disconnected(),
    );
    let area = area_k1(one_judge(), lenient_thresholds());
    let seed = one_item_source().items_for_shard(0).remove(0);
    let cancel = CancellationToken::new();
    // In-memory stores have exactly one connection. Holding it suspends the awaited lookup.
    let connection = store.raw_pool().acquire().await.unwrap();
    let mut group = Box::pin(run_group(
        "lookup",
        0,
        &seed,
        &cl,
        &area,
        RunControl::new(&cancel, BudgetBreach::Drain),
    ));
    assert!(futures::poll!(&mut group).is_pending());
    cancel.cancel();
    drop(connection);
    let outcome = group.await.unwrap();
    assert!(outcome.interrupted);
    assert!(outcome.siblings.is_empty());
    assert_eq!(teacher.call_count(), 0);
}

#[tokio::test]
async fn cancellation_during_missing_retry_lookup_keeps_revising_original() {
    let store = Store::open_in_memory().await.unwrap();
    store.create_run("retry", "{}", None).await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(vec![], 2));
    let cl = clients(
        store.clone(),
        teacher.clone(),
        Arc::new(ScriptedJudge::new(vec![&judge_body(0.65, "revise")])),
        25.0,
        EventSink::disconnected(),
    );
    let area = area_k1(one_judge(), lenient_thresholds());
    let seed = one_item_source().items_for_shard(0).remove(0);
    let cancel = CancellationToken::new();
    let control = RunControl::new(&cancel, BudgetBreach::Drain);
    let group = run_group("retry", 0, &seed, &cl, &area, control)
        .await
        .unwrap();
    let original = &group.siblings[0];
    assert_eq!(original.lifecycle.state, LifecycleState::Revising);
    let connection = store.raw_pool().acquire().await.unwrap();
    let mut retry = Box::pin(revise_once(
        "retry", 0, &seed, original, &cl, &area, control,
    ));
    assert!(futures::poll!(&mut retry).is_pending());
    cancel.cancel();
    drop(connection);
    let result = retry.await.unwrap();
    assert_eq!(&result, original);
    assert_eq!(teacher.call_count(), 1);
}

#[tokio::test]
async fn cancellation_during_retry_preparation_keeps_revising_original() {
    let store = Store::open_in_memory().await.unwrap();
    store.create_run("prepare-retry", "{}", None).await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(vec![], 2));
    let mut cl = clients(
        store,
        teacher.clone(),
        Arc::new(ScriptedJudge::new(vec![&judge_body(0.65, "revise")])),
        25.0,
        EventSink::disconnected(),
    );
    let area = area_k1(one_judge(), lenient_thresholds());
    let seed = one_item_source().items_for_shard(0).remove(0);
    let cancel = CancellationToken::new();
    let control = RunControl::new(&cancel, BudgetBreach::Drain);
    let group = run_group("prepare-retry", 0, &seed, &cl, &area, control)
        .await
        .unwrap();
    let original = &group.siblings[0];
    assert_eq!(original.lifecycle.state, LifecycleState::Revising);
    cl.embedder = Arc::new(CancelOnEmbed(cancel.clone()));
    let result = revise_once("prepare-retry", 0, &seed, original, &cl, &area, control)
        .await
        .unwrap();
    assert_eq!(&result, original);
    assert_eq!(teacher.call_count(), 1);
}
