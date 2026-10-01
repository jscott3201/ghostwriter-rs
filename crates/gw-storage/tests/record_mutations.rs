//! Full-snapshot record commands must be atomic and safe to retry after later progress.
use gw_schema::{LifecycleState, TrainingRecord};
use gw_storage::Store;

fn record() -> TrainingRecord {
    serde_json::from_value(serde_json::json!({
        "record_id":"record", "schema_version":"1.0.0", "training_area":"fixture",
        "messages":[{"role":"user","content":"2+3"},{"role":"assistant","content":"5","reasoning":"addition"}],
        "provenance":{"run_id":"run","teacher":{"provider":"fixture","slug":"fixture"},"harness_version":"fixture"},
        "generation":{}, "lifecycle":{"state":"assistant_generated","history":[
            {"state":"user_synthesized","at":"2026-01-01T00:00:00Z","attempt":0},
            {"state":"assistant_generated","at":"2026-01-01T00:00:00Z","attempt":0}
        ]}
    })).unwrap()
}

async fn store() -> Store {
    let store = Store::open_in_memory().await.unwrap();
    store
        .insert_historical_run("run", "{}", None)
        .await
        .unwrap();
    store
}

async fn insert(store: &Store, record: &TrainingRecord) -> gw_storage::Result<()> {
    store.insert_record(record).await.map(|_| ())
}

async fn transition(
    store: &Store,
    expected: &TrainingRecord,
    updated: &TrainingRecord,
    to: LifecycleState,
) -> gw_storage::Result<()> {
    store
        .transition_record(expected, updated, to, None)
        .await
        .map(|_| ())
}

#[tokio::test]
async fn initial_insertion_mirrors_both_existing_generation_facts() {
    let store = store().await;
    insert(&store, &record()).await.unwrap();
    let history = store.lifecycle_history("record").await.unwrap();
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].0, "user_synthesized");
    assert_eq!(history[1].0, "assistant_generated");
}

#[tokio::test]
async fn compatible_initial_retry_preserves_later_progress() {
    let store = store().await;
    let initial = record();
    insert(&store, &initial).await.unwrap();
    let before = store.get("record").await.unwrap();
    transition(&store, &before, &before, LifecycleState::Verified)
        .await
        .unwrap();
    let progressed = store.get("record").await.unwrap();
    let history = store.lifecycle_history("record").await.unwrap();
    insert(&store, &initial).await.unwrap();
    assert_eq!(store.get("record").await.unwrap(), progressed);
    assert_eq!(store.lifecycle_history("record").await.unwrap(), history);
}

#[tokio::test]
async fn committed_command_retry_after_later_progress_is_a_noop() {
    let store = store().await;
    insert(&store, &record()).await.unwrap();
    let original = store.get("record").await.unwrap();
    let mut verified = original.clone();
    verified.tags.push("verified data".into());
    transition(&store, &original, &verified, LifecycleState::Verified)
        .await
        .unwrap();
    let before = store.get("record").await.unwrap();
    transition(&store, &before, &before, LifecycleState::Judged)
        .await
        .unwrap();
    let progressed = store.get("record").await.unwrap();
    let history = store.lifecycle_history("record").await.unwrap();
    transition(&store, &original, &verified, LifecycleState::Verified)
        .await
        .unwrap();
    assert_eq!(store.get("record").await.unwrap(), progressed);
    assert_eq!(store.lifecycle_history("record").await.unwrap(), history);
}

#[tokio::test]
async fn stale_noncontent_snapshot_cannot_overwrite_newer_progress() {
    let store = store().await;
    insert(&store, &record()).await.unwrap();
    let stale = store.get("record").await.unwrap();
    let mut winner = stale.clone();
    winner.cost.prompt_tokens = 7;
    transition(&store, &stale, &winner, LifecycleState::Verified)
        .await
        .unwrap();
    let current = store.get("record").await.unwrap();
    assert_eq!(current.hashes.record_hash, stale.hashes.record_hash);
    assert!(
        transition(&store, &stale, &stale, LifecycleState::Rejected)
            .await
            .is_err()
    );
    assert_eq!(store.get("record").await.unwrap(), current);
}

#[tokio::test]
async fn incompatible_initial_identity_and_changed_command_identity_are_conflicts() {
    use gw_storage::StorageError;
    let store = store().await;
    let initial = record();
    let expected = store.insert_record(&initial).await.unwrap().record;
    let mut other = initial.clone();
    other.cost.prompt_tokens = 99;
    assert!(matches!(
        store.insert_record(&other).await,
        Err(StorageError::RecordConflict { .. })
    ));
    let mut identity = expected.clone();
    identity.record_id = "another".into();
    assert!(matches!(
        store
            .transition_record(&expected, &identity, LifecycleState::Verified, None)
            .await,
        Err(StorageError::RecordConflict { .. })
    ));
    let mut history = expected.clone();
    history.lifecycle.history.clear();
    assert!(matches!(
        store
            .transition_record(&expected, &history, LifecycleState::Verified, None)
            .await,
        Err(StorageError::RecordConflict { .. })
    ));
    assert_eq!(store.get("record").await.unwrap(), expected);
    assert_eq!(store.lifecycle_history("record").await.unwrap().len(), 2);
}
