//! Run partition manifest tests for sound resume/replay.

use gw_storage::{StorageError, Store, prompts_hash};
use sqlx::Row;

async fn run_manifest(store: &Store, run_id: &str) -> (Option<i64>, Option<String>, String) {
    let row = sqlx::query("SELECT shard_count, prompts_hash, status FROM runs WHERE run_id = ?1")
        .bind(run_id)
        .fetch_one(store.raw_pool())
        .await
        .unwrap();
    (
        row.get::<Option<i64>, _>("shard_count"),
        row.get::<Option<String>, _>("prompts_hash"),
        row.get::<String, _>("status"),
    )
}

#[tokio::test]
async fn run_partition_manifest_columns_exist_after_migration() {
    let store = Store::open_in_memory().await.unwrap();
    let cols = sqlx::query("PRAGMA table_info(runs)")
        .fetch_all(store.raw_pool())
        .await
        .unwrap();
    let names: Vec<String> = cols
        .into_iter()
        .map(|row| row.get::<String, _>("name"))
        .collect();
    assert!(names.contains(&"shard_count".to_string()));
    assert!(names.contains(&"prompts_hash".to_string()));
}

#[tokio::test]
async fn prompts_hash_preserves_order_and_content() {
    let prompts = vec!["q1".to_string(), "q2".to_string()];
    let same = vec!["q1".to_string(), "q2".to_string()];
    let reordered = vec!["q2".to_string(), "q1".to_string()];
    let edited = vec!["q1".to_string(), "q2 edited".to_string()];

    assert_eq!(
        prompts_hash(&prompts).unwrap(),
        prompts_hash(&same).unwrap()
    );
    assert_ne!(
        prompts_hash(&prompts).unwrap(),
        prompts_hash(&reordered).unwrap()
    );
    assert_ne!(
        prompts_hash(&prompts).unwrap(),
        prompts_hash(&edited).unwrap()
    );
}

#[tokio::test]
async fn validate_or_record_persists_manifest_on_first_launch() {
    let store = Store::open_in_memory().await.unwrap();
    let hash = prompts_hash(&["q1".to_string(), "q2".to_string()]).unwrap();

    store
        .validate_or_record_run_partition("run-manifest", 2, &hash)
        .await
        .unwrap();

    let (shard_count, prompts_hash, status) = run_manifest(&store, "run-manifest").await;
    assert_eq!(shard_count, Some(2));
    assert_eq!(prompts_hash.as_deref(), Some(hash.as_str()));
    assert_eq!(status, "running");
}

#[tokio::test]
async fn create_run_does_not_overwrite_existing_manifest() {
    let store = Store::open_in_memory().await.unwrap();
    let hash = prompts_hash(&["q1".to_string(), "q2".to_string()]).unwrap();
    store
        .validate_or_record_run_partition("run-create", 2, &hash)
        .await
        .unwrap();

    store
        .create_run("run-create", "{\"cap\":1}", Some(1.0))
        .await
        .unwrap();

    let (shard_count, prompts_hash, status) = run_manifest(&store, "run-create").await;
    assert_eq!(shard_count, Some(2));
    assert_eq!(prompts_hash.as_deref(), Some(hash.as_str()));
    assert_eq!(status, "running");
}

#[tokio::test]
async fn validate_or_record_rejects_mismatch_and_legacy_null_manifest() {
    let store = Store::open_in_memory().await.unwrap();
    let hash = prompts_hash(&["q1".to_string(), "q2".to_string()]).unwrap();
    store
        .validate_or_record_run_partition("run-mismatch", 2, &hash)
        .await
        .unwrap();

    let err = store
        .validate_or_record_run_partition("run-mismatch", 3, &hash)
        .await
        .unwrap_err();
    assert!(matches!(err, StorageError::RunPartitionMismatch { .. }));

    store.create_run("legacy", "{}", Some(1.0)).await.unwrap();
    let err = store
        .validate_or_record_run_partition("legacy", 1, &hash)
        .await
        .unwrap_err();
    assert!(matches!(err, StorageError::RunPartitionMismatch { .. }));
}
