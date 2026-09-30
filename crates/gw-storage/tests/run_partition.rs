//! Legacy partition columns remain readable; historical insertion cannot overwrite a run.
use gw_storage::Store;
use sqlx::Row;
#[tokio::test]
async fn run_partition_columns_exist_after_migration() {
    let store = Store::open_in_memory().await.unwrap();
    let columns = sqlx::query("PRAGMA table_info(runs)")
        .fetch_all(store.raw_pool())
        .await
        .unwrap();
    let names: Vec<String> = columns.into_iter().map(|r| r.get("name")).collect();
    assert!(names.contains(&"shard_count".into()));
    assert!(names.contains(&"prompts_hash".into()));
}
#[tokio::test]
async fn historical_import_is_insert_only_and_does_not_pin_execution() {
    let store = Store::open_in_memory().await.unwrap();
    store
        .insert_historical_run("r", "original", None)
        .await
        .unwrap();
    assert!(
        store
            .insert_historical_run("r", "overwrite", None)
            .await
            .is_err()
    );
    let row: (String, Option<i64>, Option<String>) =
        sqlx::query_as("SELECT config_json,shard_count,prompts_hash FROM runs WHERE run_id='r'")
            .fetch_one(store.raw_pool())
            .await
            .unwrap();
    assert_eq!(row, ("original".into(), None, None));
}
