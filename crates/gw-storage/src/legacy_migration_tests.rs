//! Upgrading an existing accounting database never rewrites acknowledged bytes or old history.
use crate::{
    Store,
    durability_support::*,
    receipt_recovery_tests::{intent, request},
};
use gw_schema::*;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions};
use std::borrow::Cow;
#[tokio::test]
async fn migration_six_retains_old_envelopes_history_and_accounting_bytes() {
    let dir = Directory::new("legacy-migration");
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(
            SqliteConnectOptions::new()
                .filename(dir.db())
                .create_if_missing(true)
                .journal_mode(SqliteJournalMode::Wal)
                .foreign_keys(true),
        )
        .await
        .unwrap();
    let migrations = sqlx::migrate!("./migrations");
    let prefix = sqlx::migrate::Migrator {
        migrations: Cow::Owned(migrations.iter().take(5).cloned().collect()),
        ..sqlx::migrate::Migrator::DEFAULT
    };
    prefix.run(&pool).await.unwrap();
    let manifest = serde_json::to_string(&request().manifest).unwrap();
    sqlx::query("INSERT INTO runs(run_id,config_json,status,created_at,shard_count,prompts_hash) VALUES ('run',?,'halted','original',1,?)").bind(&manifest).bind("a".repeat(64)).execute(&pool).await.unwrap();
    let expected = crate::record_data::normalize(&record()).unwrap();
    let envelope = serde_json::to_string_pretty(&expected).unwrap();
    sqlx::query("INSERT INTO records(record_id,run_id,lifecycle_state,record_hash,prompt_hash,record_json,updated_at) VALUES ('record','run','assistant_generated',?,?,?,'original')").bind(&expected.hashes.record_hash).bind(&expected.hashes.prompt_hash).bind(&envelope).execute(&pool).await.unwrap();
    sqlx::query("INSERT INTO lifecycle_history(record_id,state,at,detail) VALUES ('record','assistant_generated','original','legacy detail')").execute(&pool).await.unwrap();
    let policy = PolicyState {
        version: 1,
        epoch: 1,
        policy: AccountingPolicy::ObservationOnly,
    };
    let coverage = LaunchCoverage {
        version: 1,
        run_id: "run".into(),
        launch_id: "original-launch".into(),
        history: AccountingHistory::RecordedFromCreation,
        policy: Some(policy.clone()),
        teacher: AccountingCapability::PhysicalAttemptsV1,
        judge: AccountingCapability::NoModelRequests,
        embedding: AccountingCapability::NoModelRequests,
    };
    let coverage_json = serde_json::to_string_pretty(&coverage).unwrap();
    let receipt = AttemptReceipt {
        attempt_id: "original-attempt".into(),
        policy_epoch: Some(1),
        intent: intent(&coverage),
        metadata: AttemptMetadata::default(),
        observations: vec![],
        conflicts: vec![],
        transport: None,
        interpretation: None,
    };
    let receipt_json = serde_json::to_string_pretty(&receipt).unwrap();
    let policy_json = serde_json::to_string_pretty(&policy).unwrap();
    sqlx::query("INSERT INTO model_launches(launch_id,run_id,coverage_json,created_at) VALUES ('original-launch','run',?,'original')").bind(&coverage_json).execute(&pool).await.unwrap();
    sqlx::query("INSERT INTO model_attempts(attempt_id,run_id,launch_id,receipt_json,created_at,updated_at) VALUES ('original-attempt','run','original-launch',?,'original','original')").bind(&receipt_json).execute(&pool).await.unwrap();
    sqlx::query(
        "INSERT INTO run_accounting(run_id,policy_json,history_complete) VALUES ('run',?,1)",
    )
    .bind(&policy_json)
    .execute(&pool)
    .await
    .unwrap();
    pool.close().await;
    let store = Store::open(dir.db()).await.unwrap();
    integrity(&store).await;
    for (query, bytes) in [
        ("SELECT config_json FROM runs WHERE run_id='run'", manifest),
        (
            "SELECT record_json FROM records WHERE record_id='record'",
            envelope,
        ),
        (
            "SELECT coverage_json FROM model_launches WHERE launch_id='original-launch'",
            coverage_json,
        ),
        (
            "SELECT receipt_json FROM model_attempts WHERE attempt_id='original-attempt'",
            receipt_json,
        ),
        (
            "SELECT policy_json FROM run_accounting WHERE run_id='run'",
            policy_json,
        ),
    ] {
        let actual: String = sqlx::query_scalar(query)
            .fetch_one(store.pool())
            .await
            .unwrap();
        assert_eq!(actual, bytes);
    }
    assert_eq!(store.get("record").await.unwrap(), expected);
    assert_eq!(store.model_attempts("run").await.unwrap(), vec![receipt]);
    assert_eq!(store.model_launches("run").await.unwrap(), vec![coverage]);
    assert_eq!(
        store.lifecycle_history("record").await.unwrap(),
        vec![(
            "assistant_generated".into(),
            "original".into(),
            Some("legacy detail".into())
        )]
    );
    let nullable: (Option<String>, Option<i64>, Option<i64>) =
        sqlx::query_as("SELECT mutation_id,history_ordinal,attempt FROM lifecycle_history")
            .fetch_one(store.pool())
            .await
            .unwrap();
    assert_eq!(nullable, (None, None, None));
    let outcome = store
        .advance_lifecycle(&expected, LifecycleState::Verified, None)
        .await
        .unwrap();
    assert_eq!(outcome.record.lifecycle.history.len(), 3);
    assert_eq!(
        &outcome.record.lifecycle.history[..2],
        expected.lifecycle.history.as_slice()
    );
    assert_eq!(store.lifecycle_history("record").await.unwrap().len(), 2);
    integrity(&store).await;
    store.close().await;
}
