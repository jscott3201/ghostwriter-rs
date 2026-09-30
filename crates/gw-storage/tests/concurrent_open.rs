//! Startup must finish one consistent migration set before any Store escapes.
use gw_storage::Store;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions};
use std::{borrow::Cow, sync::Arc};

static MIGRATIONS: sqlx::migrate::Migrator = sqlx::migrate!("./migrations");

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_fresh_opens_complete_consistently() {
    concurrent_open(0).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_pending_migration_opens_complete_consistently() {
    concurrent_open(2).await;
}

async fn concurrent_open(applied: usize) {
    let directory = std::env::temp_dir().join(format!("gw-open-{}-{applied}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let path = directory.join("state.sqlite");
    if applied > 0 {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(
                SqliteConnectOptions::new()
                    .filename(&path)
                    .create_if_missing(true)
                    .journal_mode(SqliteJournalMode::Wal)
                    .foreign_keys(true),
            )
            .await
            .unwrap();
        let prefix = sqlx::migrate::Migrator {
            migrations: Cow::Owned(MIGRATIONS.iter().take(applied).cloned().collect()),
            ..sqlx::migrate::Migrator::DEFAULT
        };
        prefix.run(&pool).await.unwrap();
        sqlx::query("INSERT INTO runs(run_id, config_json, status, created_at) VALUES ('preserved', '{\"fixture\":true}', 'halted', 'original')").execute(&pool).await.unwrap();
        pool.close().await;
    }
    let barrier = Arc::new(tokio::sync::Barrier::new(8));
    let handles = (0..8)
        .map(|_| {
            let path = path.clone();
            let barrier = barrier.clone();
            tokio::spawn(async move {
                barrier.wait().await;
                let store = Store::open(path).await?;
                let count: i64 =
                    sqlx::query_scalar("SELECT count(*) FROM _sqlx_migrations WHERE success=1")
                        .fetch_one(store.raw_pool())
                        .await?;
                assert_eq!(count as usize, MIGRATIONS.iter().count());
                let checksums: Vec<(i64, Vec<u8>)> = sqlx::query_as(
                    "SELECT version,checksum FROM _sqlx_migrations ORDER BY version",
                )
                .fetch_all(store.raw_pool())
                .await?;
                for ((version, checksum), migration) in checksums.iter().zip(MIGRATIONS.iter()) {
                    assert_eq!(*version, migration.version);
                    assert_eq!(checksum.as_slice(), migration.checksum.as_ref());
                }
                if applied > 0 {
                    let preserved: (String, String, String) = sqlx::query_as(
                        "SELECT config_json,status,created_at FROM runs WHERE run_id='preserved'",
                    )
                    .fetch_one(store.raw_pool())
                    .await?;
                    assert_eq!(
                        preserved,
                        (
                            "{\"fixture\":true}".into(),
                            "halted".into(),
                            "original".into()
                        )
                    );
                }
                let integrity: String = sqlx::query_scalar("PRAGMA integrity_check")
                    .fetch_one(store.raw_pool())
                    .await?;
                assert_eq!(integrity, "ok");
                assert!(
                    sqlx::query("PRAGMA foreign_key_check")
                        .fetch_all(store.raw_pool())
                        .await?
                        .is_empty()
                );
                store.close().await;
                gw_storage::Result::Ok(())
            })
        })
        .collect::<Vec<_>>();
    let mut failures = Vec::new();
    for handle in handles {
        if let Err(error) = handle.await.unwrap() {
            failures.push(error.to_string());
        }
    }
    assert!(
        failures.is_empty(),
        "startup failures {failures:?}; preserved {}",
        directory.display()
    );
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_accounting_database_upgrade_preserves_applied_checksums_and_data() {
    concurrent_open(5).await;
}
