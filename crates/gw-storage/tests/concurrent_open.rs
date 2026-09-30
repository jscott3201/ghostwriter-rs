//! Startup must finish one consistent migration set before any Store escapes.
use gw_storage::Store;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions};
use std::{borrow::Cow, error::Error, sync::Arc, time::Instant};

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
        .map(|opener| {
            let path = path.clone();
            let barrier = barrier.clone();
            tokio::spawn(async move {
                barrier.wait().await;
                let started = Instant::now();
                let mut phase = "store_open";
                let result: gw_storage::Result<()> = async {
                    let store = Store::open(path).await?;
                    phase = "acquire_check_connection";
                    let mut connection = store.raw_pool().acquire().await?;
                    phase = "migration_count";
                    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM _sqlx_migrations WHERE success=1")
                        .fetch_one(&mut *connection).await?;
                    assert_eq!(count as usize, MIGRATIONS.iter().count(), "opener={opener} phase={phase}");
                    phase = "migration_checksums";
                    let checksums: Vec<(i64, Vec<u8>)> = sqlx::query_as("SELECT version,checksum FROM _sqlx_migrations ORDER BY version")
                        .fetch_all(&mut *connection).await?;
                    for ((version, checksum), migration) in checksums.iter().zip(MIGRATIONS.iter()) {
                        assert_eq!(*version, migration.version, "opener={opener} phase={phase}");
                        assert_eq!(checksum.as_slice(), migration.checksum.as_ref(), "opener={opener} phase={phase}");
                    }
                    if applied > 0 {
                        phase = "preserved_run";
                        let preserved: (String, String, String) = sqlx::query_as("SELECT config_json,status,created_at FROM runs WHERE run_id='preserved'")
                            .fetch_one(&mut *connection).await?;
                        assert_eq!(preserved, ("{\"fixture\":true}".into(), "halted".into(), "original".into()), "opener={opener} phase={phase}");
                    }
                    phase = "journal_mode";
                    let mode: String = sqlx::query_scalar("PRAGMA journal_mode").fetch_one(&mut *connection).await?;
                    phase = "synchronous";
                    let synchronous: i64 = sqlx::query_scalar("PRAGMA synchronous").fetch_one(&mut *connection).await?;
                    phase = "foreign_keys";
                    let foreign_keys: i64 = sqlx::query_scalar("PRAGMA foreign_keys").fetch_one(&mut *connection).await?;
                    assert_eq!((mode.as_str(), synchronous, foreign_keys), ("wal", 1, 1), "opener={opener} effective PRAGMAs");
                    phase = "integrity_check";
                    let integrity: String = sqlx::query_scalar("PRAGMA integrity_check").fetch_one(&mut *connection).await?;
                    assert_eq!(integrity, "ok", "opener={opener} phase={phase}");
                    phase = "foreign_key_check";
                    assert!(sqlx::query("PRAGMA foreign_key_check").fetch_all(&mut *connection).await?.is_empty(), "opener={opener} phase={phase}");
                    drop(connection);
                    store.close().await;
                    Ok(())
                }.await;
                match result {
                    Ok(()) => Ok(format!("opener={opener} phase=ready elapsed_ms={}", started.elapsed().as_millis())),
                    Err(error) => Err(format!("opener={opener} phase={phase} elapsed_ms={} code={:?} source={error}", started.elapsed().as_millis(), sqlite_code(&error))),
                }
            })
        })
        .collect::<Vec<_>>();
    let mut failures = Vec::new();
    let mut diagnostics = Vec::new();
    for handle in handles {
        match handle.await.unwrap() {
            Ok(ready) => diagnostics.push(ready),
            Err(error) => {
                diagnostics.push(error.clone());
                failures.push(error);
            }
        }
    }
    std::fs::write(
        directory.join("startup-diagnostics.txt"),
        diagnostics.join("\n"),
    )
    .unwrap();
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

fn sqlite_code(mut error: &(dyn Error + 'static)) -> Option<String> {
    loop {
        if let Some(error) = error.downcast_ref::<sqlx::Error>()
            && let Some(code) = error.as_database_error().and_then(|error| error.code())
        {
            return Some(code.into_owned());
        }
        error = error.source()?;
    }
}
