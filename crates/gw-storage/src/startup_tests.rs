//! Real connection contention at the WAL switch, with an error-boundary barrier.
use crate::{
    StartupPhase, StorageError, Store,
    durability_support::{Directory, integrity},
    startup::StartupHook,
};
use sqlx::{
    Connection, Executor, SqliteConnection,
    sqlite::{SqliteConnectOptions, SqliteJournalMode},
};
use std::{path::Path, time::Duration};

async fn held_delete_reader(path: &Path) -> SqliteConnection {
    let mut reader = SqliteConnection::connect_with(
        &SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Delete),
    )
    .await
    .unwrap();
    reader
        .execute("CREATE TABLE fixture(value INTEGER); INSERT INTO fixture VALUES (7);")
        .await
        .unwrap();
    reader.execute("BEGIN").await.unwrap();
    let value: i64 = sqlx::query_scalar("SELECT value FROM fixture")
        .fetch_one(&mut reader)
        .await
        .unwrap();
    assert_eq!(value, 7);
    let mode: String = sqlx::query_scalar("PRAGMA journal_mode")
        .fetch_one(&mut reader)
        .await
        .unwrap();
    assert_eq!(mode, "delete");
    reader
}

struct ContendedOpen {
    opener: tokio::task::JoinHandle<crate::Result<Store>>,
    release: tokio::sync::oneshot::Sender<()>,
}

async fn observed_busy(path: &Path) -> ContendedOpen {
    let path = path.to_owned();
    let (observed, observation) = tokio::sync::oneshot::channel();
    let (release, released) = tokio::sync::oneshot::channel();
    let opener = tokio::spawn(async move {
        Store::open_with_startup_hook(
            &path,
            StartupHook {
                observed,
                release: released,
            },
        )
        .await
    });
    let observation = tokio::time::timeout(Duration::from_secs(10), observation)
        .await
        .unwrap()
        .unwrap();
    println!(
        "connection boundary phase={} attempt={} elapsed_ms={} code={:?}",
        observation.phase, observation.attempt, observation.elapsed_ms, observation.code
    );
    assert_eq!(observation.phase, StartupPhase::Connect);
    assert_eq!(observation.attempt, 1);
    assert_eq!(observation.code.as_deref(), Some("5"));
    ContendedOpen { opener, release }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn released_delete_reader_allows_open_after_real_connection_busy() {
    let directory = Directory::new("startup-release");
    let path = directory.db();
    let mut reader = held_delete_reader(&path).await;
    let ContendedOpen { opener, release } = observed_busy(&path).await;
    let migration_tables: i64 =
        sqlx::query_scalar("SELECT count(*) FROM sqlite_master WHERE name='_sqlx_migrations'")
            .fetch_one(&mut reader)
            .await
            .unwrap();
    assert_eq!(
        migration_tables, 0,
        "contention must precede migration setup"
    );
    reader.execute("ROLLBACK").await.unwrap();
    reader.close().await.unwrap();
    release.send(()).unwrap();
    let store = opener
        .await
        .unwrap()
        .expect("released connection contention should be retried");
    integrity(&store).await;
    let preserved: i64 = sqlx::query_scalar("SELECT value FROM fixture")
        .fetch_one(store.raw_pool())
        .await
        .unwrap();
    assert_eq!(preserved, 7);
    let busy_timeout: i64 = sqlx::query_scalar("PRAGMA busy_timeout")
        .fetch_one(store.raw_pool())
        .await
        .unwrap();
    assert_eq!(busy_timeout, 5_000);
    assert_eq!(
        store.raw_pool().options().get_acquire_timeout(),
        Duration::from_secs(30),
        "startup deadline must not alter subsequent pool acquisitions"
    );
    store.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sustained_delete_reader_exhausts_connection_deadline_with_original_busy() {
    let directory = Directory::new("startup-deadline");
    let path = directory.db();
    let mut reader = held_delete_reader(&path).await;
    let started = tokio::time::Instant::now();
    let ContendedOpen { opener, release } = observed_busy(&path).await;
    release.send(()).unwrap();
    let error = tokio::time::timeout_at(started + Duration::from_secs(12), opener)
        .await
        .expect("connection deadline must include time inside SQLite")
        .unwrap()
        .expect_err("sustained reader must prevent the WAL switch");
    let StorageError::Startup {
        phase,
        connection_attempts,
        elapsed_ms,
        source,
    } = error
    else {
        panic!("missing startup context: {error}");
    };
    assert_eq!(phase, StartupPhase::Connect);
    assert!(connection_attempts >= 2);
    assert!((10_000..12_000).contains(&elapsed_ms), "{elapsed_ms}");
    let StorageError::StartupTimeout {
        last_busy: Some(last_busy),
    } = *source
    else {
        panic!("missing deadline or original SQLite BUSY: {source}");
    };
    assert_eq!(
        last_busy.as_database_error().unwrap().code().as_deref(),
        Some("5")
    );
    let migration_tables: i64 =
        sqlx::query_scalar("SELECT count(*) FROM sqlite_master WHERE name='_sqlx_migrations'")
            .fetch_one(&mut reader)
            .await
            .unwrap();
    assert_eq!(migration_tables, 0);
    reader.execute("ROLLBACK").await.unwrap();
    reader.close().await.unwrap();
    let store = Store::open(&path).await.unwrap();
    integrity(&store).await;
    store.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelling_at_connection_busy_leaves_no_migrations_and_can_reopen() {
    let directory = Directory::new("startup-cancel");
    let path = directory.db();
    let mut reader = held_delete_reader(&path).await;
    let ContendedOpen { opener, release } = observed_busy(&path).await;
    opener.abort();
    assert!(opener.await.unwrap_err().is_cancelled());
    assert!(release.send(()).is_err());
    let migration_tables: i64 =
        sqlx::query_scalar("SELECT count(*) FROM sqlite_master WHERE name='_sqlx_migrations'")
            .fetch_one(&mut reader)
            .await
            .unwrap();
    assert_eq!(migration_tables, 0);
    reader.execute("ROLLBACK").await.unwrap();
    reader.close().await.unwrap();
    let store = Store::open(&path).await.unwrap();
    integrity(&store).await;
    let preserved: i64 = sqlx::query_scalar("SELECT value FROM fixture")
        .fetch_one(store.raw_pool())
        .await
        .unwrap();
    assert_eq!(preserved, 7);
    store.close().await;
}

#[tokio::test]
async fn corrupt_database_returns_original_nonbusy_error_without_retry() {
    let directory = Directory::new("startup-corrupt");
    std::fs::write(directory.db(), b"This is not a SQLite database.").unwrap();
    let error = Store::open(directory.db()).await.unwrap_err();
    let StorageError::Startup {
        phase,
        connection_attempts,
        elapsed_ms,
        source,
    } = error
    else {
        panic!("missing startup context: {error}");
    };
    assert_eq!(phase, StartupPhase::Connect);
    assert_eq!(connection_attempts, 1);
    assert!(
        elapsed_ms < 5_000,
        "nonbusy error was delayed: {elapsed_ms}"
    );
    let StorageError::Sqlx(source) = *source else {
        panic!("original SQLx error replaced: {source}");
    };
    assert_eq!(
        source.as_database_error().unwrap().code().as_deref(),
        Some("26")
    );
}

#[tokio::test]
async fn migration_checksum_error_retains_phase_and_source_without_retry() {
    let directory = Directory::new("startup-checksum");
    let store = Store::open(directory.db()).await.unwrap();
    sqlx::query("UPDATE _sqlx_migrations SET checksum=x'00' WHERE version=1")
        .execute(store.raw_pool())
        .await
        .unwrap();
    store.close().await;
    let error = Store::open(directory.db()).await.unwrap_err();
    let StorageError::Startup {
        phase,
        connection_attempts,
        source,
        ..
    } = error
    else {
        panic!("missing startup context: {error}");
    };
    assert_eq!(phase, StartupPhase::MigrationApply);
    assert_eq!(connection_attempts, 1);
    assert!(matches!(
        *source,
        StorageError::Migrate(sqlx::migrate::MigrateError::VersionMismatch(1))
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn migration_lock_contention_is_not_replayed_as_connection_contention() {
    let directory = Directory::new("startup-migration-lock");
    let holder = Store::open(directory.db()).await.unwrap();
    let transaction = holder
        .raw_pool()
        .begin_with("BEGIN IMMEDIATE")
        .await
        .unwrap();
    let error = Store::open(directory.db()).await.unwrap_err();
    let StorageError::Startup {
        phase,
        connection_attempts,
        source,
        ..
    } = error
    else {
        panic!("missing startup context: {error}");
    };
    assert_eq!(phase, StartupPhase::MigrationBegin);
    assert_eq!(connection_attempts, 1);
    let StorageError::Sqlx(source) = *source else {
        panic!("migration lock error was replaced: {source}");
    };
    assert_eq!(
        source.as_database_error().unwrap().code().as_deref(),
        Some("5")
    );
    transaction.rollback().await.unwrap();
    holder.close().await;
    let store = Store::open(directory.db()).await.unwrap();
    integrity(&store).await;
    store.close().await;
}
