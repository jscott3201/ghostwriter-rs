//! [`Store`] — the async SQLite store over a [`sqlx::SqlitePool`].
//!
//! [`Store::open`] opens a file-backed database (WAL journal, foreign keys ON, created if
//! missing) and runs the embedded migrations. [`Store::open_in_memory`] opens an ephemeral
//! in-memory database for tests.
//!
//! **In-memory gotcha:** a `sqlite::memory:` pool gives each pooled connection its OWN private
//! database, so a multi-connection pool would see inconsistent state. [`Store::open_in_memory`]
//! pins `max_connections(1)` so every query in a test sees one consistent database.
//!
//! All operations use the sqlx **runtime** query API (`sqlx::query` / `sqlx::query_as` +
//! `.bind`), never the compile-time `query!` macros, so the build is hermetic — no
//! `DATABASE_URL` or live database is needed to compile or in CI. `sqlx::migrate!` only reads
//! the migrations directory at compile time and needs no database.

use std::path::Path;
use std::str::FromStr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use sqlx::sqlite::{
    SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqlitePoolOptions, SqliteSynchronous,
};

use crate::error::Result;

/// How long a connection waits for the write lock before returning `SQLITE_BUSY` (the WAL
/// single-writer model means concurrent writers queue here rather than failing immediately).
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

/// The embedded migration set, read from `./migrations` at compile time (no database needed).
static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("./migrations");

/// An async handle to the reproducible relational data plane.
///
/// Cheap to clone — internally an `Arc`-backed [`sqlx::SqlitePool`] — so the handle can be shared
/// across spawned generation workers. **Concurrency model:** SQLite in WAL mode allows many
/// concurrent readers but only ONE writer at a time; the pool serializes write transactions and a
/// blocked writer waits up to a 5-second busy timeout for the lock before surfacing
/// `SQLITE_BUSY`.
/// Cloning the handle shares the same pool — it does not grant additional write parallelism.
/// The full operation set is implemented across the crate's modules (records, cache, run-ledger)
/// but presents as inherent methods on this one type.
#[derive(Debug, Clone)]
pub struct Store {
    pool: SqlitePool,
}

impl Store {
    /// Open (creating if absent) a file-backed database at `path`, enabling WAL journaling and
    /// foreign-key enforcement, then run all pending migrations.
    ///
    /// WAL gives concurrent readers a snapshot while a single writer commits; `synchronous =
    /// NORMAL` is the WAL-safe durability setting. A 5-second busy timeout lets a blocked
    /// writer wait for the lock instead of failing immediately under contention. Foreign keys are
    /// enforced so the `records → runs` / `lifecycle_history → records` cascades hold. See the
    /// [`Store`] doc for the single-writer concurrency model.
    ///
    /// # Errors
    /// Returns [`StorageError`](crate::StorageError) if the file cannot be opened or a migration
    /// fails to apply.
    pub async fn open(path: impl AsRef<Path>) -> Result<Self> {
        let opts = SqliteConnectOptions::new()
            .filename(path.as_ref())
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Normal)
            .busy_timeout(BUSY_TIMEOUT)
            .foreign_keys(true);
        let pool = SqlitePoolOptions::new().connect_with(opts).await?;
        let store = Self { pool };
        store.migrate().await?;
        Ok(store)
    }

    /// Open an ephemeral in-memory database (for tests) and run all migrations.
    ///
    /// Pins `max_connections(1)` so every query shares one database — a multi-connection
    /// `sqlite::memory:` pool would give each connection its own private, empty database.
    ///
    /// # Errors
    /// Returns [`StorageError`](crate::StorageError) if the in-memory database cannot be opened
    /// or a migration fails to apply.
    pub async fn open_in_memory() -> Result<Self> {
        let opts = SqliteConnectOptions::from_str("sqlite::memory:")?.foreign_keys(true);
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(opts)
            .await?;
        let store = Self { pool };
        store.migrate().await?;
        Ok(store)
    }

    /// Run all pending embedded migrations against the pool.
    async fn migrate(&self) -> Result<()> {
        MIGRATOR.run(&self.pool).await?;
        Ok(())
    }

    /// Borrow the underlying pool. Crate-internal: sibling modules (records / cache / run-ledger)
    /// run their queries against it.
    pub(crate) fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Borrow the underlying [`sqlx::SqlitePool`] for advanced callers that need to compose their
    /// own queries or transactions within the same connection pool.
    #[must_use]
    pub fn raw_pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Close the pool, waiting for in-flight connections to be released. Optional — dropping the
    /// `Store` also closes the pool — but useful for a deterministic shutdown in tests.
    pub async fn close(&self) {
        self.pool.close().await;
    }
}

/// An RFC 3339 / ISO-8601 UTC timestamp string (e.g. `2026-06-21T12:34:56.789Z`).
///
/// Hand-formatted from the wall clock to avoid pulling a date-time dependency into the storage
/// layer. Timestamps are recorded verbatim and never participate in content hashing, so this is
/// for human/audit display only and need not be monotonic.
#[must_use]
pub fn now_rfc3339() -> String {
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs() as i64;
    let millis = dur.subsec_millis();
    let (y, mo, d, h, mi, s) = civil_from_unix(secs);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}.{millis:03}Z")
}

/// Convert Unix seconds (UTC) to civil `(year, month, day, hour, min, sec)` via Howard Hinnant's
/// `days_from_civil` inverse — no external date-time crate.
fn civil_from_unix(secs: i64) -> (i64, u32, u32, u32, u32, u32) {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (h, mi, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);

    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (year, m as u32, d as u32, h as u32, mi as u32, s as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn migrations_apply_in_memory() {
        let store = Store::open_in_memory().await.unwrap();
        // All five tables exist after migration.
        let names: Vec<(String,)> =
            sqlx::query_as("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                .fetch_all(store.pool())
                .await
                .unwrap();
        let tables: Vec<String> = names.into_iter().map(|(n,)| n).collect();
        for expected in [
            "cache",
            "checkpoints",
            "lifecycle_history",
            "records",
            "runs",
        ] {
            assert!(tables.contains(&expected.to_string()), "missing {expected}");
        }
    }

    #[test]
    fn rfc3339_known_epoch() {
        // 1_700_000_000 = 2023-11-14T22:13:20Z (a known fixed point).
        let (y, mo, d, h, mi, s) = civil_from_unix(1_700_000_000);
        assert_eq!((y, mo, d, h, mi, s), (2023, 11, 14, 22, 13, 20));
    }

    #[test]
    fn rfc3339_shape() {
        let ts = now_rfc3339();
        assert!(ts.ends_with('Z'));
        assert_eq!(ts.len(), 24, "want YYYY-MM-DDTHH:MM:SS.mmmZ");
    }
}
