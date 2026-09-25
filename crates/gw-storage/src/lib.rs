//! `gw-storage` — the reproducible relational data plane.
//!
//! An async [`Store`] over a [`sqlx`] `SqlitePool` (SQLite is the v1 source of truth for
//! run/lifecycle/cache state, DATA-SCHEMA §4.2) plus an Arrow/Parquet columnar export of
//! admitted records. The store uses the sqlx **runtime** query API (`sqlx::query` /
//! `sqlx::query_as` + `.bind`), never the compile-time `query!` macros, so the build is hermetic
//! — no `DATABASE_URL` or live database is needed to compile or in CI.
//!
//! ## Layout
//!
//! - **error** — [`StorageError`], the single typed error (no `anyhow`).
//! - **store** — [`Store`]: [`open`](Store::open) (file DB, WAL, foreign keys) and
//!   [`open_in_memory`](Store::open_in_memory) (tests); migrations run on open via
//!   `sqlx::migrate!`.
//! - **records** — [`put`](Store::put) (idempotent UPSERT by `record_id`),
//!   [`get`](Store::get), [`scan`](Store::scan) / [`scan_stream`](Store::scan_stream) with a
//!   [`RecordFilter`], and [`advance_lifecycle`](Store::advance_lifecycle) (state + history in
//!   one transaction).
//! - **cache** — content hashing ([`record_hash`], [`prompt_hash`], [`completion_hash`]) and the
//!   "never re-spend" call cache ([`cache_get`](Store::cache_get) /
//!   [`cache_put`](Store::cache_put)).
//! - **runledger** — [`validate_or_record_run_partition`](Store::validate_or_record_run_partition),
//!   [`create_run`](Store::create_run), [`set_run_status`](Store::set_run_status),
//!   [`checkpoint`](Store::checkpoint), and [`resume_cursor`](Store::resume_cursor) for crash
//!   recovery.
//! - **export** — [`export_parquet`] / [`export_parquet_bytes`]: a lossless columnar dump of
//!   admitted records to Parquet (one canonical `messages_json` column per row, versioned by
//!   [`gw_schema::ExportSchemaVersion`]), returning a [`gw_schema::ExportManifest`].
//!
//! ## Scope (v1)
//!
//! This is a minimal `runs` + `records` + `lifecycle_history` + `cache` + `checkpoints` schema —
//! v1-enough for the deterministic write path. The following are intentionally DEFERRED:
//!
//! - **The full provenance DAG** (relational nodes/edges + recursive-CTE lineage). There is NO
//!   lineage edge table and NO lineage query API in this crate yet: parent record ids are stored
//!   only as an opaque field inside each record's JSON envelope (`provenance.parent_ids`), not as
//!   queryable edges. A dedicated DAG + recursive-CTE lineage queries are a follow-up
//!   (DATA-SCHEMA §4.5). selene-db and AionforgeMemory are derived/advisory only and are NOT on
//!   the v1 write path.
//! - **`push_to_hub` / the Hub exporter** — `export_parquet` writes a local shard; pushing to a
//!   versioned HF dataset revision is a follow-up.
//! - **Dedup / MinHash / decontamination** and the embedding vector index — separate concerns.
//!
//! ```no_run
//! use gw_storage::{Store, RecordFilter};
//! use gw_schema::{Verdict, LifecycleState};
//!
//! # async fn run() -> Result<(), gw_storage::StorageError> {
//! let store = Store::open("gw-run.sqlite").await?;
//! store.create_run("run-1", "{}", Some(25.0)).await?;
//! // ... put records, then advance their lifecycle ...
//! store
//!     .advance_lifecycle("rec-1", LifecycleState::Admitted, Some("passed gate"))
//!     .await?;
//! let admitted = store
//!     .scan(&RecordFilter::new().run_id("run-1").verdict(Verdict::Admit))
//!     .await?;
//! # let _ = admitted;
//! # Ok(())
//! # }
//! ```

mod cache;
mod error;
mod export;
mod records;
mod runledger;
mod store;

pub use cache::{completion_hash, prompt_hash, prompts_hash, record_hash};
pub use error::{Result, StorageError};
pub use export::{clean_messages_json, export_parquet, export_parquet_bytes};
pub use records::RecordFilter;
pub use runledger::{ResumePoint, RunStatus};
pub use store::{Store, now_rfc3339};
