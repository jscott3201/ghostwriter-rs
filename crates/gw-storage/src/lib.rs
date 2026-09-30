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
//! - **records** — [`insert_record`](Store::insert_record) (insert or recognize an initial command),
//!   [`get`](Store::get), [`scan`](Store::scan) / [`scan_stream`](Store::scan_stream) with a
//!   [`RecordFilter`], and [`transition_record`](Store::transition_record) (full-envelope guarded,
//!   idempotent envelope + projection + history commit).
//! - **cache** — content hashing ([`record_hash`], [`prompt_hash`], [`completion_hash`]) and the
//!   "never re-spend" call cache ([`cache_get`](Store::cache_get) /
//!   [`cache_put`](Store::cache_put)).
//! - **runledger** — [`validate_run_manifest`](Store::validate_run_manifest),
//!   [`insert_historical_run`](Store::insert_historical_run), [`set_run_status`](Store::set_run_status),
//!   [`checkpoint`](Store::checkpoint), and [`resume_cursor`](Store::resume_cursor) for crash
//!   recovery.
//! - **export** — [`Store::publish_export`] / [`export_parquet_bytes`]: a lossless columnar dump of
//!   admitted records to Parquet (one canonical `messages_json` column per row, versioned by
//!   [`gw_schema::ExportSchemaVersion`]), returning a [`gw_schema::ExportManifest`].
//!
//! ## Persistence acknowledgments
//!
//! A successful write acknowledges a committed SQLite transaction. File stores use WAL with
//! `synchronous=NORMAL`: committed writes survive application-process termination, but an OS crash
//! or power loss can discard acknowledged transactions. Failed, canceled or lost acknowledgments
//! do not prove rollback; SQLx may already have processed COMMIT when the caller disappears.
//!
//! Record commands commit the envelope, all indexed projections, mutation receipt and corresponding
//! lifecycle history together. They compare the complete expected envelope and recognize the same
//! committed command before checking that snapshot. A retry returns the current record, including
//! later progress, without another transition. New inserts mirror supplied generation facts once;
//! legacy history is retained as written. [`Store::replace_record_for_import`] is an explicit
//! unguarded replacement for fixtures/imports and clears that record's command/history ledger.
//!
//! Manifest comparison, launch coverage and policy registration share their launch transaction.
//! Attempt intent, metadata observations, transport settlement and output interpretation each have
//! their own acknowledgment; a settled receipt does not establish persisted output or cache data.
//! Cache entries, run-status writes and shard cursors also commit independently. An engine crash
//! after saving a record but before advancing its cursor re-drives the item from persisted work.
//! These operations do not make a sibling group, provider transmission or file publication atomic.
//! Publication retains its prepared receipt and one transaction acknowledging its selected members.
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
//! - **`push_to_hub` / the Hub exporter** — `Store::publish_export` writes a local shard; pushing to a
//!   versioned HF dataset revision is a follow-up.
//! - **Dedup / MinHash / decontamination** and the embedding vector index — separate concerns.
//!
//! ```no_run
//! use gw_storage::{Store, RecordFilter};
//! use gw_schema::{Verdict, LifecycleState};
//!
//! # async fn run() -> Result<(), gw_storage::StorageError> {
//! let store = Store::open("gw-run.sqlite").await?;
//! store.insert_historical_run("run-1", "{}", Some(25.0)).await?;
//! // ... insert records, then advance using the snapshot whose state was inspected ...
//! let expected = store.get("rec-1").await?;
//! store
//!     .advance_lifecycle(&expected, LifecycleState::Admitted, Some("passed gate"))
//!     .await?;
//! let admitted = store
//!     .scan(&RecordFilter::new().run_id("run-1").verdict(Verdict::Admit))
//!     .await?;
//! # let _ = admitted;
//! # Ok(())
//! # }
//! ```

mod accounting;
mod admission;
mod artifact;
mod attempts;
pub use admission::{AttemptAdmission, LaunchRequest};
mod cache;
mod error;
mod export;
mod publication;
mod receipts;
mod record_data;
mod record_mutations;
mod records;
mod run_manifest;
mod runledger;
pub use run_manifest::RunMode;
mod store;
#[cfg(test)]
mod test_hooks;

pub use artifact::{ARTIFACT_METADATA_KEY, ArtifactVerification, ExportPlan, verify_artifact};
pub use cache::{canonical_json_hash, completion_hash, prompt_hash, record_hash};
pub use error::{Result, StorageError};
pub use export::{clean_messages_json, export_parquet_bytes};
pub use publication::{ExportPublication, PublicationDisposition};
pub use receipts::ExportPurpose;
pub use record_mutations::{RecordWriteOutcome, RecordWriteStatus};
pub use records::RecordFilter;
pub use runledger::{ResumePoint, RunStatus};
pub use store::{Store, now_rfc3339};

#[cfg(test)]
mod durability_support;
#[cfg(test)]
mod record_durability_tests;

#[cfg(test)]
mod process_recovery_tests;
#[cfg(test)]
mod receipt_recovery_tests;

#[cfg(test)]
mod receipt_evidence_tests;

#[cfg(test)]
mod legacy_migration_tests;

#[cfg(test)]
mod publication_integrity_tests;
