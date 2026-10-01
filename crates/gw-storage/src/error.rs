//! [`StorageError`] — the single error type surfaced by every [`crate::Store`] operation.
//!
//! Variants distinguish the failure classes a caller must reason about: SQL/transport faults
//! (`Sqlx`), migration failures (`Migrate`), JSON (de)serialization of the record envelope and
//! cache payloads (`Serde`), the columnar export path (`Arrow` / `Parquet` / `Io`), run manifest
//! mismatches (`RunManifest`), and the lookup-miss sentinel (`NotFound`). No `anyhow` —
//! this crate surfaces a typed error.

use thiserror::Error;

/// The startup operation that failed before a store could be returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartupPhase {
    /// Opening the pool and applying connection PRAGMAs.
    Connect,
    /// Taking the startup-only migration write lock.
    MigrationBegin,
    /// Discovering, validating, or applying embedded migrations.
    MigrationApply,
    /// Committing the complete migration transaction.
    MigrationCommit,
}
impl std::fmt::Display for StartupPhase {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Connect => "connect",
            Self::MigrationBegin => "migration_begin",
            Self::MigrationApply => "migration_apply",
            Self::MigrationCommit => "migration_commit",
        })
    }
}

/// Everything that can go wrong in the storage layer.
///
/// `#[non_exhaustive]` so new variants can be added without a breaking change. Most variants
/// carry the underlying source via `#[from]`; `NotFound` and `Export` are constructed directly.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum StorageError {
    /// Startup failed before a usable store escaped; the original error remains the source.
    #[error(
        "storage startup {phase} failed after {connection_attempts} connection attempts and {elapsed_ms} ms: {source}"
    )]
    Startup {
        /// Bounded phase label containing no database path or user data.
        phase: StartupPhase,
        /// Number of connection initialization attempts made by this opener.
        connection_attempts: u32,
        /// Elapsed time for this opener, measured with a monotonic clock.
        elapsed_ms: u64,
        /// Original connection, transaction, or migration error.
        #[source]
        source: Box<StorageError>,
    },
    /// Connection initialization exhausted its ten-second deadline before migrations began.
    #[error("connection initialization exceeded its ten-second deadline")]
    StartupTimeout {
        /// Last observed SQLite BUSY, when an attempt completed before the deadline.
        #[source]
        last_busy: Option<Box<sqlx::Error>>,
    },
    /// A guarded record command did not match its complete expected snapshot or initial identity.
    #[error("record conflict for {record_id}: {reason}")]
    RecordConflict {
        /// The conflicting record, without its private payload.
        record_id: String,
        /// Non-secret conflict category.
        reason: String,
    },
    /// Persisted record data, projections or normalized history contradict each other.
    #[error("record integrity error for {record_id}: {reason}")]
    RecordIntegrity {
        /// The affected record.
        record_id: String,
        /// Non-secret integrity failure.
        reason: String,
    },
    /// Execution cannot reuse the run's immutable semantic evidence. Inspection/export remains valid.
    #[error(
        "run semantic manifest error for {run_id}: {reason}; use a new run ID for changed or unpinned semantics"
    )]
    RunManifest {
        /// Stable run identity.
        run_id: String,
        /// Non-secret reason; never raw configuration, endpoint or credentials.
        reason: String,
    },
    /// Operational policy cannot authorize the requested launch or physical send.
    #[error("request admission denied: {0}")]
    Admission(gw_schema::AdmissionDenial),
    /// Invalid or contradictory physical model request evidence.
    #[error("model attempt evidence error: {0}")]
    Attempt(String),

    /// Reference admission was cancelled before its durable commit decision.
    #[error("reference import cancelled before commit")]
    ReferenceImportCancelled,

    /// An artifact, receipt, or selected source snapshot failed integrity validation.
    #[error("export integrity error: {0}")]
    Export(String),

    /// A prepared publication failed; its durable ID supports exact provider-free recovery.
    #[error("export publication {publication_id} failed: {source}")]
    Publication {
        /// Durable local receipt identity, distinct from content identity.
        publication_id: String,
        /// The original failure, retained without replacement by cleanup errors.
        #[source]
        source: Box<StorageError>,
    },
    /// A SQL query, connection, or transaction failed.
    #[error("sqlx error: {0}")]
    Sqlx(#[from] sqlx::Error),

    /// A schema migration failed to apply (corrupt history, checksum mismatch, SQL error).
    #[error("migration error: {0}")]
    Migrate(#[from] sqlx::migrate::MigrateError),

    /// (De)serializing a [`gw_schema::TrainingRecord`], cache value, or config snapshot failed.
    #[error("serde_json error: {0}")]
    Serde(#[from] serde_json::Error),

    /// Building or writing the Arrow `RecordBatch` projection failed.
    #[error("arrow error: {0}")]
    Arrow(String),

    /// Encoding or flushing the Parquet writer failed.
    #[error("parquet error: {0}")]
    Parquet(String),

    /// Filesystem I/O while writing a Parquet shard failed.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// A `spawn_blocking` task driving the sync Arrow/Parquet writers was cancelled or panicked.
    #[error("blocking export task failed to join: {0}")]
    Join(String),

    /// A lookup (`get` / `resume_cursor`) found no matching row where one was required.
    #[error("not found: {0}")]
    NotFound(String),
}

impl From<arrow::error::ArrowError> for StorageError {
    fn from(err: arrow::error::ArrowError) -> Self {
        StorageError::Arrow(err.to_string())
    }
}

impl From<parquet::errors::ParquetError> for StorageError {
    fn from(err: parquet::errors::ParquetError) -> Self {
        StorageError::Parquet(err.to_string())
    }
}

impl From<tokio::task::JoinError> for StorageError {
    fn from(err: tokio::task::JoinError) -> Self {
        StorageError::Join(err.to_string())
    }
}

/// Convenience alias for results returned by storage operations.
pub type Result<T> = std::result::Result<T, StorageError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn not_found_renders_message() {
        let e = StorageError::NotFound("record 01J8".into());
        assert!(e.to_string().contains("01J8"));
    }

    #[test]
    fn serde_error_converts() {
        let bad: serde_json::Error = serde_json::from_str::<i32>("not json").unwrap_err();
        let e: StorageError = bad.into();
        assert!(matches!(e, StorageError::Serde(_)));
    }

    #[test]
    fn arrow_error_converts() {
        let a = arrow::error::ArrowError::SchemaError("boom".into());
        let e: StorageError = a.into();
        assert!(matches!(e, StorageError::Arrow(_)));
        assert!(e.to_string().contains("boom"));
    }
}
