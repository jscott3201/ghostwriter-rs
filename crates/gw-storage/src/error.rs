//! [`StorageError`] — the single error type surfaced by every [`crate::Store`] operation.
//!
//! Variants distinguish the failure classes a caller must reason about: SQL/transport faults
//! (`Sqlx`), migration failures (`Migrate`), JSON (de)serialization of the record envelope and
//! cache payloads (`Serde`), the columnar export path (`Arrow` / `Parquet` / `Io`), and the
//! lookup-miss sentinel (`NotFound`). No `anyhow` — this crate surfaces a typed error.

use thiserror::Error;

/// Everything that can go wrong in the storage layer.
///
/// `#[non_exhaustive]` so new variants can be added without a breaking change. Most variants
/// carry the underlying source via `#[from]`; `NotFound` and `Export` are constructed directly.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum StorageError {
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
