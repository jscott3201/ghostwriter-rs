//! The single typed error for `gw-eval` ([`EvalError`]) and the crate [`Result`] alias.
//!
//! Library code never uses `anyhow`: every fallible path returns a typed [`EvalError`] so a
//! caller (the `gen audit-separation` / `gen promote` CLI seams) can match on the cause. The
//! pure analysis cores ([`crate::separation::analyze`], [`crate::promote::promote`]) are
//! INFALLIBLE — they return their reports directly — so the only error sources are the thin
//! async Store-reading wrapper (storage faults bubble up) and `eval_results.json` parsing.

use thiserror::Error;

/// Everything that can go wrong in an off-path evaluation/diagnostics run.
#[derive(Debug, Error)]
pub enum EvalError {
    /// A read from the underlying [`gw_storage::Store`] failed (SQL fault, decode error, …).
    ///
    /// Only the async wrappers (e.g. [`crate::separation::analyze_store`]) can raise this; the
    /// pure cores never touch storage.
    #[error("storage read failed: {0}")]
    Storage(#[from] gw_storage::StorageError),

    /// A B2 `eval_results.json` artifact could not be parsed into [`crate::promote::EvalResults`].
    #[error("failed to parse eval_results.json: {0}")]
    EvalResultsParse(#[source] serde_json::Error),
}

/// The crate-wide result type: `Result<T, EvalError>`.
pub type Result<T> = std::result::Result<T, EvalError>;
