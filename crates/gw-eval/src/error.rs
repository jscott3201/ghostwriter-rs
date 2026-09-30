//! The single typed error for `gw-eval` ([`EvalError`]) and the crate [`Result`] alias.
//!
//! Library code never uses `anyhow`: every fallible path returns a typed [`EvalError`] so a
//! caller (the `eval audit-separation` / `eval promote` CLI seams) can match on the cause. The
//! pure analysis cores ([`crate::separation::analyze`], [`crate::promote::promote`]) return
//! invalid evidence in their reports. Fallible entry points cover store reads,
//! canonical candidate binding, and evidence JSON parsing.

use thiserror::Error;

/// Everything that can go wrong in an off-path evaluation/diagnostics run.
#[derive(Debug, Error)]
pub enum EvalError {
    /// A store read or canonical record hash failed (SQL fault, decode/serialization error, …).
    ///
    /// Raised by store wrappers and candidate binding. The pure analyses perform no I/O.
    #[error("record storage or hashing failed: {0}")]
    Storage(#[from] gw_storage::StorageError),

    /// A B2 `eval_results.json` artifact could not be parsed into [`crate::promote::EvalResults`].
    #[error("failed to parse eval_results.json: {0}")]
    EvalResultsParse(#[source] serde_json::Error),

    /// An independent outcome envelope could not be decoded.
    #[error("failed to parse independent outcome evidence: {0}")]
    OutcomeEvidenceParse(#[source] serde_json::Error),
}

/// The crate-wide result type: `Result<T, EvalError>`.
pub type Result<T> = std::result::Result<T, EvalError>;
