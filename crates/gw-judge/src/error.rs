//! [`JudgeError`] — the single typed error surfaced by every `gw-judge` operation.
//!
//! Variants distinguish the failure classes a caller (`gw-engine`) must reason about: a judge
//! model call that failed ([`Provider`](JudgeError::Provider), which carries the underlying
//! [`ProviderError`] and so preserves its retryability classification), a storage/cache fault
//! ([`Storage`](JudgeError::Storage)), a malformed judge response that could not be parsed into a
//! grade ([`JudgeParse`](JudgeError::JudgeParse)), an empty panel handed to the consensus math
//! ([`EmptyPanel`](JudgeError::EmptyPanel)), and a grading invariant the caller tried to violate
//! ([`Invariant`](JudgeError::Invariant), a programmer/config error, never retryable). No
//! `anyhow` — this crate surfaces a typed error like the sibling crates.

use thiserror::Error;

use gw_providers::ProviderError;
use gw_storage::StorageError;

/// Everything that can go wrong verifying a trace or grading it with the judge panel.
///
/// `#[non_exhaustive]` so new variants can be added without a breaking change. `Provider` and
/// `Storage` carry their underlying source via `#[from]`; the rest are constructed directly with a
/// human-readable message. No secret (API key) is ever placed in a variant — `Provider` inherits
/// the providers crate's safe-to-log stance.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum JudgeError {
    /// A judge-panel model call failed. Carries the [`ProviderError`] verbatim so the engine can
    /// consult [`ProviderError::is_retryable`] and [`ProviderError::retry_after`].
    #[error("provider error: {0}")]
    Provider(#[from] ProviderError),

    /// A storage / call-cache operation failed (a SQL fault, or a cached value that would not
    /// deserialize). Carries the [`StorageError`] verbatim.
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),

    /// A judge stream completed but its response could not be parsed into a [`Grade`](crate::Grade)
    /// (no score token, malformed JSON verdict). Terminal — a malformed body will not fix itself on
    /// retry, but the panel can still reach consensus from the remaining judges.
    #[error("could not parse judge response into a grade: {0}")]
    JudgeParse(String),

    /// The consensus math was handed an empty panel (zero grades). The caller must never run
    /// aggregation on no votes — a `0/0` design effect is undefined, and an empty panel means the
    /// upstream routing produced no judges. Surfaced loud rather than returning a silent `NaN`.
    #[error("consensus called on an empty panel: {0}")]
    EmptyPanel(String),

    /// A grading INVARIANT was violated by the caller — e.g. a calibration-weight vector whose
    /// length does not match the panel, or a non-finite weight. A programmer/config fault: terminal,
    /// surfaced loud at the seam rather than silently producing a wrong consensus.
    #[error("judge invariant violated: {0}")]
    Invariant(String),
}

/// Convenience alias for results returned by `gw-judge` operations.
pub type Result<T> = std::result::Result<T, JudgeError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_error_converts_and_preserves_retryability() {
        let pe = ProviderError::RateLimited { retry_after: None };
        let e: JudgeError = pe.into();
        match e {
            JudgeError::Provider(inner) => assert!(inner.is_retryable()),
            other => panic!("expected Provider, got {other:?}"),
        }
    }

    #[test]
    fn empty_panel_names_the_hazard() {
        let e = JudgeError::EmptyPanel("no judges routed for area math_cot".into());
        assert!(e.to_string().contains("empty panel"));
    }

    #[test]
    fn invariant_renders_message() {
        let e = JudgeError::Invariant("weights.len() != panel.len()".into());
        assert!(e.to_string().contains("weights.len()"));
    }
}
