//! [`GenerateError`] — the single typed error surfaced by every `gw-generate` operation.
//!
//! Variants distinguish the failure classes a caller (`gw-engine`) must reason about: a
//! request-builder invariant that the caller tried to violate ([`Invariant`](GenerateError::Invariant),
//! a programmer/config error, never retryable), a teacher / user-synth model call that failed
//! ([`Provider`](GenerateError::Provider), which carries the underlying [`ProviderError`] and so
//! preserves its retryability classification), the round-trip ingest of a streamed response
//! ([`Format`](GenerateError::Format)), a teacher response that produced no usable assistant turn
//! ([`EmptyResponse`](GenerateError::EmptyResponse)), a chain-of-thought truncated mid-channel
//! ([`TruncatedReasoning`](GenerateError::TruncatedReasoning), the `<|channel>thought` hazard), and
//! an embedder fault surfaced through the injected diversity seam
//! ([`Embed`](GenerateError::Embed)). No `anyhow` — this crate surfaces a typed error like the
//! sibling crates.

use thiserror::Error;

use gw_format::FormatError;
use gw_providers::ProviderError;

/// Everything that can go wrong synthesizing a user turn or generating an assistant turn.
///
/// `#[non_exhaustive]` so new variants can be added without a breaking change. `Provider` and
/// `Format` carry their underlying source via `#[from]`; the rest are constructed directly with a
/// human-readable message. No secret (API key) is ever placed in a variant — `Provider` inherits
/// the providers crate's safe-to-log stance.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GenerateError {
    /// A request-builder INVARIANT was violated by the caller — e.g. building a teacher request
    /// with no `max_tokens`, or asking for BOTH a reasoning `effort` and a `reasoning_max_tokens`
    /// budget (they are mutually exclusive on the wire). A programmer/config fault: terminal, and
    /// surfaced loud at the data-entry seam rather than silently dropped (INVARIANT-g).
    #[error("generation invariant violated: {0}")]
    Invariant(String),

    /// A teacher or user-synth model call failed. Carries the [`ProviderError`] verbatim so the
    /// engine can consult [`ProviderError::is_retryable`] and [`ProviderError::retry_after`].
    #[error("provider error: {0}")]
    Provider(#[from] ProviderError),

    /// Converting the streamed provider response into a clean [`gw_schema::Message`] via the
    /// `gw-format` ingest path failed (a control-token leak the stripper could not model, or a
    /// malformed payload). Terminal — a malformed body will not fix itself on retry.
    #[error("format/ingest error: {0}")]
    Format(#[from] FormatError),

    /// The teacher stream completed but yielded no usable assistant turn (no content AND no
    /// reasoning at all). Distinct from a transport reset; signals an empty/degenerate completion.
    #[error("teacher produced an empty completion: {0}")]
    EmptyResponse(String),

    /// The chain-of-thought was truncated mid-stream: the terminal chunk's `finish_reason` was
    /// `"length"` while reasoning was still being emitted, so the `<|channel>thought` block is
    /// unterminated (ARCHITECTURE §3.2, the truncation hazard). A record carrying a truncated CoT
    /// must NOT be admitted; the producer fails loud so the engine can retry with a larger
    /// `max_tokens` or route the record to `revising`.
    #[error("reasoning truncated (finish_reason=length): {0}")]
    TruncatedReasoning(String),

    /// The injected [`Embedder`](crate::Embedder) used by the `diverse` dedup check failed. Carries
    /// a human-readable message from the embedder backend.
    #[error("embedder error: {0}")]
    Embed(String),

    /// A synthesized USER turn carried a raw chat control token (`<|turn>`, `<think>`, …) — an
    /// upstream elicitation/template stage leaked channel markup. The QC gate fails loud rather than
    /// embed it (defeating dedup) or admit it (a later render would double-frame the target). Carries
    /// the offending token.
    #[error("synthesized user turn contains leaked control token `{0}`")]
    LeakedUserTurn(&'static str),
}

/// Convenience alias for results returned by `gw-generate` operations.
pub type Result<T> = std::result::Result<T, GenerateError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invariant_renders_message() {
        let e = GenerateError::Invariant("max_tokens must be set".into());
        assert!(e.to_string().contains("max_tokens must be set"));
    }

    #[test]
    fn provider_error_converts_and_preserves_retryability() {
        let pe = ProviderError::RateLimited { retry_after: None };
        let e: GenerateError = pe.into();
        // The underlying ProviderError is reachable for the engine's retry decision.
        match e {
            GenerateError::Provider(inner) => assert!(inner.is_retryable()),
            other => panic!("expected Provider, got {other:?}"),
        }
    }

    #[test]
    fn format_error_converts() {
        let fe = FormatError::Ingest("no message".into());
        let e: GenerateError = fe.into();
        assert!(matches!(e, GenerateError::Format(_)));
        assert!(e.to_string().contains("no message"));
    }

    #[test]
    fn truncated_reasoning_names_the_hazard() {
        let e = GenerateError::TruncatedReasoning("3200 reasoning tokens then length".into());
        let msg = e.to_string();
        assert!(msg.contains("finish_reason=length"));
        assert!(msg.contains("3200 reasoning tokens"));
    }
}
