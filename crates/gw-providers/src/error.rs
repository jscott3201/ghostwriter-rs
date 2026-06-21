//! [`ProviderError`] — the single error type surfaced by every provider operation.
//!
//! Variants distinguish the failure classes the retry layer ([`mod@crate::retry`]) must reason
//! about: transient transport faults and HTTP 429/5xx (retryable) versus configuration and
//! decode faults (terminal). A parsed `Retry-After` rides on the rate-limit variant so the
//! backoff helper can honor the server's hint. No secret (API key) is ever placed in a
//! variant — error rendering is safe to log.

use std::time::Duration;

use thiserror::Error;

/// Everything that can go wrong talking to an OpenAI-compatible endpoint.
///
/// The [`ProviderError::is_retryable`] predicate is the contract the retry helper relies on:
/// transport faults, HTTP 429, HTTP 5xx, and mid-stream resets are retryable; config, decode,
/// and 4xx (other than 429) are terminal.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ProviderError {
    /// A required secret was absent from the environment. Carries the **env var name**, never
    /// a value, so the message is safe to log.
    #[error("missing API key: environment variable `{0}` is not set")]
    MissingApiKey(String),

    /// The configured base URL / request could not be built (e.g. invalid header value).
    #[error("provider configuration error: {0}")]
    Config(String),

    /// A transport-level failure (connect, DNS, TLS, read timeout) before/around the response.
    /// Retryable.
    #[error("transport error: {0}")]
    Transport(String),

    /// A non-2xx HTTP status that is **not** a 429. `retryable` is set for 5xx. The optional
    /// `body` is a truncated snippet of the error payload for diagnostics.
    #[error("http status {status}{}", body.as_deref().map(|b| format!(": {b}")).unwrap_or_default())]
    Status {
        /// The HTTP status code.
        status: u16,
        /// Whether this status is in the retryable class (5xx).
        retryable: bool,
        /// A truncated snippet of the response body, if one was read.
        body: Option<String>,
    },

    /// HTTP 429 Too Many Requests. Carries the parsed `Retry-After` if the header was present
    /// and understood. Always retryable.
    #[error("rate limited (429){}", retry_after.map(|d| format!(", retry after {}s", d.as_secs())).unwrap_or_default())]
    RateLimited {
        /// Parsed `Retry-After` hint, if any.
        retry_after: Option<Duration>,
    },

    /// The SSE byte stream ended or reset mid-response (no `[DONE]` sentinel observed, or a
    /// transport read error during streaming). Retryable.
    #[error("stream reset before completion: {0}")]
    StreamReset(String),

    /// A `data:` chunk failed to parse as a [`crate::StreamDelta`], or another JSON decode
    /// failed. Terminal — a malformed payload will not fix itself on retry.
    #[error("decode error: {0}")]
    Decode(String),
}

impl ProviderError {
    /// `true` for the classes the retry helper should retry: transport faults, HTTP 429,
    /// HTTP 5xx, and mid-stream resets. Config / decode / non-429 4xx are terminal.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        match self {
            ProviderError::Transport(_)
            | ProviderError::RateLimited { .. }
            | ProviderError::StreamReset(_) => true,
            ProviderError::Status { retryable, .. } => *retryable,
            ProviderError::MissingApiKey(_)
            | ProviderError::Config(_)
            | ProviderError::Decode(_) => false,
        }
    }

    /// The server-supplied retry delay, if this is a rate-limit error that carried one.
    #[must_use]
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            ProviderError::RateLimited { retry_after } => *retry_after,
            _ => None,
        }
    }

    /// Build a [`ProviderError`] from an HTTP status:
    /// - **429** maps to [`ProviderError::RateLimited`] (no `Retry-After` context here; the
    ///   client path parses the header and constructs that variant directly when available).
    /// - **408** (Request Timeout) and **5xx** are [`ProviderError::Status`] flagged retryable.
    /// - everything else is a terminal [`ProviderError::Status`].
    #[must_use]
    pub fn from_status(status: u16, body: Option<String>) -> Self {
        if status == 429 {
            return ProviderError::RateLimited { retry_after: None };
        }
        let retryable = status == 408 || (500..600).contains(&status);
        ProviderError::Status {
            status,
            retryable,
            body,
        }
    }
}

/// Map a `reqwest::Error` to a [`ProviderError`], preserving status when present.
impl From<reqwest::Error> for ProviderError {
    fn from(err: reqwest::Error) -> Self {
        if let Some(status) = err.status() {
            return ProviderError::from_status(status.as_u16(), Some(err.to_string()));
        }
        ProviderError::Transport(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryable_classification() {
        assert!(ProviderError::Transport("eof".into()).is_retryable());
        assert!(ProviderError::RateLimited { retry_after: None }.is_retryable());
        assert!(ProviderError::StreamReset("x".into()).is_retryable());
        assert!(ProviderError::from_status(503, None).is_retryable());
        assert!(ProviderError::from_status(500, None).is_retryable());
        // 408 Request Timeout is retryable.
        assert!(ProviderError::from_status(408, None).is_retryable());

        assert!(!ProviderError::from_status(400, None).is_retryable());
        assert!(!ProviderError::from_status(404, None).is_retryable());
        assert!(!ProviderError::MissingApiKey("OPENROUTER_API_KEY".into()).is_retryable());
        assert!(!ProviderError::Decode("bad json".into()).is_retryable());
        assert!(!ProviderError::Config("bad url".into()).is_retryable());
    }

    #[test]
    fn status_429_maps_to_rate_limited() {
        // 429 is folded into the RateLimited variant (retryable), not a bare Status.
        let e = ProviderError::from_status(429, Some("too many".into()));
        assert!(matches!(
            e,
            ProviderError::RateLimited { retry_after: None }
        ));
        assert!(e.is_retryable());
    }

    #[test]
    fn retry_after_threads_through() {
        let e = ProviderError::RateLimited {
            retry_after: Some(Duration::from_secs(7)),
        };
        assert_eq!(e.retry_after(), Some(Duration::from_secs(7)));
        assert_eq!(ProviderError::Transport("x".into()).retry_after(), None);
    }

    #[test]
    fn missing_key_message_names_var_not_value() {
        let e = ProviderError::MissingApiKey("OPENROUTER_API_KEY".into());
        let msg = e.to_string();
        assert!(msg.contains("OPENROUTER_API_KEY"));
    }
}
