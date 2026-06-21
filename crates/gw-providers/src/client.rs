//! [`OpenRouterProvider`] — the concrete OpenAI-compatible streaming client.
//!
//! Wires the pieces together: a [`RateLimiter`] gate, a [`retry`] loop around the POST,
//! `reqwest`'s `bytes_stream()`, and the [`decode_sse`] decoder.
//! The API key is read from an **environment variable** (default `OPENROUTER_API_KEY`) and is
//! never logged. The base URL defaults to OpenRouter's `/api/v1` but is configurable for OMLX
//! / any OpenAI-compatible remote.

use std::sync::Arc;

use reqwest::Client as HttpClient;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue, RETRY_AFTER};

use crate::error::ProviderError;
use crate::limiter::RateLimiter;
use crate::request::ChatRequest;
use crate::retry::{RetryPolicy, retry};
use crate::sse::decode_sse;
use crate::{DeltaStream, Provider};

/// The default OpenRouter base URL.
pub const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";
/// The default environment variable holding the API key.
pub const DEFAULT_API_KEY_ENV: &str = "OPENROUTER_API_KEY";

/// Builder for an [`OpenRouterProvider`]: configure base URL, key env var, optional OpenRouter
/// attribution headers, and the retry policy before reading the key from the environment.
#[derive(Debug, Clone)]
pub struct OpenRouterProviderBuilder {
    base_url: String,
    api_key_env: String,
    referer: Option<String>,
    title: Option<String>,
    rpm: u32,
    policy: RetryPolicy,
}

impl Default for OpenRouterProviderBuilder {
    fn default() -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key_env: DEFAULT_API_KEY_ENV.to_string(),
            referer: None,
            title: None,
            rpm: 60,
            policy: RetryPolicy::default(),
        }
    }
}

impl OpenRouterProviderBuilder {
    /// Override the OpenAI-compatible base URL (e.g. an OMLX local endpoint). Trailing slashes
    /// are trimmed.
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into().trim_end_matches('/').to_string();
        self
    }

    /// Override the environment variable the API key is read from.
    #[must_use]
    pub fn api_key_env(mut self, var: impl Into<String>) -> Self {
        self.api_key_env = var.into();
        self
    }

    /// Set the OpenRouter `HTTP-Referer` attribution header.
    #[must_use]
    pub fn referer(mut self, referer: impl Into<String>) -> Self {
        self.referer = Some(referer.into());
        self
    }

    /// Set the OpenRouter `X-Title` attribution header.
    #[must_use]
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set the per-lane requests-per-minute budget for this provider's limiter.
    #[must_use]
    pub fn rpm(mut self, rpm: u32) -> Self {
        self.rpm = rpm;
        self
    }

    /// Override the retry policy.
    #[must_use]
    pub fn retry_policy(mut self, policy: RetryPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Read the API key from the configured environment variable and build the provider.
    ///
    /// # Errors
    /// - [`ProviderError::MissingApiKey`] if the env var is unset (the message names the var,
    ///   never a value).
    /// - [`ProviderError::Config`] if the key is not a valid HTTP header value, or the HTTP
    ///   client cannot be constructed.
    pub fn build(self) -> Result<OpenRouterProvider, ProviderError> {
        let key = std::env::var(&self.api_key_env)
            .map_err(|_| ProviderError::MissingApiKey(self.api_key_env.clone()))?;
        self.build_with_key(&key)
    }

    /// Build with an explicitly supplied key (bypassing the environment). Used in tests; the
    /// key is moved into a header value and never logged.
    ///
    /// # Errors
    /// [`ProviderError::Config`] if the key / headers are invalid or the client fails to build.
    pub fn build_with_key(self, key: &str) -> Result<OpenRouterProvider, ProviderError> {
        let mut headers = HeaderMap::new();
        let mut auth = HeaderValue::from_str(&format!("Bearer {key}"))
            .map_err(|_| ProviderError::Config("invalid API key (not a valid header)".into()))?;
        auth.set_sensitive(true);
        headers.insert(AUTHORIZATION, auth);
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Some(r) = &self.referer {
            let v = HeaderValue::from_str(r)
                .map_err(|_| ProviderError::Config("invalid HTTP-Referer header".into()))?;
            headers.insert("HTTP-Referer", v);
        }
        if let Some(t) = &self.title {
            let v = HeaderValue::from_str(t)
                .map_err(|_| ProviderError::Config("invalid X-Title header".into()))?;
            headers.insert("X-Title", v);
        }

        let http = HttpClient::builder()
            .default_headers(headers)
            .build()
            .map_err(|e| ProviderError::Config(format!("http client build failed: {e}")))?;

        Ok(OpenRouterProvider {
            http,
            base_url: self.base_url,
            limiter: Arc::new(RateLimiter::per_minute(self.rpm)),
            policy: self.policy,
        })
    }
}

/// An OpenAI-compatible streaming provider with full chain-of-thought capture.
///
/// `stream_chat` gates on the rate limiter, retries the POST on transient faults, and returns a
/// stream of [`StreamDelta`](crate::StreamDelta)s decoded from the SSE response.
#[derive(Clone)]
pub struct OpenRouterProvider {
    http: HttpClient,
    base_url: String,
    limiter: Arc<RateLimiter>,
    policy: RetryPolicy,
}

/// Redacted `Debug` — never prints the HTTP client (which holds the `Authorization` header).
impl std::fmt::Debug for OpenRouterProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenRouterProvider")
            .field("base_url", &self.base_url)
            .field("rpm", &self.limiter.rpm())
            .field("policy", &self.policy)
            .finish_non_exhaustive()
    }
}

impl OpenRouterProvider {
    /// Start configuring a provider.
    #[must_use]
    pub fn builder() -> OpenRouterProviderBuilder {
        OpenRouterProviderBuilder::default()
    }

    /// Convenience: build the default OpenRouter provider, reading `OPENROUTER_API_KEY`.
    ///
    /// # Errors
    /// See [`OpenRouterProviderBuilder::build`].
    pub fn from_env() -> Result<Self, ProviderError> {
        Self::builder().build()
    }

    /// The configured chat-completions URL.
    fn completions_url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    /// Issue the POST and return the raw response, classifying non-2xx statuses (and parsing a
    /// `Retry-After` on 429) into [`ProviderError`]. Retried by the caller.
    async fn send_once(&self, req: &ChatRequest) -> Result<reqwest::Response, ProviderError> {
        self.limiter.until_ready().await;
        let resp = self
            .http
            .post(self.completions_url())
            .json(req)
            .send()
            .await
            .map_err(ProviderError::from)?;

        let status = resp.status();
        if status.is_success() {
            return Ok(resp);
        }
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let retry_after = parse_retry_after(resp.headers());
            return Err(ProviderError::RateLimited { retry_after });
        }
        let code = status.as_u16();
        let body = resp.text().await.ok().map(|b| truncate(&b, 512));
        Err(ProviderError::from_status(code, body))
    }
}

impl Provider for OpenRouterProvider {
    async fn stream_chat(&self, req: ChatRequest) -> Result<DeltaStream, ProviderError> {
        // Retry the *connection* (governed + backed off); once a 2xx response is in hand, the
        // stream itself is decoded. A mid-stream reset surfaces as a terminal item to the
        // consumer (the engine decides whether to re-dispatch the whole job).
        let response = retry(self.policy, || self.send_once(&req)).await?;
        let byte_stream = response.bytes_stream();
        let decoded = decode_sse(byte_stream);
        Ok(Box::pin(decoded))
    }
}

/// Parse a `Retry-After` header (RFC 7231 delta-seconds form) into a [`std::time::Duration`].
/// HTTP-date form is not parsed (rare for 429s); returns `None` when absent/unparseable.
fn parse_retry_after(headers: &HeaderMap) -> Option<std::time::Duration> {
    let raw = headers.get(RETRY_AFTER)?.to_str().ok()?;
    let secs: u64 = raw.trim().parse().ok()?;
    Some(std::time::Duration::from_secs(secs))
}

/// Truncate a body snippet to `max` bytes on a char boundary for safe diagnostics.
fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_env_var_is_clean_error() {
        // A var name extremely unlikely to be set in CI.
        let res = OpenRouterProvider::builder()
            .api_key_env("GW_PROVIDERS_DEFINITELY_UNSET_KEY_XYZ")
            .build();
        match res {
            Err(ProviderError::MissingApiKey(var)) => {
                assert_eq!(var, "GW_PROVIDERS_DEFINITELY_UNSET_KEY_XYZ");
            }
            other => panic!("expected MissingApiKey, got {other:?}"),
        }
    }

    #[test]
    fn build_with_key_succeeds_and_sets_base_url() {
        let p = OpenRouterProvider::builder()
            .base_url("https://example.test/api/v1/")
            .referer("https://ghostwriter.example")
            .title("ghostwriter-rs")
            .build_with_key("sk-test-not-a-real-key")
            .expect("builds");
        // Trailing slash trimmed; URL composed correctly.
        assert_eq!(
            p.completions_url(),
            "https://example.test/api/v1/chat/completions"
        );
    }

    #[test]
    fn invalid_key_is_config_error_not_panic() {
        // A newline in a header value is rejected by reqwest's HeaderValue.
        let res = OpenRouterProvider::builder().build_with_key("bad\nkey");
        assert!(matches!(res, Err(ProviderError::Config(_))));
    }

    #[test]
    fn parse_retry_after_seconds() {
        let mut h = HeaderMap::new();
        h.insert(RETRY_AFTER, HeaderValue::from_static("12"));
        assert_eq!(
            parse_retry_after(&h),
            Some(std::time::Duration::from_secs(12))
        );

        let mut bad = HeaderMap::new();
        bad.insert(
            RETRY_AFTER,
            HeaderValue::from_static("Wed, 21 Oct 2026 07:28:00 GMT"),
        );
        assert_eq!(parse_retry_after(&bad), None);

        assert_eq!(parse_retry_after(&HeaderMap::new()), None);
    }

    #[test]
    fn truncate_respects_char_boundaries() {
        let s = "héllo wörld this is long enough to clip";
        let t = truncate(s, 5);
        assert!(t.ends_with('…'));
        // Did not panic on the multibyte boundary.
        assert!(t.len() <= s.len() + 3);
    }

    #[test]
    fn default_constants_are_openrouter() {
        assert_eq!(DEFAULT_BASE_URL, "https://openrouter.ai/api/v1");
        assert_eq!(DEFAULT_API_KEY_ENV, "OPENROUTER_API_KEY");
    }
}
