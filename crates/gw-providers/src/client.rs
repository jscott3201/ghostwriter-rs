//! [`ChatCompletionsProvider`] — the concrete OpenAI-compatible streaming client.
//!
//! Wires the pieces together: a [`RateLimiter`] gate, a [`retry`] loop around the POST,
//! `reqwest`'s `bytes_stream()`, and the [`decode_sse`] decoder.
//! The API key is read from an **environment variable** (default `MODEL_API_KEY`) and is
//! never logged. The base URL defaults to OpenRouter's `/api/v1` but is configurable for OMLX
//! / any OpenAI-compatible remote.

use std::sync::Arc;

use reqwest::Client as HttpClient;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue, RETRY_AFTER};

use crate::CallObservation;
use crate::error::ProviderError;
use crate::limiter::RateLimiter;
use crate::observation::ActiveAttempt;
use crate::request::ChatRequest;
use crate::retry::{RetryPolicy, retry};
use crate::sse::decode_sse;
use crate::{DeltaStream, Provider, StreamChatFuture};
use gw_schema::TransportOutcome;

/// The default OpenRouter base URL.
pub const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";
/// The default environment variable holding the API key.
pub const DEFAULT_API_KEY_ENV: &str = "MODEL_API_KEY";

/// Builder for a [`ChatCompletionsProvider`]: configure base URL, key env var, optional OpenRouter
/// attribution headers, and the retry policy before reading the key from the environment.
#[derive(Clone)]
pub struct ChatCompletionsProviderBuilder {
    base_url: String,
    api_key_env: String,
    referer: Option<String>,
    title: Option<String>,
    rpm: u32,
    policy: RetryPolicy,
    #[cfg(test)]
    http2_prior_knowledge: bool,
}

impl std::fmt::Debug for ChatCompletionsProviderBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let reference = if validate_api_key_env(&self.api_key_env).is_ok() {
            self.api_key_env.as_str()
        } else {
            "[invalid]"
        };
        f.debug_struct("ChatCompletionsProviderBuilder")
            .field("base_url", &self.base_url)
            .field("api_key_env", &reference)
            .field("referer", &self.referer)
            .field("title", &self.title)
            .field("rpm", &self.rpm)
            .field("policy", &self.policy)
            .finish_non_exhaustive()
    }
}

/// Validate an API key environment variable name without accessing its value.
///
/// Names use portable ASCII identifiers: a letter or underscore followed by letters, digits,
/// or underscores. Invalid supplied names are omitted from lookup errors and debug output.
///
/// # Errors
/// Returns a static configuration error for an invalid name, without echoing the supplied input.
pub fn validate_api_key_env(name: &str) -> Result<(), ProviderError> {
    let mut bytes = name.bytes();
    if !bytes
        .next()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_')
        || !bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return Err(ProviderError::Config(
            "API key environment variable must be a name starting with an ASCII letter or underscore and containing only ASCII letters, digits, or underscores".into(),
        ));
    }
    Ok(())
}

impl Default for ChatCompletionsProviderBuilder {
    fn default() -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key_env: DEFAULT_API_KEY_ENV.to_string(),
            referer: None,
            title: None,
            rpm: 60,
            policy: RetryPolicy::default(),
            #[cfg(test)]
            http2_prior_knowledge: false,
        }
    }
}

impl ChatCompletionsProviderBuilder {
    #[cfg(test)]
    pub(crate) fn http2_for_test(mut self) -> Self {
        self.http2_prior_knowledge = true;
        self
    }

    /// Override the OpenAI-compatible base URL (e.g. an OMLX local endpoint). Trailing slashes
    /// are trimmed.
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into().trim_end_matches('/').to_string();
        self
    }

    /// Override the environment variable the API key is read from. Its name is validated by
    /// [`Self::semantic_declaration`], [`Self::build`], and [`Self::build_with_key`].
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
    pub fn build(self) -> Result<ChatCompletionsProvider, ProviderError> {
        self.semantic_declaration()?;
        let key = std::env::var(&self.api_key_env)
            .map_err(|_| ProviderError::MissingApiKey(self.api_key_env.clone()))?;
        self.build_with_key(&key)
    }

    /// Build with an explicitly supplied key (bypassing the environment). Used in tests; the
    /// key is moved into a header value and never logged.
    ///
    /// # Errors
    /// [`ProviderError::Config`] if the key / headers are invalid or the client fails to build.
    pub fn build_with_key(self, key: &str) -> Result<ChatCompletionsProvider, ProviderError> {
        let declaration = self.semantic_declaration()?;
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
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never());
        #[cfg(test)]
        let http = if self.http2_prior_knowledge {
            http.http2_prior_knowledge()
        } else {
            http
        };
        let http = http
            .build()
            .map_err(|e| ProviderError::Config(format!("http client build failed: {e}")))?;

        Ok(ChatCompletionsProvider {
            http,
            base_url: crate::normalize_endpoint(&self.base_url)?,
            declaration,
            limiter: Arc::new(RateLimiter::per_minute(self.rpm)),
            policy: self.policy,
        })
    }

    /// Prepare the same immutable adapter declaration the built client exposes, without reading
    /// credentials or constructing an HTTP client.
    ///
    /// # Errors
    /// Rejects unsupported or credential-bearing endpoint forms and invalid API key environment
    /// variable names without echoing the input.
    pub fn semantic_declaration(&self) -> Result<gw_schema::SemanticDeclaration, ProviderError> {
        validate_api_key_env(&self.api_key_env)?;
        crate::identity::chat(&self.base_url, self.policy.max_attempts)
    }
}

/// An OpenAI-compatible streaming provider with full chain-of-thought capture.
///
/// `stream_chat` gates on the rate limiter, retries the POST on transient faults, and returns a
/// stream of [`StreamDelta`](crate::StreamDelta)s decoded from the SSE response.
#[derive(Clone)]
pub struct ChatCompletionsProvider {
    http: HttpClient,
    base_url: String,
    limiter: Arc<RateLimiter>,
    policy: RetryPolicy,
    declaration: gw_schema::SemanticDeclaration,
}

/// Redacted `Debug` — never prints the HTTP client (which holds the `Authorization` header).
impl std::fmt::Debug for ChatCompletionsProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatCompletionsProvider")
            .field("base_url", &self.base_url)
            .field("rpm", &self.limiter.rpm())
            .field("policy", &self.policy)
            .finish_non_exhaustive()
    }
}

impl ChatCompletionsProvider {
    /// Start configuring a provider.
    #[must_use]
    pub fn builder() -> ChatCompletionsProviderBuilder {
        ChatCompletionsProviderBuilder::default()
    }

    /// Convenience: build the default OpenRouter provider, reading `MODEL_API_KEY`.
    ///
    /// # Errors
    /// See [`ChatCompletionsProviderBuilder::build`].
    pub fn from_env() -> Result<Self, ProviderError> {
        Self::builder().build()
    }

    /// The configured chat-completions URL.
    fn completions_url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    /// Issue the POST and return the raw response, classifying non-2xx statuses (and parsing a
    /// `Retry-After` on 429) into [`ProviderError`]. Retried by the caller.
    async fn send_once(
        &self,
        req: &ChatRequest,
        body: &[u8],
        observation: Option<&CallObservation>,
        ordinal: u32,
    ) -> Result<(reqwest::Response, Option<ActiveAttempt>), ProviderError> {
        self.limiter.until_ready().await;
        let endpoint = self.completions_url();
        let mut attempt = match observation {
            Some(call) => Some(call.begin(body, &req.model, &endpoint, ordinal).await?),
            None => None,
        };
        let response = self.http.post(&endpoint).body(body.to_vec()).send().await;
        let response = match response {
            Ok(response) => response,
            Err(error) => {
                let error = ProviderError::from(error);
                if let Some(attempt) = attempt {
                    attempt
                        .settle(TransportOutcome::Failed, None, Some(error.to_string()))
                        .await?;
                }
                return Err(error);
            }
        };
        let status = response.status();
        if status.is_success() {
            return Ok((response, attempt));
        }
        let retry_after = parse_retry_after(response.headers());
        let bytes = response.bytes().await;
        let error = match bytes {
            Ok(bytes) => {
                if let Some(attempt) = &mut attempt
                    && let Some(metadata) = crate::metadata::extract_json(&bytes)
                {
                    attempt.metadata(metadata).await.map_err(|error| {
                        error.with_primary(format!("http status {}", status.as_u16()))
                    })?;
                }
                if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
                    ProviderError::RateLimited { retry_after }
                } else {
                    ProviderError::from_status(
                        status.as_u16(),
                        Some(truncate(&String::from_utf8_lossy(&bytes), 512)),
                    )
                }
            }
            Err(error) => ProviderError::from(error),
        };
        if let Some(attempt) = attempt {
            attempt
                .settle(
                    TransportOutcome::HttpError,
                    Some(status.as_u16()),
                    Some(error.to_string()),
                )
                .await?;
        }
        Err(error)
    }

    async fn stream_chat_impl(
        &self,
        req: ChatRequest,
        observation: Option<CallObservation>,
    ) -> Result<DeltaStream, ProviderError> {
        let body = serde_json::to_vec(&req).map_err(|e| ProviderError::Config(e.to_string()))?;
        let mut ordinal = 0;
        let (response, attempt) = retry(self.policy, || {
            let index = ordinal;
            ordinal += 1;
            self.send_once(&req, &body, observation.as_ref(), index)
        })
        .await?;
        match attempt {
            Some(attempt) => Ok(crate::observed_sse::stream(response, attempt)),
            None => Ok(Box::pin(decode_sse(response.bytes_stream()))),
        }
    }
}

impl Provider for ChatCompletionsProvider {
    fn semantic_declaration(&self) -> Option<gw_schema::SemanticDeclaration> {
        Some(self.declaration.clone())
    }
    fn accounting_capability(&self) -> gw_schema::AccountingCapability {
        gw_schema::AccountingCapability::PhysicalAttemptsV1
    }
    fn stream_chat_observed(
        &self,
        req: ChatRequest,
        observation: CallObservation,
    ) -> StreamChatFuture<'_> {
        Box::pin(self.stream_chat_impl(req, Some(observation)))
    }

    fn stream_chat(&self, req: ChatRequest) -> StreamChatFuture<'_> {
        Box::pin(self.stream_chat_impl(req, None))
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
    fn provider_is_dyn_compatible() {
        // Coercing to `&dyn Provider` / `Box<dyn Provider>` only compiles if the trait is
        // object-safe — the runtime witness for the compile-time assertion in lib.rs.
        let p = ChatCompletionsProvider::builder()
            .build_with_key("sk-test-not-a-real-key")
            .expect("builds");
        let boxed: Box<dyn Provider> = Box::new(p.clone());
        let dynref: &dyn Provider = &p;
        // Use both bindings so the coercions are load-bearing (and Debug stays redacted).
        assert!(format!("{p:?}").contains("ChatCompletionsProvider"));
        let _ = (boxed, dynref);
    }

    #[test]
    fn missing_env_var_is_clean_error() {
        // A var name extremely unlikely to be set in CI.
        let res = ChatCompletionsProvider::builder()
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
        let p = ChatCompletionsProvider::builder()
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
        let res = ChatCompletionsProvider::builder().build_with_key("bad\nkey");
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
    fn default_endpoint_and_key_reference() {
        assert_eq!(DEFAULT_BASE_URL, "https://openrouter.ai/api/v1");
        assert_eq!(DEFAULT_API_KEY_ENV, "MODEL_API_KEY");
    }
}
