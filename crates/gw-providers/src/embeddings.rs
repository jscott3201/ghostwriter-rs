//! OpenAI-compatible embeddings client.

use std::time::Duration;

use reqwest::{
    Client, StatusCode,
    header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue},
};
use serde::{Deserialize, Serialize};

use crate::ProviderError;

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Builder for [`EmbeddingsClient`].
#[derive(Debug, Clone)]
pub struct EmbeddingsClientBuilder {
    base_url: String,
    model: String,
    dim: usize,
    api_key_env: Option<String>,
    timeout: Duration,
}

impl EmbeddingsClientBuilder {
    /// Configure an OpenAI-compatible `/v1` base URL.
    #[must_use]
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Configure the model identifier.
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Configure the required output dimension.
    #[must_use]
    pub fn dim(mut self, dim: u32) -> Self {
        self.dim = dim as usize;
        self
    }

    /// Read an optional bearer key from this environment variable at build time.
    #[must_use]
    pub fn api_key_env(mut self, api_key_env: Option<String>) -> Self {
        self.api_key_env = api_key_env;
        self
    }

    /// Configure the request timeout.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Build, reading the configured optional key environment variable.
    ///
    /// # Errors
    /// Returns a configuration error or names a configured key variable that is absent.
    pub fn build(self) -> Result<EmbeddingsClient, ProviderError> {
        let key = self
            .api_key_env
            .as_ref()
            .map(|name| std::env::var(name).map_err(|_| ProviderError::MissingApiKey(name.clone())))
            .transpose()?;
        self.build_with_key(key.as_deref())
    }

    /// Build with an explicit optional key, bypassing environment lookup for tests.
    ///
    /// # Errors
    /// Returns [`ProviderError::Config`] for invalid headers or client configuration.
    pub fn build_with_key(self, key: Option<&str>) -> Result<EmbeddingsClient, ProviderError> {
        let headers = embedding_headers(key)?;
        let http = Client::builder()
            .default_headers(headers)
            .timeout(self.timeout)
            .build()
            .map_err(|e| ProviderError::Config(format!("http client build failed: {e}")))?;
        Ok(EmbeddingsClient {
            http,
            base_url: self.base_url.trim_end_matches('/').to_string(),
            model: self.model,
            dim: self.dim,
        })
    }
}

/// Construct embedding request headers with an optional sensitive bearer key.
pub fn embedding_headers(key: Option<&str>) -> Result<HeaderMap, ProviderError> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    if let Some(key) = key {
        let mut auth = HeaderValue::from_str(&format!("Bearer {key}"))
            .map_err(|_| ProviderError::Config("invalid API key (not a valid header)".into()))?;
        auth.set_sensitive(true);
        headers.insert(AUTHORIZATION, auth);
    }
    Ok(headers)
}

/// Async OpenAI-compatible embeddings client.
#[derive(Clone)]
pub struct EmbeddingsClient {
    http: Client,
    base_url: String,
    model: String,
    dim: usize,
}

impl std::fmt::Debug for EmbeddingsClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingsClient")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("dim", &self.dim)
            .finish_non_exhaustive()
    }
}

impl EmbeddingsClient {
    /// Start configuring an embeddings client.
    #[must_use]
    pub fn builder() -> EmbeddingsClientBuilder {
        EmbeddingsClientBuilder {
            base_url: gw_schema::DEFAULT_EMBEDDING_ENDPOINT.to_string(),
            model: gw_schema::DEFAULT_EMBEDDING_MODEL.to_string(),
            dim: gw_schema::DEFAULT_EMBEDDING_DIM as usize,
            api_key_env: None,
            timeout: DEFAULT_TIMEOUT,
        }
    }

    /// Embed a batch, returning vectors ordered by response `index`.
    ///
    /// This v1 client intentionally performs no retries.
    ///
    /// # Errors
    /// Returns transport, HTTP, decode, duplicate/missing-index, or dimension errors.
    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, ProviderError> {
        let response = self
            .http
            .post(format!("{}/embeddings", self.base_url))
            .json(&EmbeddingRequest {
                model: &self.model,
                input: texts,
            })
            .send()
            .await
            .map_err(ProviderError::from)?;
        let status = response.status();
        let bytes = response.bytes().await.map_err(ProviderError::from)?;
        if !status.is_success() {
            return Err(status_error(status, &bytes));
        }
        let decoded: EmbeddingResponse =
            serde_json::from_slice(&bytes).map_err(|e| ProviderError::Decode(e.to_string()))?;
        order_and_validate(decoded, texts.len(), self.dim)
    }
}

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingDatum>,
}

#[derive(Deserialize)]
struct EmbeddingDatum {
    embedding: Vec<f32>,
    index: usize,
}

fn order_and_validate(
    response: EmbeddingResponse,
    expected_count: usize,
    dim: usize,
) -> Result<Vec<Vec<f32>>, ProviderError> {
    let mut ordered = vec![None; expected_count];
    for datum in response.data {
        if datum
            .embedding
            .iter()
            .any(|component| !component.is_finite())
        {
            return Err(ProviderError::Decode(format!(
                "embedding at index {} contains a non-finite component",
                datum.index
            )));
        }
        if datum.embedding.len() != dim {
            return Err(ProviderError::Decode(format!(
                "embedding dimension mismatch: expected {dim}, got {}",
                datum.embedding.len()
            )));
        }
        let slot = ordered.get_mut(datum.index).ok_or_else(|| {
            ProviderError::Decode(format!("embedding index {} out of range", datum.index))
        })?;
        if slot.replace(datum.embedding).is_some() {
            return Err(ProviderError::Decode(format!(
                "duplicate embedding index {}",
                datum.index
            )));
        }
    }
    ordered
        .into_iter()
        .enumerate()
        .map(|(index, vector)| {
            vector.ok_or_else(|| ProviderError::Decode(format!("missing embedding index {index}")))
        })
        .collect()
}

fn status_error(status: StatusCode, bytes: &[u8]) -> ProviderError {
    let body = String::from_utf8_lossy(bytes);
    ProviderError::from_status(status.as_u16(), Some(body.chars().take(1024).collect()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str, count: usize, dim: usize) -> Result<Vec<Vec<f32>>, ProviderError> {
        let response = serde_json::from_str(json).expect("fixture parses");
        order_and_validate(response, count, dim)
    }

    #[test]
    fn parses_fixture_and_orders_by_index() {
        let vectors = parse(
            r#"{"data":[{"embedding":[3.0,4.0],"index":1},{"embedding":[1.0,2.0],"index":0}]}"#,
            2,
            2,
        )
        .expect("valid response");
        assert_eq!(vectors, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    #[test]
    fn dimension_mismatch_names_expected_and_actual() {
        let error = parse(r#"{"data":[{"embedding":[1.0],"index":0}]}"#, 1, 2)
            .expect_err("wrong dimension fails");
        assert!(error.to_string().contains("expected 2, got 1"));
    }

    #[test]
    fn non_finite_component_names_embedding_index() {
        let error = order_and_validate(
            EmbeddingResponse {
                data: vec![EmbeddingDatum {
                    embedding: vec![f32::INFINITY],
                    index: 7,
                }],
            },
            8,
            1,
        )
        .expect_err("infinite component fails");
        assert!(error.to_string().contains("index 7"));
    }

    #[test]
    fn auth_header_is_present_only_with_key() {
        let absent = embedding_headers(None).expect("keyless headers");
        assert!(!absent.contains_key(AUTHORIZATION));
        let present = embedding_headers(Some("secret")).expect("keyed headers");
        assert_eq!(present[AUTHORIZATION], "Bearer secret");
        assert!(present[AUTHORIZATION].is_sensitive());
    }
}
