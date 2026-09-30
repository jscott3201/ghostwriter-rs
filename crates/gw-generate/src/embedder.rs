//! Async embedding boundary, with typed physical-request failures.
use gw_providers::{CallObservation, ProviderError};
use gw_schema::AccountingCapability;
use std::{future::Future, pin::Pin};

/// Object-safe embedding operation; no executor thread is blocked waiting for model I/O.
pub type EmbeddingFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Vec<f32>, ProviderError>> + Send + 'a>>;

/// Injected diversity embedder. Accounting and cancellation errors must retain their types.
/// The built-in [`gw_providers::EmbeddingsClient`] implements this asynchronous boundary directly.
pub trait Embedder: Send + Sync {
    /// Pure immutable implementation/configuration declaration, independent of accounting.
    /// Missing declarations cannot authorize engine run preparation.
    fn semantic_declaration(&self) -> Option<gw_schema::SemanticDeclaration> {
        None
    }
    /// Embed text asynchronously. Malformed vectors and accounting failures remain distinguishable.
    fn embed<'a>(&'a self, text: &'a str) -> EmbeddingFuture<'a>;
    /// Cooperative capability of the actual implementation.
    fn accounting_capability(&self) -> AccountingCapability {
        AccountingCapability::Unknown
    }
    /// Embed through a per-transmission observer when supported by this implementation.
    fn embed_observed<'a>(
        &'a self,
        text: &'a str,
        _observation: CallObservation,
    ) -> EmbeddingFuture<'a> {
        self.embed(text)
    }
}

/// Pure default that emits no model request and declares every prompt novel.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullEmbedder;
impl Embedder for NullEmbedder {
    fn semantic_declaration(&self) -> Option<gw_schema::SemanticDeclaration> {
        Some(gw_schema::SemanticDeclaration::new(
            "gw-generate/null-embedder",
            "1",
            serde_json::json!({"vectors": "empty", "model_requests": false, "diversity": "all-novel-at-default-positive-threshold"}),
        ))
    }
    fn embed<'a>(&'a self, _text: &'a str) -> EmbeddingFuture<'a> {
        Box::pin(async { Ok(Vec::new()) })
    }
    fn accounting_capability(&self) -> AccountingCapability {
        AccountingCapability::NoModelRequests
    }
}

impl Embedder for gw_providers::EmbeddingsClient {
    fn semantic_declaration(&self) -> Option<gw_schema::SemanticDeclaration> {
        Some(gw_providers::EmbeddingsClient::semantic_declaration(self))
    }
    fn embed<'a>(&'a self, text: &'a str) -> EmbeddingFuture<'a> {
        Box::pin(async move { first(self.embed_batch(&[text]).await?) })
    }
    fn embed_observed<'a>(
        &'a self,
        text: &'a str,
        observation: CallObservation,
    ) -> EmbeddingFuture<'a> {
        Box::pin(async move { first(self.embed_batch_observed(&[text], observation).await?) })
    }
    fn accounting_capability(&self) -> AccountingCapability {
        AccountingCapability::PhysicalAttemptsV1
    }
}
fn first(vectors: Vec<Vec<f32>>) -> Result<Vec<f32>, ProviderError> {
    vectors
        .into_iter()
        .next()
        .ok_or_else(|| ProviderError::Decode("embedding response contained no vector".into()))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn http_embedder_preserves_typed_errors_without_blocking() {
        let client = gw_providers::EmbeddingsClient::builder()
            .base_url("http://127.0.0.1:1/v1")
            .dim(2)
            .timeout(std::time::Duration::from_millis(100))
            .build_with_key(None)
            .unwrap();
        let error = client.embed("hello").await.unwrap_err();
        assert!(matches!(error, ProviderError::Transport(_)));
    }
}
