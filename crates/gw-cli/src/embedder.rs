//! Synchronous engine adapter for the async embeddings HTTP client.

use gw_generate::Embedder;
use gw_providers::EmbeddingsClient;

/// Bridges the synchronous generation seam to the CLI's multi-thread Tokio runtime.
#[derive(Debug, Clone)]
pub(crate) struct HttpEmbedder {
    client: EmbeddingsClient,
}

impl HttpEmbedder {
    pub(crate) fn new(client: EmbeddingsClient) -> Self {
        Self { client }
    }
}

impl Embedder for HttpEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(self.client.embed_batch(&[text]))
                .map_err(|error| error.to_string())?
                .into_iter()
                .next()
                .ok_or_else(|| "embedding response contained no vector".to_string())
        })
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn adapter_surfaces_client_error_without_panicking() {
        let client = EmbeddingsClient::builder()
            .base_url("http://127.0.0.1:1/v1")
            .dim(2)
            .timeout(Duration::from_millis(100))
            .build_with_key(None)
            .expect("client builds");
        let error = HttpEmbedder::new(client)
            .embed("hello")
            .expect_err("unroutable endpoint fails");
        assert!(error.contains("transport error"), "got: {error}");
    }
}
