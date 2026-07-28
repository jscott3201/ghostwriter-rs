//! Opt-in smoke test for an OpenAI-compatible embeddings endpoint.

use gw_providers::EmbeddingsClient;

#[tokio::test]
#[ignore = "requires GW_EMBEDDINGS_LIVE=1 and a live embeddings endpoint"]
async fn live_embeddings_smoke() {
    if std::env::var("GW_EMBEDDINGS_LIVE").as_deref() != Ok("1") {
        return;
    }
    let base_url = std::env::var("GW_EMBEDDINGS_BASE_URL")
        .unwrap_or_else(|_| gw_schema::DEFAULT_EMBEDDING_ENDPOINT.to_string());
    let model = std::env::var("GW_EMBEDDINGS_MODEL")
        .unwrap_or_else(|_| gw_schema::DEFAULT_EMBEDDING_MODEL.to_string());
    let key = std::env::var("GW_EMBEDDINGS_API_KEY").ok();
    let client = EmbeddingsClient::builder()
        .base_url(base_url)
        .model(model)
        .dim(gw_schema::DEFAULT_EMBEDDING_DIM)
        .build_with_key(key.as_deref())
        .expect("client builds");
    let vectors = client
        .embed_batch(&["A short embeddings smoke test."])
        .await
        .expect("endpoint embeds");
    assert_eq!(vectors.len(), 1);
    assert_eq!(vectors[0].len(), gw_schema::DEFAULT_EMBEDDING_DIM as usize);
}
