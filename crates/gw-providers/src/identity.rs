//! Pure, credential-free endpoint and built-in adapter declarations.
use crate::ProviderError;

/// Normalize a supported HTTP(S) base route without losing meaningful origin/path differences.
/// Userinfo, queries and fragments are unsupported; errors never repeat the input URL.
///
/// # Errors
/// Returns a non-secret configuration error for an unsupported endpoint form.
pub fn normalize_endpoint(input: &str) -> Result<String, ProviderError> {
    let mut url = reqwest::Url::parse(input)
        .map_err(|_| ProviderError::Config("endpoint must be an absolute HTTP(S) URL".into()))?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err(ProviderError::Config(
            "endpoint must be an absolute HTTP(S) URL".into(),
        ));
    }
    let userinfo = input.split_once("://").is_some_and(|(_, rest)| {
        rest.split(['/', '?', '#', '\\'])
            .next()
            .is_some_and(|authority| authority.contains('@'))
    });
    if userinfo
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(ProviderError::Config("endpoint userinfo, query, and fragment forms are unsupported; provide a credential-free base route".into()));
    }
    let path = url.path().trim_end_matches('/').to_string();
    url.set_path(&path);
    Ok(url.to_string().trim_end_matches('/').to_string())
}

pub(crate) fn chat(
    base_url: &str,
    attempts: u32,
) -> Result<gw_schema::SemanticDeclaration, ProviderError> {
    Ok(gw_schema::SemanticDeclaration::new(
        "gw-providers/openai-compatible-chat-sse",
        "1",
        serde_json::json!({
            "base_endpoint": normalize_endpoint(base_url)?, "route": "chat/completions", "request_method": "POST",
            "stream_contract": "content-and-reasoning-details-v1", "transport_attempts": attempts.max(1),
            "retry_classification": "provider-error-retryable-v1", "redirects": "reject"
        }),
    ))
}

pub(crate) fn embedding(
    base_url: &str,
    model: &str,
    dim: usize,
    revision: Option<&str>,
    index: Option<gw_schema::VectorIndex>,
) -> Result<gw_schema::SemanticDeclaration, ProviderError> {
    if dim == 0 {
        return Err(ProviderError::Config(
            "embedding dimension must be positive".into(),
        ));
    }
    Ok(gw_schema::SemanticDeclaration::new(
        "gw-providers/openai-compatible-embeddings",
        "1",
        serde_json::json!({
            "base_endpoint": normalize_endpoint(base_url)?, "route": "embeddings", "requested_model": model,
            "dimension": dim, "normalization": "none-preserve-finite-nonzero-vectors-v1", "batch_order": "response-index-v1", "retries": 0,
            "configured_declarations": {"model_revision": revision, "model_revision_enforced": false, "index": index, "index_selects_runtime_implementation": false}
        }),
    ))
}
