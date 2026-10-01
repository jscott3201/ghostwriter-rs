//! Pure, explicit behavior projections of supported built-in client declarations.
use crate::{ProviderError, normalize_endpoint};
use gw_schema::{ModelAdapterBehavior, SemanticDeclaration, VectorIndex};
use serde::Deserialize;
use serde_json::{Value, json};

/// Derive route-independent behavior from a supported built-in v1 client declaration.
///
/// This reads the same full declaration exposed by the built-in builders and clients. It decodes
/// their specific contracts, not an arbitrary adapter's JSON. Endpoints remain in the original
/// full declaration. Embedding aliases belong in `ModelExecutionSemantics::alias`; configured
/// model-revision/index labels do not affect this adapter's behavior because v1 does not enforce
/// them. This helper neither constructs a client nor verifies execution or model eligibility.
/// Describing supplied behavior is not evidence that a request used it.
///
/// # Errors
/// Rejects unsupported implementations, revisions, fields, credential-bearing endpoints, and
/// structurally invalid built-in declarations without including supplied values in errors.
pub fn builtin_adapter_behavior(
    client: &SemanticDeclaration,
) -> Result<ModelAdapterBehavior, ProviderError> {
    if client.revision != "1" {
        return Err(unsupported());
    }
    let configuration = match client.implementation.as_str() {
        "gw-providers/openai-compatible-chat-sse" => {
            let fields: ChatDescriptor = decode(&client.configuration)?;
            normalize_endpoint(&fields.base_endpoint)?;
            if fields.transport_attempts == 0 {
                return Err(unsupported());
            }
            json!({
                "route": fields.route, "request_method": fields.request_method,
                "stream_contract": fields.stream_contract, "transport_attempts": fields.transport_attempts,
                "retry_classification": fields.retry_classification, "redirects": fields.redirects
            })
        }
        "gw-providers/openai-compatible-embeddings" => {
            let fields: EmbeddingDescriptor = decode(&client.configuration)?;
            normalize_endpoint(&fields.base_endpoint)?;
            if fields.dimension == 0
                || fields.configured_declarations.model_revision_enforced
                || fields
                    .configured_declarations
                    .index_selects_runtime_implementation
            {
                return Err(unsupported());
            }
            json!({
                "route": fields.route, "dimension": fields.dimension,
                "normalization": fields.normalization, "batch_order": fields.batch_order,
                "retries": fields.retries
            })
        }
        _ => return Err(unsupported()),
    };
    Ok(ModelAdapterBehavior {
        declaration: SemanticDeclaration::new(
            &client.implementation,
            &client.revision,
            configuration,
        ),
    })
}

fn unsupported() -> ProviderError {
    ProviderError::Config("unsupported built-in v1 adapter behavior declaration".into())
}
fn decode<T: serde::de::DeserializeOwned>(value: &Value) -> Result<T, ProviderError> {
    serde_json::from_value(value.clone()).map_err(|_| unsupported())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatDescriptor {
    base_endpoint: String,
    route: String,
    request_method: String,
    stream_contract: String,
    transport_attempts: u32,
    retry_classification: String,
    redirects: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EmbeddingDescriptor {
    base_endpoint: String,
    route: String,
    // The surrounding execution identity owns the requested alias.
    #[serde(rename = "requested_model")]
    _requested_model: String,
    dimension: usize,
    normalization: String,
    batch_order: String,
    retries: u32,
    configured_declarations: ConfiguredDeclarations,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfiguredDeclarations {
    // These explicit v1 labels are not enforced by the adapter.
    #[serde(rename = "model_revision")]
    _model_revision: Option<String>,
    model_revision_enforced: bool,
    #[serde(rename = "index")]
    _index: Option<VectorIndex>,
    index_selects_runtime_implementation: bool,
}
