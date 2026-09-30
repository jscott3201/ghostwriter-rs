//! `EmbeddingConfig` — data-plane embedder + vector index (CONFIG §6.1, DATA-SCHEMA §4.6,
//! REMEDIATION ITEM 8).
//!
//! Default = `OpenAiCompatible` @ OMLX-local `Qwen3-Embedding-4B-4bit-DWQ` (2560-dim) reached
//! over the standard `/v1/embeddings` path. The revision and index are configured declarations;
//! the HTTP client does not enforce a served revision, and the engine uses its own in-memory priors.

use serde::{Deserialize, Serialize};

/// The pinned OMLX-local embedder endpoint (default).
pub const DEFAULT_EMBEDDING_ENDPOINT: &str = "http://127.0.0.1:7700/v1";
/// The default OMLX embedding model.
pub const DEFAULT_EMBEDDING_MODEL: &str = "Qwen3-Embedding-4B-4bit-DWQ";
/// The default native embedding dimension (Matryoshka/MRL-truncatable).
pub const DEFAULT_EMBEDDING_DIM: u32 = 2560;

/// Requested embedder configuration and index label. Immutable run manifests keep these
/// declarations separate from the actual client behavior and unknown served model identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Requested backend; the CLI currently constructs only the OpenAI-compatible variant.
    pub backend: EmbeddingBackend,
    /// OpenAI-compatible base_url; DEFAULT OMLX-local; None for `CandleLocal`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    /// DEFAULT `Qwen3-Embedding-4B-4bit-DWQ` (OMLX); `BAAI/bge-small-en-v1.5` if `CandleLocal`.
    pub model: String,
    /// Declared model/quant revision; the HTTP client does not send or enforce this value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    /// DEFAULT 2560 (Qwen3-Embedding-4B native; MRL-truncatable).
    pub dim: u32,
    /// Configured index label; this does not select the engine's runtime prior implementation.
    pub index: VectorIndex,
    /// Env var name for a REMOTE OpenAI-compatible endpoint; None for local OMLX (no key).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            backend: EmbeddingBackend::OpenAiCompatible,
            endpoint: Some(DEFAULT_EMBEDDING_ENDPOINT.to_string()),
            model: DEFAULT_EMBEDDING_MODEL.to_string(),
            revision: None,
            dim: DEFAULT_EMBEDDING_DIM,
            index: VectorIndex::Usearch,
            api_key_env: None,
        }
    }
}

/// Embedding backend mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingBackend {
    #[default]
    OpenAiCompatible,
    CandleLocal,
}

/// Configured vector-index vocabulary. These labels do not attest the runtime implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorIndex {
    #[default]
    Usearch,
    HnswRs,
    SqliteVec,
}
