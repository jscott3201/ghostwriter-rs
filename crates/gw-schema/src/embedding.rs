//! `EmbeddingConfig` — data-plane embedder + vector index (CONFIG §6.1, DATA-SCHEMA §4.6,
//! REMEDIATION ITEM 8).
//!
//! Default = `OpenAiCompatible` @ OMLX-local `Qwen3-Embedding-4B-4bit-DWQ` (2560-dim) reached
//! over the standard `/v1/embeddings` path, with a pinned `revision`; default vector index
//! `usearch`. A pinned LOCAL endpoint is reproducible AND zero-marginal-cost.

use serde::{Deserialize, Serialize};

/// The pinned OMLX-local embedder endpoint (default).
pub const DEFAULT_EMBEDDING_ENDPOINT: &str = "http://127.0.0.1:7700/v1";
/// The default OMLX embedding model.
pub const DEFAULT_EMBEDDING_MODEL: &str = "Qwen3-Embedding-4B-4bit-DWQ";
/// The default native embedding dimension (Matryoshka/MRL-truncatable).
pub const DEFAULT_EMBEDDING_DIM: u32 = 2560;

/// Embedder + vector-index configuration. Recorded in `ExportManifest` / `decontam_index_id`
/// so embedding-decontam results are reproducible.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub backend: EmbeddingBackend,
    /// OpenAI-compatible base_url; DEFAULT OMLX-local; None for `CandleLocal`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    /// DEFAULT `Qwen3-Embedding-4B-4bit-DWQ` (OMLX); `BAAI/bge-small-en-v1.5` if `CandleLocal`.
    pub model: String,
    /// Pinned model/quant revision for reproducibility (recorded in provenance).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    /// DEFAULT 2560 (Qwen3-Embedding-4B native; MRL-truncatable).
    pub dim: u32,
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

/// The vector index backing the near-dup / decontam path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorIndex {
    #[default]
    Usearch,
    HnswRs,
    SqliteVec,
}
