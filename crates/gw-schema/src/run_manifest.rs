//! Cooperative configured identities for immutable generation and admission runs.
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The supported immutable run-manifest contract. New meaning requires a new version.
pub const RUN_MANIFEST_VERSION: u32 = 1;

/// An implementation's pure declaration of its immutable semantic configuration.
///
/// The implementation and behavior revision must be explicit, stable identifiers. Configuration
/// must identify behavior or the immutable evidence collection, never credentials, a mutable path,
/// Rust type/debug/pointer identity, or accounting capability. This cooperative declaration is not
/// an attestation of a server, deployment, or model's eligibility.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticDeclaration {
    /// Stable implementation/adapter identifier chosen by its owner.
    pub implementation: String,
    /// Explicit behavior contract revision, independent of runtime state.
    pub revision: String,
    /// Immutable, non-secret semantic parameters or an evidence-collection digest.
    pub configuration: Value,
}
impl SemanticDeclaration {
    /// Construct an explicit declaration. Callers validate it before persistence.
    #[must_use]
    pub fn new(
        implementation: impl Into<String>,
        revision: impl Into<String>,
        configuration: Value,
    ) -> Self {
        Self {
            implementation: implementation.into(),
            revision: revision.into(),
            configuration,
        }
    }
    /// Whether the declaration has explicit implementation, revision, and object configuration.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.implementation.trim().is_empty()
            && !self.revision.trim().is_empty()
            && self.configuration.is_object()
    }
}

/// Semantic declarations read from each actual injected client, independently of accounting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClientSemantics {
    /// Teacher chat adapter and requested route.
    pub teacher: SemanticDeclaration,
    /// Judge chat adapter and requested route.
    pub judge: SemanticDeclaration,
    /// Embedding implementation and effective vector contract.
    pub embedding: SemanticDeclaration,
    /// Ground-truth sandbox implementation and immutable inputs.
    pub sandbox: SemanticDeclaration,
    /// Precomputed evidence collection; a path or backend name alone is insufficient.
    pub execution_evidence: SemanticDeclaration,
}

/// Digest of the full captured, ordered SeedItem vectors, including empty shards.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputPlanIdentity {
    /// BLAKE3 of the canonical seed-plan-v1 encoding, including all candidate fields.
    pub content_hash: String,
    /// Number of items in every effective shard, in shard order.
    pub shard_items: Vec<u64>,
}

/// Served deployment facts that configured requests cannot attest.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnattestedDeployment {
    /// Unavailable immutable served-weight identity.
    pub weights: Option<String>,
    /// Unavailable tokenizer identity.
    pub tokenizer: Option<String>,
    /// Unavailable server chat-template identity.
    pub chat_template: Option<String>,
    /// Unavailable server parser identity.
    pub parser: Option<String>,
    /// Unavailable served quantization identity.
    pub quantization: Option<String>,
    /// Unavailable deployment revision.
    pub deployment_revision: Option<String>,
}

/// Immutable effective generation/admission contract stored in `runs.config_json`.
/// Operational accounting, rates, concurrency, UI and export settings do not belong here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunManifest {
    /// Envelope contract version; only [`RUN_MANIFEST_VERSION`] is supported.
    pub version: u32,
    /// Full captured input-plan identity.
    pub input_plan: InputPlanIdentity,
    /// Effective request builders and generation/admission behavior contract.
    pub execution: SemanticDeclaration,
    /// Actual injected implementation declarations.
    pub clients: ClientSemantics,
    /// Unknown served facts remain explicitly unknown; they are never inferred from requested IDs.
    pub unattested_deployment: UnattestedDeployment,
}
impl RunManifest {
    /// Validate the supported envelope before it authorizes execution.
    ///
    /// # Errors
    /// Returns a non-secret reason for unsupported or incomplete declarations.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.version != RUN_MANIFEST_VERSION {
            return Err("unsupported manifest version");
        }
        if self.input_plan.shard_items.is_empty()
            || i64::try_from(self.input_plan.shard_items.len()).is_err()
        {
            return Err("invalid captured shard count");
        }
        if self.input_plan.content_hash.len() != 64
            || !self
                .input_plan
                .content_hash
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err("invalid captured input digest");
        }
        if [
            &self.execution,
            &self.clients.teacher,
            &self.clients.judge,
            &self.clients.embedding,
            &self.clients.sandbox,
            &self.clients.execution_evidence,
        ]
        .iter()
        .any(|declaration| !declaration.is_valid())
        {
            return Err("incomplete semantic declaration");
        }
        if self.unattested_deployment != UnattestedDeployment::default() {
            return Err("this manifest version cannot attest served deployment facts");
        }
        Ok(())
    }
}
