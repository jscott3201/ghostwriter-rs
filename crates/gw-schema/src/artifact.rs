//! Self-contained export artifact metadata; no filesystem or database identity travels here.

use serde::{Deserialize, Serialize};

use crate::{CotPolicy, ExportManifest, TrlFormat};

/// Population from which an artifact's admitted rows were selected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExportScope {
    /// Every record in the source store at preparation time.
    Store,
    /// Records belonging to one generation run.
    Run {
        /// Stable generation run identifier, never a local path.
        run_id: String,
    },
    /// An explicitly supplied in-memory record population.
    Records,
}

/// Export policies fixed before projecting or encoding any artifact bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExportOptions {
    /// Training template the downstream renderer should use.
    pub target: TrlFormat,
    /// Downstream reasoning-loss policy.
    pub cot_policy: CotPolicy,
    /// Dataset version, if the producer has assigned one.
    pub dataset_version: Option<semver::Version>,
    /// Exact selection scope; excludes machine-local destination information.
    pub scope: ExportScope,
}

/// Authoritative footer envelope for a self-contained Parquet artifact.
///
/// The identity covers the version, complete manifest, scope and every projected row. It is not a
/// hash of the Parquet file and excludes this envelope's own `artifact_id` field. Future provenance
/// additions require an explicit versioned contract; missing model identities are not inferred.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExportArtifact {
    /// Footer and artifact-identity algorithm version, separate from the column schema version.
    pub metadata_version: u32,
    /// Domain-separated BLAKE3 identity of the artifact's logical content.
    pub artifact_id: String,
    /// Population used to prepare the artifact.
    pub scope: ExportScope,
    /// Complete export manifest, also returned by command-line export.
    pub manifest: ExportManifest,
}

impl ExportArtifact {
    /// The footer and identity version written and verified by this build.
    pub const CURRENT_VERSION: u32 = 1;
}
