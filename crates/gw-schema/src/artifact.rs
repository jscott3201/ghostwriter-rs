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
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ExportArtifact {
    /// Footer and artifact-identity algorithm version, separate from the column schema version.
    pub metadata_version: u32,
    /// Domain-separated BLAKE3 identity of the artifact's logical content.
    pub artifact_id: String,
    /// Population used to prepare the artifact.
    pub scope: ExportScope,
    /// Complete export manifest, also returned by command-line export.
    pub manifest: ExportManifest,
    /// Transaction-checked source screening witness, present only in screened metadata v3.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub screening: Option<Box<crate::ScreenedExportQualification>>,
}

impl ExportArtifact {
    /// Raw footer and identity version, preserved for historical and automatic exports.
    pub const CURRENT_VERSION: u32 = 1;
    /// Source-screened footer and identity version over reviewed-task columns.
    pub const SCREENED_VERSION: u32 = 3;
}

// A present null is different from an absent key: historical metadata cannot carry a witness,
// and screened metadata cannot omit one. Direct typed decoding preserves exact numeric tags.
#[derive(Default)]
struct ScreeningField(Option<Box<crate::ScreenedExportQualification>>);
impl<'de> Deserialize<'de> for ScreeningField {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        crate::ScreenedExportQualification::deserialize(d).map(|value| Self(Some(Box::new(value))))
    }
}
impl<'de> Deserialize<'de> for ExportArtifact {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            #[serde(deserialize_with = "export_metadata_version")]
            metadata_version: u32,
            artifact_id: String,
            scope: ExportScope,
            manifest: ExportManifest,
            #[serde(default)]
            screening: ScreeningField,
        }
        let wire = Wire::deserialize(d)?;
        match (wire.metadata_version, wire.screening.0.is_some()) {
            (1, false) | (3, true) => {}
            _ => {
                return Err(serde::de::Error::custom(
                    "unsupported export metadata shape/version",
                ));
            }
        }
        Ok(Self {
            metadata_version: wire.metadata_version,
            artifact_id: wire.artifact_id,
            scope: wire.scope,
            manifest: wire.manifest,
            screening: wire.screening.0,
        })
    }
}

fn export_metadata_version<'de, D: serde::Deserializer<'de>>(d: D) -> Result<u32, D::Error> {
    let version = u32::deserialize(d)?;
    match version {
        1 | 3 => Ok(version),
        _ => Err(serde::de::Error::custom(
            "unsupported export metadata version; earlier screened plans/artifacts must be regenerated",
        )),
    }
}
