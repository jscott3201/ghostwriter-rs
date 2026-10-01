//! Frozen projection plans and independent, complete Parquet readback verification.

use std::collections::BTreeSet;
use std::fs::File;
use std::path::Path;

use arrow::array::{Array, Float64Array, RecordBatch, StringArray, UInt32Array};
use blake3::Hasher;
use gw_schema::{
    ExportArtifact, ExportManifest, ExportOptions, ExportSchemaVersion, ExportScope,
    ExportTaskProjection, Message, MultiTurnLoss, TrainingRecord,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::reader::ChunkReader;
use serde::{Deserialize, Serialize};

use crate::export::{Projected, export_schema, is_selected_admitted, project, shard_content_hash};
use crate::{Result, StorageError};

/// The sole authoritative custom footer entry. Adjacent files are never consulted.
pub const ARTIFACT_METADATA_KEY: &str = "ghostwriter.export_artifact";

/// A complete, immutable export plan. Destination identity is deliberately separate.
#[derive(Debug, Clone)]
pub struct ExportPlan {
    pub(crate) rows: Vec<Projected>,
    pub(crate) artifact: ExportArtifact,
}

/// One exact selected member of a prepared receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Member {
    pub record_id: String,
    pub projected_hash: String,
}

impl ExportPlan {
    /// Freeze policies, scope, counts and every projected value before encoding.
    ///
    /// # Errors
    /// Rejects duplicate record IDs, a run-scope mismatch, or an unprojectable record.
    pub fn prepare(records: &[TrainingRecord], options: ExportOptions) -> Result<Self> {
        let mut ids = BTreeSet::new();
        for record in records {
            if !ids.insert(&record.record_id) {
                return Err(integrity("duplicate record ID in export population"));
            }
            if let ExportScope::Run { run_id } = &options.scope
                && record.provenance.run_id != *run_id
            {
                return Err(integrity("record does not belong to the export run"));
            }
        }
        let mut rows: Vec<Projected> = records
            .iter()
            .filter(|r| is_selected_admitted(r))
            .map(|record| project(record, ExportSchemaVersion::CURRENT))
            .collect::<Result<_>>()?;
        rows.sort_by(|a, b| a.record_id.cmp(&b.record_id));
        let manifest = ExportManifest {
            column_schema_version: ExportSchemaVersion::CURRENT,
            target: options.target,
            cot_policy: options.cot_policy,
            dataset_version: options.dataset_version,
            hub_commit_sha: None,
            n_records: records.len() as u64,
            n_admitted: rows.len() as u64,
            decontam_index_id: None,
            build_inputs_hash: shard_content_hash(&rows),
            multi_turn_loss: MultiTurnLoss::default(),
            diversity: None,
        };
        let mut artifact = ExportArtifact {
            metadata_version: ExportArtifact::CURRENT_VERSION,
            artifact_id: String::new(),
            scope: options.scope,
            manifest,
        };
        artifact.artifact_id = artifact_identity(&artifact, &rows)?;
        Ok(Self { rows, artifact })
    }

    /// The complete immutable metadata to embed in the footer.
    #[must_use]
    pub fn artifact(&self) -> &ExportArtifact {
        &self.artifact
    }

    pub(crate) fn members(&self) -> Result<Vec<Member>> {
        self.rows
            .iter()
            .map(|row| {
                Ok(Member {
                    record_id: row.record_id.clone(),
                    projected_hash: projected_hash(
                        row,
                        self.artifact.manifest.column_schema_version,
                    )?,
                })
            })
            .collect()
    }
}

pub(crate) fn integrity(message: impl Into<String>) -> StorageError {
    StorageError::Export(message.into())
}

/// Length framing and explicit presence bytes distinguish empty strings, nulls and field boundaries.
fn frame(hash: &mut Hasher, bytes: &[u8]) {
    hash.update(&(bytes.len() as u64).to_be_bytes());
    hash.update(bytes);
}

pub(crate) fn projected_hash(row: &Projected, version: ExportSchemaVersion) -> Result<String> {
    let domain = match version {
        ExportSchemaVersion::CanonicalMessages if row.task_json.is_none() => {
            "ghostwriter.export.projected-row.v1"
        }
        ExportSchemaVersion::ReviewedTasks => "ghostwriter.export.projected-row.v2-reviewed-tasks",
        _ => return Err(integrity("unsupported projected row/schema combination")),
    };
    let mut hash = Hasher::new_derive_key(domain);
    for value in [
        &row.record_id,
        &row.training_area,
        &row.record_hash,
        &row.prompt_hash,
    ] {
        frame(&mut hash, value.as_bytes());
    }
    match &row.verdict {
        Some(value) => {
            hash.update(&[1]);
            frame(&mut hash, value.as_bytes());
        }
        None => {
            hash.update(&[0]);
        }
    }
    match row.judge_aggregate {
        Some(value) => {
            hash.update(&[1]);
            hash.update(&value.to_bits().to_be_bytes());
        }
        None => {
            hash.update(&[0]);
        }
    }
    hash.update(&row.reasoning_tokens.to_be_bytes());
    frame(&mut hash, row.messages_json.as_bytes());
    if version == ExportSchemaVersion::ReviewedTasks {
        match &row.task_json {
            Some(value) => {
                hash.update(&[1]);
                frame(&mut hash, value.as_bytes());
            }
            None => {
                hash.update(&[0]);
            }
        }
    }
    Ok(hash.finalize().to_hex().to_string())
}

pub(crate) fn artifact_identity(artifact: &ExportArtifact, rows: &[Projected]) -> Result<String> {
    let mut hash = Hasher::new_derive_key("ghostwriter.export.artifact.v1");
    hash.update(&artifact.metadata_version.to_be_bytes());
    frame(&mut hash, &canonical_metadata_json(&artifact.scope)?);
    frame(&mut hash, &canonical_metadata_json(&artifact.manifest)?);
    hash.update(&(rows.len() as u64).to_be_bytes());
    for row in rows {
        frame(
            &mut hash,
            projected_hash(row, artifact.manifest.column_schema_version)?.as_bytes(),
        );
    }
    Ok(hash.finalize().to_hex().to_string())
}

fn canonical_metadata_json(value: &impl Serialize) -> Result<Vec<u8>> {
    let mut value = serde_json::to_value(value)?;
    value.sort_all_objects();
    Ok(serde_json::to_vec(&value)?)
}

/// Explicit metadata status for historical files and current verified artifacts.
#[derive(Debug, Clone, PartialEq)]
pub enum ArtifactVerification {
    /// No authoritative footer entry exists. Historical row readers may still read this file.
    MissingLegacyMetadata,
    /// Footer, schema and every row passed verification.
    Verified(ExportArtifact),
}

/// Verify a newly opened artifact through all record batches; never consult a sidecar.
///
/// # Errors
/// Fails on duplicate/unsupported metadata, schema mismatch, malformed messages, duplicate IDs,
/// count/hash mismatch, corrupt batches, or filesystem errors. Missing metadata is explicit.
pub fn verify_artifact(path: impl AsRef<Path>) -> Result<ArtifactVerification> {
    verify_reader(File::open(path)?)
}

/// A successful verification bound to exact Parquet bytes, distinct from the logical artifact ID.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactSnapshotReport {
    /// Version of this stdin/snapshot report contract.
    pub report_version: u32,
    /// Fully verified self-contained artifact metadata.
    pub artifact: ExportArtifact,
    /// Length of the exact verified Parquet encoding.
    pub byte_length: u64,
    /// Ordinary BLAKE3 of the raw Parquet bytes, without the logical identity framing.
    pub snapshot_blake3: String,
}

/// Verify one owned immutable snapshot without reopening a filesystem path.
///
/// # Errors
/// Returns the same integrity failures as [`verify_artifact`], and rejects missing legacy metadata.
pub fn verify_artifact_snapshot(snapshot: Vec<u8>) -> Result<ArtifactSnapshotReport> {
    let byte_length = snapshot.len() as u64;
    let snapshot_blake3 = blake3::hash(&snapshot).to_hex().to_string();
    let ArtifactVerification::Verified(artifact) = verify_reader(bytes::Bytes::from(snapshot))?
    else {
        return Err(integrity(
            "snapshot verification requires authoritative artifact metadata",
        ));
    };
    Ok(ArtifactSnapshotReport {
        report_version: 1,
        artifact,
        byte_length,
        snapshot_blake3,
    })
}

fn verify_reader<R: ChunkReader + 'static>(reader: R) -> Result<ArtifactVerification> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(reader)?;
    let entries: Vec<_> = builder
        .metadata()
        .file_metadata()
        .key_value_metadata()
        .into_iter()
        .flatten()
        .filter(|entry| entry.key == ARTIFACT_METADATA_KEY)
        .collect();
    if entries.is_empty() {
        return Ok(ArtifactVerification::MissingLegacyMetadata);
    }
    if entries.len() != 1 {
        return Err(integrity("duplicate authoritative footer metadata"));
    }
    let artifact: ExportArtifact = serde_json::from_str(
        entries[0]
            .value
            .as_deref()
            .ok_or_else(|| integrity("missing footer metadata value"))?,
    )?;
    if artifact.metadata_version != ExportArtifact::CURRENT_VERSION {
        return Err(integrity("unsupported export metadata version"));
    }
    let version = artifact.manifest.column_schema_version;
    if builder.schema().fields() != export_schema(version)?.fields() {
        return Err(integrity(
            "export column schema does not match the manifest",
        ));
    }
    let footer_count = builder.metadata().file_metadata().num_rows();
    let mut rows = Vec::new();
    for batch in builder.with_batch_size(1024).build()? {
        rows.extend(read_batch(&batch?, version)?);
    }
    validate_rows(&artifact, &rows)?;
    if footer_count < 0 || footer_count as u64 != artifact.manifest.n_admitted {
        return Err(integrity("footer row count does not match the manifest"));
    }
    Ok(ArtifactVerification::Verified(artifact))
}

pub(crate) fn validate_rows(artifact: &ExportArtifact, rows: &[Projected]) -> Result<()> {
    export_schema(artifact.manifest.column_schema_version)?;
    if artifact.manifest.n_admitted != rows.len() as u64
        || artifact.manifest.n_records < artifact.manifest.n_admitted
    {
        return Err(integrity(
            "artifact population counts do not match its rows",
        ));
    }
    if rows
        .windows(2)
        .any(|pair| pair[0].record_id >= pair[1].record_id)
    {
        return Err(integrity(
            "artifact record IDs are duplicated or not sorted",
        ));
    }
    if artifact.manifest.build_inputs_hash != shard_content_hash(rows) {
        return Err(integrity("artifact build_inputs_hash mismatch"));
    }
    if artifact.artifact_id != artifact_identity(artifact, rows)? {
        return Err(integrity("artifact identity mismatch"));
    }
    Ok(())
}

fn read_batch(batch: &RecordBatch, version: ExportSchemaVersion) -> Result<Vec<Projected>> {
    if batch.schema().fields() != export_schema(version)?.fields() {
        return Err(integrity("record batch schema mismatch"));
    }
    for index in [0, 1, 2, 3, 6, 7] {
        if batch.column(index).null_count() != 0 {
            return Err(integrity("null in a required export column"));
        }
    }
    let string = |index| {
        batch
            .column(index)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| integrity("invalid string column"))
    };
    let ids = string(0)?;
    let areas = string(1)?;
    let hashes = string(2)?;
    let prompts = string(3)?;
    let verdicts = string(4)?;
    let messages = string(7)?;
    let tasks = if version == ExportSchemaVersion::ReviewedTasks {
        Some(string(8)?)
    } else {
        None
    };
    let scores = batch
        .column(5)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| integrity("invalid aggregate column"))?;
    let tokens = batch
        .column(6)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| integrity("invalid token column"))?;
    (0..batch.num_rows())
        .map(|i| {
            let parsed_messages: Vec<Message> = serde_json::from_str(messages.value(i))?;
            let task_json = tasks
                .filter(|column| !column.is_null(i))
                .map(|column| column.value(i).to_owned());
            if let Some(json) = &task_json {
                let task: ExportTaskProjection = serde_json::from_str(json)?;
                task.validate(&parsed_messages).map_err(integrity)?;
                if crate::export::canonical_task_json(&task)? != *json {
                    return Err(integrity(
                        "task projection must use its exact canonical typed JSON",
                    ));
                }
            }
            Ok(Projected {
                record_id: ids.value(i).into(),
                training_area: areas.value(i).into(),
                record_hash: hashes.value(i).into(),
                prompt_hash: prompts.value(i).into(),
                verdict: (!verdicts.is_null(i)).then(|| verdicts.value(i).into()),
                judge_aggregate: (!scores.is_null(i)).then(|| scores.value(i)),
                reasoning_tokens: tokens.value(i),
                messages_json: messages.value(i).into(),
                task_json,
            })
        })
        .collect()
}

#[cfg(test)]
#[path = "legacy_artifact_tests.rs"]
mod legacy_tests;
#[cfg(test)]
#[path = "task_artifact_tests.rs"]
mod task_tests;
#[cfg(test)]
#[path = "artifact_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "snapshot_tests.rs"]
mod snapshot_tests;
