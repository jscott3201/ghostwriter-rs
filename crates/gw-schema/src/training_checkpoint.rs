//! Completed full-SFT declarations. Byte verification and fresh execution remain separate evidence.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Fixed completion framing: magic, 32-byte body digest, big-endian manifest length, manifest,
/// then the five captured files in manifest order. There are no archive paths or trailing bytes.
pub const CHECKPOINT_MAGIC: &[u8; 8] = b"GWCKPT01";
/// Domain for the exact length-framed manifest and captured files.
pub const CHECKPOINT_HASH_DOMAIN: &str = "ghostwriter.completed-full-sft.v1";
/// Complete package bound, streamed by native consumers rather than allocated as one buffer.
pub const MAX_CHECKPOINT_BYTES: u64 = 7 * 1024 * 1024 * 1024;
/// Strict JSON manifest bound.
pub const MAX_CHECKPOINT_MANIFEST_BYTES: usize = 2 * 1024 * 1024;
/// Supported individual model weight bound.
pub const MAX_CHECKPOINT_WEIGHT_BYTES: u64 = 3 * 1024 * 1024 * 1024;
/// Supported model configuration bound.
pub const MAX_CHECKPOINT_CONFIG_BYTES: usize = 65_536;
/// Exact portable captured file set. The original prepared build retains its independent identity.
pub const CHECKPOINT_FILES: [&str; 5] = [
    "checkpoint/config.json",
    "checkpoint/model.safetensors",
    "initial/config.json",
    "initial/model.safetensors",
    "prepared.gwsft",
];

/// One exact file declaration; native verification measures its captured bytes independently.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointFile {
    /// One fixed identifier from the enclosing completion contract.
    pub path: String,
    /// Complete exact byte length.
    pub byte_length: u64,
    /// Ordinary BLAKE3 of those bytes.
    pub blake3: String,
}

/// Declared CPU recipe, separate from the preparation recipe and input build identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FullSftRecipe {
    /// Supported recipe version.
    pub version: u32,
    /// Complete installed training subpackage source identity.
    pub training_source_sha256: String,
    /// Unchanged preparation package identity already bound by the prepared build.
    pub preparation_source_sha256: String,
    /// Exact qualified Python dependency pins.
    pub dependencies: BTreeMap<String, String>,
    /// Producer Python/platform declarations, without local paths.
    pub runtime: BTreeMap<String, String>,
    /// Exactly CPU execution.
    pub device: String,
    /// Full model float32 training; no adapters or quantization.
    pub precision: String,
    /// One trainer process.
    pub processes: u32,
    /// Bounded CPU intra-operation threads.
    pub threads: u32,
    /// Actual successful optimizer updates required before publication.
    pub max_steps: u32,
    /// Maximum examples in one microbatch, preserving the final partial batch.
    pub batch_size: u32,
    /// Microbatches accumulated per update; epoch tails flush partial accumulation.
    pub accumulation: u32,
    /// Positive learning rate in millionths, avoiding floating JSON identity ambiguity.
    pub learning_rate_millionths: u32,
    /// Explicit deterministic sampler/model RNG seed.
    pub seed: u32,
    /// Largest permitted complete sequence; overflow is rejected, never truncated.
    pub max_sequence_length: u32,
    /// Supported optimizer semantics.
    pub optimizer: String,
    /// Supported schedule semantics.
    pub scheduler: String,
    /// Deterministic repeated full-dataset order.
    pub sampler: String,
    /// Must remain false.
    pub packing: bool,
    /// Must remain false.
    pub truncation: bool,
}

/// A successfully completed actual forward/backward microbatch, as declared by the producer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingMicrobatch {
    /// One-based optimizer update to which this accumulation group contributed.
    pub update: u32,
    /// Ordered prepared example identities, including repeated use in later epochs.
    pub example_ids: Vec<String>,
    /// Actual nonpadding input tokens in this batch.
    pub input_tokens: u64,
    /// Actual nonmasked labels after the causal shift.
    pub shifted_supervised_tokens: u64,
}

/// Recorded execution accounting. Deserializing this document never authenticates execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingObservations {
    /// Number of completed actual forward/backward calls.
    pub successful_microbatches: u64,
    /// Successfully consumed example occurrences, including repeats and partial batches.
    pub consumed_examples: u64,
    /// Sum of actual nonmasked causal-shift labels across successful microbatches.
    pub shifted_supervised_tokens: u64,
    /// Successful underlying optimizer calls, independent of Trainer.global_step.
    pub optimizer_updates: u64,
    /// Complete ordered microbatch declarations.
    pub microbatches: Vec<TrainingMicrobatch>,
    /// Parameter tensor content changed after optimization, independent of file metadata.
    pub parameter_content_changed: bool,
}

/// Measured supported tensor inventory. BF16 bytes are normalized to exact float32 bits for
/// content comparison; tied embedding/head aliases contribute one trainable parameter population.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointModelSummary {
    /// Digest of exact named tensor shapes and normalized float32 parameter content.
    pub tensor_content_id: String,
    /// Distinct parameter elements, excluding persistent buffers and a tied head alias.
    pub parameter_count: u64,
    /// Complete logical named tensor count, including the supported tied alias.
    pub tensor_count: u64,
}

/// Strict completion document. It binds captured bytes and consistent recorded observations,
/// while the separate fresh producer capability owns any locally observed-training claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingCheckpointManifest {
    /// Supported contract version.
    pub version: u32,
    /// Original immutable prepared input identity, never replaced by a completion identity.
    pub prepared_build_id: String,
    /// Fixed bounded trainer recipe.
    pub recipe: FullSftRecipe,
    /// Initial captured model measurements.
    pub initial_model: CheckpointModelSummary,
    /// Saved final inference model measurements.
    pub checkpoint_model: CheckpointModelSummary,
    /// Complete declared execution accounting.
    pub observations: TrainingObservations,
    /// Scoped source class; native inspection verifies declarations, never grants authority.
    pub source_authorization: String,
    /// Upstream pretrained lineage remains unknown; local bytes do not establish parent lineage.
    pub upstream_lineage: String,
    /// Final inference weights only, with no optimizer-resume claim or pickled state.
    pub checkpoint_kind: String,
    /// Exact fixed inventory in lexicographic order.
    pub files: Vec<CheckpointFile>,
}

impl TrainingCheckpointManifest {
    /// Decode bounded strict integer-only JSON and reject unknown, duplicate or omitted fields.
    ///
    /// # Errors
    /// Rejects invalid JSON, unsupported versions/recipes, or inconsistent bounded declarations.
    pub fn from_json(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() > MAX_CHECKPOINT_MANIFEST_BYTES {
            return Err("checkpoint manifest exceeds its byte bound");
        }
        let raw =
            crate::strict_coding_json(bytes).map_err(|_| "invalid checkpoint manifest JSON")?;
        let value: Self = serde_json::from_value(raw.clone())
            .map_err(|_| "invalid checkpoint manifest fields")?;
        if serde_json::to_value(&value).map_err(|_| "invalid checkpoint manifest")? != raw {
            return Err("checkpoint manifest contains unsupported or omitted fields");
        }
        value.validate()?;
        Ok(value)
    }

    /// Check the shape and bounds without reading files or authenticating historical execution.
    ///
    /// # Errors
    /// Rejects unsupported recipes, invalid identities, unsafe inventory, or contradictory totals.
    pub fn validate(&self) -> Result<(), &'static str> {
        let r = &self.recipe;
        if self.version != 1
            || r.version != 1
            || !hash(&self.prepared_build_id)
            || !hash(&r.training_source_sha256)
            || !hash(&r.preparation_source_sha256)
            || r.device != "cpu"
            || r.precision != "float32"
            || r.processes != 1
            || r.threads != 1
            || !(1..=32).contains(&r.max_steps)
            || !(1..=8).contains(&r.batch_size)
            || !(1..=8).contains(&r.accumulation)
            || !(1..=10_000).contains(&r.learning_rate_millionths)
            || !(2..=2048).contains(&r.max_sequence_length)
            || r.seed != 0
            || r.optimizer != "adamw_torch_full_v1"
            || r.scheduler != "constant"
            || r.sampler != "sequential_epoch_v1"
            || r.packing
            || r.truncation
            || !matches!(
                self.source_authorization.as_str(),
                "owned_software_fixture" | "qwen3_0_6b_c1899de_student_training_v1"
            )
            || self.upstream_lineage != "unknown"
            || self.checkpoint_kind != "full_inference"
        {
            return Err("unsupported full-SFT checkpoint recipe or declaration");
        }
        if r.dependencies.is_empty()
            || r.dependencies.len() > 128
            || r.runtime.len() != 4
            || ["python", "implementation", "system", "machine"]
                .iter()
                .any(|key| !r.runtime.contains_key(*key))
            || r.dependencies.iter().chain(&r.runtime).any(|(key, value)| {
                key.is_empty()
                    || value.is_empty()
                    || key.len() > 128
                    || value.len() > 128
                    || key.chars().chain(value.chars()).any(char::is_control)
            })
        {
            return Err("invalid checkpoint runtime or dependency declarations");
        }
        for model in [&self.initial_model, &self.checkpoint_model] {
            if !hash(&model.tensor_content_id)
                || !(1..=800_000_000).contains(&model.parameter_count)
                || !(1..=1024).contains(&model.tensor_count)
            {
                return Err("invalid checkpoint model summary");
            }
        }
        if self.initial_model.parameter_count != self.checkpoint_model.parameter_count
            || self.initial_model.tensor_count != self.checkpoint_model.tensor_count
            || self.initial_model.tensor_content_id == self.checkpoint_model.tensor_content_id
        {
            return Err("completed training requires changed full parameter content");
        }
        let o = &self.observations;
        if !o.parameter_content_changed
            || o.optimizer_updates != u64::from(r.max_steps)
            || o.microbatches.is_empty()
            || o.microbatches.len() > 256
            || o.successful_microbatches != o.microbatches.len() as u64
            || o.consumed_examples == 0
            || o.consumed_examples > 2048
            || o.shifted_supervised_tokens == 0
            || o.shifted_supervised_tokens > 4_194_304
        {
            return Err("invalid completed training accounting");
        }
        let mut rows = 0_u64;
        let mut tokens = 0_u64;
        for batch in &o.microbatches {
            if batch.update == 0
                || batch.update > r.max_steps
                || batch.example_ids.is_empty()
                || batch.example_ids.len() > r.batch_size as usize
                || batch.example_ids.iter().any(|id| !hash(id))
                || batch.input_tokens == 0
                || batch.input_tokens > u64::from(r.batch_size) * u64::from(r.max_sequence_length)
                || batch.shifted_supervised_tokens == 0
                || batch.shifted_supervised_tokens > batch.input_tokens
            {
                return Err("invalid completed training microbatch");
            }
            rows += batch.example_ids.len() as u64;
            tokens += batch.shifted_supervised_tokens;
        }
        if rows != o.consumed_examples || tokens != o.shifted_supervised_tokens {
            return Err("completed training totals differ from microbatch observations");
        }
        if self.files.len() != CHECKPOINT_FILES.len() {
            return Err("incomplete checkpoint file inventory");
        }
        let mut total = 0_u64;
        for (file, expected) in self.files.iter().zip(CHECKPOINT_FILES) {
            let bound = if expected.ends_with(".json") {
                MAX_CHECKPOINT_CONFIG_BYTES as u64
            } else if expected.ends_with(".safetensors") {
                MAX_CHECKPOINT_WEIGHT_BYTES
            } else {
                crate::MAX_PREPARED_SFT_BYTES as u64
            };
            if file.path != expected
                || file.byte_length == 0
                || file.byte_length > bound
                || !hash(&file.blake3)
            {
                return Err("invalid or oversized checkpoint captured file");
            }
            total += file.byte_length;
        }
        if total + MAX_CHECKPOINT_MANIFEST_BYTES as u64 + 48 > MAX_CHECKPOINT_BYTES {
            return Err("complete checkpoint exceeds its byte bound");
        }
        Ok(())
    }
}
fn hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
