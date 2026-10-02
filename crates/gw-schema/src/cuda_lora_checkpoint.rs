//! Separate CUDA checkpoint declarations. CPU v1 framing and interpretation stay unchanged.
use crate::{
    CheckpointFile, CheckpointModelSummary, CudaStateSet, GEMMA_FIXTURE, GEMMA_RELEASE,
    LoraObservations,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Separate completion framing domain; full-SFT readers reject this magic.
pub const CUDA_LORA_MAGIC: &[u8; 8] = b"GWCUDA01";
/// Digest domain of the length-framed manifest and complete captured files.
pub const CUDA_LORA_HASH_DOMAIN: &str = "ghostwriter.completed-gemma-cuda-lora.v1";
/// Complete streamed storage bound; no whole-base allocation is required by native inspection.
pub const MAX_CUDA_LORA_BYTES: u64 = 12 * 1024 * 1024 * 1024;
/// Fixed ordered inventory. The captured base appears exactly once.
pub const CUDA_LORA_FILES: [&str; 7] = [
    "base/config.json",
    "base/model.safetensors",
    "final/config.json",
    "final/adapter_model.safetensors",
    "initial/config.json",
    "initial/adapter_model.safetensors",
    "prepared.gwsft",
];

/// Declared CUDA recipe, separate from the preparation recipe and input build identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CudaLoraRecipe {
    /// Supported recipe version.
    pub version: u32,
    /// Recursive LoRA source plus consumed shared capture and publication helper identity.
    pub training_source_sha256: String,
    /// Unchanged preparation package identity already bound by the prepared build.
    pub preparation_source_sha256: String,
    /// Exact qualified Python dependency pins.
    pub dependencies: BTreeMap<String, String>,
    /// Producer Python/platform declarations, without local paths.
    pub runtime: BTreeMap<String, String>,
    /// Exactly CUDA execution.
    pub device: String,
    /// Frozen base and trainable adapters use BF16 and float32 respectively; quantization is unsupported.
    pub precision: String,
    /// One trainer process.
    pub processes: u32,
    /// Bounded CPU intra-operation thread count.
    pub threads: u32,
    /// Two to 32 actual successful optimizer updates required before publication.
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

/// Complete narrow LoRA artifact; it conveys no authority to claim fresh training.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CudaLoraCheckpointManifest {
    /// Exactly version one of the separate CUDA Gemma LoRA contract.
    pub version: u32,
    /// Original preparation identity.
    pub prepared_build_id: String,
    /// Bounded CUDA trainer policy; optimizer must be adamw_cuda_lora_v1.
    pub recipe: CudaLoraRecipe,
    /// Complete normalized frozen base state, including persistent buffers and aliases.
    pub base_model: CheckpointModelSummary,
    /// Initial safe adapter state.
    pub initial_adapter: CheckpointModelSummary,
    /// Final safe adapter state.
    pub final_adapter: CheckpointModelSummary,
    /// Declared historical operations and exact trainables.
    pub observations: LoraObservations,
    /// Exact release approval or explicit owned random fixture.
    pub source_authorization: String,
    /// Publisher parent remains unknown, without an invented immutable revision.
    pub upstream_lineage: String,
    /// Exactly gemma_cuda_qv_lora_v1.
    pub checkpoint_kind: String,
    /// Measured parameter, adapter and all-buffer declarations before optimization.
    pub initial_state: CudaStateSet,
    /// Complete state after optimization; frozen parameters and all buffers must match.
    pub final_state: CudaStateSet,
    /// Actual operator output dtype counts, declared when read from saved bytes.
    pub precision_operations: BTreeMap<String, u64>,
    /// Strict policy for an independently allocated model's probe; no CPU tolerance is inherited.
    pub probe_policy: String,
    /// Observed CUDA implementation/device facts, not hardware attestation.
    pub cuda_runtime: BTreeMap<String, String>,
    /// Additional NVIDIA/Triton exact metadata versions for this CUDA policy.
    pub cuda_dependencies: BTreeMap<String, String>,
    /// Complete fixed inventory, independently hashed by readers.
    pub files: Vec<CheckpointFile>,
}
impl CudaLoraCheckpointManifest {
    /// Strict bounded decode; duplicate, missing and unsupported fields are rejected.
    ///
    /// # Errors
    /// Rejects invalid JSON, unsafe inventory or inconsistent declarations.
    pub fn from_json(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() > crate::MAX_CHECKPOINT_MANIFEST_BYTES {
            return Err("LoRA manifest exceeds bound");
        }
        let raw = crate::strict_coding_json(bytes).map_err(|_| "invalid LoRA JSON")?;
        let value: Self = serde_json::from_value(raw.clone()).map_err(|_| "invalid LoRA fields")?;
        if serde_json::to_value(&value).map_err(|_| "invalid LoRA manifest")? != raw {
            return Err("omitted LoRA fields");
        }
        value.validate()?;
        Ok(value)
    }
    /// Validate bounded typed declarations, without model execution or filesystem access.
    ///
    /// # Errors
    /// Rejects unsupported recipes, false accounting relationships or unsafe file populations.
    pub fn validate(&self) -> Result<(), &'static str> {
        self.initial_state.validate()?;
        self.final_state.validate()?;
        if self.initial_state.frozen != self.final_state.frozen
            || self.initial_state.buffers != self.final_state.buffers
            || self.initial_state.adapters.state_id == self.final_state.adapters.state_id
            || self.probe_policy != "cuda_bf16_exact_probe_v1"
            || self.precision_operations.is_empty()
            || self.precision_operations.len() > 512
            || self
                .precision_operations
                .iter()
                .any(|(k, v)| k.len() > 256 || *v == 0)
            || self.cuda_runtime.len() != 5
            || [
                "torch_cuda",
                "torch_build",
                "device_name",
                "capability",
                "installed_wheel_records_sha256",
            ]
            .iter()
            .any(|k| !self.cuda_runtime.contains_key(*k))
            || self.cuda_runtime.get("torch_cuda").map(String::as_str) != Some("12.8")
            || !matches!(
                self.cuda_runtime.get("torch_build").map(String::as_str),
                Some("2.8.0" | "2.8.0+cu128")
            )
            || !self
                .cuda_runtime
                .get("installed_wheel_records_sha256")
                .is_some_and(|v| hash(v))
            || self.cuda_dependencies.len() != 15
            || self.cuda_dependencies.iter().any(|(k, v)| {
                k.is_empty()
                    || k.len() > 128
                    || v.is_empty()
                    || v.len() > 128
                    || k.chars().chain(v.chars()).any(char::is_control)
            })
            || self
                .cuda_runtime
                .values()
                .any(|v| v.is_empty() || v.len() > 128 || v.chars().any(char::is_control))
        {
            return Err("invalid CUDA state or execution declarations");
        }
        for (label, operation) in [
            ("norm:", "pow"),
            ("rope:", "bmm"),
            ("softmax:", "softmax"),
            ("loss:", "nll_loss"),
        ] {
            if !self.precision_operations.keys().any(|k| {
                k.starts_with(label) && k.contains(operation) && k.ends_with(":torch.float32:cuda")
            }) {
                return Err("missing CUDA precision observations");
            }
        }
        if !self
            .precision_operations
            .keys()
            .any(|key| key.starts_with("matmul:") && key.ends_with(":torch.bfloat16:cuda"))
        {
            return Err("missing CUDA BF16 matrix multiplication observations");
        }
        let r = &self.recipe;
        if self.version != 1
            || r.version != 1
            || !hash(&self.prepared_build_id)
            || !hash(&r.training_source_sha256)
            || !hash(&r.preparation_source_sha256)
            || r.device != "cuda:0"
            || r.precision != "bf16_frozen_fp32_adapter_buffers_v1"
            || r.processes != 1
            || r.threads != 1
            || !(2..=32).contains(&r.max_steps)
            || !(1..=8).contains(&r.batch_size)
            || !(1..=8).contains(&r.accumulation)
            || !(1..=10_000).contains(&r.learning_rate_millionths)
            || !(2..=2048).contains(&r.max_sequence_length)
            || r.seed != 0
            || r.optimizer != "adamw_cuda_lora_v1"
            || r.scheduler != "constant"
            || r.sampler != "sequential_epoch_v1"
            || r.packing
            || r.truncation
            || ![GEMMA_RELEASE, GEMMA_FIXTURE].contains(&self.source_authorization.as_str())
            || self.upstream_lineage != "unknown"
            || self.checkpoint_kind != "gemma_cuda_qv_lora_v1"
        {
            return Err("unsupported Gemma LoRA recipe");
        }
        if r.dependencies.is_empty()
            || r.dependencies.len() > 128
            || r.runtime.len() != 4
            || r.runtime.get("system").map(String::as_str) != Some("Linux")
            || r.runtime.get("machine").map(String::as_str) != Some("x86_64")
            || ["python", "implementation", "system", "machine"]
                .iter()
                .any(|key| !r.runtime.contains_key(*key))
            || r.dependencies.iter().chain(&r.runtime).any(|(k, v)| {
                k.is_empty()
                    || v.is_empty()
                    || k.len() > 128
                    || v.len() > 128
                    || k.chars()
                        .chain(v.chars())
                        .any(|c| c.is_control() || c == '/' || c == '\\')
            })
        {
            return Err("invalid LoRA dependency/runtime declarations");
        }
        for model in [&self.base_model, &self.initial_adapter, &self.final_adapter] {
            if !hash(&model.tensor_content_id)
                || !(1..=6_000_000_000).contains(&model.parameter_count)
                || !(1..=4096).contains(&model.tensor_count)
            {
                return Err("invalid LoRA tensor summary");
            }
        }
        if self.initial_adapter.parameter_count != self.final_adapter.parameter_count
            || self.initial_adapter.tensor_count != self.final_adapter.tensor_count
            || self.initial_adapter.tensor_content_id == self.final_adapter.tensor_content_id
        {
            return Err("LoRA requires changed adapter content");
        }
        let o = &self.observations;
        if !o.base_unchanged
            || !o.adapter_content_changed
            || o.optimizer_updates != u64::from(r.max_steps)
            || o.microbatches.is_empty()
            || o.microbatches.len() > 256
            || o.successful_microbatches != o.microbatches.len() as u64
            || o.successful_forwards != o.successful_microbatches
            || o.consumed_examples == 0
            || o.consumed_examples > 2048
            || o.shifted_supervised_tokens == 0
            || o.shifted_supervised_tokens > 4_194_304
            || o.trainables.is_empty()
            || o.trainables.len() > 100
        {
            return Err("invalid LoRA observations");
        }
        let mut rows = 0;
        let mut tokens = 0;
        for b in &o.microbatches {
            let loss =
                u64::from_str_radix(&b.loss_binary64, 16).map_err(|_| "invalid loss bits")?;
            if b.loss_binary64.len() != 16
                || !f64::from_bits(loss).is_finite()
                || b.update == 0
                || b.update > r.max_steps
                || b.example_ids.is_empty()
                || b.example_ids.len() > r.batch_size as usize
                || b.example_ids.iter().any(|id| !hash(id))
                || b.input_tokens == 0
                || b.input_tokens > u64::from(r.batch_size) * u64::from(r.max_sequence_length)
                || b.shifted_supervised_tokens == 0
                || b.shifted_supervised_tokens > b.input_tokens
                || b.finite_adapter_gradients != o.trainables.len() as u64
            {
                return Err("invalid LoRA microbatch");
            }
            rows += b.example_ids.len() as u64;
            tokens += b.shifted_supervised_tokens;
        }
        if rows != o.consumed_examples || tokens != o.shifted_supervised_tokens {
            return Err("LoRA totals mismatch");
        }
        if self.files.len() != CUDA_LORA_FILES.len() {
            return Err("incomplete LoRA files");
        }
        let mut total = 0;
        for (file, expected) in self.files.iter().zip(CUDA_LORA_FILES) {
            let bound = if expected == "base/model.safetensors" {
                11 * 1024 * 1024 * 1024
            } else if expected.ends_with(".json") {
                65_536
            } else if expected.ends_with(".safetensors") {
                64 * 1024 * 1024
            } else {
                crate::MAX_PREPARED_SFT_BYTES as u64
            };
            if file.path != expected
                || file.byte_length == 0
                || file.byte_length > bound
                || !hash(&file.blake3)
            {
                return Err("invalid LoRA file inventory");
            }
            total += file.byte_length;
        }
        if total + crate::MAX_CHECKPOINT_MANIFEST_BYTES as u64 + 48 > MAX_CUDA_LORA_BYTES {
            return Err("LoRA exceeds bound");
        }
        Ok(())
    }
}
fn hash(s: &str) -> bool {
    crate::coding_value::coding_hash_valid(s)
}
