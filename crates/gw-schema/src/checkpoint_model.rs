//! Narrow, pure Qwen3 configuration and safetensors shape validation for full CPU checkpoints.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Maximum captured safetensors JSON header; tensor payloads are streamed separately.
pub const MAX_CHECKPOINT_TENSOR_HEADER_BYTES: usize = 1024 * 1024;

/// Supported Qwen3 shape/configuration. Unknown execution and remote-code options are rejected.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointModelConfig {
    /// Must identify the known local Qwen3 causal model implementation.
    pub architectures: Vec<String>,
    /// Must be `qwen3`.
    pub model_type: String,
    /// Token embedding/head rows, including any reserved vocabulary rows.
    pub vocab_size: u64,
    /// Decoder residual width.
    pub hidden_size: u64,
    /// Feed-forward projection width.
    pub intermediate_size: u64,
    /// Number of complete decoder layers.
    pub num_hidden_layers: u64,
    /// Number of query heads.
    pub num_attention_heads: u64,
    /// Number of key/value heads.
    pub num_key_value_heads: u64,
    /// Even rotary head width, independently declared by Qwen3.
    pub head_dim: u64,
    /// Maximum supported full sequence length.
    pub max_position_embeddings: u64,
    /// Only the Qwen3 SiLU feed-forward activation is supported.
    pub hidden_act: String,
    /// Positive finite RMS normalization epsilon.
    pub rms_norm_eps: f64,
    /// Positive finite rotary embedding base.
    pub rope_theta: f64,
    /// Must be zero for this deterministic CPU recipe.
    pub attention_dropout: f64,
    /// Attention projection biases are unsupported.
    pub attention_bias: bool,
    /// Whether the output head shares the input embedding parameters.
    pub tie_word_embeddings: bool,
    /// Inference cache default; training explicitly disables cache in every forward call.
    pub use_cache: bool,
    /// Sliding-window attention is unsupported.
    pub use_sliding_window: bool,
    /// Must be null or omitted when sliding windows are disabled.
    #[serde(default)]
    pub sliding_window: Option<u64>,
    /// Optional explicit all-full-attention layer list.
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    /// Unsupported rotary scaling must be absent or null.
    #[serde(default)]
    pub rope_scaling: Option<BTreeMap<String, serde_json::Value>>,
    /// Explicit beginning token or none.
    pub bos_token_id: Option<u64>,
    /// Explicit end token from the pinned tokenizer policy.
    pub eos_token_id: u64,
    /// Explicit right-padding token from the pinned tokenizer policy.
    #[serde(default)]
    pub pad_token_id: Option<u64>,
    /// Optional declared input storage precision, independently measured from tensor headers.
    #[serde(default)]
    pub torch_dtype: Option<String>,
    /// Newer spelling of the same optional declaration.
    #[serde(default)]
    pub dtype: Option<String>,
    /// Inert initialization default; all captured model parameters must load strictly.
    #[serde(default)]
    pub initializer_range: Option<f64>,
    /// Inert disabled-sliding-window setting.
    #[serde(default)]
    pub max_window_layers: Option<u64>,
    /// Optional producer metadata, not runtime qualification.
    #[serde(default)]
    pub transformers_version: Option<String>,
}
impl CheckpointModelConfig {
    /// Parse a bounded known configuration without importing Python or reading weights.
    ///
    /// # Errors
    /// Rejects unknown/duplicate fields, unsupported options, nonfinite numbers or excessive shapes.
    pub fn from_json(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() > crate::MAX_CHECKPOINT_CONFIG_BYTES {
            return Err("checkpoint model configuration exceeds its byte bound");
        }
        let config: Self = serde_json::from_slice(bytes)
            .map_err(|_| "invalid or unsupported checkpoint model configuration")?;
        config.validate()?;
        Ok(config)
    }
    /// Validate this exact bounded Qwen3 subset.
    ///
    /// # Errors
    /// Rejects unsupported configurations or more than 800 million trainable parameters.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.architectures != ["Qwen3ForCausalLM"]
            || self.model_type != "qwen3"
            || !(151_669..=152_064).contains(&self.vocab_size)
            || !(8..=4096).contains(&self.hidden_size)
            || !self.hidden_size.is_multiple_of(8)
            || !(8..=24_576).contains(&self.intermediate_size)
            || !(1..=32).contains(&self.num_hidden_layers)
            || !(1..=32).contains(&self.num_attention_heads)
            || self.num_key_value_heads == 0
            || self.num_key_value_heads > self.num_attention_heads
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
            || !(2..=256).contains(&self.head_dim)
            || !self.head_dim.is_multiple_of(2)
            || !(2..=131_072).contains(&self.max_position_embeddings)
            || self.hidden_act != "silu"
            || !self.rms_norm_eps.is_finite()
            || !(0.0..=0.1).contains(&self.rms_norm_eps)
            || self.rms_norm_eps == 0.0
            || !self.rope_theta.is_finite()
            || !(1.0..=1_000_000_000.0).contains(&self.rope_theta)
            || self.attention_dropout != 0.0
            || self.attention_bias
            || self.use_sliding_window
            || self.sliding_window.is_some()
            || self.rope_scaling.is_some()
            || self.layer_types.as_ref().is_some_and(|layers| {
                layers.len() as u64 != self.num_hidden_layers
                    || layers.iter().any(|kind| kind != "full_attention")
            })
            || self.bos_token_id.is_some_and(|id| id >= self.vocab_size)
            || self.eos_token_id >= self.vocab_size
            || self.pad_token_id.is_some_and(|id| id >= self.vocab_size)
            || [&self.torch_dtype, &self.dtype]
                .into_iter()
                .flatten()
                .any(|kind| !matches!(kind.as_str(), "float32" | "bfloat16"))
            || self
                .torch_dtype
                .as_ref()
                .zip(self.dtype.as_ref())
                .is_some_and(|(a, b)| a != b)
            || self
                .initializer_range
                .is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v))
            || self.max_window_layers.is_some_and(|n| n > 64)
            || self.transformers_version.as_ref().is_some_and(|value| {
                value.is_empty() || value.len() > 64 || value.chars().any(char::is_control)
            })
        {
            return Err("unsupported checkpoint Qwen3 configuration");
        }
        if self.parameter_count() > 800_000_000 {
            return Err("checkpoint model exceeds the parameter bound");
        }
        Ok(())
    }
    /// Complete logical state-dictionary shapes, including the output head alias when tied.
    #[must_use]
    pub fn tensor_shapes(&self) -> BTreeMap<String, Vec<u64>> {
        let mut values = BTreeMap::from([
            (
                "model.embed_tokens.weight".into(),
                vec![self.vocab_size, self.hidden_size],
            ),
            ("model.norm.weight".into(), vec![self.hidden_size]),
            (
                "lm_head.weight".into(),
                vec![self.vocab_size, self.hidden_size],
            ),
        ]);
        for layer in 0..self.num_hidden_layers {
            let prefix = format!("model.layers.{layer}.");
            for (name, shape) in [
                ("input_layernorm.weight", vec![self.hidden_size]),
                ("post_attention_layernorm.weight", vec![self.hidden_size]),
                (
                    "self_attn.q_proj.weight",
                    vec![self.num_attention_heads * self.head_dim, self.hidden_size],
                ),
                (
                    "self_attn.k_proj.weight",
                    vec![self.num_key_value_heads * self.head_dim, self.hidden_size],
                ),
                (
                    "self_attn.v_proj.weight",
                    vec![self.num_key_value_heads * self.head_dim, self.hidden_size],
                ),
                (
                    "self_attn.o_proj.weight",
                    vec![self.hidden_size, self.num_attention_heads * self.head_dim],
                ),
                ("self_attn.q_norm.weight", vec![self.head_dim]),
                ("self_attn.k_norm.weight", vec![self.head_dim]),
                (
                    "mlp.gate_proj.weight",
                    vec![self.intermediate_size, self.hidden_size],
                ),
                (
                    "mlp.up_proj.weight",
                    vec![self.intermediate_size, self.hidden_size],
                ),
                (
                    "mlp.down_proj.weight",
                    vec![self.hidden_size, self.intermediate_size],
                ),
            ] {
                values.insert(format!("{prefix}{name}"), shape);
            }
        }
        values
    }
    /// Distinct trainable element count for the supported full model.
    #[must_use]
    pub fn parameter_count(&self) -> u64 {
        self.tensor_shapes()
            .iter()
            .filter(|(name, _)| !self.tie_word_embeddings || *name != "lm_head.weight")
            .map(|(_, shape)| shape.iter().product::<u64>())
            .sum()
    }
    /// Compare inference architecture/configuration, permitting only storage dtype and version
    /// metadata to differ between initial and final snapshots.
    #[must_use]
    pub fn same_architecture(&self, other: &Self) -> bool {
        let normalize = |value: &Self| {
            let mut value = value.clone();
            value.torch_dtype = None;
            value.dtype = None;
            value.transformers_version = None;
            value
        };
        normalize(self) == normalize(other)
    }
}

/// Supported dense little-endian floating tensor encodings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckpointTensorDtype {
    /// IEEE binary32.
    F32,
    /// Upper 16 bits of IEEE binary32; conversion to float32 is exact.
    BF16,
}
impl CheckpointTensorDtype {
    /// Number of stored bytes per element.
    #[must_use]
    pub fn width(self) -> u64 {
        if self == Self::F32 { 4 } else { 2 }
    }
}
/// Validated tensor coordinate inside one captured safetensors file.
#[derive(Debug, Clone)]
pub struct CheckpointTensor {
    /// Complete expected Qwen3 state-dictionary name.
    pub name: String,
    /// Supported storage representation.
    pub dtype: CheckpointTensorDtype,
    /// Complete exact dimensions.
    pub shape: Vec<u64>,
    /// Start of this tensor in the file's data region.
    pub offset: u64,
    /// Exact tensor byte length.
    pub byte_length: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorWire {
    dtype: CheckpointTensorDtype,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

/// Validate exact safe tensor keys, dense nonoverlapping ranges, dtype and complete model shapes.
///
/// # Errors
/// Rejects duplicate/unknown tensor fields, gaps, overlaps, extra/missing parameters or bad bounds.
pub fn parse_checkpoint_tensor_header(
    header: &[u8],
    data_length: u64,
    config: &CheckpointModelConfig,
) -> Result<Vec<CheckpointTensor>, &'static str> {
    config.validate()?;
    if header.len() > MAX_CHECKPOINT_TENSOR_HEADER_BYTES {
        return Err("checkpoint tensor header exceeds bound");
    }
    let mut raw = crate::strict_coding_json(header)
        .map_err(|_| "invalid checkpoint tensor header JSON")?
        .as_object()
        .cloned()
        .ok_or("checkpoint tensor header must be an object")?;
    if let Some(metadata) = raw.remove("__metadata__")
        && !metadata.as_object().is_some_and(|values| {
            values.len() <= 8
                && values.iter().all(|(key, value)| {
                    key.len() <= 128 && value.as_str().is_some_and(|text| text.len() <= 256)
                })
        })
    {
        return Err("unsupported checkpoint tensor metadata");
    }
    let mut expected = config.tensor_shapes();
    if config.tie_word_embeddings && !raw.contains_key("lm_head.weight") {
        expected.remove("lm_head.weight");
    }
    if raw.len() != expected.len() {
        return Err("checkpoint tensor inventory is incomplete or excessive");
    }
    let mut tensors = Vec::new();
    for (name, value) in raw {
        let tensor: TensorWire =
            serde_json::from_value(value).map_err(|_| "invalid checkpoint tensor fields")?;
        if expected.get(&name) != Some(&tensor.shape) {
            return Err("checkpoint tensor shape or name mismatch");
        }
        let byte_length = tensor.shape.iter().product::<u64>() * tensor.dtype.width();
        if tensor.data_offsets[0].checked_add(byte_length) != Some(tensor.data_offsets[1]) {
            return Err("checkpoint tensor byte range disagrees with its shape");
        }
        tensors.push(CheckpointTensor {
            name,
            dtype: tensor.dtype,
            shape: tensor.shape,
            offset: tensor.data_offsets[0],
            byte_length,
        });
    }
    tensors.sort_by_key(|tensor| tensor.offset);
    let mut end = 0;
    for tensor in &tensors {
        if tensor.offset != end {
            return Err("checkpoint tensors contain gaps or overlaps");
        }
        end += tensor.byte_length;
    }
    if end != data_length {
        return Err("checkpoint tensor payload length mismatch");
    }
    Ok(tensors)
}
