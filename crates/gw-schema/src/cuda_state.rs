//! Pure typed state declarations for BF16 frozen parameters, FP32 adapters and all buffers.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Complete dense logical tensor declaration, including alias and buffer persistence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CudaTensor {
    /// Every dimension, with bounded checked multiplication.
    pub shape: Vec<u64>,
    /// Explicit scalar storage encoding.
    pub dtype: String,
    /// Hash of the little-endian dense raw tensor bytes.
    pub blake3: String,
    /// Lexicographically first name sharing this tensor storage.
    pub alias: String,
    /// Whether this tensor participates in the model's persistent state dictionary.
    pub persistent: bool,
}
/// One independently committed parameter or buffer population.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CudaState {
    /// Exactly version one of the separate typed state domain.
    pub version: u32,
    /// Domain-separated hash of the complete tensor map; native readers recompute it.
    pub state_id: String,
    /// Every logical tensor, including aliases.
    pub tensors: BTreeMap<String, CudaTensor>,
}
/// Three disjoint runtime state populations; original approved input bytes are bound separately.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CudaStateSet {
    /// All frozen parameters, including tied names and frozen modality towers.
    pub frozen: CudaState,
    /// Exact trainable q/v adapter parameters.
    pub adapters: CudaState,
    /// All persistent and nonpersistent registered buffers.
    pub buffers: CudaState,
}
impl CudaStateSet {
    /// Validate typed, bounded declarations without filesystem access or model execution.
    ///
    /// # Errors
    /// Rejects unsupported versions, shape overflow, precision or alias inconsistencies.
    pub fn validate(&self) -> Result<(), &'static str> {
        for (kind, state) in [
            ("frozen", &self.frozen),
            ("adapters", &self.adapters),
            ("buffers", &self.buffers),
        ] {
            if state.version != 1
                || !hash(&state.state_id)
                || state.tensors.is_empty()
                || state.tensors.len() > 4096
            {
                return Err("invalid CUDA state population");
            }
            for (name, tensor) in &state.tensors {
                let elements = tensor.shape.iter().try_fold(1u64, |n, d| n.checked_mul(*d));
                if name.is_empty()
                    || name.len() > 256
                    || name
                        .chars()
                        .any(|c| c.is_control() || c == '/' || c == '\\')
                    || tensor.shape.len() > 8
                    || elements.is_none_or(|n| n > 6_000_000_000)
                    || !hash(&tensor.blake3)
                    || !match kind {
                        "frozen" => tensor.dtype == "bfloat16" && tensor.persistent,
                        "adapters" => tensor.dtype == "float32" && tensor.persistent,
                        _ => ["float32", "int64", "int32", "bool"].contains(&tensor.dtype.as_str()),
                    }
                    || tensor.alias > *name
                    || state.tensors.get(&tensor.alias).is_none_or(|other| {
                        other.alias != tensor.alias
                            || other.shape != tensor.shape
                            || other.dtype != tensor.dtype
                            || other.blake3 != tensor.blake3
                    })
                {
                    return Err("invalid CUDA tensor declaration");
                }
            }
        }
        Ok(())
    }
}
fn hash(value: &str) -> bool {
    crate::coding_value::coding_hash_valid(value)
}
