//! Stream safe tensors, validate all finite values, and hash logical float32 content.
use crate::{Result, artifact::integrity};
use gw_schema::{
    CheckpointModelConfig, CheckpointModelSummary, CheckpointTensorDtype,
    MAX_CHECKPOINT_TENSOR_HEADER_BYTES, parse_checkpoint_tensor_header,
};
use serde::Serialize;
use std::{collections::BTreeMap, io::Read};

#[derive(Clone, PartialEq, Eq, Serialize)]
struct Content {
    shape: Vec<u64>,
    f32_blake3: String,
}

pub(crate) fn measure(
    reader: &mut impl Read,
    length: u64,
    config: &CheckpointModelConfig,
    require_f32: bool,
) -> Result<CheckpointModelSummary> {
    let mut prefix = [0; 8];
    reader.read_exact(&mut prefix)?;
    let header_length = u64::from_le_bytes(prefix);
    if header_length == 0
        || header_length > MAX_CHECKPOINT_TENSOR_HEADER_BYTES as u64
        || header_length
            .checked_add(8)
            .is_none_or(|value| value > length)
    {
        return Err(integrity("invalid checkpoint tensor header length"));
    }
    let mut header = vec![0; header_length as usize];
    reader.read_exact(&mut header)?;
    let tensors = parse_checkpoint_tensor_header(&header, length - 8 - header_length, config)
        .map_err(integrity)?;
    let mut content = BTreeMap::new();
    let mut buffer = vec![0; 65_536];
    let mut normalized = Vec::with_capacity(131_072);
    for tensor in tensors {
        if require_f32 && tensor.dtype != CheckpointTensorDtype::F32 {
            return Err(integrity(
                "completed checkpoint must contain full float32 weights",
            ));
        }
        let mut hasher = blake3::Hasher::new();
        let mut remaining = tensor.byte_length;
        while remaining > 0 {
            let count = remaining.min(buffer.len() as u64) as usize;
            reader.read_exact(&mut buffer[..count])?;
            normalized.clear();
            for bytes in buffer[..count].chunks_exact(tensor.dtype.width() as usize) {
                let bits = if tensor.dtype == CheckpointTensorDtype::F32 {
                    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                } else {
                    u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16
                };
                if bits & 0x7f80_0000 == 0x7f80_0000 {
                    return Err(integrity("checkpoint contains nonfinite parameter values"));
                }
                normalized.extend_from_slice(&bits.to_le_bytes());
            }
            hasher.update(&normalized);
            remaining -= count as u64;
        }
        content.insert(
            tensor.name,
            Content {
                shape: tensor.shape,
                f32_blake3: hasher.finalize().to_hex().to_string(),
            },
        );
    }
    if config.tie_word_embeddings {
        let embedding = content
            .get("model.embed_tokens.weight")
            .ok_or_else(|| integrity("missing embedding"))?
            .clone();
        if content
            .get("lm_head.weight")
            .is_some_and(|head| *head != embedding)
        {
            return Err(integrity("tied checkpoint head differs from embedding"));
        }
        content.insert("lm_head.weight".into(), embedding);
    }
    // Value recursively orders object keys, matching the language-neutral canonical JSON.
    let bytes = serde_json::to_vec(&serde_json::to_value(&content)?)?;
    let mut hash = blake3::Hasher::new_derive_key("ghostwriter.checkpoint-tensors.v1");
    hash.update(&bytes);
    Ok(CheckpointModelSummary {
        tensor_content_id: hash.finalize().to_hex().to_string(),
        parameter_count: config.parameter_count(),
        tensor_count: content.len() as u64,
    })
}
