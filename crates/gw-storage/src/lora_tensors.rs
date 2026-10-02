//! Bounded Gemma tensor measurement, including exact official clipping-buffer sentinels.
use crate::{Result, artifact::integrity};
use gw_schema::{
    CheckpointModelSummary, CheckpointTensorDtype, GEMMA_CLIPS, GEMMA_EMBEDDING, GEMMA_HEAD,
    GemmaBaseKind,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, io::Read};

#[derive(Clone, PartialEq, Eq, Serialize)]
pub(crate) struct Content {
    pub(crate) shape: Vec<u64>,
    pub(crate) f32_blake3: String,
}
pub(crate) type Contents = BTreeMap<String, Content>;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    dtype: CheckpointTensorDtype,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

pub(crate) fn measure(
    reader: &mut impl Read,
    length: u64,
    kind: GemmaBaseKind,
    base: bool,
) -> Result<(CheckpointModelSummary, Contents)> {
    let (summary, contents, _) = measure_inner(reader, length, kind, base, false)?;
    Ok((summary, contents))
}

pub(crate) fn measure_cuda_source(
    reader: &mut impl Read,
    length: u64,
    kind: GemmaBaseKind,
) -> Result<(
    CheckpointModelSummary,
    Contents,
    BTreeMap<String, gw_schema::CudaTensor>,
)> {
    measure_inner(reader, length, kind, true, true)
}

fn measure_inner(
    reader: &mut impl Read,
    length: u64,
    kind: GemmaBaseKind,
    base: bool,
    cuda: bool,
) -> Result<(
    CheckpointModelSummary,
    Contents,
    BTreeMap<String, gw_schema::CudaTensor>,
)> {
    let mut typed = BTreeMap::new();
    let mut prefix = [0; 8];
    reader.read_exact(&mut prefix)?;
    let header_length = u64::from_le_bytes(prefix);
    if header_length == 0 || header_length > 1_048_576 || header_length + 8 >= length {
        return Err(integrity("invalid Gemma tensor header length"));
    }
    let mut bytes = vec![0; header_length as usize];
    reader.read_exact(&mut bytes)?;
    let mut raw = gw_schema::strict_coding_json(&bytes)
        .map_err(integrity)?
        .as_object()
        .cloned()
        .ok_or_else(|| integrity("invalid Gemma tensor header"))?;
    if let Some(metadata) = raw.remove("__metadata__")
        && metadata != serde_json::json!({})
        && metadata != serde_json::json!({"format":"pt"})
    {
        return Err(integrity("unsupported Gemma tensor metadata"));
    }
    let mut expected = if base {
        kind.base_shapes()
    } else {
        kind.adapter_shapes()
    };
    if base && !raw.contains_key(GEMMA_HEAD) {
        expected.remove(GEMMA_HEAD);
    }
    if raw.len() != expected.len() {
        return Err(integrity("Gemma tensor inventory mismatch"));
    }
    let mut tensors = Vec::new();
    for (name, value) in raw {
        let t: Wire = serde_json::from_value(value)
            .map_err(|_| integrity("unsupported Gemma tensor fields"))?;
        if expected.get(&name) != Some(&t.shape) || (!base && t.dtype != CheckpointTensorDtype::F32)
        {
            return Err(integrity("Gemma tensor key, shape or dtype mismatch"));
        }
        let count = t.shape.iter().product::<u64>() * t.dtype.width();
        if t.data_offsets[0].checked_add(count) != Some(t.data_offsets[1]) {
            return Err(integrity("Gemma tensor range mismatch"));
        }
        tensors.push((name, t));
    }
    tensors.sort_by_key(|(_, t)| t.data_offsets[0]);
    let mut end = 0;
    let mut content = BTreeMap::new();
    let mut clips = BTreeMap::new();
    let mut buffer = vec![0; 65536];
    let mut normalized = Vec::with_capacity(131072);
    let mut typed_bytes = Vec::with_capacity(131072);
    for (name, t) in tensors {
        if t.data_offsets[0] != end || t.data_offsets[1] > length - 8 - header_length {
            return Err(integrity(
                "Gemma tensors have gaps, overlaps or excessive ranges",
            ));
        }
        // Exact known inventory and scalar shape were checked before this exception.
        let clip = base
            && t.shape.is_empty()
            && GEMMA_CLIPS.contains(&name.rsplit('.').next().unwrap_or(""));
        let mut remaining = t.data_offsets[1] - t.data_offsets[0];
        let mut hash = blake3::Hasher::new();
        let mut typed_hash = blake3::Hasher::new();
        let is_buffer = clip || name.ends_with(".layer_scalar");
        while remaining > 0 {
            let count = remaining.min(buffer.len() as u64) as usize;
            reader.read_exact(&mut buffer[..count])?;
            normalized.clear();
            typed_bytes.clear();
            for b in buffer[..count].chunks_exact(t.dtype.width() as usize) {
                let bits = if t.dtype == CheckpointTensorDtype::F32 {
                    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
                } else {
                    u32::from(u16::from_le_bytes([b[0], b[1]])) << 16
                };
                let nonfinite = bits & 0x7f80_0000 == 0x7f80_0000;
                let sentinel = if name.ends_with("_min") {
                    0xff80_0000
                } else {
                    0x7f80_0000
                };
                if nonfinite && (!clip || bits != sentinel) {
                    return Err(integrity(
                        "nonfinite Gemma weight or invalid clipping sentinel",
                    ));
                }
                if clip {
                    clips.insert(name.clone(), f32::from_bits(bits));
                }
                normalized.extend_from_slice(&bits.to_le_bytes());
                if cuda {
                    if is_buffer {
                        typed_bytes.extend_from_slice(&bits.to_le_bytes());
                    } else {
                        // IEEE round-to-nearest-even, matching the finite F32 -> BF16 cast.
                        typed_bytes.extend_from_slice(&bf16_bits(bits)?.to_le_bytes());
                    }
                }
            }
            hash.update(&normalized);
            if cuda {
                typed_hash.update(&typed_bytes);
            }
            remaining -= count as u64;
        }
        end = t.data_offsets[1];
        if cuda {
            typed.insert(
                name.clone(),
                gw_schema::CudaTensor {
                    shape: t.shape.clone(),
                    dtype: if is_buffer { "float32" } else { "bfloat16" }.into(),
                    blake3: typed_hash.finalize().to_hex().to_string(),
                    alias: name.clone(),
                    persistent: true,
                },
            );
        }
        content.insert(
            name,
            Content {
                shape: t.shape,
                f32_blake3: hash.finalize().to_hex().to_string(),
            },
        );
    }
    if end != length - 8 - header_length {
        return Err(integrity("Gemma tensor payload length mismatch"));
    }
    for (name, lower) in &clips {
        if let Some(prefix) = name.strip_suffix("_min")
            && clips
                .get(&format!("{prefix}_max"))
                .is_none_or(|upper| lower > upper)
        {
            return Err(integrity("Gemma clipping interval mismatch"));
        }
    }
    if base {
        let embedding = content
            .get(GEMMA_EMBEDDING)
            .ok_or_else(|| integrity("missing Gemma embedding"))?
            .clone();
        if content
            .get(GEMMA_HEAD)
            .is_some_and(|head| *head != embedding)
        {
            return Err(integrity("Gemma tied head mismatch"));
        }
        content.insert(GEMMA_HEAD.into(), embedding);
        if cuda {
            let mut tied = typed
                .get(GEMMA_EMBEDDING)
                .ok_or_else(|| integrity("missing typed Gemma embedding"))?
                .clone();
            tied.alias = GEMMA_HEAD.into();
            typed.insert(GEMMA_EMBEDDING.into(), tied.clone());
            typed.insert(GEMMA_HEAD.into(), tied);
        }
    }
    let bytes = serde_json::to_vec(&serde_json::to_value(&content)?)?;
    let mut hash = blake3::Hasher::new_derive_key("ghostwriter.gemma-lora-tensors.v1");
    hash.update(&bytes);
    Ok((
        CheckpointModelSummary {
            tensor_content_id: hash.finalize().to_hex().to_string(),
            parameter_count: if base {
                kind.parameter_count()
            } else {
                kind.adapter_shapes()
                    .values()
                    .map(|s| s.iter().product::<u64>())
                    .sum()
            },
            tensor_count: content.len() as u64,
        },
        content,
        typed,
    ))
}

fn bf16_bits(bits: u32) -> Result<u16> {
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    let output = (rounded >> 16) as u16;
    if bits & 0x7f80_0000 == 0x7f80_0000 || output & 0x7f80 == 0x7f80 {
        return Err(integrity(
            "source parameter cannot materialize as finite BF16",
        ));
    }
    Ok(output)
}

#[cfg(test)]
#[path = "lora_tensor_tests.rs"]
mod tests;
