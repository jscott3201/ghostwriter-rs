//! Independent materialized-state bindings; runtime-generated buffer contents remain declared.
use crate::{Result, artifact::integrity, lora_tensors::Contents};
use gw_schema::{CudaLoraCheckpointManifest, CudaTensor, GemmaBaseKind};
use std::collections::BTreeMap;

pub(crate) fn validate(
    manifest: &CudaLoraCheckpointManifest,
    source: &BTreeMap<String, CudaTensor>,
    contents: &[Contents],
    kind: GemmaBaseKind,
) -> Result<()> {
    for state in [&manifest.initial_state, &manifest.final_state] {
        for (kind, population) in [
            ("frozen", &state.frozen),
            ("adapters", &state.adapters),
            ("buffers", &state.buffers),
        ] {
            let bytes = serde_json::to_vec(&serde_json::to_value(&population.tensors)?)?;
            let mut hash =
                blake3::Hasher::new_derive_key(&format!("ghostwriter.gemma-cuda-{kind}.v1"));
            hash.update(&bytes);
            if hash.finalize().to_hex().as_str() != population.state_id {
                return Err(integrity("CUDA typed state identity mismatch"));
            }
        }
        let frozen: BTreeMap<_, _> = source
            .iter()
            .filter(|(_, t)| t.dtype == "bfloat16")
            .map(|(n, t)| (n.clone(), t.clone()))
            .collect();
        if state.frozen.tensors != frozen {
            return Err(integrity(
                "CUDA frozen state differs from source-derived BF16 conversion",
            ));
        }
        let persistent: BTreeMap<_, _> = source
            .iter()
            .filter(|(_, t)| t.dtype == "float32")
            .map(|(n, t)| (n.clone(), t.clone()))
            .collect();
        let actual: BTreeMap<_, _> = state
            .buffers
            .tensors
            .iter()
            .filter(|(_, t)| t.persistent)
            .map(|(n, t)| (n.clone(), t.clone()))
            .collect();
        if actual != persistent {
            return Err(integrity(
                "CUDA persistent buffers differ from original FP32 source",
            ));
        }
        let expected_buffers = nonpersistent_shapes(kind);
        let actual_buffers: BTreeMap<_, _> = state
            .buffers
            .tensors
            .iter()
            .filter(|(_, t)| !t.persistent)
            .map(|(name, t)| (name.clone(), t.shape.clone()))
            .collect();
        if actual_buffers != expected_buffers
            || state
                .buffers
                .tensors
                .iter()
                .any(|(name, t)| !t.persistent && (t.dtype != "float32" || t.alias != *name))
        {
            return Err(integrity(
                "CUDA nonpersistent buffer inventory differs from the complete known architecture",
            ));
        }
        if state
            .buffers
            .tensors
            .keys()
            .any(|name| state.frozen.tensors.contains_key(name))
        {
            return Err(integrity("CUDA buffer and parameter populations overlap"));
        }
    }
    for (state, content) in [
        (&manifest.final_state, &contents[1]),
        (&manifest.initial_state, &contents[2]),
    ] {
        let expected: BTreeMap<_, _> = content
            .iter()
            .map(|(name, t)| {
                let name = name.replace(".weight", ".default.weight");
                (
                    name.clone(),
                    CudaTensor {
                        shape: t.shape.clone(),
                        dtype: "float32".into(),
                        blake3: t.f32_blake3.clone(),
                        alias: name,
                        persistent: true,
                    },
                )
            })
            .collect();
        if state.adapters.tensors != expected {
            return Err(integrity(
                "CUDA adapter state differs from independently measured captured tensors",
            ));
        }
    }
    Ok(())
}

fn nonpersistent_shapes(kind: GemmaBaseKind) -> BTreeMap<String, Vec<u64>> {
    let release = kind == GemmaBaseKind::Release;
    let mut shapes = BTreeMap::new();
    for name in ["embed_tokens", "embed_tokens_per_layer"] {
        shapes.insert(format!("model.language_model.{name}.embed_scale"), vec![]);
    }
    for (layer, width) in [
        ("sliding_attention", if release { 128 } else { 4 }),
        ("full_attention", if release { 256 } else { 8 }),
    ] {
        for suffix in ["inv_freq", "original_inv_freq"] {
            shapes.insert(
                format!("model.language_model.rotary_emb.{layer}_{suffix}"),
                vec![width],
            );
        }
    }
    if release {
        for suffix in ["inv_freq", "original_inv_freq"] {
            shapes.insert(
                format!("model.vision_tower.encoder.rotary_emb.{suffix}"),
                vec![16],
            );
        }
        shapes.insert(
            "model.audio_tower.rel_pos_enc.inv_timescales".into(),
            vec![1, 1, 512],
        );
        for index in 0..12 {
            shapes.insert(
                format!("model.audio_tower.layers.{index}.self_attn.softcap"),
                vec![],
            );
        }
    }
    shapes
}
