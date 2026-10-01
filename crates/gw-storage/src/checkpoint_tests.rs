//! Native fixtures are built independently from constant Qwen3 tensors, never Python reports.
use super::*;
use serde_json::{Value, json};
use std::{collections::BTreeMap, io::Cursor, path::Path};

fn tensor_shapes() -> BTreeMap<String, Vec<u64>> {
    let mut shapes = BTreeMap::from([
        ("model.embed_tokens.weight".into(), vec![151_669, 8]),
        ("lm_head.weight".into(), vec![151_669, 8]),
        ("model.norm.weight".into(), vec![8]),
    ]);
    for (name, shape) in [
        ("input_layernorm.weight", vec![8]),
        ("post_attention_layernorm.weight", vec![8]),
        ("self_attn.q_proj.weight", vec![8, 8]),
        ("self_attn.k_proj.weight", vec![8, 8]),
        ("self_attn.v_proj.weight", vec![8, 8]),
        ("self_attn.o_proj.weight", vec![8, 8]),
        ("self_attn.q_norm.weight", vec![8]),
        ("self_attn.k_norm.weight", vec![8]),
        ("mlp.gate_proj.weight", vec![8, 8]),
        ("mlp.up_proj.weight", vec![8, 8]),
        ("mlp.down_proj.weight", vec![8, 8]),
    ] {
        shapes.insert(format!("model.layers.0.{name}"), shape);
    }
    shapes
}

fn tensors(bf16: bool, value: u32) -> (Vec<u8>, Value) {
    let mut data = Vec::new();
    let mut header = serde_json::Map::new();
    let mut content = serde_json::Map::new();
    for (name, shape) in tensor_shapes() {
        let count = shape.iter().product::<u64>() as usize;
        let normalized = value.to_le_bytes().repeat(count);
        content.insert(
            name.clone(),
            json!({"shape": shape,
            "f32_blake3": blake3::hash(&normalized).to_hex().to_string()}),
        );
        // Initial BF16 includes both tied names. Final F32 omits its output alias.
        if !bf16 && name == "lm_head.weight" {
            continue;
        }
        let bytes = if bf16 {
            ((value >> 16) as u16).to_le_bytes().repeat(count)
        } else {
            normalized
        };
        header.insert(
            name,
            json!({"dtype": if bf16 { "BF16" } else { "F32" },
            "shape": shape, "data_offsets": [data.len(), data.len() + bytes.len()]}),
        );
        data.extend(bytes);
    }
    let mut encoded = serde_json::to_vec(&header).unwrap();
    encoded.resize(encoded.len().next_multiple_of(8), b' ');
    let mut bytes = (encoded.len() as u64).to_le_bytes().to_vec();
    bytes.extend(encoded);
    bytes.extend(data);
    let mut digest = blake3::Hasher::new_derive_key("ghostwriter.checkpoint-tensors.v1");
    digest.update(&serde_json::to_vec(&content).unwrap());
    (
        bytes,
        json!({"tensor_content_id": digest.finalize().to_hex().to_string(),
        "parameter_count": 1_213_840, "tensor_count": 14}),
    )
}

fn fixture() -> (Value, BTreeMap<String, Vec<u8>>) {
    let prepared = std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../adapters/trl/tests/fixtures/prepared-all.gwsft"),
    )
    .unwrap();
    let parsed = verify_prepared_sft_snapshot(prepared.clone()).unwrap();
    let source_recipe = &parsed.payload().manifest.recipe;
    let (initial, initial_summary) = tensors(true, 0);
    let (checkpoint, checkpoint_summary) = tensors(false, 1_f32.to_bits());
    let initial_config = json!({"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3",
        "vocab_size":151669,"hidden_size":8,"intermediate_size":8,"num_hidden_layers":1,
        "num_attention_heads":1,"num_key_value_heads":1,"head_dim":8,"max_position_embeddings":2048,
        "hidden_act":"silu","rms_norm_eps":0.000001,"rope_theta":1000000,"attention_dropout":0,
        "attention_bias":false,"tie_word_embeddings":true,"use_cache":true,"use_sliding_window":false,
        "bos_token_id":null,"eos_token_id":151645,"pad_token_id":151643,"torch_dtype":"bfloat16"});
    let mut final_config = initial_config.clone();
    final_config["torch_dtype"] = "float32".into();
    let files = BTreeMap::from([
        (
            "checkpoint/config.json".into(),
            serde_json::to_vec(&final_config).unwrap(),
        ),
        ("checkpoint/model.safetensors".into(), checkpoint),
        (
            "initial/config.json".into(),
            serde_json::to_vec(&initial_config).unwrap(),
        ),
        ("initial/model.safetensors".into(), initial),
        ("prepared.gwsft".into(), prepared),
    ]);
    let mut batches = Vec::new();
    for update in 1..=3 {
        for chunk in parsed.payload().examples.chunks(3) {
            batches.push(json!({"update": update,
                "example_ids": chunk.iter().map(|e| &e.example_id).collect::<Vec<_>>(),
                "input_tokens": chunk.iter().map(|e| e.input_ids.len()).sum::<usize>(),
                "shifted_supervised_tokens": chunk.iter().map(|e| e.labels.iter().skip(1).filter(|l| **l != -100).count()).sum::<usize>()}));
        }
    }
    let tokens: u64 = batches
        .iter()
        .map(|b| b["shifted_supervised_tokens"].as_u64().unwrap())
        .sum();
    let manifest = json!({"version":1,"prepared_build_id": parsed.report().build_id,
        "recipe":{"version":1,"training_source_sha256":"1".repeat(64),
            "preparation_source_sha256":source_recipe.adapter_source_sha256,"dependencies":source_recipe.dependencies,
            "runtime":{"python":"3.12","implementation":"CPython","system":"test","machine":"test"},
            "device":"cpu","precision":"float32","processes":1,"threads":1,"max_steps":3,
            "batch_size":3,"accumulation":2,"learning_rate_millionths":100,"seed":0,"max_sequence_length":2048,
            "optimizer":"adamw_torch_full_v1","scheduler":"constant","sampler":"sequential_epoch_v1",
            "packing":false,"truncation":false},
        "initial_model":initial_summary,"checkpoint_model":checkpoint_summary,
        "observations":{"successful_microbatches":6,"consumed_examples":12,"shifted_supervised_tokens":tokens,
            "optimizer_updates":3,"microbatches":batches,"parameter_content_changed":true},
        "source_authorization":"owned_software_fixture","upstream_lineage":"unknown","checkpoint_kind":"full_inference",
        "files":[]});
    (manifest, files)
}

fn frame(mut manifest: Value, files: &BTreeMap<String, Vec<u8>>) -> Vec<u8> {
    manifest["files"] = files
        .iter()
        .map(|(path, data)| {
            json!({"path":path,"byte_length":data.len(),
        "blake3":blake3::hash(data).to_hex().to_string()})
        })
        .collect::<Vec<_>>()
        .into();
    let raw = serde_json::to_vec(&manifest).unwrap();
    let mut body = (raw.len() as u64).to_be_bytes().to_vec();
    body.extend(raw);
    for data in files.values() {
        body.extend(data);
    }
    let mut hash = blake3::Hasher::new_derive_key(CHECKPOINT_HASH_DOMAIN);
    hash.update(&body);
    let mut output = CHECKPOINT_MAGIC.to_vec();
    output.extend(hash.finalize().as_bytes());
    output.extend(body);
    output
}

#[test]
fn native_complete_inspection_measures_aliases_bf16_and_keeps_history_declared() {
    let (manifest, files) = fixture();
    let bytes = frame(manifest, &files);
    let report = verify_training_checkpoint(Cursor::new(&bytes)).unwrap();
    assert_eq!(report.byte_length, bytes.len() as u64);
    assert_eq!(report.initial_model.parameter_count, 1_213_840);
    assert_eq!(report.initial_model.tensor_count, 14);
    assert_ne!(
        report.initial_model.tensor_content_id,
        report.checkpoint_model.tensor_content_id
    );
    assert_eq!(report.historical_training, "declared");
    assert_eq!(report.model_reload, "not_run");
    assert_eq!(report.tokenizer_replay, "not_run");
}

#[test]
fn native_rejects_rehashed_accounting_and_authorization_forgeries() {
    let (original, files) = fixture();
    for (pointer, value) in [
        ("/prepared_build_id", json!("0".repeat(64))),
        ("/observations/optimizer_updates", json!(4)),
        ("/observations/consumed_examples", json!(4)),
        ("/observations/shifted_supervised_tokens", json!(1)),
        ("/observations/microbatches/0/update", json!(2)),
        (
            "/observations/microbatches/0/example_ids/0",
            json!("0".repeat(64)),
        ),
        ("/recipe/accumulation", json!(1)),
        (
            "/source_authorization",
            json!("qwen3_0_6b_c1899de_student_training_v1"),
        ),
        ("/initial_model/tensor_content_id", json!("0".repeat(64))),
        ("/checkpoint_kind", json!("resumable_optimizer")),
    ] {
        let mut manifest = original.clone();
        *manifest.pointer_mut(pointer).unwrap() = value;
        assert!(
            verify_training_checkpoint(Cursor::new(frame(manifest, &files))).is_err(),
            "{pointer}"
        );
    }
}

#[test]
fn native_rejects_unknown_config_nonfinite_and_conflicting_tied_values() {
    let (manifest, original) = fixture();
    for case in ["config", "nonfinite", "alias", "pickle", "trailing"] {
        let mut files = original.clone();
        if case == "config" {
            let mut config: Value = serde_json::from_slice(&files["initial/config.json"]).unwrap();
            config["auto_map"] = json!({});
            files.insert(
                "initial/config.json".into(),
                serde_json::to_vec(&config).unwrap(),
            );
        } else if case == "pickle" {
            let bytes = files.remove("checkpoint/model.safetensors").unwrap();
            files.insert("checkpoint/training_args.bin".into(), bytes);
        } else {
            let bytes = files.get_mut("initial/model.safetensors").unwrap();
            let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
            match case {
                "nonfinite" => bytes[n + 8..n + 10].copy_from_slice(&0x7f80_u16.to_le_bytes()),
                "alias" => bytes[n + 8..n + 10].copy_from_slice(&0x3f80_u16.to_le_bytes()),
                _ => bytes.extend([0, 0]),
            }
        }
        assert!(
            verify_training_checkpoint(Cursor::new(frame(manifest.clone(), &files))).is_err(),
            "{case}"
        );
    }
}

#[test]
fn native_rejects_truncation_trailing_bytes_and_manifest_length_before_allocation() {
    let (manifest, files) = fixture();
    let original = frame(manifest, &files);
    let mut trailing = original.clone();
    trailing.push(0);
    let mut length = original.clone();
    length[40..48].fill(255);
    let mut stale = original.clone();
    stale[8] ^= 1;
    for bytes in [
        &original[..47],
        &original[..original.len() - 1],
        &trailing,
        &length,
        &stale,
    ] {
        assert!(verify_training_checkpoint(Cursor::new(bytes)).is_err());
    }
}
