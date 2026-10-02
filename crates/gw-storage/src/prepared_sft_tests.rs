//! Consume the exact Python-produced golden bytes and independent shared adversarial corpus.
use super::*;
use gw_schema::encode_prepared_sft_frame;
use serde_json::Value;
use std::path::Path;

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../adapters/trl/tests/fixtures")
            .join(name),
    )
    .unwrap()
}

#[test]
fn imported_python_builds_preserve_order_source_and_long_examples() {
    for (name, count) in [
        ("prepared-all.gwsft", 4),
        ("prepared-empty.gwsft", 0),
        ("prepared-long.gwsft", 4),
        ("prepared-gemma.gwsft", 4),
    ] {
        let data = fixture(name);
        let frame = decode_prepared_sft_frame(&data).unwrap();
        let verified = verify_prepared_sft_snapshot(data.clone()).unwrap();
        assert_eq!(verified.report().example_count, count);
        assert_eq!(verified.report().build_id, frame.build_id);
        assert_eq!(verified.report().byte_length, data.len() as u64);
        assert_eq!(
            verified.report().snapshot_blake3,
            blake3::hash(&data).to_hex().to_string()
        );
        assert_eq!(
            verified.report().source_verification.snapshot_blake3,
            blake3::hash(frame.source).to_hex().to_string()
        );
        assert_eq!(verified.report().tokenizer_replay, "not_run");
        assert_eq!(
            verified.payload().manifest.example_ids,
            verified
                .payload()
                .examples
                .iter()
                .map(|e| e.example_id.clone())
                .collect::<Vec<_>>()
        );
        if name.contains("long") {
            assert!(
                verified
                    .payload()
                    .examples
                    .iter()
                    .all(|e| e.input_ids.len() > 1024)
            );
            assert_ne!(
                verified.payload().examples[0].input_ids.len(),
                verified.payload().examples[3].input_ids.len()
            );
        }
    }
}

fn mutation(data: &[u8], case: &Value) -> Vec<u8> {
    let decoded = decode_prepared_sft_frame(data).unwrap();
    let mut payload: Value = serde_json::from_slice(decoded.payload).unwrap();
    let mut source = decoded.source.to_vec();
    let op = case["op"].as_str().unwrap();
    match op {
        "replace" => {
            *payload
                .pointer_mut(case["pointer"].as_str().unwrap())
                .unwrap() = case["value"].clone()
        }
        "duplicate_example" => {
            let examples = payload["examples"].as_array_mut().unwrap();
            examples.push(examples.last().unwrap().clone());
        }
        "missing_example" => {
            payload["examples"].as_array_mut().unwrap().pop();
        }
        "reorder_examples" | "reorder_and_refs" => {
            payload["examples"].as_array_mut().unwrap().reverse();
            if op == "reorder_and_refs" {
                payload["manifest"]["example_ids"]
                    .as_array_mut()
                    .unwrap()
                    .reverse();
            }
        }
        "missing_null" => {
            payload["examples"][3]["source"]
                .as_object_mut()
                .unwrap()
                .remove("task_json");
        }
        _ => {}
    }
    let mut raw = serde_json::to_string(&payload).unwrap();
    match op {
        "duplicate_field" => raw = format!("{{\"version\":1,{}", &raw[1..]),
        "nested_duplicate_field" => {
            raw = raw.replacen(
                "\"max_length\":2048",
                "\"max_length\":2048,\"max_length\":2048",
                1,
            )
        }
        "float" => raw = raw.replacen("\"version\":1", "\"version\":1.0", 1),
        "boolean" => raw = raw.replacen("\"version\":1", "\"version\":true", 1),
        "source" => source = b"not a parquet artifact".to_vec(),
        _ => {}
    }
    let raw = if op == "utf8" {
        vec![b'"', 0xff, b'"']
    } else {
        raw.into_bytes()
    };
    let mut changed = encode_prepared_sft_frame(&raw, &source).unwrap();
    match op {
        "trailing" => changed.push(b'!'),
        "truncated" => {
            changed.pop();
        }
        "length" => changed[40..48].fill(0xff),
        "magic" => changed[..8].copy_from_slice(b"GWSFT002"),
        "stale" => changed[60] ^= 1,
        _ => {}
    }
    changed
}

#[test]
fn shared_independent_corpus_rejects_whole_build_without_partial_import() {
    let cases: Vec<Value> = serde_json::from_str(include_str!(
        "../../../adapters/trl/tests/prepared_cases.json"
    ))
    .unwrap();
    for name in ["prepared-all.gwsft", "prepared-gemma.gwsft"] {
        let data = fixture(name);
        for case in &cases {
            let result = verify_prepared_sft_snapshot(mutation(&data, case));
            assert_eq!(
                result.is_ok(),
                case["rust_accept"].as_bool().unwrap(),
                "{name} {}: {result:?}",
                case["name"]
            );
        }
    }
}

#[test]
fn gemma_profile_rejects_rehashed_control_and_policy_changes() {
    let data = fixture("prepared-gemma.gwsft");
    for (pointer, value) in [
        (
            "/manifest/recipe/preparation_profile/name",
            serde_json::json!("other"),
        ),
        (
            "/manifest/recipe/preparation_profile/controls/enable_thinking",
            serde_json::json!(false),
        ),
        (
            "/manifest/recipe/tokenizer/revision",
            serde_json::json!("0".repeat(40)),
        ),
        (
            "/manifest/recipe/dependencies/transformers",
            serde_json::json!("4.56.2"),
        ),
        (
            "/manifest/recipe/tokenizer_policy/backend_sha256",
            serde_json::json!("0".repeat(64)),
        ),
        ("/examples/0/input_ids/0", serde_json::json!(1)),
    ] {
        let case = serde_json::json!({"op": "replace", "pointer": pointer, "value": value});
        assert!(
            verify_prepared_sft_snapshot(mutation(&data, &case)).is_err(),
            "{pointer}"
        );
    }
}

#[test]
fn captured_build_is_independent_of_replaced_bundle_path() {
    let path = std::env::temp_dir().join(format!("gw-prepared-{}.gwsft", std::process::id()));
    std::fs::write(&path, fixture("prepared-all.gwsft")).unwrap();
    let captured = std::fs::read(&path).unwrap();
    std::fs::write(&path, fixture("prepared-empty.gwsft")).unwrap();
    let first = verify_prepared_sft_snapshot(captured).unwrap();
    let second = verify_prepared_sft_snapshot(std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(first.report().example_count, 4);
    assert_eq!(second.report().example_count, 0);
    assert_ne!(first.report().build_id, second.report().build_id);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn prepared_source_binds_tools_presence_and_exact_bytes() {
    let (report, mut rows) =
        crate::artifact::verify_snapshot_with_rows(fixture("v5-text.parquet")).unwrap();
    let row = &mut rows[0];
    let mut source: PreparedSftExampleSource = serde_json::from_value(serde_json::json!({
        "record_id":row.record_id,"record_hash":row.record_hash,"prompt_hash":row.prompt_hash,
        "messages_json":row.messages_json,"task_json":row.task_json,"origin_json":row.origin_json,
        "tools_json":null,"artifact_id":report.artifact.artifact_id,
        "group_kind":"source_record","group_id":"fixture"
    }))
    .unwrap();
    assert_eq!(source.tools_json, Some(None));
    validate_source(&source, &report.artifact, row).unwrap();
    source.tools_json = None;
    assert!(validate_source(&source, &report.artifact, row).is_err());
    row.tools_json = Some("[]".into());
    source.tools_json = Some(None);
    assert!(validate_source(&source, &report.artifact, row).is_err());
    source.tools_json = Some(Some("[]".into()));
    validate_source(&source, &report.artifact, row).unwrap();
    source.tools_json = Some(Some("[ ]".into()));
    assert!(validate_source(&source, &report.artifact, row).is_err());
}
