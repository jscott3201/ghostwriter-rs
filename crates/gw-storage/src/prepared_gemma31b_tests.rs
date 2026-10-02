//! Independently inspect captured official-tokenizer builds and rehashed ownership mutations.
use super::*;
use gw_schema::encode_prepared_sft_frame;
use serde_json::Value;
fn fixture() -> Vec<u8> {
    std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../adapters/trl/tests/fixtures/prepared-gemma31b.gwsft"),
    )
    .unwrap()
}
#[test]
fn official_serial_tool_build_imports_all_four_targets() {
    let verified = verify_prepared_sft_snapshot(fixture()).unwrap();
    let payload = verified.payload();
    assert_eq!(verified.report().example_count, 4);
    assert_eq!(
        payload.manifest.effective_shifted_call_token_count,
        Some(53)
    );
    assert_eq!(payload.manifest.effective_shifted_answer_token_count, 13);
    assert!(payload.examples[0].rendered.contains("κλειδί value"));
    for (example, (target, call)) in
        payload
            .examples
            .iter()
            .zip([(2, true), (4, true), (6, false), (8, false)])
    {
        assert_eq!(example.target_index, target);
        assert_eq!(
            example.target_kind.as_deref(),
            Some(if call { "tool_call" } else { "text_answer" })
        );
        assert_eq!(example.shifted_answer_token_indices.is_empty(), call);
        assert_eq!(
            example
                .shifted_call_token_indices
                .as_ref()
                .unwrap()
                .is_empty(),
            !call
        );
        if call {
            assert_eq!(example.labels.last(), Some(&50));
        } else {
            assert!(example.labels.ends_with(&[106, -100]));
        }
        assert!(!example.rendered.contains("RETAINED_RAW_ONLY"));
        for span in &example.spans {
            if span.kind == "observation"
                || span.kind == "definition"
                || span.message_index < target
            {
                assert!(!span.supervised);
            }
        }
    }
}
#[test]
fn serial_native_reader_rejects_rehashed_calls_arguments_and_ownership() {
    let bytes = fixture();
    let frame = decode_prepared_sft_frame(&bytes).unwrap();
    for mutation in ["name", "argument", "observation", "handoff"] {
        let mut payload: Value = serde_json::from_slice(frame.payload).unwrap();
        match mutation {
            "name" => {
                let e = &mut payload["examples"][0];
                e["rendered"] = Value::String(e["rendered"].as_str().unwrap().replacen(
                    "call:lookup",
                    "call:lookuq",
                    1,
                ));
            }
            "argument" => {
                let e = &mut payload["examples"][0];
                let text = e["rendered"].as_str().unwrap();
                let call = text.find("<|tool_call>").unwrap();
                e["rendered"] = Value::String(format!(
                    "{}{}",
                    &text[..call],
                    text[call..].replacen("café", "cafè", 1)
                ));
            }
            "observation" => {
                let e = &mut payload["examples"][1];
                let span = e["spans"]
                    .as_array_mut()
                    .unwrap()
                    .iter_mut()
                    .find(|s| s["kind"] == "observation")
                    .unwrap();
                span["kind"] = Value::String("context".into());
            }
            "handoff" => {
                let spans = payload["examples"][0]["spans"].as_array_mut().unwrap();
                spans.last_mut().unwrap()["kind"] = Value::String("call_wrapper".into());
            }
            _ => unreachable!(),
        }
        // Keep dependent token-kind metadata coherent: the original canonical source,
        // not stale counters or a stale enclosing digest, must reject this edit.
        for e in payload["examples"].as_array_mut().unwrap() {
            let kinds: Vec<_> = e["ownership_offsets"]
                .as_array()
                .unwrap()
                .iter()
                .map(|offset| {
                    let start = offset[0].as_u64().unwrap();
                    let end = offset[1].as_u64().unwrap();
                    let owned: std::collections::BTreeSet<_> = e["spans"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .filter(|s| {
                            s["start"].as_u64().unwrap() < end && s["end"].as_u64().unwrap() > start
                        })
                        .map(|s| s["kind"].as_str().unwrap().to_owned())
                        .collect();
                    serde_json::to_value(owned).unwrap()
                })
                .collect();
            e["token_kinds"] = Value::Array(kinds);
        }
        let altered =
            encode_prepared_sft_frame(&serde_json::to_vec(&payload).unwrap(), frame.source)
                .unwrap();
        let error = verify_prepared_sft_snapshot(altered).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("serial rendering/ownership differs"),
            "{mutation}: {error}"
        );
    }
}
#[test]
fn serial_profile_still_rejects_raw_duplicate_payload_fields() {
    let bytes = fixture();
    let frame = decode_prepared_sft_frame(&bytes).unwrap();
    let text = std::str::from_utf8(frame.payload).unwrap();
    let duplicate = format!("{{\"version\":1,{}", &text[1..]);
    let changed = encode_prepared_sft_frame(duplicate.as_bytes(), frame.source).unwrap();
    assert!(verify_prepared_sft_snapshot(changed).is_err());
}
