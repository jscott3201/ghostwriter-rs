//! Literal generated-v1 wire and strict origin regression cases, independent of the writer.
use gw_schema::{RecordOrigin, TrainingRecord};

const GENERATED_V1: &str = r#"{"record_id":"literal-1","schema_version":"1.0.0","training_area":"fixture","tags":[],"messages":[{"role":"user","content":"A"},{"role":"assistant","content":"B"}],"provenance":{"run_id":"run-1","parent_ids":[],"teacher":{"provider":"fixture","slug":"teacher"},"judge_models":[],"harness_version":"fixture"},"generation":{},"verification":{"checks":[],"all_passed":false},"judging":{"panel":[]},"lifecycle":{"state":"seeded","history":[],"attempts":0},"hashes":{"prompt_hash":"","completion_hash":"","record_hash":""},"cost":{"prompt_tokens":0,"completion_tokens":0,"reasoning_tokens":0,"usd":0.0,"latency_ms":0}}"#;

#[test]
fn generated_v1_wire_bytes_do_not_change() {
    let record: TrainingRecord = serde_json::from_str(GENERATED_V1).unwrap();
    assert!(matches!(record.origin, RecordOrigin::Generated(_)));
    assert_eq!(serde_json::to_string(&record).unwrap(), GENERATED_V1);
}

#[test]
fn reference_cannot_be_mixed_into_generated_v1_or_hide_unknown_fields() {
    let mut value: serde_json::Value = serde_json::from_str(GENERATED_V1).unwrap();
    value["origin"] = serde_json::json!({"version":1,"approved":true});
    assert!(serde_json::from_value::<TrainingRecord>(value.clone()).is_err());
    value.as_object_mut().unwrap().remove("origin");
    value["operator_approved_reference"] = true.into();
    assert!(serde_json::from_value::<TrainingRecord>(value).is_err());
}

#[test]
fn versioned_reference_requires_every_binding_and_no_generation_fields() {
    let mut value: serde_json::Value = serde_json::from_str(GENERATED_V1).unwrap();
    value.as_object_mut().unwrap().remove("provenance");
    value.as_object_mut().unwrap().remove("generation");
    value["origin"] = serde_json::json!({"version":1,"catalogue_id":"a".repeat(64),
        "registration_id":"b".repeat(64),"batch_id":"c".repeat(64),"member_id":"d".repeat(64),
        "reference_code_id":"e".repeat(64),"suite_id":"f".repeat(64),"native_result_id":"0".repeat(64),
        "authorship":{"author":"agent","reviewer":"agent"},"component":{"namespace":"fixture","id":"component"},"permitted_use":"training"});
    let record: TrainingRecord = serde_json::from_value(value.clone()).unwrap();
    assert!(record.origin.generated().is_none());
    assert_eq!(record.run_id(), "c".repeat(64));
    let wire = serde_json::to_value(&record).unwrap();
    assert!(wire.get("provenance").is_none() && wire.get("generation").is_none());
    for field in ["version", "suite_id", "authorship"] {
        let mut missing = value.clone();
        missing["origin"].as_object_mut().unwrap().remove(field);
        assert!(serde_json::from_value::<TrainingRecord>(missing).is_err());
    }
    value["origin"]["version"] = 2.into();
    assert!(serde_json::from_value::<TrainingRecord>(value).is_err());
}
