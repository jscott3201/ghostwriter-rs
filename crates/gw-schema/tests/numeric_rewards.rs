//! Frozen corpus identity preserves reviewed evidence while projecting declared Train tasks.
use gw_schema::{NumericRewardArtifact, NumericTaskDocument};
use serde_json::{Value, json};

const SOURCE: &str = include_str!("../../../examples/reviewed-numeric-tasks.json");

fn source() -> Value {
    serde_json::from_str(SOURCE).unwrap()
}

fn document(value: &Value) -> NumericTaskDocument {
    NumericTaskDocument::from_json(&value.to_string()).unwrap()
}

fn artifact(value: &Value) -> NumericRewardArtifact {
    NumericRewardArtifact::from_documents(vec![document(value)]).unwrap()
}

#[test]
fn corpus_preserves_all_task_evidence_and_order_without_held_out_tasks() {
    let mut value = source();
    let first = artifact(&value);
    assert_eq!(
        serde_json::to_value(&first.tasks[0].task).unwrap(),
        value["tasks"][0]
    );
    assert_eq!(first, artifact(&value));
    value["tasks"].as_array_mut().unwrap().reverse();
    let reversed = artifact(&value);
    assert_ne!(first.artifact_id, reversed.artifact_id);
    assert_eq!(first.reward_contract_id, reversed.reward_contract_id);
    assert_eq!(first.tasks[0], reversed.tasks[1]);
    value["tasks"][0]["split"]["role"] = json!("test");
    let selected = artifact(&value);
    assert_eq!(selected.tasks.len(), 1);
    assert_eq!(selected.tasks[0].task.task_id, "addition-001");
}

#[test]
fn corpus_identity_includes_rights_review_and_split_declarations() {
    let value = source();
    let baseline = artifact(&value);
    for pointer in [
        "/tasks/0/rights/reviewer",
        "/tasks/0/rights/evidence/0",
        "/tasks/0/split/revision",
        "/tasks/0/observations/qc/evidence/0",
    ] {
        let mut changed = value.clone();
        *changed.pointer_mut(pointer).unwrap() = json!("independently-reviewed-revision");
        let changed = artifact(&changed);
        assert_ne!(baseline.artifact_id, changed.artifact_id, "{pointer}");
        assert_eq!(
            baseline.tasks[0].task_identity,
            changed.tasks[0].task_identity
        );
    }
}

#[test]
fn all_documents_and_cross_document_conflicts_are_checked_before_train_projection() {
    let value = source();
    assert!(
        NumericRewardArtifact::from_documents(vec![document(&value), document(&value)]).is_err()
    );
    let mut held_out = value.clone();
    for row in held_out["tasks"].as_array_mut().unwrap() {
        row["split"]["role"] = json!("test");
    }
    assert!(
        NumericRewardArtifact::from_documents(vec![document(&value), document(&held_out)]).is_err()
    );
    assert!(NumericRewardArtifact::from_documents(vec![document(&held_out)]).is_err());
    let mut conflict = value.clone();
    conflict["tasks"][1]["group"] = conflict["tasks"][0]["group"].clone();
    conflict["tasks"][1]["split"]["role"] = json!("test");
    assert!(NumericTaskDocument::from_json(&conflict.to_string()).is_err());
}

#[test]
fn rewards_require_authoritative_answers_training_rights_and_passing_qc() {
    for (pointer, replacement) in [
        ("/tasks/0/verification/answer_policy", json!("advisory")),
        ("/tasks/0/rights/permitted_uses", json!(["evaluation"])),
        ("/tasks/0/observations/qc/answerable", json!(false)),
        ("/tasks/0/observations/qc/difficulty_targeted", json!(false)),
        ("/tasks/0/observations/qc/in_scope", json!(false)),
    ] {
        let mut value = source();
        *value.pointer_mut(pointer).unwrap() = replacement;
        assert!(
            NumericRewardArtifact::from_documents(vec![document(&value)]).is_err(),
            "{pointer}"
        );
    }
}

#[test]
fn original_json_duplicate_fields_unknown_fields_and_wrong_types_are_rejected() {
    let artifact = artifact(&source());
    let raw = serde_json::to_string(&artifact).unwrap();
    for (needle, replacement) in [
        (
            "\"artifact_version\":1",
            "\"artifact_version\":1,\"artifact_version\":1",
        ),
        (
            "\"absolute\":{\"binary64\":\"0000000000000000\"}",
            "\"absolute\":{\"binary64\":\"0000000000000000\"},\"absolute\":{\"binary64\":\"0000000000000000\"}",
        ),
        (
            "\"artifact_version\":1",
            "\"artifact_version\":1,\"teacher_evidence\":true",
        ),
        ("\"artifact_version\":1", "\"artifact_version\":true"),
        ("\"artifact_version\":1", "\"artifact_version\":1.0"),
        ("\"expected\":\"5\"", "\"expected\":null"),
    ] {
        assert!(raw.contains(needle));
        assert!(
            NumericRewardArtifact::from_json(raw.replace(needle, replacement).as_bytes()).is_err()
        );
    }
    let compact = NumericRewardArtifact::verify_snapshot(raw.as_bytes()).unwrap();
    let spaced = format!("\n{raw}\n");
    let spaced = NumericRewardArtifact::verify_snapshot(spaced.as_bytes()).unwrap();
    assert_eq!(compact.artifact_id, spaced.artifact_id);
    assert_ne!(compact.snapshot, spaced.snapshot);
}

#[test]
fn changed_or_missing_oracle_contract_identity_and_task_identity_fail_verification() {
    let baseline = serde_json::to_value(artifact(&source())).unwrap();
    for (pointer, replacement) in [
        ("/artifact_id", json!("0".repeat(64))),
        ("/reward_contract_id", json!("0".repeat(64))),
        (
            "/reward_contract/verification_interpretation_version",
            json!(0),
        ),
        ("/tasks/0/task_identity/digest", json!("0".repeat(64))),
        ("/tasks/0/task/verification/oracle/expected", json!("6")),
        ("/tasks/0/task/verification/oracle/expected", json!(null)),
        ("/tasks/0/task/verification/oracle/expected", json!("NaN")),
    ] {
        let mut changed = baseline.clone();
        *changed.pointer_mut(pointer).unwrap() = replacement;
        assert!(
            NumericRewardArtifact::from_json(changed.to_string().as_bytes()).is_err(),
            "{pointer}"
        );
    }
}

#[test]
fn exact_tolerance_bits_survive_typed_corpus_serialization_and_snapshot_verification() {
    let mut identities = std::collections::HashSet::new();
    for bits in [
        0x3ff8_9d89_d89d_89d9,
        0x3ff8_9d89_d89d_89d8,
        0x0000_0000_0000_0000,
        0x8000_0000_0000_0000,
    ] {
        let mut document = document(&source());
        document.tasks[0].verification.numeric.tolerance.absolute = f64::from_bits(bits);
        document.tasks[0].verification.numeric.tolerance.relative = f64::from_bits(bits);
        // The reviewed-task input contract remains ordinary decimal JSON numbers.
        let task_json = serde_json::to_value(&document).unwrap();
        assert!(
            task_json["tasks"][0]["verification"]["numeric"]["tolerance"]["absolute"].is_number()
        );
        assert!(
            task_json["tasks"][0]["verification"]["numeric"]["tolerance"]["relative"].is_number()
        );
        let original = NumericRewardArtifact::from_documents(vec![document]).unwrap();
        assert!(identities.insert(original.artifact_id.clone()));
        let encoded = serde_json::to_vec(&original).unwrap();
        let wire: Value = serde_json::from_slice(&encoded).unwrap();
        let tolerance = &wire["tasks"][0]["task"]["verification"]["numeric"]["tolerance"];
        assert_eq!(
            tolerance["absolute"],
            json!({"binary64": format!("{bits:016x}")})
        );
        assert_eq!(tolerance["relative"], tolerance["absolute"]);
        let restored = NumericRewardArtifact::from_json(&encoded).unwrap();
        assert_eq!(
            restored.tasks[0]
                .task
                .verification
                .numeric
                .tolerance
                .absolute
                .to_bits(),
            bits
        );
        assert_eq!(
            restored.tasks[0]
                .task
                .verification
                .numeric
                .tolerance
                .relative
                .to_bits(),
            bits
        );
        assert_eq!(
            NumericRewardArtifact::verify_snapshot(&encoded)
                .unwrap()
                .artifact_id,
            original.artifact_id
        );
        assert_eq!(serde_json::to_vec(&restored).unwrap(), encoded);
    }
}

#[test]
fn exact_tolerance_wire_rejects_malformed_nonfinite_and_duplicate_bit_objects() {
    let artifact = artifact(&source());
    let value = serde_json::to_value(&artifact).unwrap();
    for replacement in [
        json!(0.0),
        json!("0000000000000000"),
        json!(null),
        json!({}),
        json!({"binary64": 0}),
        json!({"binary64": "000000000000000"}),
        json!({"binary64": "3FF89D89D89D89D9"}),
        json!({"binary64": "zzzzzzzzzzzzzzzz"}),
        json!({"binary64": "7ff0000000000000"}),
        json!({"binary64": "fff0000000000000"}),
        json!({"binary64": "7ff8000000000001"}),
        json!({"binary64": "0000000000000000", "extra": true}),
    ] {
        for field in ["absolute", "relative"] {
            let mut changed = value.clone();
            changed["tasks"][0]["task"]["verification"]["numeric"]["tolerance"][field] =
                replacement.clone();
            assert!(
                NumericRewardArtifact::from_json(changed.to_string().as_bytes()).is_err(),
                "{field} {replacement}"
            );
        }
    }
    let raw = serde_json::to_string(&artifact).unwrap();
    let duplicate = raw.replacen(
        "\"binary64\":\"0000000000000000\"",
        "\"binary64\":\"0000000000000000\",\"binary64\":\"0000000000000000\"",
        1,
    );
    assert!(NumericRewardArtifact::from_json(duplicate.as_bytes()).is_err());
}
