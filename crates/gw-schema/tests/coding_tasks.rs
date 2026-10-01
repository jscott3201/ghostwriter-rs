//! Independent oracle/type/identity contracts for reviewed coding task intake.
use gw_schema::{CodingTaskDocument, CodingValue, ExportTaskProjection, TaskProvenance};

const FIXTURES: &[u8] = include_bytes!("../../../examples/reviewed-coding-tasks.json");

#[test]
fn held_out_projection_cannot_be_relabelled_for_training() {
    let document = CodingTaskDocument::from_json(FIXTURES).unwrap();
    let task = &document.tasks[2];
    let mut projection = ExportTaskProjection {
        provenance: TaskProvenance::from_coding_task(task).unwrap(),
        verification_contract: task.contract(),
    };
    projection.provenance.split.role = gw_schema::TaskSplitRole::Train;
    assert!(
        projection.validate(&[task.prompt()]).is_err(),
        "protected oracle partition must remain held out after redaction"
    );
}

#[test]
fn private_cases_change_identity_without_entering_prompt_or_export() {
    let document = CodingTaskDocument::from_json(FIXTURES).unwrap();
    let task = &document.tasks[2];
    let provenance = TaskProvenance::from_coding_task(task).unwrap();
    let projection = ExportTaskProjection {
        provenance: provenance.clone(),
        verification_contract: task.contract(),
    };
    projection.validate(&[task.prompt()]).unwrap();
    for text in [
        serde_json::to_string(&task.prompt()).unwrap(),
        serde_json::to_string(&projection).unwrap(),
    ] {
        assert!(!text.contains("ORACLE_SENTINEL_HELDOUT"));
    }
    let mut changed = task.clone();
    changed.protected_cases[0].expected = CodingValue::String("different".into());
    assert_ne!(
        provenance.identity,
        TaskProvenance::from_coding_task(&changed).unwrap().identity
    );
    assert_eq!(task.prompt(), changed.prompt());
    assert!(
        provenance
            .validate_for(&task.prompt(), &changed.contract())
            .is_err()
    );
}

#[test]
fn exact_typed_results_reject_unsupported_values() {
    assert_ne!(CodingValue::Boolean(true), CodingValue::Integer(1));
    assert_ne!(
        CodingValue::String("é".into()),
        CodingValue::String("e\u{301}".into())
    );
    assert_ne!(
        CodingValue::Array(vec![]),
        CodingValue::Array(vec![CodingValue::Null])
    );
    for invalid in [
        r#"{"type":"integer","value":1.0}"#,
        r#"{"type":"integer","value":true}"#,
        r#"{"type":"integer","value":9223372036854775808}"#,
        r#"{"type":"null","value":null}"#,
        r#"{"type":"object","value":{"x":{"type":"null"},"x":{"type":"null"}}}"#,
        r#"{"type":"float","value":0}"#,
        r#"{"type":"null"} {}"#,
    ] {
        assert!(
            CodingValue::from_json(invalid.as_bytes()).is_err(),
            "{invalid}"
        );
    }
    let a = CodingValue::from_json(
        br#"{"type":"object","value":{"a":{"type":"null"},"b":{"type":"boolean","value":true}}}"#,
    )
    .unwrap();
    let b = CodingValue::from_json(
        br#"{"type":"object","value":{"b":{"type":"boolean","value":true},"a":{"type":"null"}}}"#,
    )
    .unwrap();
    assert_eq!(a, b);
}

#[test]
fn intake_rejects_mixed_partitions_duplicate_keys_groups_and_unbounded_suites() {
    let original = CodingTaskDocument::from_json(FIXTURES).unwrap();
    let mut mixed = original.clone();
    mixed.tasks[0].protected_cases = mixed.tasks[0].train_cases.clone();
    assert!(mixed.validate().is_err());
    let mut split = original.clone();
    split.tasks[2].group = split.tasks[0].group.clone();
    assert!(split.validate().is_err());
    let mut duplicate = original.clone();
    duplicate.tasks[0]
        .train_cases
        .push(original.tasks[0].visible_examples[0].clone());
    assert!(duplicate.validate().is_err());
    let mut oversized = original.clone();
    oversized.tasks[0].train_cases = (0..65)
        .map(|i| {
            let mut c = original.tasks[0].train_cases[0].clone();
            c.label = format!("case-{i}");
            c
        })
        .collect();
    assert!(oversized.validate().is_err());
    let duplicate_key = String::from_utf8(FIXTURES.to_vec()).unwrap().replacen(
        "\"version\": 1",
        "\"version\": 1, \"version\": 1",
        1,
    );
    assert!(CodingTaskDocument::from_json(duplicate_key.as_bytes()).is_err());
}
