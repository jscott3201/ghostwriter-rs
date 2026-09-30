//! Strict task wire validation and semantic identity, without any runtime or I/O.
use gw_schema::{NumericTaskDocument, TaskProvenance, TaskSplitRole};
use serde_json::{Value, json};
const FIXTURE: &str = include_str!("../../../examples/reviewed-numeric-tasks.json");

#[test]
fn strict_document_rejects_unknown_versions_fields_and_unsupported_structures() {
    let good: Value = serde_json::from_str(FIXTURE).unwrap();
    for (pointer, replacement) in [
        ("/version", json!(2)),
        ("/tasks", json!([])),
        ("/tasks/0/prompt/role", json!("assistant")),
        (
            "/tasks/0/prompt/content",
            json!([{"type":"text","text":"question"}]),
        ),
        ("/tasks/0/verification/kind", json!("set_match")),
        (
            "/tasks/0/verification/oracle/oracle",
            json!("sandbox_execution"),
        ),
        ("/tasks/0/verification/execution_policy", json!("advisory")),
        (
            "/tasks/0/verification/numeric/extraction/mode",
            json!("search"),
        ),
        (
            "/tasks/0/verification/numeric/tolerance/absolute",
            json!(-1),
        ),
        ("/tasks/0/verification/numeric/extraction/marker", json!("")),
        (
            "/tasks/0/verification/numeric/extraction/marker",
            json!("FINAL:\n"),
        ),
        ("/tasks/0/verification/oracle/expected", json!("1e9999")),
        ("/tasks/0/verification/oracle/expected", json!("1e-9999")),
        ("/tasks/0/verification/oracle/expected", json!("NaN")),
        ("/tasks/0/verification/oracle/expected", json!("$5")),
        ("/tasks/0/rights/permitted_uses", json!([])),
        (
            "/tasks/0/rights/permitted_uses",
            json!(["training", "training"]),
        ),
        ("/tasks/0/rights/evidence", json!([])),
        ("/tasks/0/observations/qc/evidence", json!([])),
    ] {
        let mut changed = good.clone();
        *changed.pointer_mut(pointer).unwrap() = replacement;
        assert!(
            NumericTaskDocument::from_json(&changed.to_string()).is_err(),
            "accepted {pointer}: {changed}"
        );
    }
    for pointer in [
        "",
        "/tasks/0",
        "/tasks/0/source",
        "/tasks/0/rights",
        "/tasks/0/group",
        "/tasks/0/split",
        "/tasks/0/split/manifest",
        "/tasks/0/prompt",
        "/tasks/0/verification",
        "/tasks/0/verification/oracle",
        "/tasks/0/verification/numeric",
        "/tasks/0/verification/numeric/extraction",
        "/tasks/0/verification/numeric/tolerance",
        "/tasks/0/observations",
        "/tasks/0/observations/difficulty",
        "/tasks/0/observations/qc",
    ] {
        let mut changed = good.clone();
        changed
            .pointer_mut(pointer)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("unexpected".into(), json!(true));
        assert!(
            NumericTaskDocument::from_json(&changed.to_string()).is_err(),
            "unknown field accepted at {pointer}"
        );
    }
}

#[test]
fn required_identities_and_reviewed_text_are_nonempty() {
    let good: Value = serde_json::from_str(FIXTURE).unwrap();
    for pointer in [
        "/tasks/0/task_id",
        "/tasks/0/source/namespace",
        "/tasks/0/source/item",
        "/tasks/0/source/revision",
        "/tasks/0/source/citation",
        "/tasks/0/rights/reviewer",
        "/tasks/0/rights/evidence/0",
        "/tasks/0/group/namespace",
        "/tasks/0/group/id",
        "/tasks/0/split/manifest/namespace",
        "/tasks/0/split/manifest/id",
        "/tasks/0/split/revision",
        "/tasks/0/prompt/content",
        "/tasks/0/observations/domain",
        "/tasks/0/observations/difficulty/label",
        "/tasks/0/observations/difficulty/basis",
        "/tasks/0/observations/qc/reviewer",
        "/tasks/0/observations/qc/evidence/0",
    ] {
        let mut changed = good.clone();
        *changed.pointer_mut(pointer).unwrap() = json!(" \n ");
        assert!(
            NumericTaskDocument::from_json(&changed.to_string()).is_err(),
            "empty {pointer}"
        );
    }
}

#[test]
fn duplicates_and_conflicting_group_splits_are_rejected() {
    let good = NumericTaskDocument::from_json(FIXTURE).unwrap();
    for case in 0..4 {
        let mut changed = good.clone();
        match case {
            0 => changed.tasks[1].task_id = changed.tasks[0].task_id.clone(),
            1 => changed.tasks[1].source = changed.tasks[0].source.clone(),
            2 => {
                changed.tasks[1] = changed.tasks[0].clone();
                changed.tasks[1].task_id = "renamed".into();
            }
            _ => {
                changed.tasks[1].group = changed.tasks[0].group.clone();
                changed.tasks[1].split.role = TaskSplitRole::Test;
            }
        }
        assert!(changed.validate().is_err(), "case {case}");
    }
}

#[test]
fn semantic_identity_uses_typed_content_and_excludes_split_rights_and_labels() {
    let doc = NumericTaskDocument::from_json(FIXTURE).unwrap();
    let compact = NumericTaskDocument::from_json(&serde_json::to_string(&doc).unwrap()).unwrap();
    let first = TaskProvenance::from_task(&doc.tasks[0]).unwrap();
    assert_eq!(first, TaskProvenance::from_task(&compact.tasks[0]).unwrap());
    let mut changed = doc.tasks[0].clone();
    changed.task_id = "another-label".into();
    changed.split.role = TaskSplitRole::Test;
    changed.rights.reviewer = "another-reviewer".into();
    assert_eq!(
        first.identity,
        TaskProvenance::from_task(&changed).unwrap().identity
    );
    changed.source.revision = "revision-two".into();
    assert_ne!(
        first.identity,
        TaskProvenance::from_task(&changed).unwrap().identity
    );
    let mut changed = doc.tasks[0].clone();
    changed.verification.oracle = gw_schema::NumericTaskOracle::Literal {
        expected: "5.0e0".into(),
    };
    assert_eq!(
        first.identity,
        TaskProvenance::from_task(&changed).unwrap().identity
    );
}

#[test]
fn library_values_reject_nonfinite_and_overflowing_tolerances() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        for relative in [false, true] {
            let mut doc = NumericTaskDocument::from_json(FIXTURE).unwrap();
            let tolerance = &mut doc.tasks[0].verification.numeric.tolerance;
            if relative {
                tolerance.relative = value;
            } else {
                tolerance.absolute = value;
            }
            assert!(doc.validate().is_err());
        }
    }
    let mut doc = NumericTaskDocument::from_json(FIXTURE).unwrap();
    doc.tasks[0].verification.numeric.tolerance.relative = f64::MAX;
    assert!(doc.validate().is_err());
}
