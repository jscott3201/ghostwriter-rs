//! Global population and exact review coverage are checked before runtime observation is possible.
mod reference_support;
use gw_schema::*;

#[test]
fn complete_synthetic_population_is_bound_to_every_captured_byte() {
    let capture = reference_support::capture();
    let validated = capture.validate().unwrap();
    assert_eq!(validated.members().len(), 112);
    assert_eq!(
        validated
            .members()
            .iter()
            .filter(|m| m.task.split.role == TaskSplitRole::Train)
            .count(),
        64
    );
    assert_eq!(
        validated
            .members()
            .iter()
            .filter(|m| m.task.split.role == TaskSplitRole::Validation)
            .count(),
        16
    );
    assert_eq!(
        validated
            .members()
            .iter()
            .filter(|m| m.task.split.role == TaskSplitRole::Test)
            .count(),
        32
    );
    let mut whitespace = capture.clone();
    whitespace.reviews[111].push('\n');
    assert_ne!(
        validated.catalogue_id(),
        whitespace.validate().unwrap().catalogue_id()
    );
    let mut changed_code = capture.clone();
    changed_code.modules[111].push('\n');
    assert!(changed_code.validate().is_err());
    let mut claimed = capture;
    let mut review: serde_json::Value = serde_json::from_str(&claimed.reviews[111]).unwrap();
    review["approved"] = true.into();
    claimed.reviews[111] = serde_json::to_string(&review).unwrap();
    assert!(claimed.validate().is_err());
}
#[test]
fn cross_document_duplicates_and_component_split_conflicts_reject_entire_population() {
    let original = reference_support::capture();
    let mut duplicate = original.clone();
    let first = CodingTaskDocument::from_json(duplicate.task_documents[0].as_bytes()).unwrap();
    let mut second = CodingTaskDocument::from_json(duplicate.task_documents[1].as_bytes()).unwrap();
    second.tasks[0].source = first.tasks[0].source.clone();
    duplicate.task_documents[1] = serde_json::to_string(&second).unwrap();
    assert!(duplicate.validate().unwrap_err().contains("duplicate"));
    let mut mixed = original;
    let mut catalogue: ReferenceCatalogue = serde_json::from_str(&mixed.catalogue).unwrap();
    catalogue.members[111].component = catalogue.members[0].component.clone();
    mixed.catalogue = serde_json::to_string(&catalogue).unwrap();
    assert!(mixed.validate().unwrap_err().contains("component crosses"));
}
