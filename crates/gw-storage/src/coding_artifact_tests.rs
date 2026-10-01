//! Reviewed coding identities and private-oracle redaction cross actual Parquet export/import.
use super::*;
use crate::export::write_parquet;
use gw_schema::{CodingTaskDocument, Content, CotPolicy, TaskProvenance, TrlFormat};

#[test]
fn coding_projection_roundtrips_without_private_oracles() {
    let document = CodingTaskDocument::from_json(include_bytes!(
        "../../../examples/reviewed-coding-tasks.json"
    ))
    .unwrap();
    let modules = [
        include_str!("../../../examples/coding/merge_closed.correct.py"),
        include_str!("../../../examples/coding/runs.correct.py"),
        include_str!("../../../examples/coding/common_prefix.correct.py"),
    ];
    let records = document
        .tasks
        .iter()
        .zip(modules)
        .map(|(task, code)| {
            let mut record = super::task_tests::task_record(&task.task_id);
            record.training_area = "python-standard-library".into();
            record.messages[0] = task.prompt();
            record.messages[1].content = Content::Text(code.into());
            record.messages[1].reasoning = None;
            record.task_provenance = Some(TaskProvenance::from_coding_task(task).unwrap());
            record.verification_contract = Some(task.contract());
            record
        })
        .collect::<Vec<_>>();
    let plan = ExportPlan::prepare(
        &records,
        ExportOptions {
            target: TrlFormat::ChatML,
            cot_policy: CotPolicy::Masked,
            dataset_version: None,
            scope: ExportScope::Records,
        },
    )
    .unwrap();
    let mut relabelled = records.clone();
    relabelled[2].task_provenance.as_mut().unwrap().split.role = gw_schema::TaskSplitRole::Train;
    let error = ExportPlan::prepare(
        &relabelled,
        ExportOptions {
            target: TrlFormat::ChatML,
            cot_policy: CotPolicy::Masked,
            dataset_version: None,
            scope: ExportScope::Records,
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("private oracle partition"));
    for row in &plan.rows {
        for text in [&row.messages_json, row.task_json.as_ref().unwrap()] {
            assert!(!text.contains("ORACLE_SENTINEL_HELDOUT"));
            assert!(!text.contains("protected_cases"));
            assert!(!text.contains("train_cases"));
        }
        let task: gw_schema::ExportTaskProjection =
            serde_json::from_str(row.task_json.as_ref().unwrap()).unwrap();
        assert_eq!(task.provenance.identity.version, 2);
        let messages: Vec<gw_schema::Message> = serde_json::from_str(&row.messages_json).unwrap();
        task.validate(&messages).unwrap();
    }
    let mut bytes = Vec::new();
    write_parquet(&plan.rows, &plan.artifact, &mut bytes).unwrap();
    assert_eq!(
        verify_artifact_snapshot(bytes.clone()).unwrap().artifact,
        plan.artifact
    );
    let name = "v3-coding.parquet";
    if let Some(output) = std::env::var_os("GW_REGENERATE_TRL_FIXTURES") {
        std::fs::write(Path::new(&output).join(name), bytes).unwrap();
    }
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../adapters/trl/tests/fixtures")
        .join(name);
    assert_eq!(
        verify_artifact_snapshot(std::fs::read(fixture).unwrap())
            .unwrap()
            .artifact,
        plan.artifact
    );
    // Forge a checksum-consistent Parquet declaration to exercise downstream source consumers.
    let mut rows = plan.rows.clone();
    let row = rows
        .iter_mut()
        .find(|row| row.record_id == "common-prefix")
        .unwrap();
    let mut projection: gw_schema::ExportTaskProjection =
        serde_json::from_str(row.task_json.as_ref().unwrap()).unwrap();
    projection.provenance.split.role = gw_schema::TaskSplitRole::Train;
    row.task_json = Some(crate::export::canonical_task_json(&projection).unwrap());
    let mut claimed = plan.artifact.clone();
    claimed.artifact_id = artifact_identity(&claimed, &rows).unwrap();
    let mut forged = Vec::new();
    write_parquet(&rows, &claimed, &mut forged).unwrap();
    assert!(
        verify_artifact_snapshot(forged.clone())
            .unwrap_err()
            .to_string()
            .contains("private oracle partition")
    );
    let name = "v3-coding-relabelled.parquet";
    if let Some(output) = std::env::var_os("GW_REGENERATE_TRL_FIXTURES") {
        std::fs::write(Path::new(&output).join(name), &forged).unwrap();
    }
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../adapters/trl/tests/fixtures")
        .join(name);
    assert!(
        verify_artifact_snapshot(std::fs::read(fixture).unwrap())
            .unwrap_err()
            .to_string()
            .contains("private oracle partition")
    );
}
