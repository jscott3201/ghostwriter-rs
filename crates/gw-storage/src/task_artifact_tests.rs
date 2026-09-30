//! Self-contained v3 task projection and frozen publication member identity.
use super::*;
use crate::export::{build_batch, write_parquet};
use crate::{ExportPurpose, Store};
use arrow::datatypes::{Field, Schema};
use gw_schema::{
    Content, CotPolicy, ExportTaskProjection, LifecycleState, NumericTaskDocument, TaskProvenance,
    TrlFormat,
};
use parquet::arrow::ArrowWriter;
use parquet::file::{metadata::KeyValue, properties::WriterProperties};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
const INPUT: &str = include_str!("../../../examples/reviewed-numeric-tasks.json");

struct Temp(std::path::PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(std::env::temp_dir().join(format!(
            "gw-task-artifact-{}-{}.parquet",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )))
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
fn options() -> ExportOptions {
    ExportOptions {
        target: TrlFormat::ChatML,
        cot_policy: CotPolicy::Masked,
        dataset_version: None,
        scope: ExportScope::Run {
            run_id: "run".into(),
        },
    }
}

pub(super) fn task_record(id: &str) -> TrainingRecord {
    let document = NumericTaskDocument::from_json(INPUT).unwrap();
    let task = &document.tasks[0];
    let mut record: TrainingRecord = serde_json::from_value(serde_json::json!({
        "record_id":id,"schema_version":"1.0.0","training_area":"arithmetic",
        "messages":[{"role":"user","content":task.prompt.text()},{"role":"assistant","content":"FINAL: 5","reasoning":"2 plus 3"}],
        "provenance":{"run_id":"run","teacher":{"provider":"fixture","slug":"fixture"},"harness_version":"0.1.0"},
        "generation":{},"lifecycle":{"state":"formatted"},"judging":{"panel":[],"verdict":"admit","aggregate":0.9}
    })).unwrap();
    record.verification_contract = Some(task.verification.contract());
    record.task_provenance = Some(TaskProvenance::from_task(task).unwrap());
    record
}
fn write_raw(path: &Path, batch: RecordBatch, artifact: &ExportArtifact) {
    let props = WriterProperties::builder()
        .set_key_value_metadata(Some(vec![KeyValue::new(
            ARTIFACT_METADATA_KEY.into(),
            serde_json::to_string(artifact).unwrap(),
        )]))
        .build();
    let mut writer =
        ArrowWriter::try_new(File::create(path).unwrap(), batch.schema(), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

#[test]
fn v3_roundtrip_retains_task_contract_and_declarations_with_null_plain_prompts() {
    let record = task_record("task");
    let mut plain = task_record("plain");
    plain.task_provenance = None;
    let plan = ExportPlan::prepare(&[record.clone(), plain], options()).unwrap();
    assert_eq!(
        plan.artifact.manifest.column_schema_version,
        ExportSchemaVersion::ReviewedTasks
    );
    let output = Temp::new();
    write_parquet(&plan.rows, &plan.artifact, File::create(&output.0).unwrap()).unwrap();
    assert_eq!(
        verify_artifact(&output.0).unwrap(),
        ArtifactVerification::Verified(plan.artifact.clone())
    );
    let batch = ParquetRecordBatchReaderBuilder::try_new(File::open(&output.0).unwrap())
        .unwrap()
        .build()
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    assert_eq!(batch.num_columns(), 9);
    let tasks = batch
        .column_by_name("task_json")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert!(tasks.is_null(0));
    let decoded: ExportTaskProjection = serde_json::from_str(tasks.value(1)).unwrap();
    assert_eq!(decoded.provenance, record.task_provenance.unwrap());
    assert_eq!(
        decoded.verification_contract,
        record.verification_contract.unwrap()
    );
}

#[test]
fn v3_rejects_malformed_substituted_or_noncanonical_task_json_and_wrong_schema() {
    let plan = ExportPlan::prepare(&[task_record("task")], options()).unwrap();
    for variant in 0..5 {
        let mut rows = plan.rows.clone();
        let mut value: serde_json::Value =
            serde_json::from_str(rows[0].task_json.as_ref().unwrap()).unwrap();
        match variant {
            0 => rows[0].task_json = Some("{".into()),
            1 => {
                value["provenance"]["source"]["revision"] =
                    serde_json::json!("changed-without-digest");
                rows[0].task_json = Some(value.to_string());
            }
            2 => {
                value["unknown"] = serde_json::json!(true);
                rows[0].task_json = Some(value.to_string());
            }
            3 => {
                value["verification_contract"]["ignored"] = serde_json::json!(true);
                rows[0].task_json = Some(value.to_string());
            }
            _ => rows[0].task_json.as_mut().unwrap().push(' '),
        }
        let mut claimed = plan.artifact.clone();
        claimed.artifact_id = artifact_identity(&claimed, &rows).unwrap();
        let output = Temp::new();
        write_raw(
            &output.0,
            build_batch(&rows, ExportSchemaVersion::ReviewedTasks).unwrap(),
            &claimed,
        );
        assert!(verify_artifact(&output.0).is_err(), "variant {variant}");
    }
    let batch = build_batch(&plan.rows, ExportSchemaVersion::ReviewedTasks).unwrap();
    let mut fields: Vec<_> = batch
        .schema()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect();
    fields[8] = Field::new(
        "unrecognized_task_json",
        arrow::datatypes::DataType::Utf8,
        true,
    );
    let wrong =
        RecordBatch::try_new(Arc::new(Schema::new(fields)), batch.columns().to_vec()).unwrap();
    let output = Temp::new();
    write_raw(&output.0, wrong, &plan.artifact);
    assert!(verify_artifact(&output.0).is_err());
}

#[tokio::test]
async fn changed_task_rights_change_member_identity_and_deny_snapshot_acknowledgment() {
    let output = Temp::new();
    let store = Store::open_in_memory().await.unwrap();
    store
        .insert_historical_run("run", "{}", None)
        .await
        .unwrap();
    store.put(&task_record("selected")).await.unwrap();
    let original = store.get("selected").await.unwrap();
    let plan = ExportPlan::prepare(std::slice::from_ref(&original), options()).unwrap();
    let receipt = store
        .prepare_export_receipt(&plan, output.0.to_str().unwrap(), ExportPurpose::Engine)
        .await
        .unwrap();
    write_parquet(&plan.rows, &plan.artifact, File::create(&output.0).unwrap()).unwrap();
    let mut changed = original.clone();
    changed
        .task_provenance
        .as_mut()
        .unwrap()
        .rights
        .evidence
        .push("new reviewed terms".into());
    let new_plan = ExportPlan::prepare(&[changed.clone()], options()).unwrap();
    assert_ne!(plan.artifact.artifact_id, new_plan.artifact.artifact_id);
    assert_ne!(plan.members().unwrap(), new_plan.members().unwrap());
    assert_eq!(
        original.hashes.record_hash, changed.hashes.record_hash,
        "task provenance is outside content hash"
    );
    store.put(&changed).await.unwrap();
    let error = store
        .resume_export(&receipt.publication_id)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("selected export record changed"));
    assert!(
        store
            .lifecycle_history("selected")
            .await
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        store.get("selected").await.unwrap().lifecycle.state,
        LifecycleState::Formatted
    );
    assert_eq!(
        verify_artifact(&output.0).unwrap(),
        ArtifactVerification::Verified(plan.artifact)
    );
    // Even a syntactically valid different prompt cannot retain the source/prompt/answer digest.
    changed.messages[0].content = Content::Text("different prompt".into());
    assert!(ExportPlan::prepare(&[changed], options()).is_err());
}
