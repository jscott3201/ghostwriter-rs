//! Canonical tools are data, independently bound even when stored record hashes are stale.
use super::*;
use crate::export::{build_batch, write_parquet};
use crate::{ExportPurpose, Store};
use gw_schema::{CotPolicy, TrlFormat};
use serde_json::{Value, json};

fn record() -> TrainingRecord {
    let source: Value = serde_json::from_str(include_str!(
        "../../gw-format/tests/fixtures/tool-training-source.json"
    ))
    .unwrap();
    serde_json::from_value(json!({
        "record_id":"tool-fixture","schema_version":"1.0.0","training_area":"tools",
        "messages":source["messages"],"tools":source["tools"],
        "provenance":{"run_id":"tools-run","teacher":{"provider":"fixture","slug":"fixture"},"harness_version":"0.1.0"},
        "generation":{},"lifecycle":{"state":"formatted"},"judging":{"panel":[],"verdict":"admit","aggregate":0.9},
        "hashes":{"record_hash":"deliberately-stale","prompt_hash":"fixed-prompt"}
    })).unwrap()
}
fn plan(record: &TrainingRecord) -> ExportPlan {
    ExportPlan::prepare(
        std::slice::from_ref(record),
        ExportOptions {
            target: TrlFormat::OpenAiMessages,
            cot_policy: CotPolicy::Supervised,
            dataset_version: None,
            scope: ExportScope::Run {
                run_id: "tools-run".into(),
            },
        },
    )
    .unwrap()
}
fn bytes(plan: &ExportPlan) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_parquet(&plan.rows, &plan.artifact, &mut bytes).unwrap();
    bytes
}
#[test]
fn tool_definitions_roundtrip_and_bind_payload_without_trusting_record_hash() {
    let original = record();
    let expected = plan(&original);
    let (report, rows) = verify_snapshot_with_rows(bytes(&expected)).unwrap();
    assert_eq!(report.artifact, expected.artifact);
    assert_eq!(
        serde_json::from_str::<Vec<Value>>(rows[0].tools_json.as_ref().unwrap()).unwrap(),
        original.tools.clone().unwrap()
    );
    assert_eq!(
        serde_json::from_str::<Vec<Message>>(&rows[0].messages_json).unwrap(),
        original.messages
    );
    let mut identities = BTreeSet::new();
    for tools in [
        original.tools.clone(),
        Some(vec![]),
        None,
        Some(vec![json!({"type":"unsupported","nested":[null,3,true]})]),
    ] {
        let mut changed = original.clone();
        changed.tools = tools;
        let current = plan(&changed);
        assert_eq!(current.rows[0].record_hash, "deliberately-stale");
        assert!(identities.insert(current.artifact.artifact_id.clone()));
        verify_artifact_snapshot(bytes(&current)).unwrap();
    }
    let destination = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../adapters/trl/tests/fixtures/v5-tools.parquet");
    if let Some(directory) = std::env::var_os("GW_REGENERATE_TOOL_FIXTURES") {
        std::fs::write(
            Path::new(&directory).join("v5-tools.parquet"),
            bytes(&expected),
        )
        .unwrap();
    }
    let mut text = original.clone();
    text.messages = serde_json::from_value(json!([{"role":"user","content":"Question"},{"role":"assistant","content":"Answer","reasoning":"Reason"}])).unwrap();
    for (name, tools) in [
        ("definitions-only", original.tools.clone()),
        ("empty-tools", Some(vec![])),
    ] {
        text.tools = tools;
        let expected = plan(&text);
        let filename = format!("v5-{name}.parquet");
        if let Some(directory) = std::env::var_os("GW_REGENERATE_TOOL_FIXTURES") {
            std::fs::write(Path::new(&directory).join(&filename), bytes(&expected)).unwrap();
        }
        let path = destination.parent().unwrap().join(filename);
        assert_eq!(
            verify_artifact_snapshot(std::fs::read(path).unwrap())
                .unwrap()
                .artifact,
            expected.artifact
        );
    }
    let saved = verify_artifact_snapshot(std::fs::read(destination).unwrap()).unwrap();
    assert_eq!(saved.artifact, expected.artifact);
}
#[test]
fn independent_tool_column_mutations_fail_verification() {
    let plan = plan(&record());
    for payload in [
        "{",
        "{}",
        "null",
        "[1]",
        "[]",
        "[ ]",
        "[{\"same\":1,\"same\":2}]",
    ] {
        let mut rows = plan.rows.clone();
        rows[0].tools_json = Some(payload.into());
        let mut artifact = plan.artifact.clone();
        // Invalid shapes/canonical encodings must fail even with a recomputed outer identity.
        if !matches!(payload, "[1]" | "[]") {
            artifact.artifact_id = artifact_identity(&artifact, &rows).unwrap();
        }
        let mut output = Vec::new();
        write_parquet(&rows, &artifact, &mut output).unwrap();
        assert!(verify_artifact_snapshot(output).is_err(), "{payload}");
    }
    let batch = build_batch(&plan.rows, ExportSchemaVersion::ToolDefinitions).unwrap();
    let missing = batch.project(&(0..10).collect::<Vec<_>>()).unwrap();
    let props = parquet::file::properties::WriterProperties::builder()
        .set_key_value_metadata(Some(vec![parquet::file::metadata::KeyValue::new(
            ARTIFACT_METADATA_KEY.into(),
            serde_json::to_string(&plan.artifact).unwrap(),
        )]))
        .build();
    let mut output = Vec::new();
    let mut writer =
        parquet::arrow::ArrowWriter::try_new(&mut output, missing.schema(), Some(props)).unwrap();
    writer.write(&missing).unwrap();
    writer.close().unwrap();
    assert!(verify_artifact_snapshot(output).is_err());
    let mut corrupt = bytes(&plan);
    corrupt.truncate(corrupt.len() / 2);
    assert!(verify_artifact_snapshot(corrupt).is_err());
}
#[test]
fn historical_columns_and_identities_ignore_unexported_tools() {
    for version in [
        ExportSchemaVersion::CanonicalMessages,
        ExportSchemaVersion::ReviewedTasks,
        ExportSchemaVersion::RecordOrigins,
    ] {
        let original = record();
        let row = project(&original, version).unwrap();
        let mut changed = original.clone();
        changed.tools = Some(vec![json!({"changed":true})]);
        assert_eq!(
            projected_hash(&row, version).unwrap(),
            projected_hash(&project(&changed, version).unwrap(), version).unwrap()
        );
        assert!(row.tools_json.is_none());
        let mut historical = plan(&original);
        historical.rows = vec![row];
        historical.artifact.manifest.column_schema_version = version;
        historical.artifact.artifact_id =
            artifact_identity(&historical.artifact, &historical.rows).unwrap();
        verify_artifact_snapshot(bytes(&historical)).unwrap();
    }
}
#[tokio::test]
async fn frozen_tools_recover_and_changed_definitions_fail_before_acknowledgment() {
    for acknowledged in [false, true] {
        for changed in [false, true] {
            let store = Store::open_in_memory().await.unwrap();
            store
                .insert_historical_run("tools-run", "{}", None)
                .await
                .unwrap();
            store.replace_record_for_import(&record()).await.unwrap();
            let frozen = plan(&store.get("tool-fixture").await.unwrap());
            let output = std::env::temp_dir().join(format!(
                "gw-tools-recovery-{}-{acknowledged}-{changed}.parquet",
                std::process::id()
            ));
            let receipt = store
                .prepare_export_receipt(&frozen, output.to_str().unwrap(), ExportPurpose::Engine)
                .await
                .unwrap();
            if acknowledged {
                store.resume_export(&receipt.publication_id).await.unwrap();
                std::fs::remove_file(&output).unwrap();
            }
            if changed {
                let mut record = store.get("tool-fixture").await.unwrap();
                record.tools.as_mut().unwrap()[0]["function"]["description"] =
                    json!("changed after prepare");
                store.replace_record_for_import(&record).await.unwrap();
                assert!(store.resume_export(&receipt.publication_id).await.is_err());
                assert_eq!(
                    store
                        .load_export_receipt(&receipt.publication_id)
                        .await
                        .unwrap()
                        .acknowledged,
                    acknowledged
                );
            } else {
                let recovered = store.resume_export(&receipt.publication_id).await.unwrap();
                assert_eq!(recovered.artifact, frozen.artifact);
                assert_eq!(std::fs::read(&output).unwrap(), bytes(&frozen));
            }
            let _ = std::fs::remove_file(output);
        }
    }
}

#[test]
fn screening_projection_binds_tools_presence_and_payload_in_its_new_domain() {
    let record = record();
    let origin = record.origin.projection();
    let identity = |tools| {
        crate::screening_binding::screening_projection_id(
            &record.training_area,
            &record.messages,
            None,
            None,
            record.judging.verdict,
            Some(&origin),
            tools,
        )
        .unwrap()
    };
    let old = identity(None);
    let absent = identity(Some(None));
    let empty = identity(Some(Some(&[])));
    let actual = identity(Some(record.tools.as_deref()));
    let mut changed = record.tools.clone().unwrap();
    changed[0]["function"]["description"] = json!("different");
    let changed = identity(Some(Some(&changed)));
    assert_eq!(
        [old, absent, empty, actual, changed]
            .into_iter()
            .collect::<BTreeSet<_>>()
            .len(),
        5
    );
}

#[tokio::test]
async fn historical_v2_v3_v4_receipts_recover_their_frozen_bytes() {
    for version in [
        ExportSchemaVersion::CanonicalMessages,
        ExportSchemaVersion::ReviewedTasks,
        ExportSchemaVersion::RecordOrigins,
    ] {
        for acknowledged in [false, true] {
            let store = Store::open_in_memory().await.unwrap();
            store
                .insert_historical_run("tools-run", "{}", None)
                .await
                .unwrap();
            store.replace_record_for_import(&record()).await.unwrap();
            let original = store.get("tool-fixture").await.unwrap();
            let mut frozen = plan(&original);
            frozen.rows = vec![project(&original, version).unwrap()];
            frozen.artifact.manifest.column_schema_version = version;
            frozen.artifact.artifact_id =
                artifact_identity(&frozen.artifact, &frozen.rows).unwrap();
            let output = std::env::temp_dir().join(format!(
                "gw-tools-legacy-{}-{}-{acknowledged}.parquet",
                std::process::id(),
                version as u8
            ));
            let receipt = store
                .prepare_export_receipt(&frozen, output.to_str().unwrap(), ExportPurpose::Engine)
                .await
                .unwrap();
            if acknowledged {
                store.resume_export(&receipt.publication_id).await.unwrap();
                std::fs::remove_file(&output).unwrap();
            }
            let recovered = store.resume_export(&receipt.publication_id).await.unwrap();
            assert_eq!(recovered.artifact, frozen.artifact);
            assert_eq!(std::fs::read(&output).unwrap(), bytes(&frozen));
            std::fs::remove_file(output).unwrap();
        }
    }
}
