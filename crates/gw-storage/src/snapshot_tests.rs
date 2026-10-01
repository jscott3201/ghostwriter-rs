//! Rust-generated interoperability fixtures and snapshot/path equivalence.
use super::*;
use crate::export::write_parquet;
use gw_schema::{CotPolicy, TrlFormat};

fn plan(version: ExportSchemaVersion, empty: bool) -> ExportPlan {
    let records = if empty {
        vec![]
    } else {
        (0..2).map(|index| {
        serde_json::from_value::<TrainingRecord>(serde_json::json!({
            "record_id":format!("fixture-{index}"), "schema_version":"1.0.0", "training_area":"fixture",
            "messages":[
                {"role":"system","content":"SYSTEM same 🙂中e\u{301}"},
                {"role":"user","content":"USER same 🙂中e\u{301}"},
                {"role":"assistant","content":"ANSWER same 🙂中e\u{301}","reasoning":"REASON same 🙂中e\u{301}",
                 "reasoning_details":[{"type":"reasoning.text","index":2,"text":"REASON "},{"type":"reasoning.text","index":5,"text":"same 🙂中e\u{301}"}]},
                {"role":"user","content":"USER second"},
                {"role":"assistant","content":if index == 0 {"ANSWER final"} else {"ANSWER final substantially longer final response"},"reasoning":"REASON final"}
            ],
            "provenance":{"run_id":"fixture","teacher":{"provider":"fixture","slug":"fixture"},"harness_version":"0.1.0"},
            "generation":{},"lifecycle":{"state":"formatted"},"judging":{"panel":[],"verdict":"admit","aggregate":0.9}
        })).unwrap()
    }).collect()
    };
    let mut plan = ExportPlan::prepare(
        &records,
        ExportOptions {
            target: TrlFormat::ChatML,
            cot_policy: CotPolicy::Supervised,
            dataset_version: None,
            scope: ExportScope::Records,
        },
    )
    .unwrap();
    plan.rows = records
        .iter()
        .map(|r| project(r, version).unwrap())
        .collect();
    // Exercise raw, valid JSON bytes that must never be reserialized by the Python bridge.
    if let Some(row) = plan.rows.first_mut() {
        row.messages_json = row.messages_json.replace("same", "s\\u0061me");
    }
    plan.artifact.manifest.column_schema_version = version;
    plan.artifact.artifact_id = artifact_identity(&plan.artifact, &plan.rows).unwrap();
    plan
}

#[test]
fn snapshot_golden_artifacts_cover_empty_nonempty_v2_v3_v4_and_raw_unicode() {
    let fixture_root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../adapters/trl/tests/fixtures");
    for (version, name) in [
        (ExportSchemaVersion::CanonicalMessages, "v2"),
        (ExportSchemaVersion::ReviewedTasks, "v3"),
        (ExportSchemaVersion::RecordOrigins, "v4"),
    ] {
        for empty in [false, true] {
            let plan = plan(version, empty);
            let mut encoded = Vec::new();
            write_parquet(&plan.rows, &plan.artifact, &mut encoded).unwrap();
            let report = verify_artifact_snapshot(encoded.clone()).unwrap();
            assert_eq!(report.report_version, 1);
            assert_eq!(report.artifact, plan.artifact);
            assert_eq!(report.byte_length, encoded.len() as u64);
            assert_eq!(
                report.snapshot_blake3,
                blake3::hash(&encoded).to_hex().to_string()
            );
            assert_ne!(report.snapshot_blake3, report.artifact.artifact_id);
            let file_name = format!("{name}-{}.parquet", if empty { "empty" } else { "text" });
            // Explicit regeneration only; ordinary tests never modify fixture files.
            if let Some(output) = std::env::var_os("GW_REGENERATE_TRL_FIXTURES") {
                std::fs::write(Path::new(&output).join(&file_name), &encoded).unwrap();
            }
            if version == ExportSchemaVersion::RecordOrigins
                && let Some(output) = std::env::var_os("GW_REGENERATE_ORIGIN_TRL_FIXTURES")
            {
                std::fs::write(Path::new(&output).join(&file_name), &encoded).unwrap();
            }
            let frozen = std::fs::read(fixture_root.join(&file_name)).unwrap();
            let frozen_report = verify_artifact_snapshot(frozen).unwrap();
            assert_eq!(frozen_report.artifact, plan.artifact);
            assert_eq!(
                verify_artifact(fixture_root.join(file_name)).unwrap(),
                ArtifactVerification::Verified(plan.artifact)
            );
        }
    }
}

#[test]
fn owned_snapshot_survives_source_path_replacement() {
    let first = plan(ExportSchemaVersion::ReviewedTasks, false);
    let second = plan(ExportSchemaVersion::ReviewedTasks, true);
    let path = std::env::temp_dir().join(format!(
        "gw-snapshot-replace-{}.parquet",
        std::process::id()
    ));
    write_parquet(&first.rows, &first.artifact, File::create(&path).unwrap()).unwrap();
    let captured = std::fs::read(&path).unwrap();
    write_parquet(&second.rows, &second.artifact, File::create(&path).unwrap()).unwrap();
    assert_eq!(
        verify_artifact_snapshot(captured).unwrap().artifact,
        first.artifact
    );
    assert_eq!(
        verify_artifact(&path).unwrap(),
        ArtifactVerification::Verified(second.artifact)
    );
    std::fs::remove_file(path).unwrap();
}

#[test]
fn snapshot_large_and_reviewed_task_fixtures() {
    let fixture_root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../adapters/trl/tests/fixtures");
    let mut large = plan(ExportSchemaVersion::ReviewedTasks, false);
    let prototype = large.rows[0].clone();
    large.rows = (0..1025)
        .map(|index| {
            let mut row = prototype.clone();
            row.record_id = format!("row-{index:04}");
            row
        })
        .collect();
    large.artifact.manifest.n_records = 1025;
    large.artifact.manifest.n_admitted = 1025;
    large.artifact.manifest.build_inputs_hash = shard_content_hash(&large.rows);
    large.artifact.artifact_id = artifact_identity(&large.artifact, &large.rows).unwrap();
    let mut reviewed = ExportPlan::prepare(
        &[super::task_tests::task_record("reviewed-task")],
        ExportOptions {
            target: TrlFormat::ChatML,
            cot_policy: CotPolicy::Masked,
            dataset_version: None,
            scope: ExportScope::Records,
        },
    )
    .unwrap();
    reviewed.artifact.manifest.column_schema_version = ExportSchemaVersion::ReviewedTasks;
    for row in &mut reviewed.rows {
        row.origin_json = None;
    }
    reviewed.artifact.artifact_id = artifact_identity(&reviewed.artifact, &reviewed.rows).unwrap();
    for (name, plan) in [("v3-many.parquet", large), ("v3-task.parquet", reviewed)] {
        let mut encoded = Vec::new();
        write_parquet(&plan.rows, &plan.artifact, &mut encoded).unwrap();
        assert_eq!(
            verify_artifact_snapshot(encoded.clone()).unwrap().artifact,
            plan.artifact
        );
        if let Some(output) = std::env::var_os("GW_REGENERATE_TRL_FIXTURES") {
            std::fs::write(Path::new(&output).join(name), &encoded).unwrap();
        }
        assert_eq!(
            verify_artifact_snapshot(std::fs::read(fixture_root.join(name)).unwrap())
                .unwrap()
                .artifact,
            plan.artifact
        );
    }
}

#[test]
fn snapshot_heldout_task_fixtures_remain_valid_artifacts() {
    let fixture_root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../adapters/trl/tests/fixtures");
    for (name, role) in [
        ("validation", gw_schema::TaskSplitRole::Validation),
        ("test", gw_schema::TaskSplitRole::Test),
    ] {
        let mut record = super::task_tests::task_record("heldout-task");
        record.task_provenance.as_mut().unwrap().split.role = role;
        let mut plan = ExportPlan::prepare(
            &[record],
            ExportOptions {
                target: TrlFormat::ChatML,
                cot_policy: CotPolicy::Masked,
                dataset_version: None,
                scope: ExportScope::Records,
            },
        )
        .unwrap();
        plan.artifact.manifest.column_schema_version = ExportSchemaVersion::ReviewedTasks;
        for row in &mut plan.rows {
            row.origin_json = None;
        }
        plan.artifact.artifact_id = artifact_identity(&plan.artifact, &plan.rows).unwrap();
        let mut encoded = Vec::new();
        write_parquet(&plan.rows, &plan.artifact, &mut encoded).unwrap();
        let file_name = format!("v3-{name}.parquet");
        assert_eq!(
            verify_artifact_snapshot(encoded.clone()).unwrap().artifact,
            plan.artifact
        );
        if let Some(output) = std::env::var_os("GW_REGENERATE_TRL_FIXTURES") {
            std::fs::write(Path::new(&output).join(&file_name), &encoded).unwrap();
        }
        assert_eq!(
            verify_artifact_snapshot(std::fs::read(fixture_root.join(file_name)).unwrap())
                .unwrap()
                .artifact,
            plan.artifact
        );
    }
}

#[test]
fn snapshot_long_conversations_preserve_complete_training_context() {
    let mut plan = plan(ExportSchemaVersion::ReviewedTasks, false);
    for (index, row) in plan.rows.iter_mut().enumerate() {
        let mut messages: Vec<Message> = serde_json::from_str(&row.messages_json).unwrap();
        messages[1].content = gw_schema::Content::Text("context ".repeat(1050 + index * 10));
        row.messages_json = serde_json::to_string(&messages).unwrap();
    }
    plan.artifact.manifest.build_inputs_hash = shard_content_hash(&plan.rows);
    plan.artifact.artifact_id = artifact_identity(&plan.artifact, &plan.rows).unwrap();
    let mut encoded = Vec::new();
    write_parquet(&plan.rows, &plan.artifact, &mut encoded).unwrap();
    let name = "v3-long.parquet";
    if let Some(output) = std::env::var_os("GW_REGENERATE_TRL_FIXTURES") {
        std::fs::write(Path::new(&output).join(name), &encoded).unwrap();
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
}
