//! Independent footer and row mutations challenge verification beyond encoder round trips.

use super::*;
use crate::export::{build_batch, write_parquet};
use arrow::datatypes::{Field, Schema};
use gw_schema::{Content, CotPolicy, TrlFormat};
use parquet::arrow::ArrowWriter;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

fn options() -> ExportOptions {
    ExportOptions {
        target: TrlFormat::ChatML,
        cot_policy: CotPolicy::Masked,
        dataset_version: None,
        scope: ExportScope::Records,
    }
}

fn record(id: &str) -> TrainingRecord {
    serde_json::from_value(serde_json::json!({
        "record_id":id, "schema_version":"1.0.0", "training_area":"toy",
        "messages":[{"role":"assistant","content":null,"reasoning":"why"}],
        "provenance":{"run_id":"run","teacher":{"provider":"fixture","slug":"fixture"},"harness_version":"0.1.0"},
        "generation":{},"lifecycle":{"state":"formatted"},"judging":{"panel":[],"verdict":"admit","aggregate":0.95}
    })).unwrap()
}

struct Temp(std::path::PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(std::env::temp_dir().join(format!(
            "gw-artifact-{}-{}.parquet",
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

fn write_raw(path: &Path, batch: RecordBatch, metadata: Vec<KeyValue>) {
    let props = WriterProperties::builder()
        .set_key_value_metadata(Some(metadata))
        .build();
    let mut writer =
        ArrowWriter::try_new(File::create(path).unwrap(), batch.schema(), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn entry(artifact: &ExportArtifact) -> KeyValue {
    KeyValue::new(
        ARTIFACT_METADATA_KEY.into(),
        serde_json::to_string(artifact).unwrap(),
    )
}

#[test]
fn identity_covers_each_projected_column_and_exact_null_float_framing() {
    let row = project(&record("id")).unwrap();
    let original = projected_hash(&row);
    let variants: Vec<Projected> = (0..8)
        .map(|i| {
            let mut changed = row.clone();
            match i {
                0 => changed.record_id.push('x'),
                1 => changed.training_area.push('x'),
                2 => changed.record_hash.push('x'),
                3 => changed.prompt_hash.push('x'),
                4 => changed.verdict = None,
                5 => changed.judge_aggregate = None,
                6 => changed.reasoning_tokens += 1,
                _ => changed.messages_json.push(' '),
            }
            changed
        })
        .collect();
    for variant in variants {
        assert_ne!(projected_hash(&variant), original);
    }
    let mut left = row.clone();
    let mut right = row.clone();
    left.record_id = "ab".into();
    left.training_area = "c".into();
    right.record_id = "a".into();
    right.training_area = "bc".into();
    assert_ne!(projected_hash(&left), projected_hash(&right));
    left = row.clone();
    right = row.clone();
    left.verdict = None;
    right.verdict = Some(String::new());
    assert_ne!(projected_hash(&left), projected_hash(&right));
    left = row.clone();
    right = row;
    left.judge_aggregate = Some(0.0);
    right.judge_aggregate = Some(-0.0);
    assert_ne!(projected_hash(&left), projected_hash(&right));
    left.judge_aggregate = Some(f64::from_bits(0x7ff8_0000_0000_0001));
    right.judge_aggregate = Some(f64::from_bits(0x7ff8_0000_0000_0002));
    assert_ne!(
        projected_hash(&left),
        projected_hash(&right),
        "NaN payload bits remain distinct"
    );
}

#[test]
fn content_identity_is_order_independent_and_includes_full_manifest_and_scope() {
    let records = [record("b"), record("a")];
    let base = ExportPlan::prepare(&records, options()).unwrap();
    let reversed =
        ExportPlan::prepare(&[records[1].clone(), records[0].clone()], options()).unwrap();
    assert_eq!(base.artifact, reversed.artifact);
    assert_eq!(base.rows[0].record_id, "a");
    for i in 0..10 {
        let mut metadata = base.artifact.clone();
        match i {
            0 => metadata.scope = ExportScope::Store,
            1 => metadata.manifest.target = TrlFormat::Gemma4,
            2 => metadata.manifest.cot_policy = CotPolicy::Stripped,
            3 => metadata.manifest.dataset_version = Some(semver::Version::new(1, 2, 3)),
            4 => metadata.manifest.n_records += 1,
            5 => metadata.manifest.hub_commit_sha = Some("published-revision".into()),
            6 => metadata.manifest.decontam_index_id = Some("index".into()),
            7 => metadata.manifest.multi_turn_loss = MultiTurnLoss::FinalTurnOnly,
            8 => metadata.manifest.diversity = Some(Default::default()),
            _ => metadata.metadata_version += 1,
        }
        assert_ne!(
            artifact_identity(&metadata, &base.rows).unwrap(),
            base.artifact.artifact_id
        );
    }
    // The old build hash remains exactly sorted record hashes separated by newline.
    let mut hashes: Vec<_> = base.rows.iter().map(|r| r.record_hash.as_str()).collect();
    hashes.sort();
    let expected = blake3::hash(format!("{}\n", hashes.join("\n")).as_bytes())
        .to_hex()
        .to_string();
    assert_eq!(base.artifact.manifest.build_inputs_hash, expected);
}

#[test]
fn duplicate_record_ids_are_rejected_even_when_unselected() {
    let a = record("same");
    let mut b = a.clone();
    b.lifecycle.state = gw_schema::LifecycleState::Rejected;
    assert!(
        ExportPlan::prepare(&[a, b], options())
            .unwrap_err()
            .to_string()
            .contains("duplicate")
    );
}

#[test]
fn valid_nonempty_and_empty_files_carry_complete_metadata_and_stable_identity() {
    for records in [vec![], vec![record("r")]] {
        let temp = Temp::new();
        let mut opts = options();
        opts.dataset_version = Some(semver::Version::new(2, 3, 4));
        let plan = ExportPlan::prepare(&records, opts).unwrap();
        write_parquet(&plan.rows, &plan.artifact, File::create(&temp.0).unwrap()).unwrap();
        assert_eq!(
            verify_artifact(&temp.0).unwrap(),
            ArtifactVerification::Verified(plan.artifact.clone())
        );
        assert_eq!(plan.artifact.manifest.n_admitted, records.len() as u64);
        let metadata = serde_json::to_string(&plan.artifact).unwrap();
        assert!(!metadata.contains(temp.0.to_str().unwrap()));
        assert!(!metadata.contains("destination"));
        assert_eq!(
            plan.artifact.manifest.dataset_version,
            Some(semver::Version::new(2, 3, 4))
        );
    }
}

#[test]
fn missing_legacy_metadata_never_uses_adjacent_manifest() {
    let temp = Temp::new();
    let plan = ExportPlan::prepare(&[record("r")], options()).unwrap();
    write_raw(&temp.0, build_batch(&plan.rows).unwrap(), vec![]);
    let sidecar = temp.0.with_extension("parquet.manifest.json");
    std::fs::write(&sidecar, serde_json::to_vec(&plan.artifact).unwrap()).unwrap();
    assert_eq!(
        verify_artifact(&temp.0).unwrap(),
        ArtifactVerification::MissingLegacyMetadata
    );
    // Ordinary Parquet readers still have full access to historical rows.
    let rows: usize = ParquetRecordBatchReaderBuilder::try_new(File::open(&temp.0).unwrap())
        .unwrap()
        .build()
        .unwrap()
        .map(|b| b.unwrap().num_rows())
        .sum();
    assert_eq!(rows, 1);
    std::fs::remove_file(sidecar).unwrap();
}

#[test]
fn duplicate_unknown_or_tampered_footer_is_rejected_including_empty_artifacts() {
    for records in [vec![], vec![record("r")]] {
        let temp = Temp::new();
        let plan = ExportPlan::prepare(&records, options()).unwrap();
        write_raw(
            &temp.0,
            build_batch(&plan.rows).unwrap(),
            vec![entry(&plan.artifact), entry(&plan.artifact)],
        );
        assert!(
            verify_artifact(&temp.0)
                .unwrap_err()
                .to_string()
                .contains("duplicate")
        );
        let mut altered = plan.artifact.clone();
        altered.metadata_version = 99;
        write_raw(
            &temp.0,
            build_batch(&plan.rows).unwrap(),
            vec![entry(&altered)],
        );
        assert!(
            verify_artifact(&temp.0)
                .unwrap_err()
                .to_string()
                .contains("unsupported")
        );
        altered = plan.artifact.clone();
        altered.manifest.dataset_version = Some(semver::Version::new(9, 0, 0));
        write_raw(
            &temp.0,
            build_batch(&plan.rows).unwrap(),
            vec![entry(&altered)],
        );
        assert!(
            verify_artifact(&temp.0)
                .unwrap_err()
                .to_string()
                .contains("identity mismatch")
        );
        let mut value = serde_json::to_value(&plan.artifact).unwrap();
        value["manifest"]["invented"] = true.into();
        write_raw(
            &temp.0,
            build_batch(&plan.rows).unwrap(),
            vec![KeyValue::new(
                ARTIFACT_METADATA_KEY.into(),
                value.to_string(),
            )],
        );
        assert!(verify_artifact(&temp.0).is_err());
    }
}

#[test]
fn wrong_schema_count_duplicate_ids_and_undecodable_messages_fail_readback() {
    let temp = Temp::new();
    let plan = ExportPlan::prepare(&[record("r")], options()).unwrap();
    let batch = build_batch(&plan.rows).unwrap();
    let mut fields: Vec<Field> = batch
        .schema()
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect();
    fields[0] = Field::new("wrong_id", fields[0].data_type().clone(), false);
    let wrong =
        RecordBatch::try_new(Arc::new(Schema::new(fields)), batch.columns().to_vec()).unwrap();
    write_raw(&temp.0, wrong, vec![entry(&plan.artifact)]);
    assert!(
        verify_artifact(&temp.0)
            .unwrap_err()
            .to_string()
            .contains("schema")
    );
    let mut metadata = plan.artifact.clone();
    metadata.manifest.n_admitted = 2;
    write_raw(
        &temp.0,
        build_batch(&plan.rows).unwrap(),
        vec![entry(&metadata)],
    );
    assert!(
        verify_artifact(&temp.0)
            .unwrap_err()
            .to_string()
            .contains("counts")
    );
    let mut rows = plan.rows.clone();
    rows.push(rows[0].clone());
    metadata.manifest.n_records = 2;
    metadata.manifest.build_inputs_hash = shard_content_hash(&rows);
    metadata.artifact_id = artifact_identity(&metadata, &rows).unwrap();
    write_raw(&temp.0, build_batch(&rows).unwrap(), vec![entry(&metadata)]);
    assert!(
        verify_artifact(&temp.0)
            .unwrap_err()
            .to_string()
            .contains("duplicated")
    );
    rows = plan.rows.clone();
    rows[0].messages_json = "[{}]".into();
    metadata = plan.artifact.clone();
    metadata.artifact_id = artifact_identity(&metadata, &rows).unwrap();
    write_raw(&temp.0, build_batch(&rows).unwrap(), vec![entry(&metadata)]);
    assert!(
        verify_artifact(&temp.0)
            .unwrap_err()
            .to_string()
            .contains("serde_json")
    );
}

#[test]
fn verifier_reads_tail_batches_and_detects_changed_conversation() {
    let temp = Temp::new();
    let records: Vec<_> = (0..2050).map(|i| record(&format!("r{i:04}"))).collect();
    let mut plan = ExportPlan::prepare(&records, options()).unwrap();
    write_parquet(&plan.rows, &plan.artifact, File::create(&temp.0).unwrap()).unwrap();
    assert!(matches!(
        verify_artifact(&temp.0).unwrap(),
        ArtifactVerification::Verified(_)
    ));
    let mut last: Vec<Message> =
        serde_json::from_str(&plan.rows.last().unwrap().messages_json).unwrap();
    last[0].content = Content::Text("changed in third reader batch".into());
    plan.rows.last_mut().unwrap().messages_json = serde_json::to_string(&last).unwrap();
    write_parquet(&plan.rows, &plan.artifact, File::create(&temp.0).unwrap()).unwrap();
    assert!(
        verify_artifact(&temp.0)
            .unwrap_err()
            .to_string()
            .contains("identity mismatch")
    );
}
