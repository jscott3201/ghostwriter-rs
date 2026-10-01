//! Frozen v2 framing and publication recovery remain supported after v3 becomes the write version.
use super::*;
use crate::export::write_parquet;
use crate::{ExportPurpose, PublicationDisposition, Store};
use gw_schema::{CotPolicy, LifecycleState, TrlFormat};
use std::sync::atomic::{AtomicU64, Ordering};

struct Temp(std::path::PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(std::env::temp_dir().join(format!(
            "gw-v2-recovery-{}-{}.parquet",
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

fn record(id: &str) -> TrainingRecord {
    serde_json::from_value(serde_json::json!({
        "record_id":id,"schema_version":"1.0.0","training_area":"toy",
        "messages":[{"role":"user","content":"question"},{"role":"assistant","content":"answer","reasoning":"why"}],
        "provenance":{"run_id":"run","teacher":{"provider":"fixture","slug":"fixture"},"harness_version":"0.1.0"},
        "generation":{},"lifecycle":{"state":"formatted"},"judging":{"panel":[],"verdict":"admit","aggregate":0.9}
    })).unwrap()
}
fn legacy_plan(records: &[TrainingRecord]) -> ExportPlan {
    let mut plan = ExportPlan::prepare(
        records,
        ExportOptions {
            target: TrlFormat::ChatML,
            cot_policy: CotPolicy::Masked,
            dataset_version: None,
            scope: ExportScope::Run {
                run_id: "run".into(),
            },
        },
    )
    .unwrap();
    plan.rows = records
        .iter()
        .map(|rec| project(rec, ExportSchemaVersion::CanonicalMessages).unwrap())
        .collect();
    plan.rows.sort_by(|a, b| a.record_id.cmp(&b.record_id));
    plan.artifact.manifest.column_schema_version = ExportSchemaVersion::CanonicalMessages;
    plan.artifact.artifact_id = artifact_identity(&plan.artifact, &plan.rows).unwrap();
    plan
}

// Independent transcription of the frozen v2 wire framing. It does not call projected_hash,
// artifact_identity, canonical_metadata_json, or the encoder to derive the golden reference.
fn reference_identity(plan: &ExportPlan) -> String {
    fn put(bytes: &mut Vec<u8>, value: &[u8]) {
        bytes.extend((value.len() as u64).to_be_bytes());
        bytes.extend(value);
    }
    let mut body = Vec::new();
    body.extend(1u32.to_be_bytes());
    for value in [
        serde_json::to_value(&plan.artifact.scope).unwrap(),
        serde_json::to_value(&plan.artifact.manifest).unwrap(),
    ] {
        put(&mut body, &serde_json::to_vec(&value).unwrap());
    }
    body.extend((plan.rows.len() as u64).to_be_bytes());
    for row in &plan.rows {
        let mut bytes = Vec::new();
        for value in [
            &row.record_id,
            &row.training_area,
            &row.record_hash,
            &row.prompt_hash,
        ] {
            put(&mut bytes, value.as_bytes());
        }
        match &row.verdict {
            None => bytes.push(0),
            Some(value) => {
                bytes.push(1);
                put(&mut bytes, value.as_bytes());
            }
        }
        match row.judge_aggregate {
            None => bytes.push(0),
            Some(value) => {
                bytes.push(1);
                bytes.extend(value.to_bits().to_be_bytes());
            }
        }
        bytes.extend(row.reasoning_tokens.to_be_bytes());
        put(&mut bytes, row.messages_json.as_bytes());
        let mut hash = blake3::Hasher::new_derive_key("ghostwriter.export.projected-row.v1");
        hash.update(&bytes);
        put(&mut body, hash.finalize().to_hex().as_bytes());
    }
    let mut hash = blake3::Hasher::new_derive_key("ghostwriter.export.artifact.v1");
    hash.update(&body);
    hash.finalize().to_hex().to_string()
}

#[test]
fn fixed_v2_empty_and_nonempty_artifact_identities_and_readback() {
    let mut fixed = record("fixed");
    fixed.hashes.record_hash = "fixed-record-hash".into();
    fixed.hashes.prompt_hash = "fixed-prompt-hash".into();
    let plans = [legacy_plan(&[]), legacy_plan(&[fixed])];
    let identities: Vec<_> = plans.iter().map(reference_identity).collect();
    assert_eq!(
        identities,
        [
            "f6556fed3d6688f68044b7e16c436b195594659bcf79d15bbfc93f49737d56a4",
            "e5e39a59cc1ff94373d6d974cb9d5bb998ae038532348165bf82782dbc955a22",
        ]
    );
    for (plan, expected) in plans.into_iter().zip(identities) {
        assert_eq!(plan.artifact.artifact_id, expected);
        let output = Temp::new();
        write_parquet(&plan.rows, &plan.artifact, File::create(&output.0).unwrap()).unwrap();
        let builder =
            ParquetRecordBatchReaderBuilder::try_new(File::open(&output.0).unwrap()).unwrap();
        assert_eq!(builder.schema().fields().len(), 8);
        assert_eq!(
            verify_artifact(&output.0).unwrap(),
            ArtifactVerification::Verified(plan.artifact)
        );
    }
}

#[tokio::test]
async fn prepared_and_acknowledged_v2_recover_present_missing_and_replaced_files() {
    for acknowledged in [false, true] {
        for file_state in ["present", "missing", "replaced"] {
            let output = Temp::new();
            let store = Store::open_in_memory().await.unwrap();
            store
                .insert_historical_run("run", "{}", None)
                .await
                .unwrap();
            store
                .replace_record_for_import(&record("selected"))
                .await
                .unwrap();
            let plan = legacy_plan(&[store.get("selected").await.unwrap()]);
            let receipt = store
                .prepare_export_receipt(&plan, output.0.to_str().unwrap(), ExportPurpose::Engine)
                .await
                .unwrap();
            write_parquet(&plan.rows, &plan.artifact, File::create(&output.0).unwrap()).unwrap();
            if acknowledged {
                store.resume_export(&receipt.publication_id).await.unwrap();
            }
            let before = store.lifecycle_history("selected").await.unwrap();
            store
                .replace_record_for_import(&record("later"))
                .await
                .unwrap();
            match file_state {
                "missing" => std::fs::remove_file(&output.0).unwrap(),
                "replaced" => std::fs::write(&output.0, b"unrelated replacement").unwrap(),
                _ => (),
            }
            let result = store.resume_export(&receipt.publication_id).await.unwrap();
            assert_eq!(result.publication_id, receipt.publication_id);
            assert_eq!(result.artifact, plan.artifact);
            assert_eq!(
                result.artifact.manifest.column_schema_version,
                ExportSchemaVersion::CanonicalMessages
            );
            assert_eq!(
                result.disposition,
                if file_state == "present" {
                    PublicationDisposition::AcknowledgedExisting
                } else {
                    PublicationDisposition::Republished
                }
            );
            assert_eq!(
                verify_artifact(&output.0).unwrap(),
                ArtifactVerification::Verified(plan.artifact.clone())
            );
            let restored = store
                .load_export_receipt(&receipt.publication_id)
                .await
                .unwrap();
            assert_eq!(restored.artifact, receipt.artifact);
            assert_eq!(restored.members, receipt.members);
            assert_eq!(restored.destination, receipt.destination);
            assert_eq!(restored.purpose, receipt.purpose);
            assert!(restored.acknowledged);
            let history = store.lifecycle_history("selected").await.unwrap();
            assert_eq!(history.len(), 1);
            if acknowledged {
                assert_eq!(history, before);
            }
            assert_eq!(
                store.get("later").await.unwrap().lifecycle.state,
                LifecycleState::Formatted
            );
            let again = store.resume_export(&receipt.publication_id).await.unwrap();
            assert_eq!(again.artifact, plan.artifact);
            assert!(again.advanced_record_ids.is_empty());
            assert_eq!(store.lifecycle_history("selected").await.unwrap(), history);
        }
    }
}

#[tokio::test]
async fn v2_receipt_cannot_drop_new_task_provenance() {
    let output = Temp::new();
    let store = Store::open_in_memory().await.unwrap();
    store
        .insert_historical_run("run", "{}", None)
        .await
        .unwrap();
    let mut record = super::task_tests::task_record("selected");
    let task = record.task_provenance.take().unwrap();
    store.replace_record_for_import(&record).await.unwrap();
    let plan = legacy_plan(&[store.get("selected").await.unwrap()]);
    let receipt = store
        .prepare_export_receipt(&plan, output.0.to_str().unwrap(), ExportPurpose::Engine)
        .await
        .unwrap();
    record.task_provenance = Some(task);
    store.replace_record_for_import(&record).await.unwrap();
    let error = store
        .resume_export(&receipt.publication_id)
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("v2 publication cannot attest task provenance")
    );
    assert!(!output.0.exists());
    assert!(
        store
            .lifecycle_history("selected")
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn v2_recovery_retains_legacy_null_history_for_prepared_and_acknowledged_receipts() {
    for purpose in [ExportPurpose::Engine, ExportPurpose::Standalone] {
        for acknowledged in [false, true] {
            let output = Temp::new();
            let store = Store::open_in_memory().await.unwrap();
            store
                .insert_historical_run("run", "{}", None)
                .await
                .unwrap();
            let mut historical = record("selected");
            historical
                .lifecycle
                .history
                .push(gw_schema::StateTransition {
                    state: LifecycleState::Formatted,
                    at: "legacy timestamp".into(),
                    attempt: 0,
                });
            store.replace_record_for_import(&historical).await.unwrap();
            sqlx::query("UPDATE lifecycle_history SET history_ordinal=NULL, attempt=NULL, detail='legacy detail'").execute(store.pool()).await.unwrap();
            let legacy: String = sqlx::query_scalar("SELECT json_array(id,record_id,state,at,detail,mutation_id,history_ordinal,attempt) FROM lifecycle_history WHERE history_ordinal IS NULL").fetch_one(store.pool()).await.unwrap();
            let plan = legacy_plan(&[store.get("selected").await.unwrap()]);
            let receipt = store
                .prepare_export_receipt(&plan, output.0.to_str().unwrap(), purpose)
                .await
                .unwrap();
            write_parquet(&plan.rows, &plan.artifact, File::create(&output.0).unwrap()).unwrap();
            if acknowledged {
                store.resume_export(&receipt.publication_id).await.unwrap();
            }
            let before = std::fs::read(&output.0).unwrap();
            let recovered = store.resume_export(&receipt.publication_id).await.unwrap();
            assert_eq!(recovered.artifact, plan.artifact);
            assert_eq!(
                recovered.disposition,
                PublicationDisposition::AcknowledgedExisting
            );
            assert!(std::fs::read(&output.0).unwrap() == before);
            let retained: String = sqlx::query_scalar("SELECT json_array(id,record_id,state,at,detail,mutation_id,history_ordinal,attempt) FROM lifecycle_history WHERE history_ordinal IS NULL").fetch_one(store.pool()).await.unwrap();
            assert_eq!(retained, legacy);
            assert_eq!(
                store.lifecycle_history("selected").await.unwrap().len(),
                if purpose == ExportPurpose::Engine {
                    2
                } else {
                    1
                }
            );
            let current = store.get("selected").await.unwrap();
            assert_eq!(
                current.lifecycle.history[0],
                historical.lifecycle.history[0]
            );
            assert_eq!(
                current.lifecycle.state,
                if purpose == ExportPurpose::Engine {
                    LifecycleState::Exported
                } else {
                    LifecycleState::Formatted
                }
            );
            assert!(
                store
                    .resume_export(&receipt.publication_id)
                    .await
                    .unwrap()
                    .advanced_record_ids
                    .is_empty()
            );
        }
    }
}

fn verify_artifact(path: impl AsRef<Path>) -> Result<ArtifactVerification> {
    let disk = super::verify_artifact(path.as_ref());
    let snapshot = verify_artifact_snapshot(std::fs::read(path).unwrap());
    match (&disk, snapshot) {
        (Ok(ArtifactVerification::Verified(expected)), Ok(report)) => {
            assert_eq!(expected, &report.artifact)
        }
        (Ok(ArtifactVerification::MissingLegacyMetadata), Err(error)) => {
            assert!(error.to_string().contains("metadata"))
        }
        (Err(_), Err(_)) => {}
        _ => panic!("path and snapshot verification disagree"),
    }
    disk
}
