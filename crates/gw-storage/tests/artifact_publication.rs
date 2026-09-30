//! Publication receipts preserve exact membership across the filesystem/SQLite boundary.

use gw_schema::{
    CotPolicy, ExportArtifact, ExportOptions, ExportScope, LifecycleState, TrainingRecord,
    TrlFormat,
};
use gw_storage::{
    ArtifactVerification, ExportPurpose, PublicationDisposition, RunStatus, Store, verify_artifact,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "gw-receipt-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).unwrap();
        Self(dir)
    }
    fn output(&self) -> PathBuf {
        self.0.join("dataset.parquet")
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
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

fn options() -> ExportOptions {
    ExportOptions {
        target: TrlFormat::ChatML,
        cot_policy: CotPolicy::Masked,
        dataset_version: Some(semver::Version::new(1, 2, 3)),
        scope: ExportScope::Run {
            run_id: "run".into(),
        },
    }
}

async fn store(ids: &[&str]) -> Store {
    let store = Store::open_in_memory().await.unwrap();
    store.create_run("run", "{}", None).await.unwrap();
    for id in ids {
        store.put(&record(id)).await.unwrap();
    }
    store
}

async fn reject_ack(store: &Store) {
    sqlx::query("CREATE TRIGGER fail_ack BEFORE UPDATE ON export_receipts WHEN NEW.state = 'acknowledged' BEGIN SELECT RAISE(FAIL, 'injected acknowledgment failure'); END")
        .execute(store.raw_pool()).await.unwrap();
}

async fn allow_ack(store: &Store) {
    sqlx::query("DROP TRIGGER fail_ack")
        .execute(store.raw_pool())
        .await
        .unwrap();
}

async fn receipts(store: &Store) -> Vec<(String, String, String)> {
    sqlx::query_as("SELECT artifact_id, state, members_json FROM export_receipts ORDER BY rowid")
        .fetch_all(store.raw_pool())
        .await
        .unwrap()
}

fn verified(path: &std::path::Path) -> ExportArtifact {
    match verify_artifact(path).unwrap() {
        ArtifactVerification::Verified(artifact) => artifact,
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn postrename_database_failure_rolls_back_every_member_and_retry_is_idempotent() {
    let temp = Temp::new();
    let store = store(&["a", "b"]).await;
    reject_ack(&store).await;
    let error = store
        .publish_export(options(), temp.output(), ExportPurpose::Engine)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("injected acknowledgment"));
    let artifact = verified(&temp.output());
    for id in ["a", "b"] {
        let rec = store.get(id).await.unwrap();
        assert_eq!(rec.lifecycle.state, LifecycleState::Formatted);
        assert!(rec.lifecycle.history.is_empty());
        assert!(store.lifecycle_history(id).await.unwrap().is_empty());
    }
    assert_eq!(receipts(&store).await[0].1, "prepared");
    allow_ack(&store).await;
    let before = std::fs::read(temp.output()).unwrap();
    let recovered = store
        .publish_export(options(), temp.output(), ExportPurpose::Engine)
        .await
        .unwrap();
    assert_eq!(recovered.artifact, artifact);
    assert_eq!(
        recovered.disposition,
        PublicationDisposition::AcknowledgedExisting
    );
    assert_eq!(recovered.advanced_record_ids, ["a", "b"]);
    assert_eq!(std::fs::read(temp.output()).unwrap(), before);
    assert_eq!(receipts(&store).await[0].1, "acknowledged");
    let retried = store
        .publish_export(options(), temp.output(), ExportPurpose::Engine)
        .await
        .unwrap();
    assert_eq!(retried.artifact, artifact);
    assert!(retried.advanced_record_ids.is_empty());
    for id in ["a", "b"] {
        let history = store.lifecycle_history(id).await.unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].0, "exported");
        assert_eq!(
            history[0].2.as_deref(),
            Some(format!("artifact:{}", artifact.artifact_id).as_str())
        );
        assert_eq!(store.get(id).await.unwrap().lifecycle.history.len(), 1);
    }
}

#[tokio::test]
async fn prepared_retry_ignores_later_unrelated_admissions_and_keeps_frozen_population() {
    let temp = Temp::new();
    let store = store(&["selected"]).await;
    reject_ack(&store).await;
    store
        .publish_export(options(), temp.output(), ExportPurpose::Engine)
        .await
        .unwrap_err();
    let expected = verified(&temp.output());
    store.put(&record("later-admission")).await.unwrap();
    allow_ack(&store).await;
    let result = store
        .publish_export(options(), temp.output(), ExportPurpose::Engine)
        .await
        .unwrap();
    assert_eq!(result.artifact, expected);
    assert_eq!(result.artifact.manifest.n_records, 1);
    assert_eq!(result.advanced_record_ids, ["selected"]);
    assert_eq!(
        store.get("later-admission").await.unwrap().lifecycle.state,
        LifecycleState::Formatted
    );
    assert!(
        store
            .lifecycle_history("later-admission")
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn changed_selected_projection_or_eligibility_is_never_acknowledged() {
    for change_state in [false, true] {
        let temp = Temp::new();
        let store = store(&["selected"]).await;
        reject_ack(&store).await;
        store
            .publish_export(options(), temp.output(), ExportPurpose::Engine)
            .await
            .unwrap_err();
        let bytes = std::fs::read(temp.output()).unwrap();
        let mut changed = store.get("selected").await.unwrap();
        if change_state {
            changed.lifecycle.state = LifecycleState::Rejected;
        } else {
            changed.cost.reasoning_tokens += 1;
        }
        store.put(&changed).await.unwrap();
        allow_ack(&store).await;
        let error = store
            .publish_export(options(), temp.output(), ExportPurpose::Engine)
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains(if change_state { "eligible" } else { "changed" })
        );
        assert_eq!(std::fs::read(temp.output()).unwrap(), bytes);
        assert_eq!(receipts(&store).await[0].1, "prepared");
        assert!(
            store
                .lifecycle_history("selected")
                .await
                .unwrap()
                .is_empty()
        );
    }
}

#[tokio::test]
async fn mismatched_destination_is_republished_from_same_frozen_plan_before_ack() {
    let temp = Temp::new();
    let store = store(&["selected"]).await;
    reject_ack(&store).await;
    store
        .publish_export(options(), temp.output(), ExportPurpose::Engine)
        .await
        .unwrap_err();
    let expected = verified(&temp.output());
    std::fs::write(temp.output(), b"different previous output").unwrap();
    store.put(&record("later")).await.unwrap();
    allow_ack(&store).await;
    let result = store
        .publish_export(options(), temp.output(), ExportPurpose::Engine)
        .await
        .unwrap();
    assert_eq!(result.disposition, PublicationDisposition::Republished);
    assert_eq!(verified(&temp.output()), expected);
    assert_eq!(result.advanced_record_ids, ["selected"]);
    assert_eq!(
        store.get("later").await.unwrap().lifecycle.state,
        LifecycleState::Formatted
    );
}

#[tokio::test]
async fn empty_receipt_recovers_without_absorbing_new_admissions() {
    let temp = Temp::new();
    let store = store(&[]).await;
    std::fs::write(temp.output(), b"old artifact").unwrap();
    reject_ack(&store).await;
    store
        .publish_export(options(), temp.output(), ExportPurpose::Engine)
        .await
        .unwrap_err();
    let expected = verified(&temp.output());
    assert_eq!(expected.manifest.n_admitted, 0);
    assert_eq!(receipts(&store).await[0].2, "[]");
    store.put(&record("later")).await.unwrap();
    allow_ack(&store).await;
    let result = store
        .publish_export(options(), temp.output(), ExportPurpose::Engine)
        .await
        .unwrap();
    assert_eq!(result.artifact, expected);
    assert_eq!(
        result.disposition,
        PublicationDisposition::AcknowledgedExisting
    );
    assert!(result.advanced_record_ids.is_empty());
    assert_eq!(
        store.get("later").await.unwrap().lifecycle.state,
        LifecycleState::Formatted
    );
}

#[tokio::test]
async fn standalone_receipt_does_not_mutate_generation_state_and_destination_is_separate_identity()
{
    let temp = Temp::new();
    let store = store(&["selected"]).await;
    store
        .set_run_status("run", RunStatus::Halted)
        .await
        .unwrap();
    let before = store.get("selected").await.unwrap();
    let first = store
        .publish_export(options(), temp.output(), ExportPurpose::Standalone)
        .await
        .unwrap();
    let second = store
        .publish_export(
            options(),
            temp.0.join("second.parquet"),
            ExportPurpose::Standalone,
        )
        .await
        .unwrap();
    assert_eq!(first.artifact, second.artifact);
    assert!(first.advanced_record_ids.is_empty());
    assert_eq!(store.get("selected").await.unwrap(), before);
    assert!(
        store
            .lifecycle_history("selected")
            .await
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        store.run_status("run").await.unwrap().as_deref(),
        Some("halted")
    );
    let ledger = receipts(&store).await;
    assert_eq!(ledger.len(), 2);
    assert!(ledger.iter().all(|(_, state, _)| state == "acknowledged"));
    assert_eq!(ledger[0].0, ledger[1].0);
}

#[tokio::test]
async fn explicit_acknowledged_receipt_recovery_excludes_later_records_and_retains_history() {
    let temp = Temp::new();
    let store = store(&["selected"]).await;
    let published = store
        .publish_export(options(), temp.output(), ExportPurpose::Engine)
        .await
        .unwrap();
    let before = store.get("selected").await.unwrap();
    let history = store.lifecycle_history("selected").await.unwrap();
    store.put(&record("later")).await.unwrap();
    store
        .set_run_status("run", RunStatus::Failed)
        .await
        .unwrap();
    let recovered = store
        .resume_export(&published.publication_id)
        .await
        .unwrap();
    assert_eq!(recovered.artifact, published.artifact);
    assert_eq!(
        recovered.disposition,
        PublicationDisposition::AcknowledgedExisting
    );
    assert!(recovered.advanced_record_ids.is_empty());
    assert_eq!(store.get("selected").await.unwrap(), before);
    assert_eq!(store.lifecycle_history("selected").await.unwrap(), history);
    assert_eq!(
        store.get("later").await.unwrap().lifecycle.state,
        LifecycleState::Formatted
    );
    assert_eq!(
        store.run_status("run").await.unwrap().as_deref(),
        Some("failed")
    );
    // Exact-ID recovery can also republish a removed artifact without absorbing later admissions.
    std::fs::remove_file(temp.output()).unwrap();
    let republished = store
        .resume_export(&published.publication_id)
        .await
        .unwrap();
    assert_eq!(republished.disposition, PublicationDisposition::Republished);
    assert_eq!(republished.artifact, published.artifact);
    assert_eq!(store.lifecycle_history("selected").await.unwrap(), history);
    assert_eq!(
        store.get("later").await.unwrap().lifecycle.state,
        LifecycleState::Formatted
    );
    // An ordinary new export remains free to select the now-larger population.
    let fresh = store
        .publish_export(options(), temp.output(), ExportPurpose::Engine)
        .await
        .unwrap();
    assert_eq!(fresh.artifact.manifest.n_admitted, 2);
    assert_ne!(fresh.artifact.artifact_id, published.artifact.artifact_id);
}

#[tokio::test]
async fn existing_expected_identity_with_tampered_metadata_is_an_integrity_error() {
    use parquet::arrow::{ArrowWriter, arrow_reader::ParquetRecordBatchReaderBuilder};
    use parquet::file::{metadata::KeyValue, properties::WriterProperties};
    let temp = Temp::new();
    let store = store(&["selected"]).await;
    reject_ack(&store).await;
    let error = store
        .publish_export(options(), temp.output(), ExportPurpose::Engine)
        .await
        .unwrap_err();
    let gw_storage::StorageError::Publication { publication_id, .. } = error else {
        panic!("failure must carry its recovery ID");
    };
    let mut claimed = verified(&temp.output());
    let reader =
        ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(temp.output()).unwrap())
            .unwrap();
    let schema = reader.schema().clone();
    let batches: Vec<_> = reader.build().unwrap().map(Result::unwrap).collect();
    claimed.manifest.dataset_version = Some(semver::Version::new(99, 0, 0));
    let props = WriterProperties::builder()
        .set_key_value_metadata(Some(vec![KeyValue::new(
            gw_storage::ARTIFACT_METADATA_KEY.into(),
            serde_json::to_string(&claimed).unwrap(),
        )]))
        .build();
    let mut writer = ArrowWriter::try_new(
        std::fs::File::create(temp.output()).unwrap(),
        schema,
        Some(props),
    )
    .unwrap();
    for batch in batches {
        writer.write(&batch).unwrap();
    }
    writer.close().unwrap();
    let tampered = std::fs::read(temp.output()).unwrap();
    allow_ack(&store).await;
    let error = store.resume_export(&publication_id).await.unwrap_err();
    assert!(error.to_string().contains("identity mismatch"));
    assert!(error.to_string().contains(&publication_id));
    assert_eq!(
        std::fs::read(temp.output()).unwrap(),
        tampered,
        "integrity failures are not silently rewritten"
    );
    assert_eq!(receipts(&store).await[0].1, "prepared");
    assert_eq!(
        store.get("selected").await.unwrap().lifecycle.state,
        LifecycleState::Formatted
    );
}
