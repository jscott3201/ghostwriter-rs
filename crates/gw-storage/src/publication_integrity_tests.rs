//! Publication must reject damaged source state without repairing or acknowledging it.
use crate::{ExportPurpose, StorageError, Store, durability_support::*};
use gw_schema::{CotPolicy, ExportOptions, ExportScope, LifecycleState, TrlFormat, Verdict};

async fn snapshot(store: &Store) -> Vec<Vec<String>> {
    let mut tables = Vec::new();
    for query in [
        "SELECT json_array(record_id,run_id,lifecycle_state,verdict,judge_aggregate,record_hash,prompt_hash,record_json,updated_at) FROM records ORDER BY record_id",
        "SELECT json_array(id,record_id,state,at,detail,mutation_id,history_ordinal,attempt) FROM lifecycle_history ORDER BY id",
        "SELECT json_array(mutation_id,record_id,version,kind,history_start,history_count,committed_at) FROM record_mutations ORDER BY mutation_id",
        "SELECT json_array(publication_id,artifact_id,destination,purpose,artifact_json,members_json,state,prepared_at,acknowledged_at) FROM export_receipts ORDER BY publication_id",
    ] {
        tables.push(
            sqlx::query_scalar(query)
                .fetch_all(store.pool())
                .await
                .unwrap(),
        );
    }
    tables
}

async fn reject_corruption(purpose: ExportPurpose, acknowledged: bool, corruption: &'static str) {
    let dir = Directory::new("publication-integrity");
    let store = initialized(&dir.db()).await;
    store
        .insert_historical_run("other", "{}", None)
        .await
        .unwrap();
    // Retain a healthy member before the damaged member in publication order.
    let mut healthy = record();
    healthy.record_id = "healthy".into();
    store.insert_record(&healthy).await.unwrap();
    for id in ["healthy", "record"] {
        let expected = store.get(id).await.unwrap();
        let mut updated = expected.clone();
        updated.judging.verdict = Some(Verdict::Admit);
        updated.judging.aggregate = Some(0.95);
        store
            .transition_record(&expected, &updated, LifecycleState::Formatted, None)
            .await
            .unwrap();
    }
    if !acknowledged {
        sqlx::query("CREATE TRIGGER fail_ack BEFORE UPDATE ON export_receipts WHEN NEW.state='acknowledged' BEGIN SELECT RAISE(ABORT,'injected acknowledgment failure'); END").execute(store.pool()).await.unwrap();
    }
    let output = dir.0.join("artifact.parquet");
    let result = store
        .publish_export(
            ExportOptions {
                target: TrlFormat::ChatML,
                cot_policy: CotPolicy::Masked,
                dataset_version: None,
                scope: ExportScope::Run {
                    run_id: "run".into(),
                },
            },
            &output,
            purpose,
        )
        .await;
    let publication_id = if acknowledged {
        result.unwrap().publication_id
    } else {
        let StorageError::Publication { publication_id, .. } = result.unwrap_err() else {
            panic!("expected an interrupted publication");
        };
        sqlx::query("DROP TRIGGER fail_ack")
            .execute(store.pool())
            .await
            .unwrap();
        publication_id
    };
    assert!(matches!(
        crate::verify_artifact(&output).unwrap(),
        crate::ArtifactVerification::Verified(_)
    ));
    let bytes = std::fs::read(&output).unwrap();
    sqlx::query(corruption).execute(store.pool()).await.unwrap();
    let damaged = snapshot(&store).await;
    let state: String = sqlx::query_scalar("SELECT state FROM export_receipts")
        .fetch_one(store.pool())
        .await
        .unwrap();
    assert_eq!(
        state,
        if acknowledged {
            "acknowledged"
        } else {
            "prepared"
        }
    );
    let error = store.resume_export(&publication_id).await.unwrap_err();
    assert!(
        matches!(error, StorageError::Publication { source, .. } if matches!(*source, StorageError::RecordIntegrity { .. }))
    );
    assert_eq!(snapshot(&store).await, damaged);
    assert!(std::fs::read(&output).unwrap() == bytes);
    // A corruption between plan restoration and acknowledgment must also fail in the batch tx.
    let receipt = store.load_export_receipt(&publication_id).await.unwrap();
    assert!(matches!(
        store.acknowledge_export(&receipt, purpose).await,
        Err(StorageError::RecordIntegrity { .. })
    ));
    assert_eq!(snapshot(&store).await, damaged);
    assert!(std::fs::read(&output).unwrap() == bytes);
    store.close().await;
}

#[tokio::test]
async fn prepared_engine_rejects_partition() {
    reject_corruption(
        ExportPurpose::Engine,
        false,
        "UPDATE records SET run_id='other' WHERE record_id='record'",
    )
    .await;
}

#[tokio::test]
async fn prepared_engine_rejects_hash() {
    reject_corruption(
        ExportPurpose::Engine,
        false,
        "UPDATE records SET record_hash='damaged' WHERE record_id='record'",
    )
    .await;
}

#[tokio::test]
async fn prepared_engine_rejects_missing_history() {
    reject_corruption(
        ExportPurpose::Engine,
        false,
        "DELETE FROM lifecycle_history WHERE record_id='record' AND history_ordinal=0",
    )
    .await;
}

#[tokio::test]
async fn prepared_engine_rejects_substituted_history() {
    reject_corruption(ExportPurpose::Engine, false, "UPDATE lifecycle_history SET at='substituted' WHERE record_id='record' AND history_ordinal=0").await;
}

#[tokio::test]
async fn prepared_standalone_rejects_partition() {
    reject_corruption(
        ExportPurpose::Standalone,
        false,
        "UPDATE records SET run_id='other' WHERE record_id='record'",
    )
    .await;
}

#[tokio::test]
async fn prepared_standalone_rejects_hash() {
    reject_corruption(
        ExportPurpose::Standalone,
        false,
        "UPDATE records SET record_hash='damaged' WHERE record_id='record'",
    )
    .await;
}

#[tokio::test]
async fn prepared_standalone_rejects_missing_history() {
    reject_corruption(
        ExportPurpose::Standalone,
        false,
        "DELETE FROM lifecycle_history WHERE record_id='record' AND history_ordinal=0",
    )
    .await;
}

#[tokio::test]
async fn prepared_standalone_rejects_substituted_history() {
    reject_corruption(ExportPurpose::Standalone, false, "UPDATE lifecycle_history SET at='substituted' WHERE record_id='record' AND history_ordinal=0").await;
}

#[tokio::test]
async fn acknowledged_engine_rejects_partition() {
    reject_corruption(
        ExportPurpose::Engine,
        true,
        "UPDATE records SET run_id='other' WHERE record_id='record'",
    )
    .await;
}

#[tokio::test]
async fn acknowledged_engine_rejects_hash() {
    reject_corruption(
        ExportPurpose::Engine,
        true,
        "UPDATE records SET record_hash='damaged' WHERE record_id='record'",
    )
    .await;
}

#[tokio::test]
async fn acknowledged_engine_rejects_missing_history() {
    reject_corruption(
        ExportPurpose::Engine,
        true,
        "DELETE FROM lifecycle_history WHERE record_id='record' AND history_ordinal=0",
    )
    .await;
}

#[tokio::test]
async fn acknowledged_engine_rejects_substituted_history() {
    reject_corruption(ExportPurpose::Engine, true, "UPDATE lifecycle_history SET at='substituted' WHERE record_id='record' AND history_ordinal=0").await;
}

#[tokio::test]
async fn acknowledged_standalone_rejects_partition() {
    reject_corruption(
        ExportPurpose::Standalone,
        true,
        "UPDATE records SET run_id='other' WHERE record_id='record'",
    )
    .await;
}

#[tokio::test]
async fn acknowledged_standalone_rejects_hash() {
    reject_corruption(
        ExportPurpose::Standalone,
        true,
        "UPDATE records SET record_hash='damaged' WHERE record_id='record'",
    )
    .await;
}

#[tokio::test]
async fn acknowledged_standalone_rejects_missing_history() {
    reject_corruption(
        ExportPurpose::Standalone,
        true,
        "DELETE FROM lifecycle_history WHERE record_id='record' AND history_ordinal=0",
    )
    .await;
}

#[tokio::test]
async fn acknowledged_standalone_rejects_substituted_history() {
    reject_corruption(ExportPurpose::Standalone, true, "UPDATE lifecycle_history SET at='substituted' WHERE record_id='record' AND history_ordinal=0").await;
}
