//! Synthetic storage contracts. Material here deliberately models a trusted runtime adapter.
#[path = "../../gw-schema/tests/reference_support/mod.rs"]
mod support;
use crate::*;
use gw_schema::*;

fn observations(registered: &RegisteredReferenceCatalogue) -> Vec<ReferenceMemberObservation> {
    registered.population().members().iter().map(|member| {
        let absent = VerificationAxis { policy: VerificationPolicy::Absent, observation: None };
        let verification = VerificationInterpretation { version: 2, reasoning: absent.clone(), answer: absent,
            execution: VerificationAxis { policy: VerificationPolicy::Authoritative,
                observation: Some(VerificationObservation { outcome: VerificationOutcome::Pass, reason: "synthetic native adapter fixture".into() }) } };
        let result_id = coding_digest("synthetic-test-only", member.member_id.as_bytes());
        let evidence = serde_json::json!({"artifact_id":result_id,"input":{"task":member.task,"code":member.code,
            "code_id":member.reference_code_id,"suite":member.task.suite_binding()},"report":{"native_verification":verification}});
        ReferenceMemberObservation::from_native_observation(member.member_id.clone(), result_id, verification, evidence).unwrap()
    }).collect()
}
fn options(batch: &str) -> ExportOptions {
    ExportOptions {
        target: TrlFormat::OpenAiMessages,
        cot_policy: CotPolicy::Stripped,
        dataset_version: None,
        scope: ExportScope::Run {
            run_id: batch.into(),
        },
    }
}
#[tokio::test]
async fn registration_is_explicit_and_matches_exact_bytes_before_import() {
    let store = Store::open_in_memory().await.unwrap();
    let capture = support::capture();
    assert!(
        store
            .registered_reference_catalogue(&capture)
            .await
            .is_err()
    );
    let accepted = store.register_reference_catalogue(&capture).await.unwrap();
    let loaded = store
        .registered_reference_catalogue(&capture)
        .await
        .unwrap();
    assert_eq!(accepted.registration_id(), loaded.registration_id());
    let mut changed = capture;
    changed.reviews[111].push('\n'); // Same reviewed semantics is insufficient: exact accepted bytes changed.
    assert!(
        store
            .registered_reference_catalogue(&changed)
            .await
            .is_err()
    );
    assert!(store.scan(&RecordFilter::new()).await.unwrap().is_empty());
    assert!(
        store
            .committed_reference_import(&accepted)
            .await
            .unwrap()
            .is_none()
    );
}
#[tokio::test]
async fn complete_commit_keeps_heldouts_private_and_requires_registered_publication() {
    let dir = crate::durability_support::Directory::new("references");
    let store = Store::open(dir.db()).await.unwrap();
    let registered = store
        .register_reference_catalogue(&support::capture())
        .await
        .unwrap();
    let result = store
        .commit_reference_import(
            &registered,
            &observations(&registered),
            std::future::pending(),
        )
        .await
        .unwrap();
    assert_eq!(result.status, RecordWriteStatus::Applied);
    assert_eq!(result.records.len(), 64);
    assert_eq!(result.held_out_count, 48);
    assert_eq!(store.scan(&RecordFilter::new()).await.unwrap().len(), 64);
    assert_eq!(
        store
            .run_status(registered.batch_id())
            .await
            .unwrap()
            .as_deref(),
        Some("completed")
    );
    let record = &result.records[0];
    assert_eq!(record.judging, Judging::default());
    assert!(record.origin.generated().is_none());
    assert_eq!(
        record
            .lifecycle
            .history
            .iter()
            .map(|s| s.state)
            .collect::<Vec<_>>(),
        [LifecycleState::Verified, LifecycleState::Admitted]
    );
    assert!(ExportPlan::prepare(&result.records, options(registered.batch_id())).is_err());
    assert!(store.insert_record(record).await.is_err());
    assert!(store.replace_record_for_import(record).await.is_err());
    assert!(
        store
            .publish_export(
                options(registered.batch_id()),
                dir.0.join("engine.parquet"),
                ExportPurpose::Engine
            )
            .await
            .is_err()
    );
    let standalone = store
        .publish_export(
            options(registered.batch_id()),
            dir.0.join("standalone.parquet"),
            ExportPurpose::Standalone,
        )
        .await
        .unwrap();
    assert!(standalone.advanced_record_ids.is_empty());
    assert_eq!(
        store.get(&record.record_id).await.unwrap().lifecycle.state,
        LifecycleState::Admitted
    );
    let published = store
        .publish_export(
            options(registered.batch_id()),
            dir.0.join("reference.parquet"),
            ExportPurpose::Reference,
        )
        .await
        .unwrap();
    assert_eq!(published.advanced_record_ids.len(), 64);
    let reused = store
        .committed_reference_import(&registered)
        .await
        .unwrap()
        .unwrap();
    assert!(
        reused
            .records
            .iter()
            .all(|r| r.lifecycle.state == LifecycleState::Exported)
    );
    let history = reused.records[0].lifecycle.history.clone();
    let retried = store
        .commit_reference_import(
            &registered,
            &observations(&registered),
            std::future::pending(),
        )
        .await
        .unwrap();
    assert_eq!(retried.status, RecordWriteStatus::AlreadyApplied);
    assert_eq!(retried.records[0].lifecycle.history, history);
    let bytes = std::fs::read(dir.0.join("reference.parquet")).unwrap();
    // Explicit fixture production only; these declarations model the trusted adapter in tests.
    if let Some(output) = std::env::var_os("GW_REFERENCE_TEST_FIXTURE_OUT") {
        std::fs::write(output, &bytes).unwrap();
    }
    let (report, rows) = crate::artifact::verify_snapshot_with_rows(bytes).unwrap();
    assert_eq!(
        report.artifact.manifest.column_schema_version,
        ExportSchemaVersion::CURRENT
    );
    assert_eq!(rows.len(), 64);
    for row in rows {
        assert!(row.verdict.is_none() && row.judge_aggregate.is_none());
        for text in [
            &row.messages_json,
            row.task_json.as_ref().unwrap(),
            row.origin_json.as_ref().unwrap(),
        ] {
            assert!(!text.contains("PRIVATE_REVIEW_CANARY"));
            assert!(!text.contains("synthetic-111"));
        }
    }
}
#[tokio::test]
async fn final_member_failure_and_precommit_error_leave_zero_eligibility() {
    let store = Store::open_in_memory().await.unwrap();
    let registered = store
        .register_reference_catalogue(&support::capture())
        .await
        .unwrap();
    for outcome in [VerificationOutcome::Fail, VerificationOutcome::Unknown] {
        let mut members = observations(&registered);
        members[111]
            .verification
            .execution
            .observation
            .as_mut()
            .unwrap()
            .outcome = outcome;
        assert!(
            store
                .commit_reference_import(&registered, &members, std::future::pending())
                .await
                .is_err()
        );
        assert!(store.scan(&RecordFilter::new()).await.unwrap().is_empty());
        assert!(
            store
                .run_status(registered.batch_id())
                .await
                .unwrap()
                .is_none()
        );
    }
    store.set_test_hook(Some(crate::test_hooks::Hook {
        operation: "reference_import",
        stage: "precommit",
        action: crate::test_hooks::Action::Fail,
    }));
    assert!(
        store
            .commit_reference_import(
                &registered,
                &observations(&registered),
                std::future::pending()
            )
            .await
            .is_err()
    );
    assert!(store.scan(&RecordFilter::new()).await.unwrap().is_empty());
    assert!(
        store
            .committed_reference_import(&registered)
            .await
            .unwrap()
            .is_none()
    );
    store.set_test_hook(Some(crate::test_hooks::Hook {
        operation: "reference_import",
        stage: "committed",
        action: crate::test_hooks::Action::Fail,
    }));
    assert!(
        store
            .commit_reference_import(
                &registered,
                &observations(&registered),
                std::future::pending()
            )
            .await
            .is_err()
    );
    store.set_test_hook(None);
    let reused = store
        .committed_reference_import(&registered)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(reused.records.len(), 64);
    assert_eq!(reused.status, RecordWriteStatus::AlreadyApplied);
}
#[tokio::test]
async fn old_columns_and_rehashed_inconsistent_origins_cannot_hide_reference_facts() {
    let store = Store::open_in_memory().await.unwrap();
    let registered = store
        .register_reference_catalogue(&support::capture())
        .await
        .unwrap();
    let result = store
        .commit_reference_import(
            &registered,
            &observations(&registered),
            std::future::pending(),
        )
        .await
        .unwrap();
    for version in [
        ExportSchemaVersion::CanonicalMessages,
        ExportSchemaVersion::ReviewedTasks,
    ] {
        assert!(crate::export::project(&result.records[0], version).is_err());
    }
    for field in [
        "reference_code_id",
        "suite_id",
        "member_id",
        "kind",
        "approved",
    ] {
        let mut plan =
            ExportPlan::prepare_registered(&result.records, options(registered.batch_id()))
                .unwrap();
        let row = &mut plan.rows[0];
        let mut origin: serde_json::Value =
            serde_json::from_str(row.origin_json.as_ref().unwrap()).unwrap();
        origin[field] = if field == "kind" {
            "generated".into()
        } else {
            "0".repeat(64).into()
        };
        row.origin_json = Some(serde_json::to_string(&origin).unwrap());
        plan.artifact.artifact_id =
            crate::artifact::artifact_identity(&plan.artifact, &plan.rows).unwrap();
        assert!(
            crate::artifact::validate_rows(&plan.artifact, &plan.rows).is_err(),
            "{field}"
        );
    }
}

async fn assert_reference_tools_rejected(tools: Vec<serde_json::Value>) {
    let store = Store::open_in_memory().await.unwrap();
    let registered = store
        .register_reference_catalogue(&support::capture())
        .await
        .unwrap();
    let result = store
        .commit_reference_import(
            &registered,
            &observations(&registered),
            std::future::pending(),
        )
        .await
        .unwrap();
    let mut plan =
        ExportPlan::prepare_registered(&result.records, options(registered.batch_id())).unwrap();
    assert_eq!(
        plan.artifact.manifest.column_schema_version,
        ExportSchemaVersion::ToolDefinitions
    );
    assert!(plan.rows.iter().all(|row| row.tools_json.is_none()));
    let mut original = Vec::new();
    crate::export::write_parquet(&plan.rows, &plan.artifact, &mut original).unwrap();
    crate::verify_artifact_snapshot(original).unwrap();
    let original_id = plan.artifact.artifact_id.clone();
    plan.rows[0].tools_json = Some(crate::export::canonical_tools_json(&tools).unwrap());
    plan.artifact.artifact_id =
        crate::artifact::artifact_identity(&plan.artifact, &plan.rows).unwrap();
    assert_ne!(plan.artifact.artifact_id, original_id);
    let mut forged = Vec::new();
    crate::export::write_parquet(&plan.rows, &plan.artifact, &mut forged).unwrap();
    let checked = crate::verify_artifact_snapshot(forged);
    assert!(
        checked.is_err(),
        "reference row accepted present tools after its outer identity was recomputed"
    );
    assert!(checked.unwrap_err().to_string().contains("reference row"));
}

#[tokio::test]
async fn reference_rows_reject_rehashed_empty_tools() {
    assert_reference_tools_rejected(vec![]).await;
}

#[tokio::test]
async fn reference_rows_reject_rehashed_nonempty_tools() {
    assert_reference_tools_rejected(vec![serde_json::json!({
        "type":"function",
        "function":{"name":"lookup","parameters":{"type":"object"}}
    })])
    .await;
}

async fn reference_counts(store: &Store) -> (i64, i64, i64, i64, i64) {
    sqlx::query_as("SELECT (SELECT COUNT(*) FROM records), (SELECT COUNT(*) FROM lifecycle_history), (SELECT COUNT(*) FROM reference_members), (SELECT COUNT(*) FROM reference_batches), (SELECT COUNT(*) FROM runs)")
        .fetch_one(store.pool()).await.unwrap()
}

#[tokio::test]
async fn cancellation_waiting_for_writer_or_before_commit_rolls_back_every_reference_fact() {
    use crate::test_hooks::{Action, Hook};
    use std::{sync::Arc, time::Duration};
    use tokio::sync::Notify;
    for stage in ["transaction_wait", "precommit"] {
        let dir = crate::durability_support::Directory::new(stage);
        let store = Store::open(dir.db()).await.unwrap();
        let registered = store
            .register_reference_catalogue(&support::capture())
            .await
            .unwrap();
        let writer = if stage == "transaction_wait" {
            Some(store.pool().begin_with("BEGIN IMMEDIATE").await.unwrap())
        } else {
            None
        };
        let entered = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let cancelled = Arc::new(Notify::new());
        store.set_test_hook(Some(Hook {
            operation: "reference_import",
            stage,
            action: Action::Pause {
                entered: entered.clone(),
                release: release.clone(),
            },
        }));
        let task_store = store.clone();
        let task_registration = registered.clone();
        let task_cancelled = cancelled.clone();
        let mut task = tokio::spawn(async move {
            task_store
                .commit_reference_import(
                    &task_registration,
                    &observations(&task_registration),
                    async move { task_cancelled.notified().await },
                )
                .await
        });
        entered.notified().await;
        if stage == "transaction_wait" {
            release.notify_one();
            // Admission remains blocked while a real IMMEDIATE writer owns the database.
            assert!(
                tokio::time::timeout(Duration::from_millis(50), &mut task)
                    .await
                    .is_err()
            );
        }
        cancelled.notify_one();
        if let Some(writer) = writer {
            writer.rollback().await.unwrap();
        }
        release.notify_one();
        let result = task.await.unwrap();
        assert!(
            matches!(result, Err(StorageError::ReferenceImportCancelled)),
            "{stage}: cancellation did not prevent admission"
        );
        store.set_test_hook(None);
        assert_eq!(reference_counts(&store).await, (0, 0, 0, 0, 0));
        assert!(
            store
                .committed_reference_import(&registered)
                .await
                .unwrap()
                .is_none()
        );
        // The same accepted capture remains usable after cancellation and rollback.
        store
            .commit_reference_import(
                &registered,
                &observations(&registered),
                std::future::pending(),
            )
            .await
            .unwrap();
        assert_eq!(reference_counts(&store).await, (64, 128, 112, 1, 1));
    }
}

#[tokio::test]
async fn cancellation_after_commit_retains_truthful_outcome_or_authoritative_reuse() {
    use crate::test_hooks::{Action, Hook};
    use std::sync::Arc;
    use tokio::sync::Notify;
    for lost_ack in [false, true] {
        let dir = crate::durability_support::Directory::new("reference-committed-cancel");
        let store = Store::open(dir.db()).await.unwrap();
        let registered = store
            .register_reference_catalogue(&support::capture())
            .await
            .unwrap();
        let entered = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let cancelled = Arc::new(Notify::new());
        store.set_test_hook(Some(Hook {
            operation: "reference_import",
            stage: "committed",
            action: Action::Pause {
                entered: entered.clone(),
                release: release.clone(),
            },
        }));
        let task_store = store.clone();
        let task_registration = registered.clone();
        let task_cancelled = cancelled.clone();
        let task = tokio::spawn(async move {
            task_store
                .commit_reference_import(
                    &task_registration,
                    &observations(&task_registration),
                    async move { task_cancelled.notified().await },
                )
                .await
        });
        entered.notified().await;
        cancelled.notify_one();
        if lost_ack {
            task.abort();
            assert!(task.await.unwrap_err().is_cancelled());
        } else {
            release.notify_one();
            let outcome = task.await.unwrap().unwrap();
            assert_eq!(outcome.status, RecordWriteStatus::Applied);
            assert_eq!(outcome.records.len(), 64);
        }
        store.close().await;
        let reopened = Store::open(dir.db()).await.unwrap();
        let current = reopened
            .committed_reference_import(&registered)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(current.status, RecordWriteStatus::AlreadyApplied);
        assert_eq!(reference_counts(&reopened).await, (64, 128, 112, 1, 1));
        let retried = reopened
            .commit_reference_import(
                &registered,
                &observations(&registered),
                std::future::pending(),
            )
            .await
            .unwrap();
        assert_eq!(retried.records, current.records);
    }
}

#[tokio::test]
async fn administrative_repair_cannot_replace_a_reference_with_corrupt_json_or_run_projection() {
    let store = Store::open_in_memory().await.unwrap();
    let registered = store
        .register_reference_catalogue(&support::capture())
        .await
        .unwrap();
    let imported = store
        .commit_reference_import(
            &registered,
            &observations(&registered),
            std::future::pending(),
        )
        .await
        .unwrap();
    store
        .insert_historical_run("run", "{}", None)
        .await
        .unwrap();
    let id = &imported.records[0].record_id;
    let mut replacement = crate::durability_support::record();
    replacement.record_id = id.clone();
    // The committed member relation remains authoritative even if both row surfaces are corrupt.
    sqlx::query("UPDATE records SET record_json='{', run_id='run' WHERE record_id=?")
        .bind(id)
        .execute(store.pool())
        .await
        .unwrap();
    assert!(store.replace_record_for_import(&replacement).await.is_err());
    let stored: String = sqlx::query_scalar("SELECT record_json FROM records WHERE record_id=?")
        .bind(id)
        .fetch_one(store.pool())
        .await
        .unwrap();
    assert_eq!(stored, "{");
    assert_eq!(reference_counts(&store).await, (64, 128, 112, 1, 2));
    // Independently, a typed reference ledger protects its row even if membership is damaged.
    sqlx::query("UPDATE records SET run_id=? WHERE record_id=?")
        .bind(registered.batch_id())
        .bind(id)
        .execute(store.pool())
        .await
        .unwrap();
    sqlx::query("DELETE FROM reference_members WHERE record_id=?")
        .bind(id)
        .execute(store.pool())
        .await
        .unwrap();
    assert!(store.replace_record_for_import(&replacement).await.is_err());
    assert_eq!(reference_counts(&store).await, (64, 128, 111, 1, 2));
}
