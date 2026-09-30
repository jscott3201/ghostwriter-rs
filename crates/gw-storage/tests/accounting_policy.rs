//! Transactional policy authority, durable evidence, and conservative live ownership.
mod common;
use gw_schema::*;
use gw_storage::{AttemptAdmission, LaunchRequest, StorageError, Store};
fn request<'a>(run: &'a str, policy: &'a AccountingPolicy) -> LaunchRequest<'a> {
    LaunchRequest {
        run_id: run,
        manifest: common::manifest(),
        mode: gw_storage::RunMode::CreateOrResume,
        policy,
        teacher: AccountingCapability::PhysicalAttemptsV1,
        judge: AccountingCapability::NoModelRequests,
        embedding: AccountingCapability::NoModelRequests,
    }
}
fn intent(launch: &LaunchCoverage) -> AttemptIntent {
    AttemptIntent {
        version: 1,
        context: AttemptContext {
            run_id: launch.run_id.clone(),
            launch_id: launch.launch_id.clone(),
            shard: Some(0),
            record_id: None,
            role: AttemptRole::Teacher,
            purpose: AttemptPurpose::Initial,
        },
        request_digest: "a".repeat(64),
        retry_ordinal: 0,
        requested_model: "fixture".into(),
        endpoint: "http://localhost/fixture".into(),
    }
}
fn epoch(launch: &LaunchCoverage) -> u64 {
    launch.policy.as_ref().unwrap().epoch
}
async fn begin(store: &Store, launch: &LaunchCoverage) -> String {
    match store
        .admit_model_attempt(&intent(launch), epoch(launch), &[])
        .await
        .unwrap()
    {
        AttemptAdmission::Admitted(id) => id,
        other => panic!("{other:?}"),
    }
}
async fn settle(store: &Store, id: &str, cost: ReportedCost) {
    store
        .observe_model_attempt(
            id,
            0,
            &AttemptMetadata {
                cost_usd: cost,
                total_tokens: Some(9),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    store
        .settle_model_attempt(
            id,
            &TransportSettlement {
                outcome: TransportOutcome::Failed,
                http_status: Some(503),
                elapsed_ms: 12,
            },
        )
        .await
        .unwrap();
}
#[tokio::test]
async fn finite_admission_waits_only_for_live_owned_attempts_and_rechecks_cost() {
    let store = Store::open_in_memory().await.unwrap();
    let policy = AccountingPolicy::FiniteUsd { limit_usd: 0.5 };
    let launch = store
        .register_accounting_launch(request("r", &policy))
        .await
        .unwrap();
    let id = begin(&store, &launch).await;
    assert_eq!(
        store
            .admit_model_attempt(&intent(&launch), epoch(&launch), std::slice::from_ref(&id))
            .await
            .unwrap(),
        AttemptAdmission::Wait
    );
    assert_eq!(
        store
            .admit_model_attempt(&intent(&launch), epoch(&launch), &[])
            .await
            .unwrap(),
        AttemptAdmission::Denied(AdmissionDenial::UnresolvedAttempts)
    );
    settle(&store, &id, ReportedCost::Known(0.5)).await;
    assert_eq!(
        store
            .admit_model_attempt(&intent(&launch), epoch(&launch), &[])
            .await
            .unwrap(),
        AttemptAdmission::Denied(AdmissionDenial::LimitReached {
            known_usd: 0.5,
            limit_usd: 0.5
        })
    );
    let summary = store.accounting_snapshot("r").await.unwrap();
    assert_eq!(summary.known_usd, Some(0.5));
    assert_eq!(summary.total_tokens.known, Some(9));
    assert_eq!(summary.elapsed_ms, Some(12));
    assert_eq!(summary.history, AccountingHistory::RecordedFromCreation);
}
#[tokio::test]
async fn observation_supersedes_finite_and_old_receipts_can_settle_without_new_authority() {
    let store = Store::open_in_memory().await.unwrap();
    let first = store
        .register_accounting_launch(request(
            "r",
            &AccountingPolicy::FiniteUsd { limit_usd: 5.0 },
        ))
        .await
        .unwrap();
    let id = begin(&store, &first).await;
    let second = store
        .register_accounting_launch(request("r", &AccountingPolicy::ObservationOnly))
        .await
        .unwrap();
    assert_eq!(epoch(&second), epoch(&first) + 1);
    assert_eq!(
        store
            .admit_model_attempt(&intent(&first), epoch(&first), std::slice::from_ref(&id))
            .await
            .unwrap(),
        AttemptAdmission::Denied(AdmissionDenial::PolicySuperseded)
    );
    let _concurrent = begin(&store, &second).await;
    settle(&store, &id, ReportedCost::Known(1.0)).await;
    assert!(matches!(
        store
            .register_accounting_launch(request(
                "r",
                &AccountingPolicy::FiniteUsd { limit_usd: 5.0 }
            ))
            .await,
        Err(StorageError::Admission(AdmissionDenial::UnresolvedAttempts))
    ));
}
#[tokio::test]
async fn unknown_prices_coverage_and_overflow_cannot_enter_a_finite_policy() {
    for cost in [
        ReportedCost::Missing,
        ReportedCost::Invalid,
        ReportedCost::Known(f64::MAX),
    ] {
        let store = Store::open_in_memory().await.unwrap();
        let launch = store
            .register_accounting_launch(request("r", &AccountingPolicy::ObservationOnly))
            .await
            .unwrap();
        let first = begin(&store, &launch).await;
        settle(&store, &first, cost.clone()).await;
        let second = begin(&store, &launch).await;
        settle(&store, &second, cost.clone()).await;
        let result = store
            .register_accounting_launch(request(
                "r",
                &AccountingPolicy::FiniteUsd { limit_usd: 5.0 },
            ))
            .await;
        assert!(matches!(
            result,
            Err(StorageError::Admission(
                AdmissionDenial::UnknownCost | AdmissionDenial::InvalidEvidence
            ))
        ));
        if cost == ReportedCost::Known(f64::MAX) {
            assert_eq!(
                store.accounting_snapshot("r").await.unwrap().known_usd,
                None
            );
        }
    }
    let store = Store::open_in_memory().await.unwrap();
    let mut unknown = request("u", &AccountingPolicy::ObservationOnly);
    unknown.teacher = AccountingCapability::Unknown;
    store.register_accounting_launch(unknown).await.unwrap();
    assert!(matches!(
        store
            .register_accounting_launch(request(
                "u",
                &AccountingPolicy::FiniteUsd { limit_usd: 5.0 }
            ))
            .await,
        Err(StorageError::Admission(AdmissionDenial::UnknownCoverage))
    ));
    store
        .insert_historical_run("legacy", "{}", None)
        .await
        .unwrap();
    assert!(matches!(
        store
            .register_accounting_launch(request(
                "legacy",
                &AccountingPolicy::FiniteUsd { limit_usd: 5.0 }
            ))
            .await,
        Err(StorageError::RunManifest { .. })
    ));
}

#[tokio::test]
async fn typed_invalid_cost_remains_visible_after_known_cost_exact_retries_and_reopen() {
    for initial_policy in [
        AccountingPolicy::ObservationOnly,
        AccountingPolicy::FiniteUsd { limit_usd: 5.0 },
    ] {
        let path = std::env::temp_dir().join(format!(
            "gw-invalid-cost-{}-{}.sqlite",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let store = Store::open(&path).await.unwrap();
        let launch = store
            .register_accounting_launch(request("r", &initial_policy))
            .await
            .unwrap();
        let id = begin(&store, &launch).await;
        let invalid = AttemptMetadata {
            cost_usd: ReportedCost::Invalid,
            ..Default::default()
        };
        let known = AttemptMetadata {
            cost_usd: ReportedCost::Known(0.1),
            ..Default::default()
        };
        store.observe_model_attempt(&id, 0, &invalid).await.unwrap();
        store.observe_model_attempt(&id, 1, &known).await.unwrap();
        store
            .settle_model_attempt(
                &id,
                &TransportSettlement {
                    outcome: TransportOutcome::Complete,
                    http_status: Some(200),
                    elapsed_ms: 1,
                },
            )
            .await
            .unwrap();
        assert_eq!(
            store
                .accounting_snapshot("r")
                .await
                .unwrap()
                .invalid_cost_attempts,
            1
        );
        // Older typed callers retained Invalid in observations without the merged string marker.
        // Reopening must derive authority from that retained payload as well.
        sqlx::query("UPDATE model_attempts SET receipt_json = json_set(receipt_json, '$.metadata.invalid_fields', json('[]'))")
            .execute(store.raw_pool()).await.unwrap();
        store.raw_pool().close().await;
        let store = Store::open(&path).await.unwrap();
        let before = store.accounting_snapshot("r").await.unwrap();
        store.observe_model_attempt(&id, 0, &invalid).await.unwrap();
        store.observe_model_attempt(&id, 1, &known).await.unwrap();
        assert_eq!(
            store.accounting_snapshot("r").await.unwrap(),
            before,
            "exact retries preserve the original result and revision"
        );
        assert_eq!(
            before.known_usd,
            Some(0.0),
            "invalid attempts are excluded from the known subtotal"
        );
        assert_eq!(before.invalid_cost_attempts, 1);
        let receipt = store.model_attempts("r").await.unwrap().remove(0);
        assert_eq!(receipt.observations.len(), 2);
        assert_eq!(receipt.observations[0].metadata, invalid);
        assert_eq!(receipt.observations[1].metadata, known);
        assert!(receipt.conflicts.is_empty());
        assert_eq!(receipt.metadata.cost_usd, ReportedCost::Known(0.1));
        assert!(
            receipt
                .metadata
                .invalid_fields
                .iter()
                .any(|field| field == "cost")
        );
        if matches!(initial_policy, AccountingPolicy::FiniteUsd { .. }) {
            assert_eq!(
                store
                    .admit_model_attempt(&intent(&launch), epoch(&launch), &[])
                    .await
                    .unwrap(),
                AttemptAdmission::Denied(AdmissionDenial::InvalidEvidence)
            );
        }
        assert!(matches!(
            store
                .register_accounting_launch(request(
                    "r",
                    &AccountingPolicy::FiniteUsd { limit_usd: 6.0 }
                ))
                .await,
            Err(StorageError::Admission(AdmissionDenial::InvalidEvidence))
        ));
        store.raw_pool().close().await;
        std::fs::remove_file(&path).unwrap();
        for suffix in ["-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
        }
    }
}

#[tokio::test]
async fn concurrent_connections_admit_only_one_and_reopened_orphans_do_not_wait() {
    let path = std::env::temp_dir().join(format!(
        "gw-policy-race-{}-{}.sqlite",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let first = Store::open(&path).await.unwrap();
    let second = Store::open(&path).await.unwrap();
    let policy = AccountingPolicy::FiniteUsd { limit_usd: 5.0 };
    let a = first
        .register_accounting_launch(request("race", &policy))
        .await
        .unwrap();
    let b = second
        .register_accounting_launch(request("race", &policy))
        .await
        .unwrap();
    assert_eq!(epoch(&a), epoch(&b));
    let (ia, ib) = (intent(&a), intent(&b));
    let (left, right) = tokio::join!(
        first.admit_model_attempt(&ia, epoch(&a), &[]),
        second.admit_model_attempt(&ib, epoch(&b), &[])
    );
    let results = [left.unwrap(), right.unwrap()];
    assert_eq!(
        results
            .iter()
            .filter(|r| matches!(r, AttemptAdmission::Admitted(_)))
            .count(),
        1
    );
    assert_eq!(
        results
            .iter()
            .filter(|r| matches!(
                r,
                AttemptAdmission::Denied(AdmissionDenial::UnresolvedAttempts)
            ))
            .count(),
        1
    );
    first.raw_pool().close().await;
    second.raw_pool().close().await;
    let reopened = Store::open(&path).await.unwrap();
    let resumed = reopened
        .register_accounting_launch(request("race", &policy))
        .await
        .unwrap();
    assert_eq!(
        reopened
            .admit_model_attempt(&intent(&resumed), epoch(&resumed), &[])
            .await
            .unwrap(),
        AttemptAdmission::Denied(AdmissionDenial::UnresolvedAttempts)
    );
    assert_eq!(reopened.model_attempts("race").await.unwrap().len(), 1);
    reopened.raw_pool().close().await;
    std::fs::remove_file(&path).unwrap();
    for suffix in ["-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
    }
}

#[tokio::test]
async fn a_conflicting_price_revises_known_subtotal_downward_and_denies_admission() {
    let store = Store::open_in_memory().await.unwrap();
    let launch = store
        .register_accounting_launch(request(
            "r",
            &AccountingPolicy::FiniteUsd { limit_usd: 5.0 },
        ))
        .await
        .unwrap();
    let id = begin(&store, &launch).await;
    settle(&store, &id, ReportedCost::Known(1.0)).await;
    let before = store.accounting_snapshot("r").await.unwrap();
    store
        .observe_model_attempt(
            &id,
            1,
            &AttemptMetadata {
                cost_usd: ReportedCost::Known(0.5),
                ..Default::default()
            },
        )
        .await
        .unwrap_err();
    let after = store.accounting_snapshot("r").await.unwrap();
    assert!(after.revision > before.revision);
    assert_eq!(before.known_usd, Some(1.0));
    assert_eq!(after.known_usd, Some(0.0));
    assert_eq!(after.conflicting_attempts, 1);
    assert_eq!(
        store
            .admit_model_attempt(&intent(&launch), epoch(&launch), &[])
            .await
            .unwrap(),
        AttemptAdmission::Denied(AdmissionDenial::InvalidEvidence)
    );
}

#[tokio::test]
async fn finite_changes_are_atomic_and_legacy_history_never_upgrades() {
    let store = Store::open_in_memory().await.unwrap();
    let first = store
        .register_accounting_launch(request("r", &AccountingPolicy::ObservationOnly))
        .await
        .unwrap();
    let id = begin(&store, &first).await;
    settle(&store, &id, ReportedCost::Known(0.0)).await;
    let finite = store
        .register_accounting_launch(request(
            "r",
            &AccountingPolicy::FiniteUsd { limit_usd: 2.0 },
        ))
        .await
        .unwrap();
    assert_eq!(epoch(&finite), 2);
    let same = store
        .register_accounting_launch(request(
            "r",
            &AccountingPolicy::FiniteUsd { limit_usd: 2.0 },
        ))
        .await
        .unwrap();
    assert_eq!(epoch(&same), 2);
    let changed = store
        .register_accounting_launch(request(
            "r",
            &AccountingPolicy::FiniteUsd { limit_usd: 3.0 },
        ))
        .await
        .unwrap();
    assert_eq!(epoch(&changed), 3);
    let mut unknown = request("r", &AccountingPolicy::FiniteUsd { limit_usd: 3.0 });
    unknown.teacher = AccountingCapability::Unknown;
    assert!(matches!(
        store.register_accounting_launch(unknown).await,
        Err(StorageError::Admission(AdmissionDenial::UnknownCoverage))
    ));
    assert_eq!(store.model_launches("r").await.unwrap().len(), 4);
    store
        .insert_historical_run("legacy", "{}", None)
        .await
        .unwrap();
    for _ in 0..2 {
        assert!(matches!(
            store
                .register_accounting_launch(request("legacy", &AccountingPolicy::ObservationOnly))
                .await,
            Err(StorageError::RunManifest { .. })
        ));
    }
    assert_eq!(
        store.accounting_snapshot("legacy").await.unwrap().history,
        AccountingHistory::Unknown
    );
}
