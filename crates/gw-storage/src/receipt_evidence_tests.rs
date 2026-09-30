//! Exact observation and terminal retries after process death retain their original result.
use crate::{
    Store,
    durability_support::*,
    receipt_recovery_tests::{begin, intent, metadata, request, settlement},
    test_hooks::{Action, Hook},
};
use gw_schema::*;
fn observations() -> Vec<AttemptMetadata> {
    vec![
        metadata(ReportedCost::Known(0.0), 0),
        metadata(ReportedCost::Missing, 4),
        metadata(ReportedCost::Invalid, 8),
        metadata(ReportedCost::Known(1.0), 9),
        metadata(ReportedCost::Known(0.5), 2),
    ]
}
fn conflicting_transport() -> TransportSettlement {
    TransportSettlement {
        outcome: TransportOutcome::Failed,
        ..settlement()
    }
}
#[test]
fn child_entry() {
    let Ok(path) = std::env::var("GW_DURABILITY_DB") else {
        return;
    };
    let scenario = std::env::var("GW_DURABILITY_SCENARIO").unwrap();
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let store = Store::open(path).await.unwrap();
        if scenario == "unknown_history_ack" {
            store
                .insert_historical_run("run", "{}", None)
                .await
                .unwrap();
            let coverage = store
                .begin_model_launch(
                    "run",
                    AccountingCapability::Unknown,
                    AccountingCapability::NoModelRequests,
                    AccountingCapability::NoModelRequests,
                )
                .await
                .unwrap();
            store.begin_model_attempt(&intent(&coverage)).await.unwrap();
            pause(&store).await;
            return;
        }
        let coverage = store.register_accounting_launch(request()).await.unwrap();
        let _missing = begin(&store, &coverage).await;
        let zero = begin(&store, &coverage).await;
        store
            .observe_model_attempt(&zero, 0, &metadata(ReportedCost::Known(0.0), 0))
            .await
            .unwrap();
        let id = begin(&store, &coverage).await;
        for (sequence, observation) in observations().iter().enumerate() {
            if sequence == 4 && scenario == "metadata_conflict_committed" {
                store.set_test_hook(Some(Hook {
                    operation: "metadata",
                    stage: "committed",
                    action: Action::Process("kill"),
                }));
            }
            assert_eq!(
                store
                    .observe_model_attempt(&id, sequence as u64, observation)
                    .await
                    .is_ok(),
                sequence < 4
            );
        }
        store
            .settle_model_attempt(&id, &settlement())
            .await
            .unwrap();
        if scenario == "terminal_conflict_committed" {
            store.set_test_hook(Some(Hook {
                operation: "settlement",
                stage: "committed",
                action: Action::Process("kill"),
            }));
        }
        assert!(
            store
                .settle_model_attempt(&id, &conflicting_transport())
                .await
                .is_err()
        );
        store
            .interpret_model_attempt(&id, OutputInterpretation::Accepted)
            .await
            .unwrap();
        if scenario == "interpretation_conflict_committed" {
            store.set_test_hook(Some(Hook {
                operation: "interpretation",
                stage: "committed",
                action: Action::Process("kill"),
            }));
        }
        assert!(
            store
                .interpret_model_attempt(&id, OutputInterpretation::Invalid)
                .await
                .is_err()
        );
        pause(&store).await;
    });
}
#[tokio::test]
async fn killed_receipts_keep_zero_missing_sticky_invalid_and_exact_conflict_results() {
    for scenario in [
        "evidence_ack",
        "metadata_conflict_committed",
        "terminal_conflict_committed",
        "interpretation_conflict_committed",
    ] {
        let dir = Directory::new(scenario);
        kill_at(&dir, "receipt_evidence_tests::child_entry", scenario);
        let store = Store::open(dir.db()).await.unwrap();
        integrity(&store).await;
        let before = store.model_attempts("run").await.unwrap();
        assert_eq!(before.len(), 3);
        assert!(
            before
                .iter()
                .any(|r| r.observations.is_empty() && r.metadata.cost_usd == ReportedCost::Missing)
        );
        assert!(
            before
                .iter()
                .any(|r| r.observations.len() == 1
                    && r.metadata.cost_usd == ReportedCost::Known(0.0))
        );
        let receipt = before.iter().find(|r| r.observations.len() == 5).unwrap();
        assert!(receipt.metadata.invalid_fields.contains(&"cost".into()));
        assert!(receipt.conflicts.contains(&"total_tokens".into()));
        for (sequence, observation) in observations().iter().enumerate() {
            assert_eq!(receipt.observations[sequence].sequence, sequence as u64);
            assert_eq!(&receipt.observations[sequence].metadata, observation);
            assert_eq!(
                store
                    .observe_model_attempt(&receipt.attempt_id, sequence as u64, observation)
                    .await
                    .is_ok(),
                sequence < 4
            );
        }
        if scenario != "metadata_conflict_committed" {
            assert_eq!(receipt.transport, Some(settlement()));
            store
                .settle_model_attempt(&receipt.attempt_id, &settlement())
                .await
                .unwrap();
            for _ in 0..2 {
                assert!(
                    store
                        .settle_model_attempt(&receipt.attempt_id, &conflicting_transport())
                        .await
                        .is_err()
                );
            }
        }
        if matches!(
            scenario,
            "evidence_ack" | "interpretation_conflict_committed"
        ) {
            assert_eq!(receipt.interpretation, Some(OutputInterpretation::Accepted));
            store
                .interpret_model_attempt(&receipt.attempt_id, OutputInterpretation::Accepted)
                .await
                .unwrap();
            for _ in 0..2 {
                assert!(
                    store
                        .interpret_model_attempt(&receipt.attempt_id, OutputInterpretation::Invalid)
                        .await
                        .is_err()
                );
            }
        }
        assert_eq!(store.model_attempts("run").await.unwrap(), before);
        let summary = store.accounting_snapshot("run").await.unwrap();
        assert_eq!(summary.invalid_cost_attempts, 1);
        assert_eq!(summary.unknown_cost_attempts, 1);
        store.register_accounting_launch(request()).await.unwrap();
        assert_eq!(store.model_attempts("run").await.unwrap(), before);
        store.close().await;
    }
}
#[tokio::test]
async fn acknowledged_unknown_history_survives_death_and_later_capable_launch() {
    let dir = Directory::new("unknown-history");
    kill_at(
        &dir,
        "receipt_evidence_tests::child_entry",
        "unknown_history_ack",
    );
    let store = Store::open(dir.db()).await.unwrap();
    integrity(&store).await;
    let receipts = store.model_attempts("run").await.unwrap();
    assert_eq!(receipts.len(), 1);
    store
        .begin_model_launch(
            "run",
            AccountingCapability::PhysicalAttemptsV1,
            AccountingCapability::NoModelRequests,
            AccountingCapability::NoModelRequests,
        )
        .await
        .unwrap();
    let coverage = store.model_launches("run").await.unwrap();
    assert_eq!(coverage.len(), 2);
    assert!(
        coverage
            .iter()
            .any(|c| c.teacher == AccountingCapability::Unknown)
    );
    assert!(
        coverage
            .iter()
            .all(|c| c.history == AccountingHistory::Unknown)
    );
    let summary = store.accounting_snapshot("run").await.unwrap();
    assert_eq!(summary.history, AccountingHistory::Unknown);
    assert_eq!(summary.unresolved_attempts, 1);
    assert_eq!(store.model_attempts("run").await.unwrap(), receipts);
    store.close().await;
}
#[tokio::test]
async fn malformed_accounting_state_and_manifest_mismatch_never_reset_the_ledger() {
    let dir = Directory::new("malformed-ledger");
    let store = Store::open(dir.db()).await.unwrap();
    let coverage = store.register_accounting_launch(request()).await.unwrap();
    let id = begin(&store, &coverage).await;
    let before = store.accounting_snapshot("run").await.unwrap();
    store
        .set_run_status("run", crate::RunStatus::Completed)
        .await
        .unwrap();
    let mut mismatch = request();
    mismatch.manifest.input_plan.content_hash = "b".repeat(64);
    assert!(store.register_accounting_launch(mismatch).await.is_err());
    assert_eq!(store.accounting_snapshot("run").await.unwrap(), before);
    assert_eq!(
        store.run_status("run").await.unwrap().as_deref(),
        Some("completed")
    );
    assert_eq!(store.model_launches("run").await.unwrap(), vec![coverage]);
    sqlx::query("UPDATE model_attempts SET receipt_json='{' WHERE attempt_id=?")
        .bind(id)
        .execute(store.pool())
        .await
        .unwrap();
    assert!(store.model_attempts("run").await.is_err());
    assert!(store.accounting_snapshot("run").await.is_err());
    sqlx::query("UPDATE model_launches SET coverage_json='{'")
        .execute(store.pool())
        .await
        .unwrap();
    assert!(store.model_launches("run").await.is_err());
    sqlx::query("UPDATE run_accounting SET policy_json='{'")
        .execute(store.pool())
        .await
        .unwrap();
    assert!(store.register_accounting_launch(request()).await.is_err());
    sqlx::query("UPDATE runs SET config_json='{'")
        .execute(store.pool())
        .await
        .unwrap();
    assert!(store.register_accounting_launch(request()).await.is_err());
    assert_eq!(
        store.run_status("run").await.unwrap().as_deref(),
        Some("completed")
    );
    store.close().await;
}
