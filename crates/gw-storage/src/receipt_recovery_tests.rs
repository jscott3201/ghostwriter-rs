//! Physical-attempt receipts remain separate durable facts across actual process termination.
use crate::{
    AttemptAdmission, LaunchRequest, Store,
    durability_support::*,
    test_hooks::{Action, Hook},
};
use gw_schema::*;
#[path = "../tests/common/mod.rs"]
mod common;
pub(super) fn request() -> LaunchRequest<'static> {
    LaunchRequest {
        run_id: "run",
        manifest: common::manifest(),
        mode: crate::RunMode::CreateOrResume,
        policy: &AccountingPolicy::ObservationOnly,
        teacher: AccountingCapability::PhysicalAttemptsV1,
        judge: AccountingCapability::NoModelRequests,
        embedding: AccountingCapability::NoModelRequests,
    }
}
pub(super) fn intent(coverage: &LaunchCoverage) -> AttemptIntent {
    AttemptIntent {
        version: 1,
        context: AttemptContext {
            run_id: coverage.run_id.clone(),
            launch_id: coverage.launch_id.clone(),
            shard: Some(0),
            record_id: Some("not-created".into()),
            role: AttemptRole::Teacher,
            purpose: AttemptPurpose::Initial,
        },
        request_digest: "a".repeat(64),
        retry_ordinal: 0,
        requested_model: "fixture".into(),
        endpoint: "http://localhost/fixture".into(),
    }
}
pub(super) fn metadata(cost: ReportedCost, tokens: u64) -> AttemptMetadata {
    AttemptMetadata {
        cost_usd: cost,
        total_tokens: Some(tokens),
        ..Default::default()
    }
}
pub(super) fn settlement() -> TransportSettlement {
    TransportSettlement {
        outcome: TransportOutcome::Complete,
        http_status: Some(200),
        elapsed_ms: 7,
    }
}
pub(super) async fn begin(store: &Store, coverage: &LaunchCoverage) -> String {
    match store
        .admit_model_attempt(
            &intent(coverage),
            coverage.policy.as_ref().unwrap().epoch,
            &[],
        )
        .await
        .unwrap()
    {
        AttemptAdmission::Admitted(id) => id,
        other => panic!("{other:?}"),
    }
}
fn hook(store: &Store, scenario: &str, operation: &'static str) {
    let stage = if scenario == format!("{operation}_precommit") {
        Some("precommit")
    } else if scenario == format!("{operation}_committed") {
        Some("committed")
    } else {
        None
    };
    store.set_test_hook(stage.map(|stage| Hook {
        operation,
        stage,
        action: Action::Process("kill"),
    }));
}
#[test]
fn child_entry() {
    let Ok(path) = std::env::var("GW_DURABILITY_DB") else {
        return;
    };
    let scenario = std::env::var("GW_DURABILITY_SCENARIO").unwrap();
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let store = Store::open(path).await.unwrap();
        hook(&store, &scenario, "launch");
        let coverage = store.register_accounting_launch(request()).await.unwrap();
        if scenario == "launch_ack" {
            pause(&store).await;
        }
        hook(&store, &scenario, "intent");
        let id = begin(&store, &coverage).await;
        if scenario == "intent_ack" {
            pause(&store).await;
        } // No fake transmission has occurred.
        std::fs::write(
            std::path::Path::new(&std::env::var("GW_DURABILITY_DB").unwrap())
                .with_extension("sent"),
            "fake send",
        )
        .unwrap();
        hook(&store, &scenario, "metadata");
        store
            .observe_model_attempt(&id, 0, &metadata(ReportedCost::Known(0.0), 0))
            .await
            .unwrap();
        if scenario == "metadata_ack" {
            pause(&store).await;
        }
        hook(&store, &scenario, "settlement");
        store
            .settle_model_attempt(&id, &settlement())
            .await
            .unwrap();
        if scenario == "settlement_ack" {
            pause(&store).await;
        } // No output interpretation or cache exists.
        hook(&store, &scenario, "interpretation");
        store
            .interpret_model_attempt(&id, OutputInterpretation::Accepted)
            .await
            .unwrap();
        pause(&store).await;
    });
}
#[tokio::test]
async fn process_death_at_receipt_boundaries_preserves_only_committed_facts() {
    for scenario in [
        "launch_precommit",
        "launch_committed",
        "launch_ack",
        "intent_precommit",
        "intent_committed",
        "intent_ack",
        "metadata_precommit",
        "metadata_committed",
        "metadata_ack",
        "settlement_precommit",
        "settlement_committed",
        "settlement_ack",
        "interpretation_precommit",
        "interpretation_committed",
        "interpretation_ack",
    ] {
        let dir = Directory::new(scenario);
        kill_at(&dir, "receipt_recovery_tests::child_entry", scenario);
        let store = Store::open(dir.db()).await.unwrap();
        integrity(&store).await;
        let launches = store.model_launches("run").await.unwrap();
        if scenario == "launch_precommit" {
            assert!(launches.is_empty());
            assert!(store.run_status("run").await.unwrap().is_none());
            store.close().await;
            continue;
        }
        assert_eq!(launches.len(), 1);
        assert_eq!(launches[0].history, AccountingHistory::RecordedFromCreation);
        assert_eq!(
            launches[0].teacher,
            AccountingCapability::PhysicalAttemptsV1
        );
        assert_eq!(
            launches[0].policy.as_ref().unwrap().policy,
            AccountingPolicy::ObservationOnly
        );
        assert_eq!(
            store.run_status("run").await.unwrap().as_deref(),
            Some("running")
        );
        let receipts = store.model_attempts("run").await.unwrap();
        let no_intent = scenario.starts_with("launch_") || scenario == "intent_precommit";
        assert_eq!(receipts.len(), usize::from(!no_intent));
        assert!(!dir.db().with_extension("sent").exists() || !scenario.starts_with("intent_"));
        if no_intent {
            store.close().await;
            continue;
        }
        let receipt = &receipts[0];
        let has_metadata = !(scenario.starts_with("intent_") || scenario == "metadata_precommit");
        let has_settlement = scenario == "settlement_committed"
            || scenario == "settlement_ack"
            || scenario.starts_with("interpretation_");
        let has_interpretation =
            scenario == "interpretation_committed" || scenario == "interpretation_ack";
        assert_eq!(receipt.observations.len(), usize::from(has_metadata));
        assert_eq!(
            receipt.metadata.cost_usd,
            if has_metadata {
                ReportedCost::Known(0.0)
            } else {
                ReportedCost::Missing
            }
        );
        assert_eq!(receipt.transport.is_some(), has_settlement);
        assert_eq!(receipt.interpretation.is_some(), has_interpretation);
        assert!(
            store
                .cache_get("a", "teacher", "fixture", None)
                .await
                .unwrap()
                .is_none()
        );
        assert!(store.get("not-created").await.is_err());
        let summary = store.accounting_snapshot("run").await.unwrap();
        assert_eq!(summary.unresolved_attempts, u64::from(!has_settlement));
        // A later launch retains the historical attempt; unknown data never becomes zero.
        store.register_accounting_launch(request()).await.unwrap();
        assert_eq!(store.model_attempts("run").await.unwrap(), receipts);
        assert_eq!(
            store
                .accounting_snapshot("run")
                .await
                .unwrap()
                .unresolved_attempts,
            summary.unresolved_attempts
        );
        if has_metadata {
            store
                .observe_model_attempt(
                    &receipt.attempt_id,
                    0,
                    &metadata(ReportedCost::Known(0.0), 0),
                )
                .await
                .unwrap();
        }
        if has_settlement {
            store
                .settle_model_attempt(&receipt.attempt_id, &settlement())
                .await
                .unwrap();
        }
        if has_interpretation {
            store
                .interpret_model_attempt(&receipt.attempt_id, OutputInterpretation::Accepted)
                .await
                .unwrap();
        }
        assert_eq!(store.model_attempts("run").await.unwrap(), receipts);
        store.close().await;
    }
}
