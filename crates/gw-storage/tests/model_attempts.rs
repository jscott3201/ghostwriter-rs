//! Durable model intent, cumulative observations, conflicts, and incomplete history.
use gw_schema::{
    AccountingCapability as Cap, AttemptContext, AttemptIntent, AttemptMetadata, AttemptPurpose,
    AttemptRole, OutputInterpretation, ReportedCost, TransportOutcome, TransportSettlement,
};
use gw_storage::Store;

async fn intent(store: &Store) -> AttemptIntent {
    store
        .insert_historical_run("run", "{}", None)
        .await
        .unwrap();
    let coverage = store
        .begin_model_launch(
            "run",
            Cap::PhysicalAttemptsV1,
            Cap::Unknown,
            Cap::NoModelRequests,
        )
        .await
        .unwrap();
    AttemptIntent {
        version: 1,
        context: AttemptContext {
            run_id: "run".into(),
            launch_id: coverage.launch_id,
            shard: Some(2),
            record_id: Some("not-yet-created".into()),
            role: AttemptRole::Teacher,
            purpose: AttemptPurpose::Initial,
        },
        request_digest: "a".repeat(64),
        retry_ordinal: 0,
        requested_model: "fixture".into(),
        endpoint: "http://localhost/v1/chat/completions".into(),
    }
}
fn settlement() -> TransportSettlement {
    TransportSettlement {
        outcome: TransportOutcome::Complete,
        http_status: Some(200),
        elapsed_ms: 10,
    }
}
#[tokio::test]
async fn cumulative_updates_replace_and_repeated_updates_and_settlement_are_idempotent() {
    let store = Store::open_in_memory().await.unwrap();
    let intent = intent(&store).await;
    let id = store.begin_model_attempt(&intent).await.unwrap();
    let first = AttemptMetadata {
        prompt_tokens: Some(4),
        total_tokens: Some(4),
        cost_usd: ReportedCost::Known(0.0),
        ..Default::default()
    };
    let final_metadata = AttemptMetadata {
        total_tokens: Some(10),
        completion_tokens: Some(6),
        cost_usd: ReportedCost::Known(0.25),
        ..Default::default()
    };
    for _ in 0..2 {
        store.observe_model_attempt(&id, 0, &first).await.unwrap();
    }
    for _ in 0..2 {
        store
            .observe_model_attempt(&id, 1, &final_metadata)
            .await
            .unwrap();
    }
    // Retrying the same earlier operation does not move current totals backward.
    store.observe_model_attempt(&id, 0, &first).await.unwrap();
    for _ in 0..2 {
        store
            .settle_model_attempt(&id, &settlement())
            .await
            .unwrap();
        store
            .interpret_model_attempt(&id, OutputInterpretation::Invalid)
            .await
            .unwrap();
    }
    let receipt = store.model_attempts("run").await.unwrap().pop().unwrap();
    assert_eq!(receipt.metadata.prompt_tokens, Some(4));
    assert_eq!(receipt.metadata.total_tokens, Some(10));
    assert_eq!(receipt.metadata.cost_usd, ReportedCost::Known(0.25));
    assert_eq!(receipt.observations.len(), 2);
    assert!(receipt.conflicts.is_empty());
    assert_eq!(receipt.transport, Some(settlement()));
    assert_eq!(receipt.interpretation, Some(OutputInterpretation::Invalid));
}
#[tokio::test]
async fn contradictory_metadata_and_terminal_updates_are_durable_errors() {
    let store = Store::open_in_memory().await.unwrap();
    let id = store
        .begin_model_attempt(&intent(&store).await)
        .await
        .unwrap();
    store
        .observe_model_attempt(
            &id,
            0,
            &AttemptMetadata {
                total_tokens: Some(10),
                response_id: Some("one".into()),
                cost_usd: ReportedCost::Known(0.5),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let conflicting = AttemptMetadata {
        total_tokens: Some(9),
        response_id: Some("two".into()),
        cost_usd: ReportedCost::Known(0.2),
        ..Default::default()
    };
    assert!(
        store
            .observe_model_attempt(&id, 1, &conflicting)
            .await
            .is_err()
    );
    store
        .settle_model_attempt(&id, &settlement())
        .await
        .unwrap();
    assert!(
        store
            .settle_model_attempt(
                &id,
                &TransportSettlement {
                    outcome: TransportOutcome::Failed,
                    ..settlement()
                }
            )
            .await
            .is_err()
    );
    store
        .interpret_model_attempt(&id, OutputInterpretation::Accepted)
        .await
        .unwrap();
    assert!(
        store
            .interpret_model_attempt(&id, OutputInterpretation::Invalid)
            .await
            .is_err()
    );
    let receipt = store.model_attempts("run").await.unwrap().pop().unwrap();
    for field in [
        "total_tokens",
        "response_id",
        "cost_usd",
        "transport",
        "interpretation",
    ] {
        assert!(receipt.conflicts.contains(&field.into()));
    }
    assert_eq!(receipt.observations.len(), 2);
    assert_eq!(receipt.observations[1].metadata, conflicting);
    assert_eq!(receipt.observations[1].sequence, 1);
    assert_eq!(receipt.transport, Some(settlement()));
    assert_eq!(receipt.interpretation, Some(OutputInterpretation::Accepted));
}
#[tokio::test]
async fn known_zero_missing_invalid_and_unresolved_remain_distinct_after_file_reopen() {
    let path = std::env::temp_dir().join(format!(
        "gw-attempts-{}-{}.sqlite",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let store = Store::open(&path).await.unwrap();
    let intent = intent(&store).await;
    for cost in [
        ReportedCost::Missing,
        ReportedCost::Known(0.0),
        ReportedCost::Invalid,
    ] {
        let id = store.begin_model_attempt(&intent).await.unwrap();
        store
            .observe_model_attempt(
                &id,
                0,
                &AttemptMetadata {
                    cost_usd: cost,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
    }
    let failed_id = store.begin_model_attempt(&intent).await.unwrap();
    store
        .observe_model_attempt(
            &failed_id,
            0,
            &AttemptMetadata {
                cost_usd: ReportedCost::Known(0.75),
                total_tokens: Some(9),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    store
        .settle_model_attempt(
            &failed_id,
            &TransportSettlement {
                outcome: TransportOutcome::Failed,
                ..settlement()
            },
        )
        .await
        .unwrap();
    store
        .interpret_model_attempt(&failed_id, OutputInterpretation::Invalid)
        .await
        .unwrap();
    store.close().await;
    let reopened = Store::open(&path).await.unwrap();
    let receipts = reopened.model_attempts("run").await.unwrap();
    assert_eq!(receipts.len(), 4);
    for expected in [
        ReportedCost::Missing,
        ReportedCost::Known(0.0),
        ReportedCost::Invalid,
    ] {
        assert!(receipts.iter().any(|r| r.metadata.cost_usd == expected));
    }
    assert_eq!(
        receipts
            .iter()
            .filter(|r| r.transport.is_none() && r.interpretation.is_none())
            .count(),
        3
    );
    let failed = receipts.iter().find(|r| r.attempt_id == failed_id).unwrap();
    assert_eq!(failed.metadata.cost_usd, ReportedCost::Known(0.75));
    assert_eq!(failed.metadata.total_tokens, Some(9));
    assert_eq!(
        failed.transport.as_ref().unwrap().outcome,
        TransportOutcome::Failed
    );
    assert_eq!(failed.interpretation, Some(OutputInterpretation::Invalid));
    assert_eq!(
        reopened.model_launches("run").await.unwrap()[0].judge,
        Cap::Unknown
    );
    reopened.close().await;
    std::fs::remove_file(path).unwrap();
}
#[tokio::test]
async fn later_coverage_does_not_relabel_earlier_or_legacy_history() {
    let store = Store::open_in_memory().await.unwrap();
    store
        .insert_historical_run("run", "{}", None)
        .await
        .unwrap();
    assert!(store.model_launches("run").await.unwrap().is_empty());
    let first = store
        .begin_model_launch("run", Cap::Unknown, Cap::Unknown, Cap::NoModelRequests)
        .await
        .unwrap();
    let second = store
        .begin_model_launch(
            "run",
            Cap::PhysicalAttemptsV1,
            Cap::PhysicalAttemptsV1,
            Cap::PhysicalAttemptsV1,
        )
        .await
        .unwrap();
    assert_ne!(first.launch_id, second.launch_id);
    let launches = store.model_launches("run").await.unwrap();
    assert!(launches.contains(&first) && launches.contains(&second));
    assert!(store.model_attempts("run").await.unwrap().is_empty());
}

#[tokio::test]
async fn invalid_typed_prices_cannot_serialize_as_known_null_or_negative_cost() {
    let store = Store::open_in_memory().await.unwrap();
    let intent = intent(&store).await;
    for value in [-1.0, f64::NAN, f64::INFINITY] {
        let id = store.begin_model_attempt(&intent).await.unwrap();
        store
            .observe_model_attempt(
                &id,
                0,
                &AttemptMetadata {
                    cost_usd: ReportedCost::Known(value),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
    }
    let receipts = store.model_attempts("run").await.unwrap();
    assert_eq!(receipts.len(), 3);
    assert!(
        receipts
            .iter()
            .all(|r| r.metadata.cost_usd == ReportedCost::Invalid
                && r.metadata.invalid_fields.contains(&"cost".into()))
    );
}

#[tokio::test]
async fn fresh_decreases_and_reused_sequence_conflicts_survive_reopen_and_retries() {
    let path = std::env::temp_dir().join(format!(
        "gw-attempt-operations-{}-{}.sqlite",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let store = Store::open(&path).await.unwrap();
    let id = store
        .begin_model_attempt(&intent(&store).await)
        .await
        .unwrap();
    let first = AttemptMetadata {
        total_tokens: Some(10),
        cost_usd: ReportedCost::Known(0.1),
        ..Default::default()
    };
    let next = AttemptMetadata {
        total_tokens: Some(20),
        cost_usd: ReportedCost::Known(0.2),
        ..Default::default()
    };
    store.observe_model_attempt(&id, 0, &first).await.unwrap();
    store.observe_model_attempt(&id, 1, &next).await.unwrap();
    // Identical payload with a fresh identity is new evidence, including a decrease.
    let decreasing = store
        .observe_model_attempt(&id, 2, &first)
        .await
        .unwrap_err()
        .to_string();
    // Reusing an operation's identity with different data also retains a durable error.
    let reused = store
        .observe_model_attempt(&id, 0, &next)
        .await
        .unwrap_err()
        .to_string();
    let before = store.model_attempts("run").await.unwrap();
    assert_eq!(before[0].observations.len(), 4);
    assert_eq!(
        before[0]
            .observations
            .iter()
            .map(|entry| entry.sequence)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 0]
    );
    for field in ["cost_usd", "total_tokens", "observation_sequence"] {
        assert!(before[0].conflicts.contains(&field.into()));
    }
    store.close().await;
    let reopened = Store::open(&path).await.unwrap();
    // Lost-response retries are stable across processes and never replay older values into totals.
    reopened
        .observe_model_attempt(&id, 0, &first)
        .await
        .unwrap();
    reopened.observe_model_attempt(&id, 1, &next).await.unwrap();
    assert_eq!(
        reopened
            .observe_model_attempt(&id, 2, &first)
            .await
            .unwrap_err()
            .to_string(),
        decreasing
    );
    assert_eq!(
        reopened
            .observe_model_attempt(&id, 0, &next)
            .await
            .unwrap_err()
            .to_string(),
        reused
    );
    assert_eq!(reopened.model_attempts("run").await.unwrap(), before);
    reopened.close().await;
    std::fs::remove_file(path).unwrap();
}

#[tokio::test]
async fn skipped_observation_sequences_are_retained_as_errors_without_updating_totals() {
    let store = Store::open_in_memory().await.unwrap();
    let id = store
        .begin_model_attempt(&intent(&store).await)
        .await
        .unwrap();
    let metadata = AttemptMetadata {
        cost_usd: ReportedCost::Known(0.2),
        ..Default::default()
    };
    for _ in 0..2 {
        assert!(
            store
                .observe_model_attempt(&id, 1, &metadata)
                .await
                .is_err()
        );
    }
    let receipt = store.model_attempts("run").await.unwrap().pop().unwrap();
    assert_eq!(receipt.observations.len(), 1);
    assert_eq!(receipt.observations[0].metadata, metadata);
    assert_eq!(receipt.metadata.cost_usd, ReportedCost::Missing);
    assert_eq!(receipt.conflicts, vec!["observation_sequence"]);
}

#[tokio::test]
async fn conflicting_terminal_retries_stay_rejected_after_reopen() {
    let path = std::env::temp_dir().join(format!(
        "gw-attempt-terminals-{}-{}.sqlite",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let store = Store::open(&path).await.unwrap();
    let id = store
        .begin_model_attempt(&intent(&store).await)
        .await
        .unwrap();
    let canonical = settlement();
    let conflicting = TransportSettlement {
        outcome: TransportOutcome::Failed,
        ..canonical.clone()
    };
    let other = TransportSettlement {
        outcome: TransportOutcome::HttpError,
        ..canonical.clone()
    };
    store.settle_model_attempt(&id, &canonical).await.unwrap();
    store
        .interpret_model_attempt(&id, OutputInterpretation::Accepted)
        .await
        .unwrap();
    let mut rejected = Vec::new();
    for _ in 0..2 {
        rejected.push(store.settle_model_attempt(&id, &conflicting).await.is_err());
        rejected.push(
            store
                .interpret_model_attempt(&id, OutputInterpretation::Invalid)
                .await
                .is_err(),
        );
    }
    store.close().await;
    let reopened = Store::open(&path).await.unwrap();
    for value in [&conflicting, &other] {
        rejected.push(reopened.settle_model_attempt(&id, value).await.is_err());
    }
    for value in [
        OutputInterpretation::Invalid,
        OutputInterpretation::Truncated,
    ] {
        rejected.push(reopened.interpret_model_attempt(&id, value).await.is_err());
    }
    let canonical_retries = [
        reopened.settle_model_attempt(&id, &canonical).await.is_ok(),
        reopened
            .interpret_model_attempt(&id, OutputInterpretation::Accepted)
            .await
            .is_ok(),
    ];
    let receipt = reopened.model_attempts("run").await.unwrap().pop().unwrap();
    reopened.close().await;
    std::fs::remove_file(path).unwrap();
    assert!(rejected.iter().all(|rejected| *rejected), "{rejected:?}");
    assert_eq!(canonical_retries, [true, true]);
    assert_eq!(receipt.transport, Some(canonical));
    assert_eq!(receipt.interpretation, Some(OutputInterpretation::Accepted));
    assert_eq!(receipt.conflicts, vec!["transport", "interpretation"]);
}
