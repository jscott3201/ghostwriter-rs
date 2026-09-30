//! Queue real coordinator futures while outer error handling is deliberately absent.
use super::*;
use gw_schema::{
    AccountingCapability, AccountingPolicy, AttemptContext, AttemptPurpose, AttemptRole,
    ReportedCost, TransportOutcome,
};
use gw_storage::LaunchRequest;
use std::{
    future::{Future, poll_fn},
    sync::Arc,
    time::Duration,
};
use tokio::sync::oneshot;

async fn observer(policy: AccountingPolicy) -> Arc<StoreObserver> {
    let store = Store::open_in_memory().await.unwrap();
    let coverage = store
        .register_accounting_launch(LaunchRequest {
            run_id: "r",
            config_json: "{}",
            shard_count: 1,
            prompts_hash: "p",
            policy: &policy,
            teacher: AccountingCapability::PhysicalAttemptsV1,
            judge: AccountingCapability::NoModelRequests,
            embedding: AccountingCapability::NoModelRequests,
        })
        .await
        .unwrap();
    Arc::new(StoreObserver::new(
        store,
        coverage,
        CancellationToken::new(),
        EventSink::disconnected(),
    ))
}
fn intent(observer: &StoreObserver, model: &str) -> AttemptIntent {
    AttemptIntent {
        version: 1,
        context: AttemptContext {
            run_id: "r".into(),
            launch_id: observer.coverage.launch_id.clone(),
            shard: Some(0),
            record_id: None,
            role: AttemptRole::Teacher,
            purpose: AttemptPurpose::Initial,
        },
        request_digest: "a".repeat(64),
        retry_ordinal: 0,
        requested_model: model.into(),
        endpoint: "http://localhost/test".into(),
    }
}
fn metadata() -> AttemptMetadata {
    AttemptMetadata {
        cost_usd: ReportedCost::Known(0.1),
        ..Default::default()
    }
}
fn settlement() -> TransportSettlement {
    TransportSettlement {
        outcome: TransportOutcome::Complete,
        http_status: Some(200),
        elapsed_ms: 1,
    }
}

/// The readiness signal means `begin` was actually polled, not merely spawned. When the caller
/// holds begin_gate this is a queued lock waiter; otherwise acquiring that gate after this signal
/// joins the initial admission transaction and establishes that the finite waiter reached Wait.
async fn queued_begin(
    observer: Arc<StoreObserver>,
    model: &str,
) -> tokio::task::JoinHandle<Result<String, ObservationError>> {
    let intent = intent(&observer, model);
    queued(async move { observer.begin(intent).await }).await
}
async fn queued<T: Send + 'static>(
    future: impl Future<Output = T> + Send + 'static,
) -> tokio::task::JoinHandle<T> {
    let (ready, polled) = oneshot::channel();
    let task = tokio::spawn(async move {
        let mut future = Box::pin(future);
        let mut ready = Some(ready);
        poll_fn(|cx| {
            let result = future.as_mut().poll(cx);
            if let Some(ready) = ready.take() {
                let _ = ready.send(());
            }
            result
        })
        .await
    });
    polled.await.unwrap();
    task
}
async fn finish<T>(task: tokio::task::JoinHandle<T>) -> T {
    tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("queued caller drains")
        .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failed_begin_seals_queued_admission_before_outer_error_handling() {
    let observer = observer(AccountingPolicy::ObservationOnly).await;
    sqlx::query("CREATE TRIGGER fail_begin BEFORE INSERT ON model_attempts WHEN json_extract(NEW.receipt_json, '$.intent.requested_model') = 'fault' BEGIN SELECT RAISE(FAIL, 'begin failed'); END")
        .execute(observer.store.raw_pool()).await.unwrap();
    let gate = observer.begin_gate.lock().await;
    let failed = queued_begin(observer.clone(), "fault").await;
    let waiting = queued_begin(observer.clone(), "later").await;
    drop(gate);
    let error = finish(failed).await.unwrap_err();
    assert!(
        matches!(&error, ObservationError::Persistence(detail) if detail.contains("begin failed"))
    );
    // No executor/shard classifier has received the error; the observer must already be sealed.
    assert!(matches!(
        finish(waiting).await,
        Err(ObservationError::Cancelled)
    ));
    assert!(observer.store.model_attempts("r").await.unwrap().is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn each_failed_observation_seals_admission_but_started_attempts_still_drain() {
    for stage in ["metadata", "settlement", "interpretation"] {
        let observer = observer(AccountingPolicy::ObservationOnly).await;
        let id = observer.begin(intent(&observer, "first")).await.unwrap();
        sqlx::query("CREATE TRIGGER fail_update BEFORE UPDATE ON model_attempts BEGIN SELECT RAISE(FAIL, 'callback failed'); END")
            .execute(observer.store.raw_pool()).await.unwrap();
        let gate = observer.begin_gate.lock().await;
        let callback_observer = observer.clone();
        let callback_id = id.clone();
        let failed = queued(async move {
            match stage {
                "metadata" => callback_observer.metadata(callback_id, 0, metadata()).await,
                "settlement" => callback_observer.settle(callback_id, settlement()).await,
                _ => {
                    callback_observer
                        .interpret(callback_id, OutputInterpretation::Accepted)
                        .await
                }
            }
        })
        .await;
        let waiting = queued_begin(observer.clone(), "later").await;
        drop(gate);
        let error = finish(failed).await.unwrap_err();
        assert!(
            matches!(&error, ObservationError::Persistence(detail) if detail.contains("callback failed"))
        );
        assert!(
            matches!(finish(waiting).await, Err(ObservationError::Cancelled)),
            "{stage}"
        );
        sqlx::query("DROP TRIGGER fail_update")
            .execute(observer.store.raw_pool())
            .await
            .unwrap();
        observer.metadata(id.clone(), 0, metadata()).await.unwrap();
        observer.settle(id.clone(), settlement()).await.unwrap();
        observer
            .interpret(id.clone(), OutputInterpretation::Accepted)
            .await
            .unwrap();
        observer.released(&id);
        let receipts = observer.store.model_attempts("r").await.unwrap();
        assert_eq!(receipts.len(), 1);
        assert_eq!(receipts[0].transport, Some(settlement()));
        assert_eq!(
            receipts[0].interpretation,
            Some(OutputInterpretation::Accepted)
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn committed_settlement_with_failed_ack_seals_finite_waiter_before_release() {
    let observer = observer(AccountingPolicy::FiniteUsd { limit_usd: 5.0 }).await;
    let id = observer.begin(intent(&observer, "first")).await.unwrap();
    observer.metadata(id.clone(), 0, metadata()).await.unwrap();
    let waiting = queued_begin(observer.clone(), "later").await;
    let gate = observer.begin_gate.lock().await;
    assert!(
        !waiting.is_finished(),
        "finite caller reached the active-attempt wait"
    );
    // Inject a lost acknowledgment around the same persistence boundary all callbacks use.
    // The durable settlement is valid, so evidence denial cannot mask an unsealed coordinator.
    let error = observer
        .persist(async {
            observer
                .store
                .settle_model_attempt(&id, &settlement())
                .await?;
            Err::<(), _>(gw_storage::StorageError::Attempt(
                "settlement acknowledgement lost".into(),
            ))
        })
        .await
        .unwrap_err();
    assert!(
        matches!(&error, ObservationError::Persistence(detail) if detail.contains("acknowledgement lost"))
    );
    assert_eq!(
        observer
            .store
            .accounting_snapshot("r")
            .await
            .unwrap()
            .unresolved_attempts,
        0
    );
    observer.released(&id); // ActiveAttempt's drop callback runs before the outer error classifier.
    drop(gate);
    assert!(matches!(
        finish(waiting).await,
        Err(ObservationError::Cancelled)
    ));
    assert_eq!(observer.store.model_attempts("r").await.unwrap().len(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn queued_finite_waiter_reclassifies_dropped_owner_as_unresolved_and_drains() {
    let observer = observer(AccountingPolicy::FiniteUsd { limit_usd: 5.0 }).await;
    let id = observer.begin(intent(&observer, "first")).await.unwrap();
    let waiting = queued_begin(observer.clone(), "later").await;
    let gate = observer.begin_gate.lock().await;
    assert!(!waiting.is_finished());
    observer.released(&id);
    drop(gate);
    assert!(matches!(
        finish(waiting).await,
        Err(ObservationError::Admission(
            AdmissionDenial::UnresolvedAttempts
        ))
    ));
    assert_eq!(observer.store.model_attempts("r").await.unwrap().len(), 1);
}
