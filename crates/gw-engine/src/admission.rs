//! Live request ownership and cancellation-aware waits around short database transactions.
use crate::{EngineEvent, EventSink};
use gw_providers::{AttemptObserver, ObservationError, ObservationFuture};
use gw_schema::{
    AdmissionDenial, AttemptIntent, AttemptMetadata, LaunchCoverage, OutputInterpretation,
    TransportSettlement,
};
use gw_storage::{AttemptAdmission, Store};
use std::{collections::HashSet, sync::Mutex};
use tokio::sync::{Mutex as AsyncMutex, Notify};
use tokio_util::sync::CancellationToken;

pub(crate) struct StoreObserver {
    pub(crate) store: Store,
    pub(crate) coverage: LaunchCoverage,
    pub(crate) cancel: CancellationToken,
    pub(crate) events: EventSink,
    live: Mutex<HashSet<String>>,
    begin_gate: AsyncMutex<()>,
    snapshot_gate: AsyncMutex<()>,
    changed: Notify,
    reason: Mutex<Option<AdmissionDenial>>,
}
impl StoreObserver {
    pub(crate) fn new(
        store: Store,
        coverage: LaunchCoverage,
        cancel: CancellationToken,
        events: EventSink,
    ) -> Self {
        Self {
            store,
            coverage,
            cancel,
            events,
            live: Mutex::new(HashSet::new()),
            begin_gate: AsyncMutex::new(()),
            snapshot_gate: AsyncMutex::new(()),
            changed: Notify::new(),
            reason: Mutex::new(None),
        }
    }
    pub(crate) fn reason(&self) -> Option<AdmissionDenial> {
        self.reason
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
    fn persistence(&self, error: impl std::fmt::Display) -> ObservationError {
        // Seal the launch before releasing a transaction gate or returning to ActiveAttempt's
        // drop callback. Waiting callers must not depend on an outer shard classifying this error.
        self.cancel.cancel();
        ObservationError::Persistence(error.to_string())
    }
    async fn persist<T>(
        &self,
        operation: impl std::future::Future<Output = gw_storage::Result<T>>,
    ) -> Result<T, ObservationError> {
        operation.await.map_err(|error| self.persistence(error))
    }
    pub(crate) async fn publish(&self) {
        let _ordered = self.snapshot_gate.lock().await;
        match self.store.accounting_snapshot(&self.coverage.run_id).await {
            Ok(mut snapshot) => {
                snapshot.configured = self.coverage.policy.clone();
                self.events.emit(EngineEvent::AccountingSnapshot {
                    run_id: self.coverage.run_id.clone(),
                    snapshot,
                });
            }
            Err(error) => tracing::warn!(%error, "could not read advisory accounting snapshot"),
        }
    }
}
impl AttemptObserver for StoreObserver {
    fn begin(&self, intent: AttemptIntent) -> ObservationFuture<'_, String> {
        Box::pin(async move {
            loop {
                let changed = self.changed.notified();
                tokio::pin!(changed);
                changed.as_mut().enable();
                // Serialize this coordinator's admission/observation transactions and ownership publication.
                let gate = self.begin_gate.lock().await;
                if self.cancel.is_cancelled() {
                    return Err(ObservationError::Cancelled);
                }
                let live = self
                    .live
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>();
                let epoch = self
                    .coverage
                    .policy
                    .as_ref()
                    .ok_or_else(|| self.persistence("launch has no captured policy"))?
                    .epoch;
                let decision = self
                    .persist(self.store.admit_model_attempt(&intent, epoch, &live))
                    .await?;
                match decision {
                    AttemptAdmission::Admitted(id) => {
                        self.live
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .insert(id.clone());
                        drop(gate);
                        let mut ownership = PendingOwnership {
                            observer: self,
                            id: &id,
                            transferred: false,
                        };
                        self.publish().await;
                        ownership.transferred = true;
                        drop(ownership);
                        return Ok(id);
                    }
                    AttemptAdmission::Denied(reason) => {
                        *self
                            .reason
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner) =
                            Some(reason.clone());
                        self.cancel.cancel();
                        drop(gate);
                        self.publish().await;
                        return Err(ObservationError::Admission(reason));
                    }
                    AttemptAdmission::Wait => drop(gate),
                }
                tokio::select! {
                    biased;
                    () = self.cancel.cancelled() => return Err(ObservationError::Cancelled),
                    () = &mut changed => {},
                    // Another connection may supersede the epoch without sharing this notifier.
                    () = tokio::time::sleep(std::time::Duration::from_millis(100)) => {},
                }
            }
        })
    }
    fn metadata(
        &self,
        id: String,
        sequence: u64,
        metadata: AttemptMetadata,
    ) -> ObservationFuture<'_, ()> {
        Box::pin(async move {
            let gate = self.begin_gate.lock().await;
            self.persist(self.store.observe_model_attempt(&id, sequence, &metadata))
                .await?;
            drop(gate);
            self.publish().await;
            Ok(())
        })
    }
    fn settle(&self, id: String, settlement: TransportSettlement) -> ObservationFuture<'_, ()> {
        Box::pin(async move {
            // Admission cannot observe a committed settlement before its acknowledgment is known.
            // These gates cover short store operations only, never network consumption.
            let gate = self.begin_gate.lock().await;
            self.persist(self.store.settle_model_attempt(&id, &settlement))
                .await?;
            drop(gate);
            self.changed.notify_waiters();
            self.publish().await;
            Ok(())
        })
    }
    fn interpret(
        &self,
        id: String,
        interpretation: OutputInterpretation,
    ) -> ObservationFuture<'_, ()> {
        Box::pin(async move {
            let gate = self.begin_gate.lock().await;
            self.persist(self.store.interpret_model_attempt(&id, interpretation))
                .await?;
            drop(gate);
            self.publish().await;
            Ok(())
        })
    }
    fn released(&self, id: &str) {
        self.live
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(id);
        self.changed.notify_waiters();
    }
}

// A caller may drop its pre-send future while the advisory snapshot read is pending.
// Release only live memory ownership; the durable intent remains conservatively unresolved.
struct PendingOwnership<'a> {
    observer: &'a StoreObserver,
    id: &'a str,
    transferred: bool,
}
impl Drop for PendingOwnership<'_> {
    fn drop(&mut self) {
        if !self.transferred {
            self.observer.released(self.id);
        }
    }
}

#[cfg(test)]
#[path = "admission_tests.rs"]
mod tests;
