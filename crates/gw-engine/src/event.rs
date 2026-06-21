//! The engine event stream — the headless side of the Component+Action TUI design (ARCHITECTURE §3.1,
//! `_research/06-rust-ratatui-architecture.md`).
//!
//! `gw-engine` is HEADLESS-runnable; the TUI and the CLI are interchangeable CONSUMERS of the event
//! stream it emits. The engine never knows whether a terminal exists — it just publishes
//! [`EngineEvent`]s onto a [`tokio::sync::mpsc`] channel. A consumer (the TUI's `Action` loop, a CLI
//! progress bar, or a test) drains the receiver; if nobody is listening the send is a no-op (the
//! engine is never blocked by a slow or absent consumer).
//!
//! ## Why a bounded channel + non-blocking send
//!
//! At high token rates a slow render must not back-pressure (or OOM) the engine. The
//! [`EventSink`] uses a BOUNDED channel and a NON-BLOCKING `try_send`: when the buffer is full the
//! event is dropped (and counted) rather than awaited. Events are OBSERVABILITY, never the source of
//! truth — the SQLite run-ledger is authoritative for lifecycle, so a dropped event never loses
//! data. (D3-open in the spec: coalesce deltas per render tick at high throughput; here we drop, the
//! cheapest backpressure policy for a v1 lifecycle-transition stream that is far lower-rate than the
//! per-token delta stream the TUI sources separately.)

use gw_schema::LifecycleState;
use tokio::sync::mpsc::{Receiver, Sender, error::TrySendError};

/// The default bounded capacity of the event channel. Lifecycle transitions are low-rate (a handful
/// per record), so a few thousand in-flight events is ample headroom before the drop policy engages.
pub const DEFAULT_EVENT_CAPACITY: usize = 4096;

/// One observable event emitted by the engine as it drives records through the pipeline.
///
/// This is the engine-side analogue of the TUI's `Action` enum: a consumer maps these onto its own
/// model. Every variant carries the `record_id` (or `run`/`shard`) it pertains to, so a consumer can
/// route it without engine internals. Events are advisory — the authoritative state is in storage.
#[derive(Debug, Clone, PartialEq)]
pub enum EngineEvent {
    /// A run started: `shards` shards were planned for `run_id`.
    RunStarted {
        /// The run id.
        run_id: String,
        /// The number of shards the seed space was partitioned into.
        shards: usize,
    },
    /// A shard began processing (fresh, or resumed from a checkpoint).
    ShardStarted {
        /// The run id.
        run_id: String,
        /// The 0-based shard index.
        shard: i64,
        /// `true` if the shard resumed from a persisted checkpoint rather than starting fresh.
        resumed: bool,
    },
    /// A record transitioned to a new lifecycle state (the load-bearing event — one per persisted
    /// transition, so a consumer accumulates the exact lifecycle the storage layer recorded).
    StateAdvanced {
        /// The record id.
        record_id: String,
        /// The state the record advanced TO.
        to: LifecycleState,
    },
    /// A teacher/judge call's cost was charged against the budget meter (post-spend accounting).
    CostCharged {
        /// The record id the spend is attributed to.
        record_id: String,
        /// The marginal USD this call cost.
        usd: f64,
        /// The cumulative USD spent for the run after this charge.
        run_total_usd: f64,
    },
    /// The budget cap was reached; no new teacher work will be dispatched.
    BudgetReached {
        /// The run id.
        run_id: String,
        /// The cumulative USD spent.
        spent: f64,
        /// The configured cap.
        cap: f64,
    },
    /// A record failed unrecoverably and was parked at [`LifecycleState::Error`].
    RecordErrored {
        /// The record id.
        record_id: String,
        /// A human-readable error message (never carries a secret).
        error: String,
    },
    /// A shard finished (all its records reached a terminal state or the run drained).
    ShardFinished {
        /// The run id.
        run_id: String,
        /// The 0-based shard index.
        shard: i64,
    },
    /// The run finished (all shards drained, or the run halted on budget/cancellation).
    RunFinished {
        /// The run id.
        run_id: String,
        /// `true` if every shard drained cleanly; `false` if the run halted early.
        completed: bool,
    },
}

/// The producing half of the engine event stream: a cheap-to-clone, non-blocking publisher.
///
/// Cloning shares the underlying channel, so every spawned shard worker holds its own
/// [`EventSink`] clone and emits concurrently. [`emit`](EventSink::emit) NEVER blocks and NEVER errors
/// — a full buffer (slow/absent consumer) drops the event. Use [`subscribe`](EventSink::subscribe) to
/// create a sink + its receiver, or [`disconnected`](EventSink::disconnected) for a headless run with
/// no consumer at all (every emit is a silent no-op).
#[derive(Debug, Clone)]
pub struct EventSink {
    tx: Option<Sender<EngineEvent>>,
}

impl EventSink {
    /// Create a connected sink and its bounded receiver at [`DEFAULT_EVENT_CAPACITY`]. The caller
    /// owns the [`Receiver`] (the TUI / CLI / test); the engine holds the [`EventSink`].
    #[must_use]
    pub fn subscribe() -> (Self, Receiver<EngineEvent>) {
        Self::subscribe_with_capacity(DEFAULT_EVENT_CAPACITY)
    }

    /// Create a connected sink and its bounded receiver at an explicit `capacity`.
    #[must_use]
    pub fn subscribe_with_capacity(capacity: usize) -> (Self, Receiver<EngineEvent>) {
        let (tx, rx) = tokio::sync::mpsc::channel(capacity.max(1));
        (Self { tx: Some(tx) }, rx)
    }

    /// A disconnected sink: every [`emit`](EventSink::emit) is a silent no-op. For a fully headless
    /// run (a CI batch job) that consumes no events.
    #[must_use]
    pub fn disconnected() -> Self {
        Self { tx: None }
    }

    /// Publish an event without blocking. A full buffer (slow/absent consumer) DROPS the event and
    /// returns `false`; a successful enqueue returns `true`. A closed receiver also returns `false`.
    /// Events are observability, never the source of truth, so a drop is safe by construction.
    pub fn emit(&self, event: EngineEvent) -> bool {
        match &self.tx {
            None => false,
            Some(tx) => match tx.try_send(event) {
                Ok(()) => true,
                Err(TrySendError::Full(_)) => {
                    // The consumer is behind; drop (the ledger is authoritative).
                    tracing::trace!("engine event dropped: consumer buffer full");
                    false
                }
                Err(TrySendError::Closed(_)) => false,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn subscribe_delivers_events_in_order() {
        let (sink, mut rx) = EventSink::subscribe();
        assert!(sink.emit(EngineEvent::RunStarted {
            run_id: "r".into(),
            shards: 2,
        }));
        assert!(sink.emit(EngineEvent::StateAdvanced {
            record_id: "rec-1".into(),
            to: LifecycleState::Admitted,
        }));
        match rx.recv().await.unwrap() {
            EngineEvent::RunStarted { shards, .. } => assert_eq!(shards, 2),
            other => panic!("unexpected first event {other:?}"),
        }
        match rx.recv().await.unwrap() {
            EngineEvent::StateAdvanced { to, .. } => assert_eq!(to, LifecycleState::Admitted),
            other => panic!("unexpected second event {other:?}"),
        }
    }

    #[test]
    fn disconnected_sink_silently_drops() {
        let sink = EventSink::disconnected();
        // No consumer, no panic, no block — emit just returns false.
        assert!(!sink.emit(EngineEvent::RunFinished {
            run_id: "r".into(),
            completed: true,
        }));
    }

    #[tokio::test]
    async fn full_buffer_drops_rather_than_blocking() {
        // Capacity 1: the second emit cannot enqueue and must drop (return false), never block.
        let (sink, _rx) = EventSink::subscribe_with_capacity(1);
        assert!(sink.emit(EngineEvent::ShardFinished {
            run_id: "r".into(),
            shard: 0,
        }));
        // Buffer full now (receiver not drained); the next emit drops.
        assert!(!sink.emit(EngineEvent::ShardFinished {
            run_id: "r".into(),
            shard: 1,
        }));
    }

    #[tokio::test]
    async fn cloned_sinks_share_one_channel() {
        let (sink, mut rx) = EventSink::subscribe();
        let clone = sink.clone();
        assert!(clone.emit(EngineEvent::ShardStarted {
            run_id: "r".into(),
            shard: 3,
            resumed: true,
        }));
        match rx.recv().await.unwrap() {
            EngineEvent::ShardStarted { shard, resumed, .. } => {
                assert_eq!(shard, 3);
                assert!(resumed);
            }
            other => panic!("unexpected {other:?}"),
        }
    }
}
