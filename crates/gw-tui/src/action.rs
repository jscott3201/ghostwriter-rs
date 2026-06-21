//! The internal `Action` message type — the TUI's analogue of the engine's [`EngineEvent`].
//!
//! Per the Component+Action pattern (ARCHITECTURE §1.4, §2.2, `_research/06-rust-ratatui-architecture.md`),
//! `Action` is the SINGLE message type that flows through the model. Two sources feed it, and both are
//! mapped onto `Action` BEFORE they touch the model so that [`crate::App::update`] stays pure and
//! terminal-free:
//!
//! 1. engine [`EngineEvent`]s, via [`Action::from_engine_event`] (lifecycle dashboard data);
//! 2. terminal key/resize events + the tick/render timers, via the event loop ([`crate::event_loop`]).
//!
//! ## Deferred: the per-token trace viewer
//!
//! v1 is a LIFECYCLE dashboard. The engine's [`EngineEvent`] stream is lifecycle-level (state
//! transitions, cost charges, errors, run/shard boundaries) and emits NO per-token reasoning/answer
//! deltas — `gw-engine::event` states the TUI "sources that separately". The spec (§2.3) describes a
//! future side-by-side reasoning⟷answer live viewer; its `Action` variants
//! ([`Action::ReasoningDelta`]/[`Action::AnswerDelta`]) are reserved here as the documented extension
//! point but are NOT produced or consumed in v1.

use gw_engine::EngineEvent;
use gw_schema::LifecycleState;

/// The internal message type driving the model. Every engine event and every key/timer event is
/// translated into one of these before [`crate::App::update`] applies it.
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    /// A run started with `shards` planned shards.
    RunStarted {
        /// The run id.
        run_id: String,
        /// The number of planned shards.
        shards: usize,
    },
    /// A shard began (fresh, or `resumed` from a checkpoint).
    ShardStarted {
        /// The 0-based shard index.
        shard: i64,
        /// `true` if the shard resumed from a persisted checkpoint.
        resumed: bool,
    },
    /// A record advanced to a new lifecycle state — the load-bearing dashboard event.
    StateAdvanced {
        /// The record id.
        record_id: String,
        /// The state the record advanced TO.
        to: LifecycleState,
    },
    /// A teacher/judge call's marginal cost was charged; `run_total_usd` is the cumulative spend.
    CostCharged {
        /// The cumulative USD spent for the run after this charge.
        run_total_usd: f64,
    },
    /// The budget cap was reached; no new teacher work will dispatch.
    BudgetReached {
        /// The cumulative USD spent.
        spent: f64,
        /// The configured cap.
        cap: f64,
    },
    /// A record failed unrecoverably and was parked at [`LifecycleState::Error`].
    RecordErrored {
        /// The record id.
        record_id: String,
        /// A human-readable error message (never carries a secret, per the engine contract).
        error: String,
    },
    /// A shard finished.
    ShardFinished {
        /// The 0-based shard index.
        shard: i64,
    },
    /// The run finished; `completed` is `true` only if every shard drained cleanly.
    RunFinished {
        /// `true` if the run drained cleanly; `false` if it halted on budget/cancellation.
        completed: bool,
    },

    // --- terminal / timer actions ---
    /// Move the table selection up one row.
    SelectUp,
    /// Move the table selection down one row.
    SelectDown,
    /// Jump the table selection to the first row.
    SelectFirst,
    /// Jump the table selection to the last row.
    SelectLast,
    /// A periodic tick: advance animated/derived UI state (gauges, sparkline samples).
    Tick,
    /// The terminal was resized.
    Resize {
        /// New terminal width in columns.
        width: u16,
        /// New terminal height in rows.
        height: u16,
    },
    /// The user requested shutdown ('q' / Ctrl-C); the loop tears down.
    Quit,

    // --- DEFERRED extension point: per-token trace viewer (NOT produced/consumed in v1) ---
    /// RESERVED (v2): a chain-of-thought token delta for the future per-token trace viewer. The engine
    /// does not emit per-token deltas today; this variant marks the seam without implementing it.
    ReasoningDelta {
        /// The record id the delta belongs to.
        record_id: String,
        /// The reasoning-token text.
        text: String,
    },
    /// RESERVED (v2): a final-answer token delta for the future per-token trace viewer. Not produced in
    /// v1 (see [`Action::ReasoningDelta`]).
    AnswerDelta {
        /// The record id the delta belongs to.
        record_id: String,
        /// The answer-token text.
        text: String,
    },
}

impl Action {
    /// Map a lifecycle-level [`EngineEvent`] onto an [`Action`]. This is the engine⟷tui seam: the model
    /// only ever sees `Action`s, so it stays decoupled from the engine's event type. Run/shard ids that
    /// the model tracks once (in its header) are dropped here where redundant.
    #[must_use]
    pub fn from_engine_event(event: EngineEvent) -> Self {
        match event {
            EngineEvent::RunStarted { run_id, shards } => Action::RunStarted { run_id, shards },
            EngineEvent::ShardStarted { shard, resumed, .. } => {
                Action::ShardStarted { shard, resumed }
            }
            EngineEvent::StateAdvanced { record_id, to } => Action::StateAdvanced { record_id, to },
            EngineEvent::CostCharged { run_total_usd, .. } => Action::CostCharged { run_total_usd },
            EngineEvent::BudgetReached { spent, cap, .. } => Action::BudgetReached { spent, cap },
            EngineEvent::RecordErrored { record_id, error } => {
                Action::RecordErrored { record_id, error }
            }
            EngineEvent::ShardFinished { shard, .. } => Action::ShardFinished { shard },
            EngineEvent::RunFinished { completed, .. } => Action::RunFinished { completed },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_every_engine_event_variant() {
        let cases = [
            (
                EngineEvent::RunStarted {
                    run_id: "r".into(),
                    shards: 3,
                },
                Action::RunStarted {
                    run_id: "r".into(),
                    shards: 3,
                },
            ),
            (
                EngineEvent::ShardStarted {
                    run_id: "r".into(),
                    shard: 1,
                    resumed: true,
                },
                Action::ShardStarted {
                    shard: 1,
                    resumed: true,
                },
            ),
            (
                EngineEvent::StateAdvanced {
                    record_id: "rec".into(),
                    to: LifecycleState::Admitted,
                },
                Action::StateAdvanced {
                    record_id: "rec".into(),
                    to: LifecycleState::Admitted,
                },
            ),
            (
                EngineEvent::CostCharged {
                    record_id: "rec".into(),
                    usd: 0.1,
                    run_total_usd: 1.5,
                },
                Action::CostCharged { run_total_usd: 1.5 },
            ),
            (
                EngineEvent::BudgetReached {
                    run_id: "r".into(),
                    spent: 10.0,
                    cap: 10.0,
                },
                Action::BudgetReached {
                    spent: 10.0,
                    cap: 10.0,
                },
            ),
            (
                EngineEvent::RecordErrored {
                    record_id: "rec".into(),
                    error: "boom".into(),
                },
                Action::RecordErrored {
                    record_id: "rec".into(),
                    error: "boom".into(),
                },
            ),
            (
                EngineEvent::ShardFinished {
                    run_id: "r".into(),
                    shard: 2,
                },
                Action::ShardFinished { shard: 2 },
            ),
            (
                EngineEvent::RunFinished {
                    run_id: "r".into(),
                    completed: false,
                },
                Action::RunFinished { completed: false },
            ),
        ];
        for (event, expected) in cases {
            assert_eq!(Action::from_engine_event(event), expected);
        }
    }
}
