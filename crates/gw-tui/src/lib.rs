//! `gw-tui` — the ratatui terminal UI for ghostwriter-rs (a graded-CoT training-data harness).
//!
//! This is a LIBRARY crate; the `gw` binary lives in `gw-cli`, which calls [`run`]. The TUI consumes
//! the engine ONLY through its event interface — it depends on `gw-engine` for the
//! [`EngineEvent`](gw_engine::EngineEvent) type and on `gw-schema` for
//! [`LifecycleState`](gw_schema::LifecycleState), and reaches into NO engine internals. It does NOT call
//! `Engine::run`; the caller (gw-cli) spawns the engine run and hands this crate the receiver half of
//! the event stream plus a shared [`CancellationToken`](tokio_util::sync::CancellationToken).
//!
//! ## v1 scope: a LIFECYCLE dashboard
//!
//! The engine's [`EngineEvent`](gw_engine::EngineEvent) stream is lifecycle-level (state transitions,
//! cost charges, errors, run/shard boundaries) — it emits NO per-token reasoning/answer deltas (the
//! engine sources those separately). So v1 is a dashboard driven by that stream:
//!
//! - a run/shard progress **header** (run id, shards started/finished, resumed flags, a
//!   completed-vs-halted banner at `RunFinished`);
//! - a per-record lifecycle **table** (`record_id` → current [`LifecycleState`](gw_schema::LifecycleState),
//!   update count), with selection/scroll state held in the app model;
//! - live **counters + gauges**: per-terminal-state counts, admit-rate, and a cost gauge + spend
//!   sparkline (from `CostCharged`/`BudgetReached`);
//! - an errors/**log pane** listing recent `RecordErrored` messages + a budget banner.
//!
//! The future per-token trace viewer (spec §2.3) is a documented EXTENSION POINT, not implemented in
//! v1 — see [`Action::ReasoningDelta`]/[`Action::AnswerDelta`].
//!
//! ## Architecture (ARCHITECTURE §1.4, §2, `_research/06-rust-ratatui-architecture.md`)
//!
//! Component + Action: [`Action`] is the single internal message type; [`App::update`] is a PURE
//! transition over the model (no terminal, no I/O), so the whole dashboard is unit-testable. The view
//! ([`view()`](crate::view::view)) is a pure function of the model into a `Frame`. A single
//! `tokio::select!` loop ([`run`]) multiplexes the crossterm `EventStream`, a tick interval, a render
//! interval, and the engine event receiver; `terminal.draw` is the SOLE owner of the `Frame`. A panic
//! hook restores the terminal; a [`CancellationToken`](tokio_util::sync::CancellationToken) drives clean
//! shutdown.
//!
//! ## Tolerating dropped events
//!
//! The engine event channel is BOUNDED + drop-on-full (events are observability; the SQLite ledger is
//! truth). The model is written for this: counters DERIVE from a last-writer-wins per-record state map
//! (never running increments), and the cost meter is monotone, so a dropped event can leave a row
//! momentarily stale but can never corrupt a count or rewind the gauge.
//!
//! ## Example
//!
//! ```no_run
//! use gw_engine::{Engine, EventSink};
//! use gw_tui::{run, DEFAULT_TICK_RATE, DEFAULT_FRAME_RATE};
//! use tokio_util::sync::CancellationToken;
//!
//! # async fn demo(
//! #     engine: Engine,
//! #     source: gw_engine::InMemorySeedSource,
//! # ) -> Result<(), Box<dyn std::error::Error>> {
//! let (sink, rx) = EventSink::subscribe();
//! let cancel = CancellationToken::new();
//!
//! // The CALLER spawns the engine run; the TUI only owns the terminal + consumes the event stream.
//! let engine_cancel = cancel.clone();
//! let handle = tokio::spawn(async move {
//!     // `engine` is constructed with `sink` wired into its `Clients` by the caller.
//!     engine.run("run-1", &source, engine_cancel).await
//! });
//!
//! run(rx, cancel, DEFAULT_TICK_RATE, DEFAULT_FRAME_RATE).await?;
//! let _report = handle.await??;
//! # Ok(())
//! # }
//! ```

mod action;
mod error;
mod event_loop;
mod model;
mod view;

pub use action::Action;
pub use error::{Result, TuiError};
pub use event_loop::{DEFAULT_FRAME_RATE, DEFAULT_TICK_RATE, run};
pub use model::{App, CostMeter, RecordRow, RunHeader};

// Re-exported for callers that drive the model directly (tests, embedders) without the I/O loop.
pub use view::view;
