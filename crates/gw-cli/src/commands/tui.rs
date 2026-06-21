//! The `gen tui` handler — the SAME engine run as `gen run`, WITH the live ratatui dashboard.
//!
//! This is the engine⟷tui seam (ARCHITECTURE §6, `06-rust-ratatui-architecture.md` §6): the engine is
//! HEADLESS and the TUI is an interchangeable CONSUMER of its [`EngineEvent`](gw_engine::EngineEvent)
//! stream. The handler:
//!
//! 1. creates a connected sink + receiver via [`EventSink::subscribe`](gw_engine::EventSink);
//! 2. builds the engine with the SINK wired into its `Clients`;
//! 3. SPAWNS the engine run on a task, sharing ONE [`CancellationToken`](tokio_util::sync::CancellationToken)
//!    with the dashboard;
//! 4. runs [`gw_tui::run`] on the current task, draining the receiver — when the user quits it fires
//!    the shared token, which winds down the engine task; when the engine finishes and the sink drops,
//!    the channel closes and the dashboard loop exits.
//!
//! Like `gen run` this is a LIVE provider-spending path — NOT exercised against the network in tests
//! (the wiring is asserted structurally in [`crate::wire`]).

use anyhow::Context;
use std::time::Duration;

use gw_engine::EventSink;

use crate::cli::RunArgs;
use crate::commands::run::effective_config;
use crate::seedsource::FileSeedSource;
use crate::wire::{build_engine, new_cancel_token};

/// Run the engine over the prompts in `args.prompts` WITH the dashboard, sharing one cancellation
/// token between the engine task and the TUI loop.
///
/// # Errors
/// Propagates a config-load, seed-load, store-open, provider-construction, TUI-I/O, or engine-run
/// failure. The engine task's result is awaited AFTER the dashboard exits, so a run error surfaces
/// here (not silently swallowed by the spawned task).
pub async fn tui(args: RunArgs) -> anyhow::Result<()> {
    let config = effective_config(&args)?;
    let source = FileSeedSource::from_prompts_file(&args.prompts, args.shards)?;

    // The connected sink (into the engine's Clients) + its receiver (drained by the dashboard).
    let (sink, rx) = EventSink::subscribe();
    let (engine, _store) = build_engine(&config, sink, args.max_in_flight)
        .await
        .context("building the engine")?;

    // ONE token shared between the dashboard and the engine task (clean mutual shutdown).
    let cancel = new_cancel_token();
    let engine_cancel = cancel.clone();
    let run_id = args.run_id.clone();
    let handle = tokio::spawn(async move { engine.run(&run_id, &source, engine_cancel).await });

    let tick_rate = Duration::from_millis(config.tick_ms);
    let frame_rate = Duration::from_millis(config.frame_ms);
    // The dashboard owns the terminal for its lifetime; on quit it fires `cancel` (winding down the
    // engine task), and when the engine finishes + drops the sink the channel closes and this returns.
    let tui_result = gw_tui::run(rx, cancel, tick_rate, frame_rate)
        .await
        .context("running the TUI dashboard");

    // Await the engine task so its RunReport / error is not lost (joining regardless of how the
    // dashboard exited — a clean quit fired `cancel`, which winds the engine task down).
    let report = handle.await.context("joining the engine task")?;

    // Surface a TUI error first (it owns the terminal); otherwise surface any engine error, then print.
    tui_result?;
    let report = report.context("running the engine")?;
    println!(
        "run {} {} — admitted {}, rejected {}, errored {}",
        args.run_id,
        if report.completed {
            "completed"
        } else {
            "halted"
        },
        report.admitted,
        report.rejected,
        report.errored,
    );
    Ok(())
}
