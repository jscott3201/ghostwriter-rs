//! `gw-cli` — the `gw` command-line binary that wires the whole ghostwriter-rs stack.
//!
//! This is the INTEGRATION CAPSTONE of the orchestrator tier: the first place a REAL (non-fake)
//! engine [`Clients`](gw_engine::Clients) bundle is constructed from live providers, and the single
//! seam where the headless [`Engine`](gw_engine::Engine), the [`gw_tui`] dashboard, and the off-path
//! [`gw_eval`] diagnostics meet. The binary is a thin shell over this library so the clap tree, the
//! config loader, and the command handlers are all unit + integration testable.
//!
//! ## The command surface
//!
//! | command                   | kind   | what it does                                              |
//! |---------------------------|--------|-----------------------------------------------------------|
//! | `gw gen run`              | live   | headless engine run; print the [`RunReport`](gw_engine::RunReport) |
//! | `gw gen tui`             | live   | same run WITH the ratatui dashboard over the event stream |
//! | `gw gen export`          | pure   | admitted records → a Parquet shard (`gw_storage::export_parquet`) |
//! | `gw gen replay`          | live   | resume a run from its persisted shard checkpoints         |
//! | `gw eval audit-separation`| pure  | selector-vs-random separation diagnostic over a store     |
//! | `gw eval promote`        | pure   | variance-aware promotion gate over two `eval_results.json` |
//!
//! ## Security posture
//!
//! `OPENROUTER_API_KEY` is read ONLY from the process environment, by the provider constructor in
//! [`wire`]; it is never a config-file field, never a CLI flag, never serialized, and never logged
//! (the figment env layer is scoped to the `GW_` prefix, so it cannot even slurp the key by accident).
//! See [`config`] and [`wire`].

pub mod cli;
pub mod commands;
pub mod config;
mod embedder;
pub mod seedsource;
pub mod wire;

use clap::Parser;

use crate::cli::{Cli, Command, EvalCommand, GenCommand};

/// The process-level outcome of a successfully-run command.
///
/// Operational failures still return `Err`; this enum represents commands that ran to completion but
/// may have produced a gate decision intended for CI-style checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandOutcome {
    /// The command completed successfully, or a gate command rejected while `--check` was off.
    Success,
    /// The command ran successfully and an opt-in `--check` gate rejected.
    GateRejected,
}

/// Initialize `tracing-subscriber` from the `RUST_LOG` env filter (defaulting to `info`). Idempotent
/// across the process: a second call is a no-op (the global subscriber is set once). NEVER logs the
/// API key (it is not in scope of any traced value).
pub fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    // `try_init` returns Err if a subscriber is already set (e.g. a test installed one) — ignore it.
    let _ = fmt().with_env_filter(filter).try_init();
}

/// Parse the process arguments and dispatch to the matching command handler.
///
/// This is the single async entrypoint the binary's `main` awaits. Tracing is initialized first so
/// every downstream `tracing` event is captured.
///
/// # Errors
/// Returns the first error from the dispatched handler (config-load, store/provider construction,
/// engine run, export, eval, or I/O), as `anyhow::Error` (the binary boundary).
pub async fn run() -> anyhow::Result<CommandOutcome> {
    init_tracing();
    dispatch(Cli::parse()).await
}

/// Dispatch an already-parsed [`Cli`] to its handler — factored out of [`run`] so a test can drive a
/// constructed `Cli` without touching the process args (the LIVE handlers still need credentials /
/// network, so tests drive only the parse + the pure handlers).
///
/// # Errors
/// Propagates the dispatched handler's error.
pub async fn dispatch(cli: Cli) -> anyhow::Result<CommandOutcome> {
    match cli.command {
        Command::Gen(gen_cmd) => match gen_cmd {
            GenCommand::Run(args) => commands::run::run(args)
                .await
                .map(|()| CommandOutcome::Success),
            GenCommand::Tui(args) => commands::tui::tui(args)
                .await
                .map(|()| CommandOutcome::Success),
            GenCommand::Export(args) => commands::export::export(args)
                .await
                .map(|()| CommandOutcome::Success),
            GenCommand::Replay(args) => commands::replay::replay(args)
                .await
                .map(|()| CommandOutcome::Success),
        },
        Command::Eval(eval_cmd) => match eval_cmd {
            EvalCommand::AuditSeparation(args) => commands::eval::audit_separation(args).await,
            EvalCommand::Promote(args) => commands::eval::promote_cmd(args).await,
        },
    }
}
