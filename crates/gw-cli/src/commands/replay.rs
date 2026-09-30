//! The `gen replay` handler — resume a previously-started run from its persisted shard checkpoints.
//!
//! This is THIN: `Engine::run` is already idempotent and crash-resuming (running the SAME `run_id`
//! over the SAME [`SeedSource`](gw_engine::SeedSource) skips committed offsets and re-drives any
//! mid-flight record from its last persisted state, reusing persisted teacher output). So replay just
//! re-derives the identical seed plan from the SAME task/prompt file + shard count and re-enters
//! `Engine::run`. It is headless (no dashboard, no events). Like `gen run` it is a LIVE path and is
//! not network-tested.

use anyhow::Context;

use gw_engine::EventSink;

use crate::cli::ReplayArgs;
use crate::config::Config;
use crate::seedsource::InputSeedSource;
use crate::wire::{build_engine, new_cancel_token};

/// Resume `args.run_id` over the same seed plan, persisting onward, and print the terminal report.
///
/// # Errors
/// Propagates a config-load, seed-load, store-open, provider-construction, or engine-run failure.
pub async fn replay(args: ReplayArgs) -> anyhow::Result<()> {
    let mut config = Config::load(args.config.as_deref()).context("loading config")?;
    if let Some(db) = &args.db {
        config.db = db.clone();
    }
    if let Some(intent) = args.admission_intent {
        config.area.admission_intent = intent.into();
    }
    args.accounting.apply(&mut config)?;
    let source =
        InputSeedSource::from_files(args.prompts.as_deref(), args.tasks.as_deref(), args.shards)?;

    let mode = gw_storage::RunMode::Replay;
    let (engine, store, prepared) = build_engine(
        &config,
        EventSink::disconnected(),
        args.max_in_flight,
        &args.run_id,
        &source,
        mode,
    )
    .await
    .context("building the engine")?;

    let cancel = new_cancel_token();
    let result = engine
        .run_prepared(&args.run_id, prepared, mode, cancel)
        .await;
    super::accounting::terminal(
        &store,
        &args.run_id,
        &config.effective_policy(),
        result.as_ref().ok(),
    )
    .await;
    result.context("resuming the engine run")?;
    Ok(())
}
