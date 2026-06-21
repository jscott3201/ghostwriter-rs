//! The `gen replay` handler — resume a previously-started run from its persisted shard checkpoints.
//!
//! This is THIN: `Engine::run` is already idempotent and crash-resuming (running the SAME `run_id`
//! over the SAME [`SeedSource`](gw_engine::SeedSource) skips committed offsets and re-drives any
//! mid-flight record from its last persisted state — the teacher is never re-spent). So replay just
//! re-derives the identical seed plan from the SAME prompts file + shard count and re-enters
//! `Engine::run`. It is headless (no dashboard, no events). Like `gen run` it is a LIVE path and is
//! not network-tested.

use anyhow::Context;

use gw_engine::EventSink;

use crate::cli::ReplayArgs;
use crate::config::Config;
use crate::seedsource::FileSeedSource;
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
    config.validate_run_control()?;
    let source = FileSeedSource::from_prompts_file(&args.prompts, args.shards)?;

    let (engine, _store) = build_engine(&config, EventSink::disconnected(), args.max_in_flight)
        .await
        .context("building the engine")?;

    let cancel = new_cancel_token();
    let report = engine
        .run(&args.run_id, &source, cancel)
        .await
        .context("resuming the engine run")?;

    println!(
        "resumed run {} {} — admitted {}, rejected {}, errored {}",
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
