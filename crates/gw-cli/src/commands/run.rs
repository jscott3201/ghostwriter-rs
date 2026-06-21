//! The `gen run` handler — a HEADLESS engine run (no terminal): construct the live clients, drive the
//! engine over a file-backed seed source, and print the [`RunReport`](gw_engine::RunReport).
//!
//! This is a LIVE provider-spending path (the teacher + judge rails call OpenRouter), so it is NOT
//! exercised against the network in tests — its non-network wiring (config → `AreaConfig`, the
//! `Clients` field types, the env-sourced key) is asserted in [`crate::wire`] + [`crate::config`]. A
//! headless run consumes NO events: it wires [`EventSink::disconnected()`](gw_engine::EventSink), so
//! every engine `emit` is a silent no-op (the SQLite ledger is the authoritative record).

use anyhow::Context;

use gw_engine::EventSink;

use crate::cli::RunArgs;
use crate::config::Config;
use crate::seedsource::FileSeedSource;
use crate::wire::{build_engine, new_cancel_token};

/// Resolve the effective [`Config`] for a run: load the file + env layers, then apply the clap
/// overrides (`--db`, `--budget-usd`, `--k`) as the highest-precedence layer.
///
/// # Errors
/// Propagates a config-load (malformed TOML / bad value) failure.
pub fn effective_config(args: &RunArgs) -> anyhow::Result<Config> {
    let mut config = Config::load(args.config.as_deref()).context("loading config")?;
    if let Some(db) = &args.db {
        config.db = db.clone();
    }
    if let Some(budget) = args.budget_usd {
        config.budget_usd = budget;
    }
    if let Some(k) = args.k {
        config.area.k = k;
    }
    Ok(config)
}

/// Run the engine headlessly over the prompts in `args.prompts`, persisting to the store, and print
/// the terminal report.
///
/// # Errors
/// Propagates a config-load, seed-load, store-open, provider-construction (missing
/// `OPENROUTER_API_KEY`), or engine-run failure.
pub async fn run(args: RunArgs) -> anyhow::Result<()> {
    let config = effective_config(&args)?;
    let source = FileSeedSource::from_prompts_file(&args.prompts, args.shards)?;

    let (engine, _store) = build_engine(&config, EventSink::disconnected(), args.max_in_flight)
        .await
        .context("building the engine")?;

    let cancel = new_cancel_token();
    let report = engine
        .run(&args.run_id, &source, cancel)
        .await
        .context("running the engine")?;

    print_report(&args.run_id, &report);
    Ok(())
}

/// Print a one-block human summary of the run's terminal [`RunReport`](gw_engine::RunReport).
fn print_report(run_id: &str, report: &gw_engine::RunReport) {
    println!(
        "run {run_id} {}",
        if report.completed {
            "completed"
        } else {
            "halted (budget/cancel)"
        }
    );
    println!("  admitted    {}", report.admitted);
    println!("  exported    {}", report.exported);
    println!("  rejected    {}", report.rejected);
    println!("  needs_review {}", report.needs_review);
    println!("  revising    {}", report.revising);
    println!("  errored     {}", report.errored);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn run_args() -> RunArgs {
        RunArgs {
            config: None,
            db: Some(PathBuf::from("/tmp/override.sqlite")),
            run_id: "r1".into(),
            prompts: PathBuf::from("/tmp/prompts.txt"),
            shards: 1,
            budget_usd: Some(9.5),
            k: Some(4),
            max_in_flight: 4,
        }
    }

    #[test]
    fn clap_overrides_win_over_config_defaults() {
        let cfg = effective_config(&run_args()).expect("config");
        // The clap flags layer over the defaults (no file, no env).
        assert_eq!(cfg.db, PathBuf::from("/tmp/override.sqlite"));
        assert!((cfg.budget_usd - 9.5).abs() < 1e-12);
        assert_eq!(cfg.area.k, 4);
    }

    #[test]
    fn absent_overrides_keep_config_values() {
        let mut args = run_args();
        args.db = None;
        args.budget_usd = None;
        args.k = None;
        let cfg = effective_config(&args).expect("config");
        // Falls back to the built-in defaults when no override and no file.
        assert_eq!(cfg.db, Config::default().db);
        assert!((cfg.budget_usd - Config::default().budget_usd).abs() < 1e-12);
        assert_eq!(cfg.area.k, Config::default().area.k);
    }
}
