//! The `eval` handlers: `audit-separation` and `promote` (both PURE — model-free, no network).
//!
//! `audit-separation` scans a [`Store`] (optionally filtered to one run) and runs the closed-form
//! [`separation::analyze_store`] diagnostic; `promote` parses two `eval_results.json` artifacts plus a
//! drift exit code and runs the variance-aware [`promote`] gate. Both print their report as
//! pretty JSON to stdout — the machine-readable artifact a CI gate consumes.

use anyhow::Context;

use gw_eval::{
    PromoteConfig, SeparationConfig, promote::EvalResults, promote::promote, separation,
};
use gw_storage::{RecordFilter, Store};

use crate::CommandOutcome;
use crate::cli::{AuditSeparationArgs, PromoteArgs};

/// Run the selector-vs-random separation diagnostic over the store at `args.db`, optionally filtered
/// to `args.run_id`, and print the [`SeparationReport`](gw_eval::SeparationReport) as JSON.
///
/// # Errors
/// Propagates a store-open / scan failure or a JSON serialization failure.
pub async fn audit_separation(args: AuditSeparationArgs) -> anyhow::Result<CommandOutcome> {
    let store = Store::open(&args.db)
        .await
        .with_context(|| format!("opening the store at {}", args.db.display()))?;

    let mut filter = RecordFilter::new();
    if let Some(run_id) = &args.run_id {
        filter = filter.run_id(run_id);
    }

    let mut cfg = SeparationConfig::default();
    if let Some(n) = args.min_decidable_groups {
        cfg.min_decidable_groups = n;
    }
    if let Some(f) = args.min_decidable_fraction {
        cfg.min_decidable_fraction = f;
    }

    let report = separation::analyze_store(&store, &filter, &cfg)
        .await
        .context("running the separation diagnostic")?;
    let json =
        serde_json::to_string_pretty(&report).context("serializing the separation report")?;
    println!("{json}");
    if args.check && !report.passed() {
        Ok(CommandOutcome::GateRejected)
    } else {
        Ok(CommandOutcome::Success)
    }
}

/// Run the variance-aware promotion gate over the two `eval_results.json` files in `args` and the
/// drift exit code, and print the [`PromotionReport`](gw_eval::PromotionReport) as JSON.
///
/// The promote config is the A3 default unless `args.config` points at a TOML file with a `[promote]`
/// (or top-level) table overriding the knobs (`ab_metric`, `ab_min_delta`, `ab_sigma_k`, ...).
///
/// # Errors
/// Propagates a file-read, JSON-parse, config-parse, or serialization failure.
pub async fn promote_cmd(args: PromoteArgs) -> anyhow::Result<CommandOutcome> {
    let baseline_bytes = std::fs::read(&args.baseline)
        .with_context(|| format!("reading baseline {}", args.baseline.display()))?;
    let candidate_bytes = std::fs::read(&args.candidate)
        .with_context(|| format!("reading candidate {}", args.candidate.display()))?;

    let baseline = EvalResults::from_json(&baseline_bytes)
        .with_context(|| format!("parsing baseline {}", args.baseline.display()))?;
    let candidate = EvalResults::from_json(&candidate_bytes)
        .with_context(|| format!("parsing candidate {}", args.candidate.display()))?;

    let cfg = load_promote_config(args.config.as_deref())?;

    let report = promote(&baseline, &candidate, args.drift_exit, &cfg);
    let json = serde_json::to_string_pretty(&report).context("serializing the promotion report")?;
    println!("{json}");
    if args.check && !report.promote {
        Ok(CommandOutcome::GateRejected)
    } else {
        Ok(CommandOutcome::Success)
    }
}

/// Load a [`PromoteConfig`] from an optional TOML file (a `[promote]` table, falling back to the
/// top-level table), layered over the A3 defaults. Absent ⇒ the defaults.
fn load_promote_config(file: Option<&std::path::Path>) -> anyhow::Result<PromoteConfig> {
    use figment::{
        Figment,
        providers::{Format, Serialized, Toml},
    };
    let Some(path) = file else {
        return Ok(PromoteConfig::default());
    };
    let fig = Figment::from(Serialized::defaults(PromoteConfig::default()))
        // Prefer a nested `[promote]` table; an absent key is a no-op merge, so a flat file works too.
        .merge(Toml::file(path).nested().profile("promote"))
        .merge(Toml::file(path));
    fig.extract()
        .with_context(|| format!("parsing promote config {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promote_config_defaults_when_no_file() {
        let cfg = load_promote_config(None).expect("defaults");
        assert_eq!(cfg, PromoteConfig::default());
        assert_eq!(cfg.ab_metric, "eval_results.aggregate");
    }
}
