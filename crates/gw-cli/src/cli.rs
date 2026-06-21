//! The clap-derive command tree for the `gw` binary.
//!
//! Two top-level groups mirror the crate-dependency split: `gen` drives the engine (the LIVE
//! provider-spending paths plus the PURE `export`), and `eval` drives the off-path, model-free
//! diagnostics ([`gw_eval`]). Every leaf carries only the flags that command needs; the shared
//! `--config` / `--db` / `--run-id` knobs that LAYER over the config file are hoisted onto the
//! relevant subcommands so a single figment merge (file → env → these flags) produces the effective
//! [`Config`](crate::config::Config).
//!
//! The API key is DELIBERATELY ABSENT here: `OPENROUTER_API_KEY` is read from the environment by the
//! provider constructor ([`crate::wire`]) and is NEVER a CLI flag (so it can never land in a shell
//! history, a process listing, or a `--help` dump).

use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// ghostwriter-rs: generate graded chain-of-thought reasoning traces as fine-tuning data.
#[derive(Debug, Parser, PartialEq)]
#[command(name = "gw", version, about)]
pub struct Cli {
    /// The top-level command group.
    #[command(subcommand)]
    pub command: Command,
}

/// The top-level command groups.
#[derive(Debug, Subcommand, PartialEq)]
pub enum Command {
    /// Generation: drive the engine over a seed space (run / tui / export / replay).
    #[command(subcommand)]
    Gen(GenCommand),
    /// Evaluation: off-path, model-free diagnostics over persisted records / eval artifacts.
    #[command(subcommand)]
    Eval(EvalCommand),
}

/// The `gen` subcommands.
#[derive(Debug, Subcommand, PartialEq)]
pub enum GenCommand {
    /// Headless engine run (no terminal UI): generate → grade → admit → persist, print the report.
    Run(RunArgs),
    /// Engine run WITH the live ratatui dashboard consuming the engine event stream.
    Tui(RunArgs),
    /// Export admitted records from a store to a Parquet shard (pure: no providers).
    Export(ExportArgs),
    /// Resume a previously-started run from its persisted shard checkpoints (thin re-entry).
    Replay(ReplayArgs),
}

/// The `eval` subcommands.
#[derive(Debug, Subcommand, PartialEq)]
pub enum EvalCommand {
    /// Selector-vs-random separation diagnostic over a store; prints the report as JSON.
    AuditSeparation(AuditSeparationArgs),
    /// Variance-aware promotion gate over two `eval_results.json` files; prints the report as JSON.
    Promote(PromoteArgs),
}

/// Shared knobs for `gen run` / `gen tui` (the engine-spending paths). The config FILE supplies the
/// area/provider/budget defaults; these flags LAYER over it (highest precedence after env).
#[derive(Debug, clap::Args, PartialEq)]
pub struct RunArgs {
    /// Path to the TOML config file (figment base layer). Absent ⇒ defaults + env only.
    #[arg(long, value_name = "FILE")]
    pub config: Option<PathBuf>,
    /// Override the SQLite store path (else the config's `db` / the default).
    #[arg(long, value_name = "PATH")]
    pub db: Option<PathBuf>,
    /// The run id (idempotency key). A re-run of the SAME id over the same seeds RESUMES.
    #[arg(long, value_name = "ID")]
    pub run_id: String,
    /// Path to a newline-delimited prompts file (one user turn per line) — the v1 seed source.
    #[arg(long, value_name = "FILE")]
    pub prompts: PathBuf,
    /// Number of shards to partition the seed space into.
    #[arg(long, value_name = "N", default_value_t = 1)]
    pub shards: usize,
    /// Override the run-wide budget cap in USD (else the config's `budget_usd`).
    #[arg(long, value_name = "USD")]
    pub budget_usd: Option<f64>,
    /// Override the per-area best-of-k fan-out (else the config's `area.k`).
    #[arg(long, value_name = "K")]
    pub k: Option<u32>,
    /// Max seed items in flight across all shards (concurrency cap).
    #[arg(long, value_name = "N", default_value_t = 4)]
    pub max_in_flight: u32,
}

/// `gen export` flags (pure path: a store read + a Parquet write, no providers).
#[derive(Debug, clap::Args, PartialEq)]
pub struct ExportArgs {
    /// Path to the SQLite store to export from.
    #[arg(long, value_name = "PATH")]
    pub db: PathBuf,
    /// Destination Parquet file path.
    #[arg(long, value_name = "FILE")]
    pub out: PathBuf,
    /// Restrict the export to a single run id (else every admitted record in the store).
    #[arg(long, value_name = "ID")]
    pub run_id: Option<String>,
    /// The TRL export target template.
    #[arg(long, value_enum, default_value_t = ExportFormat::ChatMl)]
    pub format: ExportFormat,
    /// Whether reasoning enters the supervised loss region on export.
    #[arg(long, value_enum, default_value_t = ExportCot::Supervised)]
    pub cot: ExportCot,
}

/// `gen replay` flags (thin: re-enter `Engine::run` for an existing run id + store).
///
/// Resume is sound only when the SAME seed plan is re-derived, so the SAME `--prompts` file (and
/// `--shards`) the original run used MUST be supplied — the engine then skips already-committed
/// offsets and re-drives any mid-flight record from its last persisted state (the teacher is never
/// re-spent).
#[derive(Debug, clap::Args, PartialEq)]
pub struct ReplayArgs {
    /// Path to the TOML config file (must describe the same area/provider as the original run).
    #[arg(long, value_name = "FILE")]
    pub config: Option<PathBuf>,
    /// Override the SQLite store path (else the config's `db`).
    #[arg(long, value_name = "PATH")]
    pub db: Option<PathBuf>,
    /// The run id to resume (must already exist in the store with persisted checkpoints).
    #[arg(long, value_name = "ID")]
    pub run_id: String,
    /// The SAME newline-delimited prompts file the original run used (resume re-derives by offset).
    #[arg(long, value_name = "FILE")]
    pub prompts: PathBuf,
    /// The SAME shard count the original run used (REQUIRED — there is no safe default for a resume:
    /// the seed→shard partition is `index % shards`, so a different (or silently defaulted) value
    /// re-partitions the space and duplicates/orphans records against the persisted offset cursors).
    #[arg(long, value_name = "N")]
    pub shards: usize,
    /// Max seed items in flight across all shards (concurrency cap).
    #[arg(long, value_name = "N", default_value_t = 4)]
    pub max_in_flight: u32,
}

/// `eval audit-separation` flags (pure: a store scan + closed-form diagnostic).
#[derive(Debug, clap::Args, PartialEq)]
pub struct AuditSeparationArgs {
    /// Path to the SQLite store to analyze.
    #[arg(long, value_name = "PATH")]
    pub db: PathBuf,
    /// Restrict the analysis to a single run id (else the whole store).
    #[arg(long, value_name = "ID")]
    pub run_id: Option<String>,
    /// Minimum decidable (mixed) groups before the selector signal is trusted.
    #[arg(long, value_name = "N")]
    pub min_decidable_groups: Option<usize>,
    /// Floor on `decidable_fraction` below which the corpus is treated as a data ceiling.
    #[arg(long, value_name = "F")]
    pub min_decidable_fraction: Option<f64>,
}

/// `eval promote` flags (pure: two JSON artifacts + a drift exit code → a binary decision).
#[derive(Debug, clap::Args, PartialEq)]
pub struct PromoteArgs {
    /// Path to the BASELINE `eval_results.json`.
    #[arg(long, value_name = "FILE")]
    pub baseline: PathBuf,
    /// Path to the CANDIDATE `eval_results.json`.
    #[arg(long, value_name = "FILE")]
    pub candidate: PathBuf,
    /// The capability-drift probe exit code (`0` == clean). A non-zero code hard-rejects.
    #[arg(long, value_name = "N", default_value_t = 0)]
    pub drift_exit: i32,
    /// Optional path to a TOML `[promote]` config (else the A3 defaults).
    #[arg(long, value_name = "FILE")]
    pub config: Option<PathBuf>,
}

/// The TRL export target, mapped to [`gw_schema::TrlFormat`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ExportFormat {
    /// Gemma-4 byte-exact (pinned chat template).
    Gemma4,
    /// ChatML.
    ChatMl,
    /// ShareGPT.
    Sharegpt,
    /// OpenAI `messages` conversational.
    OpenaiMessages,
    /// gpt-oss Harmony channels.
    Harmony,
    /// TRL prompt/completion.
    TrlPromptCompletion,
}

impl From<ExportFormat> for gw_schema::TrlFormat {
    fn from(f: ExportFormat) -> Self {
        match f {
            ExportFormat::Gemma4 => Self::Gemma4,
            ExportFormat::ChatMl => Self::ChatML,
            ExportFormat::Sharegpt => Self::ShareGpt,
            ExportFormat::OpenaiMessages => Self::OpenAiMessages,
            ExportFormat::Harmony => Self::Harmony,
            ExportFormat::TrlPromptCompletion => Self::TrlPromptCompletion,
        }
    }
}

/// The export CoT policy, mapped to [`gw_schema::CotPolicy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ExportCot {
    /// Render reasoning into the supervised (loss) region.
    Supervised,
    /// Render reasoning but mask it out of loss.
    Masked,
    /// Drop reasoning entirely (answer-only).
    Stripped,
}

impl From<ExportCot> for gw_schema::CotPolicy {
    fn from(c: ExportCot) -> Self {
        match c {
            ExportCot::Supervised => Self::Supervised,
            ExportCot::Masked => Self::Masked,
            ExportCot::Stripped => Self::Stripped,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Cli::command()` must satisfy clap's internal invariants (arg names, conflicts, value parsers).
    #[test]
    fn cli_definition_is_valid() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }
}
