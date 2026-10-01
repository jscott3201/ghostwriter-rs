//! The clap-derive command tree for the `gw` binary.
//!
//! `gen` drives the engine (live provider paths plus pure export), `eval` drives model-free
//! diagnostics ([`gw_eval`]), and `artifact` verifies immutable provider-free snapshots.
//! Every leaf carries only the flags that command needs; the shared
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
    /// Verify a self-contained immutable Parquet snapshot (no providers or database).
    #[command(subcommand)]
    Artifact(ArtifactCommand),
    /// Generation: drive the engine over a seed space (run / tui / export / replay).
    #[command(subcommand)]
    Gen(GenCommand),
    /// Evaluation: off-path, model-free diagnostics over persisted records / eval artifacts.
    #[command(subcommand)]
    Eval(EvalCommand),
}

/// Provider-free artifact commands.
#[derive(Debug, Subcommand, PartialEq)]
pub enum ArtifactCommand {
    /// Read all Parquet bytes from stdin and emit a versioned JSON verification report.
    Verify {
        /// Require explicit binary stdin input; paths and legacy metadata are unsupported.
        #[arg(long, required = true)]
        stdin: bool,
    },
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
    /// Fit supplied judge observations against independent labels offline; never qualifies quality.
    FitCalibration(FitCalibrationArgs),
    /// Descriptive score diagnostics and an optional independent-outcome check; prints JSON.
    AuditSeparation(AuditSeparationArgs),
    /// Variance-aware promotion gate over two `eval_results.json` files; prints the report as JSON.
    Promote(PromoteArgs),
}

/// Offline calibration inputs; no database, provider credentials, or model calls are used.
#[derive(Debug, clap::Args, PartialEq)]
pub struct FitCalibrationArgs {
    /// Production TOML configuration resolving the expected area, rubric and ordered judge panel.
    #[arg(long, value_name = "FILE")]
    pub config: PathBuf,
    /// JSON lookup pool of full records with unique (run_id, record_id) pairs. Only records
    /// referenced by the evidence fit/assessment rows are evaluated and bound in the snapshot.
    #[arg(long, value_name = "FILE")]
    pub records: PathBuf,
    /// Strict versioned calibration evidence with exact numerical encodings.
    #[arg(long, value_name = "FILE")]
    pub evidence: PathBuf,
}

/// Whether a generation command requests automatic admission or collection for human review.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum AdmissionMode {
    /// Admit candidates that clear the verifier and panel safeguards.
    Automatic,
    /// Keep otherwise admitted candidates at NeedsReview, including on replay.
    ReviewOnly,
}

impl From<AdmissionMode> for gw_schema::AdmissionIntent {
    fn from(intent: AdmissionMode) -> Self {
        match intent {
            AdmissionMode::Automatic => Self::Automatic,
            AdmissionMode::ReviewOnly => Self::ReviewOnly,
        }
    }
}

/// Shared knobs for `gen run` / `gen tui` (the engine-spending paths). The config FILE supplies the
/// area/provider/accounting defaults; these flags LAYER over it (highest precedence after env).
#[derive(Debug, clap::Args, PartialEq)]
pub struct RunArgs {
    /// Override automatic admission or explicitly collect for human review.
    #[arg(long, value_enum, value_name = "INTENT")]
    pub admission_intent: Option<AdmissionMode>,
    /// Path to the TOML config file (figment base layer). Absent ⇒ defaults + env only.
    #[arg(long, value_name = "FILE")]
    pub config: Option<PathBuf>,
    /// Override the SQLite store path (else the config's `db` / the default).
    #[arg(long, value_name = "PATH")]
    pub db: Option<PathBuf>,
    /// The run id (idempotency key). A re-run of the SAME id over the same seeds RESUMES.
    #[arg(long, value_name = "ID")]
    pub run_id: String,
    /// Plain prompts, one user turn per line, with explicit judge-only verification.
    #[arg(
        long,
        value_name = "FILE",
        required_unless_present = "tasks",
        conflicts_with = "tasks"
    )]
    pub prompts: Option<PathBuf>,
    /// Strict reviewed numeric task JSON. Supply exactly one of --tasks or --prompts.
    #[arg(
        long,
        value_name = "FILE",
        required_unless_present = "prompts",
        conflicts_with = "prompts"
    )]
    pub tasks: Option<PathBuf>,
    /// Number of shards to partition the seed space into.
    #[arg(long, value_name = "N", default_value_t = 1)]
    pub shards: usize,
    /// Physical-request accounting and admission policy overrides.
    #[command(flatten)]
    pub accounting: AccountingArgs,
    /// Override the per-area best-of-k fan-out (else the config's `area.k`).
    #[arg(long, value_name = "K")]
    pub k: Option<u32>,
    /// Max seed items in flight across all shards (concurrency cap).
    #[arg(long, value_name = "N", default_value_t = 4)]
    pub max_in_flight: u32,
}

/// `gen export` flags (provider-free publication or exact receipt recovery).
#[derive(Debug, clap::Args, PartialEq)]
pub struct ExportArgs {
    /// Recover this receipt at its recorded destination (may replace that file). Uses its original
    /// acknowledgment mode; engine receipts may finish record export, but run status is unchanged.
    #[arg(long, value_name = "ID", conflicts_with_all = ["out", "run_id", "format", "cot", "dataset_version"])]
    pub resume_publication: Option<String>,
    /// Dataset version fixed in the artifact manifest before encoding.
    #[arg(long, value_name = "VERSION")]
    pub dataset_version: Option<semver::Version>,
    /// Path to the SQLite store to export from.
    #[arg(long, value_name = "PATH")]
    pub db: PathBuf,
    /// Destination Parquet file path.
    #[arg(
        long,
        value_name = "FILE",
        required_unless_present = "resume_publication"
    )]
    pub out: Option<PathBuf>,
    /// Restrict the export to a single run id (else every admitted record in the store).
    #[arg(long, value_name = "ID")]
    pub run_id: Option<String>,
    /// The TRL export target template (default: chat-ml for a new export).
    #[arg(long, value_enum)]
    pub format: Option<ExportFormat>,
    /// Reasoning loss policy (default: supervised for a new export).
    #[arg(long, value_enum)]
    pub cot: Option<ExportCot>,
}

/// `gen replay` flags (thin: re-enter `Engine::run` for an existing run id + store).
///
/// Resume is sound only when the SAME seed plan is re-derived, so the SAME task/prompt input (and
/// `--shards`) the original run used MUST be supplied — the engine then skips already-committed
/// offsets and re-drives any mid-flight record from its last persisted state, reusing stored output.
#[derive(Debug, clap::Args, PartialEq)]
pub struct ReplayArgs {
    /// Physical-request accounting and admission policy overrides.
    #[command(flatten)]
    pub accounting: AccountingArgs,
    /// Override admission intent; persisted review-only records always remain review-only.
    #[arg(long, value_enum, value_name = "INTENT")]
    pub admission_intent: Option<AdmissionMode>,
    /// Path to the TOML config file (must describe the same area/provider as the original run).
    #[arg(long, value_name = "FILE")]
    pub config: Option<PathBuf>,
    /// Override the SQLite store path (else the config's `db`).
    #[arg(long, value_name = "PATH")]
    pub db: Option<PathBuf>,
    /// The run id to resume (must already exist in the store with persisted checkpoints).
    #[arg(long, value_name = "ID")]
    pub run_id: String,
    /// The same plain prompt source used by the original run.
    #[arg(
        long,
        value_name = "FILE",
        required_unless_present = "tasks",
        conflicts_with = "tasks"
    )]
    pub prompts: Option<PathBuf>,
    /// The same reviewed numeric task document used by the original run.
    #[arg(
        long,
        value_name = "FILE",
        required_unless_present = "prompts",
        conflicts_with = "prompts"
    )]
    pub tasks: Option<PathBuf>,
    /// The SAME shard count the original run used (REQUIRED — there is no safe default for a resume:
    /// the seed→shard partition is `index % shards`, so a different (or silently defaulted) value
    /// re-partitions the space and duplicates/orphans records against the persisted offset cursors).
    #[arg(long, value_name = "N")]
    pub shards: usize,
    /// Max seed items in flight across all shards (concurrency cap).
    #[arg(long, value_name = "N", default_value_t = 4)]
    pub max_in_flight: u32,
}

/// `eval audit-separation` flags (model-free: a store scan and optional independent outcomes).
#[derive(Debug, clap::Args, PartialEq)]
pub struct AuditSeparationArgs {
    /// Path to the SQLite store to analyze.
    #[arg(long, value_name = "PATH")]
    pub db: PathBuf,
    /// Restrict the analysis to a single run id (else the whole store).
    #[arg(long, value_name = "ID")]
    pub run_id: Option<String>,
    /// Versioned independent outcomes for an explicit frozen corpus (required for qualification).
    #[arg(long, value_name = "FILE")]
    pub outcomes: Option<PathBuf>,
    /// Minimum distinct evaluated prompts for qualification (default 30).
    #[arg(long, value_name = "N")]
    pub min_evaluated_prompts: Option<usize>,
    /// One-sided confidence level in (0, 1), under independent prompt sampling (default 0.95).
    #[arg(long, value_name = "F")]
    pub confidence_level: Option<f64>,
    /// Descriptive warning threshold for mixed verifier groups; does not qualify a selector.
    #[arg(long, value_name = "N")]
    pub min_decidable_groups: Option<usize>,
    /// Descriptive warning floor on verifier mixedness; does not qualify a selector.
    #[arg(long, value_name = "F")]
    pub min_decidable_fraction: Option<f64>,
    /// Opt into decision-bearing process exits: 0 = pass, 1 = operational error, 2 = gate rejects.
    #[arg(long, default_value_t = false)]
    pub check: bool,
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
    /// Opt into decision-bearing process exits: 0 = promote, 1 = operational error, 2 = gate rejects.
    #[arg(long, default_value_t = false)]
    pub check: bool,
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

/// Operational accounting mode for all live run commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum AccountingMode {
    /// Record tokens, timing, and reported costs without monetary gating.
    ObservationOnly,
    /// Serialize physical sends below a known-spend USD threshold.
    FiniteUsd,
}

/// Shared highest-precedence overrides for run, replay, and TUI.
#[derive(Debug, Default, clap::Args, PartialEq)]
pub struct AccountingArgs {
    /// Select observation-only or finite-usd; finite-usd requires --limit-usd.
    #[arg(long, value_enum)]
    pub accounting_policy: Option<AccountingMode>,
    /// Finite, nonnegative reported-dollar dispatch threshold; requires finite-usd mode.
    #[arg(long, requires = "accounting_policy")]
    pub limit_usd: Option<f64>,
}
impl AccountingArgs {
    /// Apply an explicit mode as a complete policy, preserving file/env policy when absent.
    ///
    /// # Errors
    /// Rejects contradictory, incomplete, negative, or nonfinite monetary overrides.
    pub fn apply(&self, config: &mut crate::config::Config) -> anyhow::Result<()> {
        use gw_schema::AccountingPolicy;
        match (self.accounting_policy, self.limit_usd) {
            (None, None) => {}
            (Some(AccountingMode::ObservationOnly), None) => {
                config.accounting_policy = Some(AccountingPolicy::ObservationOnly);
            }
            (Some(AccountingMode::FiniteUsd), Some(limit_usd)) => {
                config.accounting_policy = Some(AccountingPolicy::FiniteUsd { limit_usd });
            }
            _ => anyhow::bail!(
                "--accounting-policy finite-usd requires --limit-usd; observation-only accepts no limit"
            ),
        }
        config.validate_accounting_policy()
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
