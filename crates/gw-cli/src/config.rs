//! The figment-layered run configuration (CONFIG.md: TOML file → env → clap overrides).
//!
//! [`Config`] is the effective, validated configuration a `gen run` / `gen tui` / `gen replay` reads.
//! It is assembled by [`Config::load`] from three layers, lowest precedence first:
//!
//! 1. **the built-in [`Default`]** (so a missing file still yields a runnable shape);
//! 2. **a TOML file** (`--config`), if supplied;
//! 3. **environment variables** prefixed `GW_` (e.g. `GW_BUDGET_USD=5.0`), via figment's `Env`.
//!
//! clap flags (`--db`, `--budget-usd`, `--k`) layer LAST, applied by the command handler AFTER load
//! (figment merges file+env; the flags are the final, highest-precedence override). Splitting it this
//! way keeps figment's job purely "file + env" and makes the flag precedence explicit at the call.
//!
//! ## The API key is NEVER in this struct (security INVARIANT)
//!
//! `OPENROUTER_API_KEY` is read from the process environment by the provider constructor
//! ([`crate::wire::build_provider`]) and is deliberately ABSENT from [`Config`] — it is never read
//! from the TOML file, never a clap flag, never serialized, and never logged. The figment `Env`
//! provider is scoped to the `GW_` prefix, so it cannot even accidentally slurp `OPENROUTER_API_KEY`
//! (which carries no `GW_` prefix) into the config.
//!
//! The config mirrors the leaf-crate config types ([`AreaThresholds`], [`PanelJudge`]) with its OWN
//! `serde`-deriving structs, because those leaf types do not derive `serde`; [`Config::area_config`]
//! maps them into the engine's [`AreaConfig`].

use std::path::PathBuf;

use figment::{
    Figment,
    providers::{Env, Format, Serialized, Toml},
};
use serde::{Deserialize, Serialize};

use gw_engine::AreaConfig;
use gw_judge::{AreaThresholds, PanelJudge};
use gw_schema::{BudgetBreach, CotPolicy, ReasoningEffort, TrlFormat};

/// The default SQLite store path when none is configured.
pub const DEFAULT_DB_PATH: &str = "gw-run.sqlite";

/// The effective run configuration (file + env layered). clap flags override fields AFTER load.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// The SQLite store path.
    pub db: PathBuf,
    /// The run-wide budget cap in USD (the primary spend guard).
    pub budget_usd: f64,
    /// What the engine does when the budget cap is reached.
    #[serde(default)]
    pub on_breach: BudgetBreach,
    /// The OpenAI-compatible provider base URL (default OpenRouter). The API KEY is NOT here.
    pub provider_base_url: String,
    /// The per-lane requests-per-minute budget for the provider rate limiter.
    pub provider_rpm: u32,
    /// The TUI tick interval in milliseconds (dashboard model updates).
    pub tick_ms: u64,
    /// The TUI frame interval in milliseconds (render cap).
    pub frame_ms: u64,
    /// The per-area generation + grading configuration.
    pub area: AreaSettings,
    /// Optional end-of-run Parquet shard export configuration.
    pub export: Option<ExportSettings>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            db: PathBuf::from(DEFAULT_DB_PATH),
            budget_usd: 5.0,
            on_breach: BudgetBreach::Drain,
            provider_base_url: gw_providers::DEFAULT_BASE_URL.to_string(),
            provider_rpm: 60,
            tick_ms: 250,
            frame_ms: 33,
            area: AreaSettings::default(),
            export: None,
        }
    }
}

/// Optional end-of-run Parquet shard export settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExportSettings {
    /// Destination Parquet shard path.
    pub out: PathBuf,
    /// The TRL export target template recorded in the manifest.
    pub format: TrlFormat,
    /// Whether reasoning enters the supervised loss region on export.
    #[serde(default)]
    pub cot: CotPolicy,
    /// Optional dataset version recorded in the sidecar manifest.
    #[serde(default)]
    pub dataset_version: Option<semver::Version>,
}

/// The per-area settings, mirroring the engine's [`AreaConfig`] with serde-deriving fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AreaSettings {
    /// The training-area name (stamped into provenance + the record id prefix).
    pub training_area: String,
    /// The teacher model slug.
    pub teacher_slug: String,
    /// The rubric text handed to each judge.
    pub rubric: String,
    /// Whether this area requires chain-of-thought (drives the reasoning-present Verify gate).
    pub cot_required: bool,
    /// Best-of-k fan-out size (clamped to >= 1 by the engine).
    pub k: u32,
    /// The cold-start inter-judge correlation prior `rho` (NEVER identity for `k > 1`).
    pub correlation_rho: f64,
    /// The judge panel.
    pub judges: Vec<JudgeSettings>,
    /// The admission thresholds + correlation-guard floors.
    pub thresholds: ThresholdSettings,
    /// Optional area-wide judge combined completion cap.
    #[serde(default)]
    pub judge_max_tokens: Option<u32>,
    /// Optional area-wide explicit judge reasoning-token cap.
    #[serde(default)]
    pub judge_reasoning_max_tokens: Option<u32>,
    /// Optional area-wide judge reasoning effort. Mutually exclusive with
    /// `judge_reasoning_max_tokens` in the same table.
    #[serde(default)]
    pub judge_reasoning_effort: Option<ReasoningEffort>,
}

impl Default for AreaSettings {
    fn default() -> Self {
        Self {
            training_area: "general".to_string(),
            teacher_slug: "z-ai/glm-5.2".to_string(),
            rubric: "Grade the reasoning trace for correctness, rigor, and clarity.".to_string(),
            cot_required: true,
            k: gw_engine::DEFAULT_K,
            correlation_rho: gw_engine::DEFAULT_CORRELATION_RHO,
            judges: Vec::new(),
            thresholds: ThresholdSettings::default(),
            judge_max_tokens: None,
            judge_reasoning_max_tokens: None,
            judge_reasoning_effort: None,
        }
    }
}

impl AreaSettings {
    fn apply_judge_defaults(&self, mut judge: PanelJudge) -> PanelJudge {
        if let Some(max_tokens) = self.judge_max_tokens {
            judge = judge.with_max_tokens(max_tokens);
        }
        if let Some(reasoning_max_tokens) = self.judge_reasoning_max_tokens {
            judge = judge.with_reasoning_max_tokens(reasoning_max_tokens);
        }
        if let Some(reasoning_effort) = self.judge_reasoning_effort {
            judge = judge.with_reasoning_effort(reasoning_effort);
        }
        judge
    }
}

/// One judge's config, mirroring [`PanelJudge`] (which does not derive serde).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JudgeSettings {
    /// OpenRouter model slug, e.g. `"deepseek/deepseek-v4-pro"`.
    pub slug: String,
    /// Coarse model family for same-family exclusion, e.g. `"deepseek"`.
    pub family: String,
    /// Optional rubric id (part of the cache key).
    #[serde(default)]
    pub rubric_id: Option<String>,
    /// Optional per-judge combined completion cap.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Optional per-judge explicit reasoning-token cap.
    #[serde(default)]
    pub reasoning_max_tokens: Option<u32>,
    /// Optional per-judge reasoning effort. Mutually exclusive with `reasoning_max_tokens` in the
    /// same judge table.
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
}

impl JudgeSettings {
    fn apply_overrides(&self, mut judge: PanelJudge) -> PanelJudge {
        if let Some(max_tokens) = self.max_tokens {
            judge = judge.with_max_tokens(max_tokens);
        }
        if let Some(reasoning_max_tokens) = self.reasoning_max_tokens {
            judge = judge.with_reasoning_max_tokens(reasoning_max_tokens);
        }
        if let Some(reasoning_effort) = self.reasoning_effort {
            judge = judge.with_reasoning_effort(reasoning_effort);
        }
        judge
    }
}

/// The admission thresholds, mirroring [`AreaThresholds`] (which does not derive serde). Defaults
/// match the gw-judge defaults exactly.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ThresholdSettings {
    /// Admit at or above this score.
    pub accept_threshold: f64,
    /// Reject below this score (the `[reject_below, accept_threshold)` window is the revise band).
    pub reject_below: f64,
    /// Escalate if `n_eff/k < this`.
    pub min_n_eff_ratio: f64,
    /// Absolute `n_eff` floor; escalate if `n_eff < this`.
    pub min_n_eff: f64,
}

impl Default for ThresholdSettings {
    fn default() -> Self {
        // Mirror gw_judge::AreaThresholds::default() so the config and engine agree out of the box.
        let d = AreaThresholds::default();
        Self {
            accept_threshold: d.accept_threshold,
            reject_below: d.reject_below,
            min_n_eff_ratio: d.min_n_eff_ratio,
            min_n_eff: d.min_n_eff,
        }
    }
}

impl From<ThresholdSettings> for AreaThresholds {
    fn from(t: ThresholdSettings) -> Self {
        Self {
            accept_threshold: t.accept_threshold,
            reject_below: t.reject_below,
            min_n_eff_ratio: t.min_n_eff_ratio,
            min_n_eff: t.min_n_eff,
        }
    }
}

impl Config {
    /// Load the effective config by layering: built-in defaults → an optional TOML file → `GW_`-
    /// prefixed env vars. clap flags are applied by the caller AFTER this (highest precedence).
    ///
    /// The `GW_` env prefix is split on `__` for nested keys (e.g. `GW_AREA__K=4` sets `area.k`).
    /// `OPENROUTER_API_KEY` carries no `GW_` prefix, so it is structurally unreachable from here.
    ///
    /// Returns [`anyhow::Result`] (not the raw `figment::Error`): the underlying figment error is
    /// large (clippy `result_large_err`), so it is boxed into `anyhow` — which is also the binary
    /// boundary every caller already wraps with `.context(...)`.
    ///
    /// # Errors
    /// Returns an error if the TOML file is malformed or a value fails to deserialize into [`Config`]
    /// (e.g. a non-numeric `budget_usd`).
    pub fn load(file: Option<&std::path::Path>) -> anyhow::Result<Self> {
        let mut fig = Figment::from(Serialized::defaults(Config::default()));
        if let Some(path) = file {
            fig = fig.merge(Toml::file(path));
        }
        fig = fig.merge(Env::prefixed("GW_").split("__"));
        let config: Self = fig.extract()?;
        config.validate_run_control()?;
        config.validate_judge_reasoning()?;
        Ok(config)
    }

    /// Validate run-control settings that deserialize but are not implemented yet.
    ///
    /// # Errors
    /// Returns an error if `on_breach = "pause"` is configured. Pause remains a schema variant, but
    /// the engine has not implemented parking semantics yet.
    pub fn validate_run_control(&self) -> anyhow::Result<()> {
        if self.on_breach == BudgetBreach::Pause {
            anyhow::bail!(
                "on_breach = \"pause\" is not yet supported (tracked as a follow-up); use \"drain\" or \"abort\""
            );
        }
        Ok(())
    }

    /// Validate judge reasoning settings that are individually valid TOML but ambiguous together.
    ///
    /// # Errors
    /// Returns an error when the same area or judge table asks for both effort-mode and explicit
    /// max-token reasoning.
    pub fn validate_judge_reasoning(&self) -> anyhow::Result<()> {
        if self.area.judge_reasoning_max_tokens.is_some()
            && self.area.judge_reasoning_effort.is_some()
        {
            anyhow::bail!(
                "area judge_reasoning_max_tokens and judge_reasoning_effort are mutually exclusive"
            );
        }
        for judge in &self.area.judges {
            if judge.reasoning_max_tokens.is_some() && judge.reasoning_effort.is_some() {
                anyhow::bail!(
                    "judge {} sets both reasoning_max_tokens and reasoning_effort; use only one",
                    judge.slug
                );
            }
        }
        Ok(())
    }

    /// Map the configured area into the engine's [`AreaConfig`].
    ///
    /// The `k` and `correlation_rho` are carried through the engine's builders (which clamp `k >= 1`).
    /// The judge panel + thresholds are mapped from the serde-mirror structs into the leaf types.
    #[must_use]
    pub fn area_config(&self) -> AreaConfig {
        let judges: Vec<PanelJudge> = self
            .area
            .judges
            .iter()
            .map(|j| {
                let judge = PanelJudge::new(&j.slug, &j.family);
                let judge = match &j.rubric_id {
                    Some(id) => judge.with_rubric(id),
                    None => judge,
                };
                j.apply_overrides(self.area.apply_judge_defaults(judge))
            })
            .collect();
        AreaConfig::new(
            &self.area.training_area,
            &self.area.teacher_slug,
            judges,
            &self.area.rubric,
        )
        .with_k(self.area.k)
        .with_correlation_rho(self.area.correlation_rho)
        .with_cot_required(self.area.cot_required)
        .with_thresholds(self.area.thresholds.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_runnable_without_a_file() {
        let cfg = Config::load(None).expect("defaults load");
        assert_eq!(cfg.db, PathBuf::from(DEFAULT_DB_PATH));
        assert_eq!(cfg.provider_base_url, gw_providers::DEFAULT_BASE_URL);
        assert_eq!(cfg.on_breach, BudgetBreach::Drain);
        assert_eq!(cfg.area.k, gw_engine::DEFAULT_K);
        assert!(cfg.export.is_none());
    }

    #[test]
    fn area_config_maps_judges_and_thresholds() {
        let mut cfg = Config::default();
        cfg.area.judges = vec![
            JudgeSettings {
                slug: "deepseek/deepseek-v4-pro".into(),
                family: "deepseek".into(),
                rubric_id: Some("r1".into()),
                max_tokens: None,
                reasoning_max_tokens: None,
                reasoning_effort: None,
            },
            JudgeSettings {
                slug: "qwen/qwen4-72b".into(),
                family: "qwen".into(),
                rubric_id: None,
                max_tokens: Some(4_200),
                reasoning_max_tokens: Some(1_200),
                reasoning_effort: None,
            },
        ];
        cfg.area.judge_max_tokens = Some(3_500);
        cfg.area.judge_reasoning_max_tokens = Some(2_000);
        cfg.area.k = 3;
        let area = cfg.area_config();
        assert_eq!(area.k_judges(), 2);
        assert_eq!(area.k, 3);
        assert_eq!(area.judges[0].max_tokens, 3_500);
        assert_eq!(
            area.judges[0].reasoning,
            Some(gw_judge::JudgeReasoning::MaxTokens(2_000))
        );
        assert_eq!(area.judges[1].max_tokens, 4_200);
        assert_eq!(
            area.judges[1].reasoning,
            Some(gw_judge::JudgeReasoning::MaxTokens(1_200))
        );
        // Thresholds round-trip into the leaf type.
        assert!((area.thresholds.accept_threshold - 0.80).abs() < 1e-12);
    }
}
