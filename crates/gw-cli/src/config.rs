//! The figment-layered run configuration (CONFIG.md: TOML file → env → clap overrides).
//!
//! [`Config`] is the layered configuration a `gen run` / `gen tui` / `gen replay` reads. Admission
//! preflight runs on the resolved area after CLI overrides, before live provider construction.
//! It is assembled by [`Config::load`] from three layers, lowest precedence first:
//!
//! 1. **the built-in [`Default`]** (so a missing file still yields a deserializable shape);
//! 2. **a TOML file** (`--config`), if supplied;
//! 3. **environment variables** prefixed `GW_` (e.g. `GW_ACCOUNTING_POLICY__MODE=observation_only`), via figment's `Env`.
//!
//! clap flags (`--db`, `--accounting-policy`, `--k`) layer LAST, applied by the command handler AFTER load
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
use gw_schema::{
    AccountingPolicy, AdmissionIntent, CotPolicy, EmbeddingConfig, ReasoningEffort, TrlFormat,
};

/// The default SQLite store path when none is configured.
pub const DEFAULT_DB_PATH: &str = "gw-run.sqlite";

/// The effective run configuration (file + env layered). clap flags override fields AFTER load.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// The SQLite store path.
    pub db: PathBuf,
    /// Explicit policy from file, environment, or flags. Absence defaults only after layering.
    pub accounting_policy: Option<AccountingPolicy>,
    /// The OpenAI-compatible provider base URL (default OpenRouter). The API KEY is NOT here.
    pub provider_base_url: String,
    /// The shared teacher and judge chat requests-per-minute rate limit.
    pub provider_rpm: u32,
    /// The TUI tick interval in milliseconds (dashboard model updates).
    pub tick_ms: u64,
    /// The TUI frame interval in milliseconds (render cap).
    pub frame_ms: u64,
    /// The per-area generation + grading configuration.
    pub area: AreaSettings,
    /// Optional end-of-run Parquet shard export configuration.
    pub export: Option<ExportSettings>,
    /// Optional OpenAI-compatible embedding configuration.
    pub embedding: Option<EmbeddingConfig>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            db: PathBuf::from(DEFAULT_DB_PATH),
            accounting_policy: None,
            provider_base_url: gw_providers::DEFAULT_BASE_URL.to_string(),
            provider_rpm: 60,
            tick_ms: 250,
            frame_ms: 33,
            area: AreaSettings::default(),
            export: None,
            embedding: None,
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
    /// Optional dataset version fixed in the artifact footer before encoding.
    #[serde(default)]
    pub dataset_version: Option<semver::Version>,
}

/// The per-area settings, mirroring the engine's [`AreaConfig`] with serde-deriving fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AreaSettings {
    /// Automatic admission or explicit review-only collection (numeric safeguards still apply).
    pub admission_intent: AdmissionIntent,
    /// The training-area name (stamped into provenance + the record id prefix).
    pub training_area: String,
    /// The teacher model slug.
    pub teacher_slug: String,
    /// Optional teacher combined completion cap.
    #[serde(default)]
    pub teacher_max_tokens: Option<u32>,
    /// Optional teacher reasoning-token cap.
    #[serde(default)]
    pub teacher_reasoning_max_tokens: Option<u32>,
    /// The rubric text handed to each judge.
    pub rubric: String,
    /// Whether this area requires chain-of-thought (drives the reasoning-present Verify gate).
    pub cot_required: bool,
    /// Best-of-k fan-out size (clamped to >= 1 by the engine).
    pub k: u32,
    /// The cold-start inter-judge correlation prior `rho` (nonidentity for multiple judges,
    /// independently of generation's best-of-k setting).
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
            admission_intent: AdmissionIntent::Automatic,
            training_area: "general".to_string(),
            teacher_slug: "z-ai/glm-5.2".to_string(),
            teacher_max_tokens: None,
            teacher_reasoning_max_tokens: None,
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
    /// (e.g. a malformed `accounting_policy`).
    pub fn load(file: Option<&std::path::Path>) -> anyhow::Result<Self> {
        let mut fig = Figment::from(Serialized::defaults(Config::default()));
        if let Some(path) = file {
            fig = fig.merge(Toml::file(path));
        }
        fig = fig.merge(Env::prefixed("GW_").split("__"));
        let mut value: serde_json::Value = fig.extract()?;
        let environment: serde_json::Value =
            Figment::from(Env::prefixed("GW_").split("__")).extract()?;
        // A mode selected by a later layer replaces the complete tagged policy, so ObservationOnly
        // cannot inherit a stale finite limit from the file. Contradictions within one layer error.
        if let Some(policy) = environment.get("accounting_policy")
            && policy.get("mode").is_some()
        {
            value["accounting_policy"] = policy.clone();
        }
        let config: Self = serde_json::from_value(value)?;
        config.validate_accounting_policy()?;
        config.validate_generation_budgets()?;
        config.validate_judge_reasoning()?;
        Ok(config)
    }

    /// Resolve the historical $5 default only after every configuration layer has been applied.
    #[must_use]
    pub fn effective_policy(&self) -> AccountingPolicy {
        self.accounting_policy.clone().unwrap_or_default()
    }

    /// Reject invalid monetary thresholds before constructing live clients.
    ///
    /// # Errors
    /// Returns an error for a negative, NaN, or infinite finite threshold.
    pub fn validate_accounting_policy(&self) -> anyhow::Result<()> {
        if !self.effective_policy().is_valid() {
            anyhow::bail!("accounting_policy finite_usd limit_usd must be finite and nonnegative");
        }
        Ok(())
    }

    /// Validate generation token budgets that deserialize but cannot produce a usable provider call.
    ///
    /// # Errors
    /// Returns an error if the teacher completion cap is explicitly set to zero.
    pub fn validate_generation_budgets(&self) -> anyhow::Result<()> {
        if self.area.teacher_max_tokens == Some(0) {
            anyhow::bail!("area teacher_max_tokens must be greater than zero");
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
        let mut area = AreaConfig::new(
            &self.area.training_area,
            &self.area.teacher_slug,
            judges,
            &self.area.rubric,
        )
        .with_k(self.area.k)
        .with_admission_intent(self.area.admission_intent)
        .with_correlation_rho(self.area.correlation_rho)
        .with_cot_required(self.area.cot_required)
        .with_thresholds(self.area.thresholds.into());
        if let Some(max_tokens) = self.area.teacher_max_tokens {
            area = area.with_max_tokens(max_tokens);
        }
        if let Some(reasoning_max_tokens) = self.area.teacher_reasoning_max_tokens {
            area = area.with_teacher_reasoning_max_tokens(reasoning_max_tokens);
        }
        area
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_load_without_a_file_but_need_a_generation_panel() {
        let cfg = Config::load(None).expect("defaults load");
        assert_eq!(cfg.db, PathBuf::from(DEFAULT_DB_PATH));
        assert_eq!(cfg.provider_base_url, gw_providers::DEFAULT_BASE_URL);
        assert_eq!(cfg.effective_policy(), AccountingPolicy::default());
        assert_eq!(cfg.area.k, gw_engine::DEFAULT_K);
        assert!(cfg.export.is_none());
        assert!(cfg.area_config().assess_admission().is_err());
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
        cfg.area.teacher_max_tokens = Some(20_000);
        cfg.area.teacher_reasoning_max_tokens = Some(12_000);
        cfg.area.judge_max_tokens = Some(3_500);
        cfg.area.judge_reasoning_max_tokens = Some(2_000);
        cfg.area.k = 3;
        let area = cfg.area_config();
        assert_eq!(area.k_judges(), 2);
        assert_eq!(area.k, 3);
        assert_eq!(area.max_tokens, 20_000);
        assert_eq!(area.teacher_reasoning_max_tokens, Some(12_000));
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
