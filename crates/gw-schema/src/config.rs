//! The global `Config` and its sub-configs (CONFIG §1, REMEDIATION ITEM 5).
//!
//! `Config` is the *non-secret* operational surface. The **types** live here in `gw-schema`;
//! the figment loader/validation lives in `gw-cli`. Secrets are NEVER in `Config` (CONFIG §5).
//! Defaults follow CONFIG §9.1 so an absent `gw.toml` yields a fully-populated, valid config.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::embedding::EmbeddingConfig;
use crate::generation::ReasoningEffort;
use crate::sandbox::SandboxConfig;

/// The top-level layered config handed to `gw-engine`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    // --- paths & output ---
    /// SQLite run-ledger. default `"./gw-run.sqlite"`.
    pub db_path: PathBuf,
    /// `run-<id>/shard-<k>.jsonl` etc. default `"./out"`.
    pub output_dir: PathBuf,
    /// e.g. `"user/dataset-name"`; None = no push.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hub_repo: Option<String>,
    /// tracing `EnvFilter` directive. default `"info"`.
    pub log_level: String,

    // --- concurrency & rate limits ---
    /// global tokio Semaphore permits. DEFAULT 8.
    pub max_in_flight: u32,
    /// per-provider governor GCRA fallback. DEFAULT 60.
    pub default_rpm: u32,
    /// keyed by served-by/provider slug.
    #[serde(default)]
    pub providers: BTreeMap<String, ProviderLimits>,

    // --- budget cap ---
    pub budget: BudgetConfig,

    // --- teacher routing defaults ---
    /// teacher slug -> routing.
    #[serde(default)]
    pub teacher_defaults: BTreeMap<String, TeacherRouting>,

    // --- subsystems ---
    pub embedding: EmbeddingConfig,
    pub sandbox: SandboxConfig,
    pub promote: PromoteConfig,
    /// AionforgeMemory dedicated harness namespace. Advisory sidecar only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_namespace: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from("./gw-run.sqlite"),
            output_dir: PathBuf::from("./out"),
            hub_repo: None,
            log_level: "info".to_string(),
            max_in_flight: 8,
            default_rpm: 60,
            providers: BTreeMap::new(),
            budget: BudgetConfig::default(),
            teacher_defaults: BTreeMap::new(),
            embedding: EmbeddingConfig::default(),
            sandbox: SandboxConfig::default(),
            promote: PromoteConfig::default(),
            memory_namespace: None,
        }
    }
}

/// Per-provider rate-limit + endpoint overrides. NOT `Copy` (`endpoint` is non-Copy).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProviderLimits {
    /// OpenAI-compatible base URL. None ⇒ default OpenRouter base URL; REQUIRED for omlx +
    /// any non-OpenRouter remote.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    /// requests/min (GCRA refill); overrides `default_rpm` for this slug.
    pub rpm: u32,
    /// tokens/min cap; None = no token-rate limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tpm: Option<u32>,
    /// per-provider in-flight cap; None = use global `max_in_flight`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_in_flight: Option<u32>,
}

/// Budget cap — the PRIMARY spend guard (CONFIG §4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BudgetConfig {
    /// hard cap value. default 25.0 (pilot-safe).
    pub cap_usd: f64,
    pub granularity: BudgetGranularity,
    pub on_breach: BudgetBreach,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            cap_usd: 25.0,
            granularity: BudgetGranularity::PerRun,
            on_breach: BudgetBreach::Drain,
        }
    }
}

/// The window the cost meter sums over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetGranularity {
    #[default]
    PerRun,
    PerShard,
    PerDay,
}

/// What happens at the cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetBreach {
    /// Stop dispatching new records; let in-flight finish + persist; then halt cleanly.
    #[default]
    Drain,
    /// Like Drain but the engine parks instead of exiting.
    Pause,
    /// Cancel in-flight immediately.
    Abort,
}

/// Per-teacher OpenRouter routing default (CONFIG §1.1). The serde-able mirror of OpenRouter
/// `provider:{}` + `reasoning:{}` knobs. Reuses the shared `ReasoningEffort` enum.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TeacherRouting {
    /// OpenRouter `provider.order` pin. None = OpenRouter default routing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_only: Option<Vec<String>>,
    /// reasoning effort. Teacher routing default is `Xhigh`; NEVER `"max"`.
    pub reasoning_effort: ReasoningEffort,
    /// OpenRouter `provider.data_collection`. default `Deny`.
    pub data_collection: DataCollection,
    /// OpenRouter `provider.require_parameters`. default true.
    pub require_parameters: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
}

impl Default for TeacherRouting {
    fn default() -> Self {
        Self {
            provider_only: None,
            reasoning_effort: ReasoningEffort::Xhigh,
            data_collection: DataCollection::Deny,
            require_parameters: true,
            temperature: None,
            top_p: None,
        }
    }
}

/// OpenRouter `provider.data_collection`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataCollection {
    #[default]
    Deny,
    Allow,
}

/// The promotion-gate threshold block — TOP-LEVEL, not per-area (CONFIG §1.2, REMEDIATION
/// ITEM 9). Owned by `gw-eval`, consumed by `gw-cli gen promote`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromoteConfig {
    /// default `"subliminal_v1"`.
    pub drift_probe: String,
    /// default true — probe exit code 0 required.
    pub drift_must_pass: bool,
    /// default `"eval_results.aggregate"`.
    pub ab_metric: String,
    /// default 0.0 — candidate must be >= baseline + this.
    pub ab_min_delta: f64,
    /// hf-repo-or-sha; None until a baseline exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ab_baseline: Option<String>,
}

impl Default for PromoteConfig {
    fn default() -> Self {
        Self {
            drift_probe: "subliminal_v1".to_string(),
            drift_must_pass: true,
            ab_metric: "eval_results.aggregate".to_string(),
            ab_min_delta: 0.0,
            ab_baseline: None,
        }
    }
}
