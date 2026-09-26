//! The injected client bundle ([`Clients`]) + the per-area grading config ([`AreaConfig`]).
//!
//! Every side-effecting dependency the step machine touches is INJECTED through [`Clients`], so the
//! transition function `step(rec, &clients)` is PURE given its inputs + these clients — replayable
//! and HERMETIC. Unit tests build a `Clients` over a fake [`Provider`], a fake [`Embedder`],
//! a [`Store::open_in_memory`](gw_storage::Store::open_in_memory), and a [`NullSandboxOracle`](gw_judge::NullSandboxOracle); no
//! network is touched. The bundle holds:
//!
//! - **`store`** — the [`Store`] (the authoritative data plane; persist-after-every-transition).
//! - **`teacher`** — the [`Provider`] for teacher (assistant-generation) calls.
//! - **`judge`** — the [`Provider`] for judge-panel calls (often the same provider; separate field so
//!   a run can route judges to a cheaper/local endpoint).
//! - **`embedder`** — the [`Embedder`] for the user-turn `diverse` dedup.
//! - **`sandbox`** — the [`SandboxOracle`] for `Oracle::SandboxExecution` ground truth (default
//!   [`NullSandboxOracle`](gw_judge::NullSandboxOracle); D-SANDBOX deferred).
//! - **`budget`** — the shared [`BudgetMeter`].
//! - **`events`** — the [`EventSink`].
//!
//! ## The R-prior is config, not a constant (INVARIANT — never identity for k>1)
//!
//! [`AreaConfig::correlation_rho`] is the cold-start inter-judge correlation prior (≈ 0.7). The step
//! machine builds `CorrelationMatrix::uniform_offdiagonal(k, rho)` from it — NEVER
//! `CorrelationMatrix::identity` — for a `k > 1` panel grade. Passing identity would silently degrade
//! the correlation guard to Kish-only (the gw-judge review flagged this), so the engine asserts the
//! matrix it hands the grader is non-identity for `k > 1` and fails loud otherwise. The prior lives
//! on the area config so a calibration run can later replace the cold-start `rho` per area.

use std::sync::Arc;

use gw_generate::Embedder;
use gw_judge::{
    AreaThresholds, ExecutionEvidenceSource, NullExecutionEvidenceSource, PanelJudge, SandboxOracle,
};
use gw_providers::Provider;
use gw_storage::Store;

use crate::budget::BudgetMeter;
use crate::event::EventSink;

/// The default cold-start inter-judge correlation prior `rho` (≈ 0.7, JUDGE-DESIGN §5.4). Used to
/// build `CorrelationMatrix::uniform_offdiagonal(k, rho)` for a `k > 1` panel grade — NEVER identity.
/// An unproven panel cannot inflate its independence; this mirrors a high Glicko-2 sigma.
pub const DEFAULT_CORRELATION_RHO: f64 = 0.7;

/// The default best-of-k for a v1 run (single trace, no fan-out). Matches `gw_generate::DEFAULT_K`.
pub const DEFAULT_K: u32 = 1;

/// The default combined-output `max_tokens` cap for a teacher call (generous, to dodge the
/// `<|channel>thought` truncation hazard on a reasoning model — ARCHITECTURE §3.2 / HARNESS_DESIGN §9).
pub const DEFAULT_MAX_TOKENS: u32 = 16_384;

/// Per-area grading + generation configuration: everything the step machine needs that varies by
/// training area. Carried on [`Clients`] so the step function reads it without a global config dep.
#[derive(Debug, Clone)]
pub struct AreaConfig {
    /// The training-area name (e.g. `"rust-async"`), stamped into provenance + the record id prefix.
    pub training_area: String,
    /// The teacher model slug for this area (e.g. `"z-ai/glm-5.2"`).
    pub teacher_slug: String,
    /// The combined-output token cap for the teacher call (always set; INVARIANT-g).
    pub max_tokens: u32,
    /// Optional explicit reasoning-token cap for teacher calls. `None` keeps the teacher default
    /// effort mode.
    pub teacher_reasoning_max_tokens: Option<u32>,
    /// Whether this area requires chain-of-thought (drives the reasoning-present Verify hard gate).
    pub cot_required: bool,
    /// Whether the rule-based answer comparator is authoritative (hard-reject a non-match) for this
    /// area. DEFAULT `false` (`rescue_negatives`: a rule non-match routes to judge rescue).
    pub rule_only_authoritative: bool,
    /// The judge panel for this area (its size is `k_judges`; a `k_judges > 1` panel triggers the
    /// non-identity correlation matrix).
    pub judges: Vec<PanelJudge>,
    /// The rubric text handed to each judge.
    pub rubric: String,
    /// The admission thresholds + correlation-guard floors for this area.
    pub thresholds: AreaThresholds,
    /// The cold-start inter-judge correlation prior `rho` for `CorrelationMatrix::uniform_offdiagonal`
    /// (NEVER identity for `k > 1`). Defaults to [`DEFAULT_CORRELATION_RHO`].
    pub correlation_rho: f64,
    /// Best-of-k fan-out size `k` (≥ 1; default 1 = single trace, no fan-out).
    pub k: u32,
}

impl AreaConfig {
    /// A judge-only area config with sane defaults: the given teacher, a generous token cap, CoT
    /// required, the default thresholds + cold-start `rho`, `k = 1`, and the supplied judge panel +
    /// rubric. Builder-style setters layer the rest.
    #[must_use]
    pub fn new(
        training_area: impl Into<String>,
        teacher_slug: impl Into<String>,
        judges: Vec<PanelJudge>,
        rubric: impl Into<String>,
    ) -> Self {
        Self {
            training_area: training_area.into(),
            teacher_slug: teacher_slug.into(),
            max_tokens: DEFAULT_MAX_TOKENS,
            teacher_reasoning_max_tokens: None,
            cot_required: true,
            rule_only_authoritative: false,
            judges,
            rubric: rubric.into(),
            thresholds: AreaThresholds::default(),
            correlation_rho: DEFAULT_CORRELATION_RHO,
            k: DEFAULT_K,
        }
    }

    /// Set the best-of-k fan-out size (clamped to ≥ 1). Chainable.
    #[must_use]
    pub fn with_k(mut self, k: u32) -> Self {
        self.k = k.max(1);
        self
    }

    /// Set the combined-output token cap for teacher calls. Chainable.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set an explicit teacher reasoning-token cap. Chainable.
    #[must_use]
    pub fn with_teacher_reasoning_max_tokens(mut self, max_tokens: u32) -> Self {
        self.teacher_reasoning_max_tokens = Some(max_tokens);
        self
    }

    /// Set the admission thresholds. Chainable.
    #[must_use]
    pub fn with_thresholds(mut self, thresholds: AreaThresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// Set the cold-start correlation prior `rho`. Chainable.
    #[must_use]
    pub fn with_correlation_rho(mut self, rho: f64) -> Self {
        self.correlation_rho = rho;
        self
    }

    /// Set whether CoT is required (drives the reasoning-present Verify gate). Chainable.
    #[must_use]
    pub fn with_cot_required(mut self, cot_required: bool) -> Self {
        self.cot_required = cot_required;
        self
    }

    /// The number of judges in this area's panel.
    #[must_use]
    pub fn k_judges(&self) -> usize {
        self.judges.len()
    }
}

/// The injected bundle of side-effecting clients the step machine drives, plus the per-area config.
///
/// Cheap to clone — the `Provider`s, `Embedder`, and `SandboxOracle` are `Arc`-wrapped trait objects,
/// the `Store` / `BudgetMeter` / `EventSink` are `Arc`-backed handles — so each spawned shard worker
/// holds its own clone and they all share one store, one budget meter, and one event channel.
#[derive(Clone)]
pub struct Clients {
    /// The authoritative data plane.
    pub store: Store,
    /// The teacher (assistant-generation) provider.
    pub teacher: Arc<dyn Provider>,
    /// The judge-panel provider (may be the same as `teacher`).
    pub judge: Arc<dyn Provider>,
    /// The user-turn diversity embedder.
    pub embedder: Arc<dyn Embedder + Send + Sync>,
    /// Run-scoped admitted-turn embeddings used by the diversity gate.
    pub(crate) priors: crate::priors::Priors,
    /// The sandbox ground-truth oracle (default [`NullSandboxOracle`](gw_judge::NullSandboxOracle)).
    pub sandbox: Arc<dyn SandboxOracle + Send + Sync>,
    /// The PRECOMPUTED execution-evidence source (default [`NullExecutionEvidenceSource`], which
    /// resolves nothing and leaves the execution axis inert). A keyed lookup — the harness never
    /// executes anything itself; the report already exists.
    pub execution_evidence: Arc<dyn ExecutionEvidenceSource + Send + Sync>,
    /// The shared run-wide budget meter.
    pub budget: BudgetMeter,
    /// The observability event sink.
    pub events: EventSink,
    /// The harness version stamped into provenance.
    pub harness_version: String,
    /// The optional git commit stamped into provenance.
    pub git_commit: Option<String>,
}

impl std::fmt::Debug for Clients {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The trait-object clients are not Debug; show the structural shape instead.
        f.debug_struct("Clients")
            .field("store", &self.store)
            .field("budget", &self.budget)
            .field("harness_version", &self.harness_version)
            .field("git_commit", &self.git_commit)
            .finish_non_exhaustive()
    }
}

impl Clients {
    /// Build a client bundle. `teacher` and `judge` may be the same `Arc` (one provider for both
    /// rails). The `sandbox` defaults are wired by the caller (a real run injects a `ToolExecutor`
    /// adapter; a test injects [`NullSandboxOracle`](gw_judge::NullSandboxOracle)).
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        store: Store,
        teacher: Arc<dyn Provider>,
        judge: Arc<dyn Provider>,
        embedder: Arc<dyn Embedder + Send + Sync>,
        sandbox: Arc<dyn SandboxOracle + Send + Sync>,
        budget: BudgetMeter,
        events: EventSink,
        harness_version: impl Into<String>,
    ) -> Self {
        Self {
            store,
            teacher,
            judge,
            embedder,
            priors: crate::priors::new(),
            sandbox,
            execution_evidence: Arc::new(NullExecutionEvidenceSource),
            budget,
            events,
            harness_version: harness_version.into(),
            git_commit: None,
        }
    }

    /// Set the precomputed execution-evidence source. Chainable. The default resolves nothing, so
    /// an area with no evaluator wired behaves exactly as it did before the execution axis existed.
    #[must_use]
    pub fn with_execution_evidence_source(
        mut self,
        source: Arc<dyn ExecutionEvidenceSource + Send + Sync>,
    ) -> Self {
        self.execution_evidence = source;
        self
    }

    /// Set the git commit stamped into provenance. Chainable.
    #[must_use]
    pub fn with_git_commit(mut self, commit: impl Into<String>) -> Self {
        self.git_commit = Some(commit.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn area_config_defaults_are_pinned() {
        let cfg = AreaConfig::new("math", "z-ai/glm-5.2", vec![], "rubric");
        assert_eq!(cfg.k, 1);
        assert!(cfg.cot_required);
        assert!(!cfg.rule_only_authoritative);
        assert_eq!(cfg.correlation_rho, DEFAULT_CORRELATION_RHO);
        assert_eq!(cfg.max_tokens, DEFAULT_MAX_TOKENS);
        assert_eq!(cfg.k_judges(), 0);
    }

    #[test]
    fn with_k_clamps_to_one() {
        let cfg = AreaConfig::new("math", "m", vec![], "r").with_k(0);
        assert_eq!(cfg.k, 1);
        let cfg = AreaConfig::new("math", "m", vec![], "r").with_k(8);
        assert_eq!(cfg.k, 8);
    }

    #[test]
    fn builders_layer_config() {
        let cfg = AreaConfig::new("math", "m", vec![], "r")
            .with_correlation_rho(0.5)
            .with_cot_required(false);
        assert_eq!(cfg.correlation_rho, 0.5);
        assert!(!cfg.cot_required);
    }
}
