//! `reasoning_quality{}` — the optional per-step CoT verdict block (DATA-SCHEMA §1.13, A4).
//!
//! A SOFT block: it drives ranking + a revise-band trigger only. The hard floor is OFF by
//! default and it does NOT gate admission (INVARIANT-f anti-mean pushed onto the step axis).
//! Phase-0 is schema + doc only — the `aggregate_steps` / `failed_step_fraction` compute
//! lives in `gw-judge`.

use serde::{Deserialize, Serialize};

/// Ordered per-step verdicts over the segmented CoT plus a soft aggregate.
///
/// `steps[i].index` MUST be dense and monotonic (`steps[i].index == i`); that invariant is
/// not expressible in serde/JSON-Schema and is asserted at ingest by the A4 impl.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ReasoningQuality {
    /// Ordered per-step verdicts over the segmented CoT.
    #[serde(default)]
    pub steps: Vec<StepVerdict>,
    /// SOFT aggregate `0..1` — drives ranking + a revise-band trigger ONLY; hard floor OFF.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_score: Option<f64>,
    /// Which rule produced `reasoning_score`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aggregation: Option<StepAggregation>,
    /// Failed-Step-Fraction, LABEL-derived from `steps[]`. Distinct from the §9 deterministic
    /// branch-abandonment FSF pre-filter (separate metric, separate provenance).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fsf: Option<f64>,
}

/// One step's verdict over the segmented CoT.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepVerdict {
    /// 0-based step ordinal (dense + monotonic; see [`ReasoningQuality`]).
    pub index: u32,
    /// `0..1` per-step quality.
    pub score: f64,
    pub passed: bool,
    /// Optional step label (e.g. `"abandoned"`, `"correct"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
}

/// The aggregation rule for `reasoning_score`. Per-area knob (`step_aggregation`): DEFAULT
/// `Min` for verifiable areas / `LateWeighted` for non-verifiable areas. `Mean` is NEVER the
/// default — it dilutes a single fatal step (DATA-SCHEMA §1.13, JUDGE-DESIGN §10).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepAggregation {
    Min,
    LateWeighted,
    Mean,
}
