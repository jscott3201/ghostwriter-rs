//! `judging{}` — the LLM JudgePanel rail (DATA-SCHEMA §1.7) and the config-side
//! [`JudgeSampling`] policy (JUDGE-DESIGN §4.2).
//!
//! INVARIANT f: `aggregate` is NEVER a plain majority/mean. This schema stores only the
//! inputs and outputs of the harness-side consensus compute (SP/BTS + calibration-weighting
//! + Dung + minority-veto); the algorithm lives in `gw-judge`.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Persisted panel grades + the derived verdict + the threshold at decision time, so admission
/// is re-derivable under a new threshold WITHOUT re-judging.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Judging {
    #[serde(default)]
    pub panel: Vec<JudgeVote>,
    /// SP/BTS + calibration-weighted — NEVER plain mean. (INVARIANT f)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aggregate: Option<f64>,
    /// Inter-judge agreement (variance / kappa).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agreement: Option<f64>,
    /// Effective sample size after correlation-adjusted calibration weighting (JUDGE-DESIGN §5.5).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_eff: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verdict: Option<Verdict>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verdict_reason: Option<String>,
    /// Stored so admission is re-derivable without re-judging.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold_at_decision: Option<f64>,
}

/// One judge's sealed grade. The RECORDED sampling params (`temperature`/`top_p`/`seed`)
/// mirror the teacher `Generation` shape; `temperature` also feeds the judge content-hash
/// cache key as `temperature_bits = f64::to_bits(temperature)` (A2, DATA-SCHEMA §6.2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JudgeVote {
    pub judge_model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rubric_id: Option<String>,
    /// Judge sampling temperature actually used for THIS vote.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// `Option<i64>` for consistency with `Generation.seed`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Normalized `0..1` (or `1..10` per rubric).
    pub score: f64,
    /// Per-criterion sub-scores (also carries the B7 assistant-side reject axes as keys).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<BTreeMap<String, f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_response: Option<String>,
}

/// The persisted envelope verdict (DATA-SCHEMA §1.7). Stays 3 variants; distinct from the
/// 4-variant per-grade `gw-judge::Verdict` and the panel `gw-judge::Decision` (REMEDIATION
/// ITEM 6 reconciliation). `Revise` is transient and `Uncertain` is per-judge only, so
/// neither persists here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Admit,
    Reject,
    NeedsReview,
}

/// Judge-call sampling policy (config-side; JUDGE-DESIGN §4.2). The params ACTUALLY used are
/// recorded per-vote on [`JudgeVote`]. Default policy is `temperature = 0.0` — but the
/// content-hash cache, NOT temp=0, is the replay guarantee.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JudgeSampling {
    /// Default `0.0` (bare `f64`, not `Option`).
    pub temperature: f64,
    /// Default None.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Default None. `Option<i64>` matches `Generation.seed`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
}

impl Default for JudgeSampling {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: None,
            seed: None,
        }
    }
}
