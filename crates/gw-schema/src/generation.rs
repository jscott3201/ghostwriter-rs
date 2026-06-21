//! `generation{}` — the reproducibility params (DATA-SCHEMA §1.5).

use serde::{Deserialize, Serialize};

/// The teacher-call params needed to regenerate a record up to teacher non-determinism.
///
/// Reproducibility contract (DATA-SCHEMA §1.5): `record = f(seed, persona, taxonomy_node,
/// teacher_slug, gen_params{seed,temp,top_p,reasoning_effort,max_tokens}, prompt_template_id,
/// harness_version, git_commit)`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Generation {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Official = 1.0, Precise = 0.6.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// ALWAYS set on the request. (INVARIANT g)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// ALWAYS `"xhigh"` for CoT generation; never `"max"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    /// `effort` and `reasoning_max_tokens` are mutually exclusive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_max_tokens: Option<u32>,
    /// User-synth persona.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub persona: Option<String>,
    /// Node in the seed taxonomy/skill tree.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub taxonomy_node: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_template_id: Option<String>,
    /// best-of-k `k`; None = k=1; RFT 4..=16.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_completions: Option<u32>,
    /// 0-based index within the sibling group.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_index: Option<u32>,
    /// `== prompt_hash`; links best-of-k siblings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sibling_group_id: Option<String>,
}

/// OpenRouter reasoning-effort enum. **No `Max` variant** — `"max"` is a hard HTTP 400
/// (DATA-SCHEMA §1.5, INVARIANT §5; CONFIG ITEM 11e).
///
/// Serialized `lowercase` (this enum predates the snake_case convention; the spelling is
/// pinned by the OpenRouter wire enum `none|minimal|low|medium|high|xhigh`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
}
