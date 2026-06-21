//! `provenance{}` — who/what made a record (DATA-SCHEMA §1.4).

use serde::{Deserialize, Serialize};

/// Lineage + authorship of a record. The relational DAG edges are derived from `parent_ids`
/// plus the run-ledger (DATA-SCHEMA §5.5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub run_id: String,
    /// Upstream records (seed node, prior turn) — the lineage DAG edges.
    #[serde(default)]
    pub parent_ids: Vec<String>,
    pub teacher: TeacherRef,
    /// Model that synthesized the USER turn (the harness synthesizes BOTH roles).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_synth_model: Option<String>,
    /// gw-judge-readable user-turn classification: the serialized `VerificationKind` variant
    /// (e.g. `"refusal_expected"`) — NOT a new enum. Persisted so an assistant-side
    /// `over_refusal` rubric axis can be made INERT when refusal IS the oracle (B7). DEFAULT
    /// None until classified.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_turn_kind: Option<String>,
    /// Whether the synthesized user turn is in-scope and safe (mirrors
    /// `UserTurnVerdict.in_scope_safe`, USER-SYNTHESIS §9). gw-judge-readable. DEFAULT None
    /// until the user-turn verdict is computed (B7).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub in_scope_safe: Option<bool>,
    #[serde(default)]
    pub judge_models: Vec<String>,
    pub harness_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_commit: Option<String>,
}

/// The OpenRouter teacher actually used. `served_by` is load-bearing for reproducibility:
/// routing variance changes outputs even at temperature 0 (DATA-SCHEMA §1.4).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TeacherRef {
    /// Const `"openrouter"`.
    pub provider: String,
    /// e.g. `"z-ai/glm-5.2"` (one of the four teachers).
    pub slug: String,
    /// The OpenRouter UPSTREAM provider actually routed to (response.provider / served_by),
    /// e.g. `"Parasail"`, `"Wafer"`, `"WandB"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub served_by: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_card_revision: Option<String>,
}
