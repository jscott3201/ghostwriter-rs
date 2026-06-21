//! User-synthesis verification contract + the pre-teacher-spend QC verdict
//! (USER-SYNTHESIS §8/§9).
//!
//! These types describe *what correctness means* for a synthesized USER turn and gate it
//! before any teacher tokens are spent. `gw-judge` reads `VerificationContract.kind` to gate
//! the `over_refusal` dimension; `Provenance.user_turn_kind` / `Provenance.in_scope_safe`
//! mirror the serialized variant + `UserTurnVerdict.in_scope_safe`.

use serde::{Deserialize, Serialize};

/// What the deterministic Verifier rail will check, and how the oracle answer is obtained
/// (USER-SYNTHESIS §8).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationContract {
    /// The class of correctness check.
    pub kind: VerificationKind,
    /// How ground truth is computed.
    pub oracle: Oracle,
    /// Canonical answer-format marker, chosen ONCE per corpus build (B8). Drives teacher
    /// answer-marker steering + the export-time single-marker REFUSE/quarantine guard
    /// (DATA-SCHEMA §3) and the extractor fallback (JUDGE-DESIGN §1.1). None ⇒ no single
    /// canonical marker enforced (guard inert).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer_marker: Option<String>,
}

/// The class of correctness check for a synthesized USER turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationKind {
    /// answer is a number / aggregate; compare to oracle within tolerance.
    NumericMatch,
    /// answer is a set/ranking; compare membership/order.
    SetMatch,
    /// answer derives from a SQL/tool result; compare to oracle execution.
    SqlResultMatch,
    /// adversarial-by-construction (seed-020): correct behavior is refusal.
    RefusalExpected,
    /// answer must match a described schema/columns.
    SchemaShape,
    /// open-ended; no deterministic oracle (judge-only admission).
    None,
}

/// How ground truth is computed for a [`VerificationContract`]. Tagged on `oracle`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "oracle")]
pub enum Oracle {
    /// Oracle answer computed by executing a reference query/tool in the sandbox (REMEDIATION ITEM 3).
    SandboxExecution {
        tool_or_sql: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        expected: Option<String>,
    },
    /// Oracle is a precomputed literal carried with the contract.
    Literal { expected: String },
    /// Refusal is the oracle: correct behavior is a safe refusal + explanation.
    RefusalPolicy { policy_id: String },
    /// No deterministic oracle; admission is judge-only.
    None,
}

/// QC verdict on a candidate USER turn, BEFORE teacher spend (USER-SYNTHESIS §9). A candidate
/// advances to `user_synthesized` iff all four booleans are true.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserTurnVerdict {
    /// a competent teacher could answer it; not nonsense/contradictory.
    pub answerable: bool,
    /// matches the requested difficulty band (not trivially off-target).
    pub difficulty_targeted: bool,
    /// passes embedding-dedup: not a near-repeat.
    pub diverse: bool,
    /// within Taxonomy + decontaminated + safety-classified. Adversarial-by-construction
    /// prompts (seed-020) are `in_scope_safe = true` (wanted, tagged adversarial); only
    /// out-of-scope-unsafe prompts fail. Mirrored to `Provenance.in_scope_safe`.
    pub in_scope_safe: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}
