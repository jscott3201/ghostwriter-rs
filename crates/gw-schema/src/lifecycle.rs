//! `lifecycle{}` — the per-record state machine (DATA-SCHEMA §1.8, REMEDIATION ITEM 6).

use serde::{Deserialize, Serialize};

/// Event-sourced lifecycle state. `history` doubles as a per-record checkpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lifecycle {
    pub state: LifecycleState,
    #[serde(default)]
    pub history: Vec<StateTransition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default)]
    pub attempts: u32,
}

impl Default for Lifecycle {
    fn default() -> Self {
        Self {
            state: LifecycleState::Seeded,
            history: Vec::new(),
            error: None,
            attempts: 0,
        }
    }
}

/// One recorded transition. `at` is an RFC 3339 date-time string.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateTransition {
    pub state: LifecycleState,
    /// RFC 3339 date-time.
    pub at: String,
    pub attempt: u32,
}

/// The canonical 12-variant per-record state machine (REMEDIATION ITEM 6). Serialized
/// snake_case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleState {
    /// seed/taxonomy node + persona + template chosen; `record_id` minted.
    Seeded,
    /// USER turn synthesized AND passed the `UserTurnVerdict` gate.
    UserSynthesized,
    /// teacher called; `content` + `reasoning` + `reasoning_details` captured.
    AssistantGenerated,
    /// deterministic Verifier rail incl. the `reasoning_present` hard gate.
    Verified,
    /// JudgePanel sealed pass + SP/BTS + `n_eff` computed.
    Judged,
    /// judge emitted `Revise`; bounded single retry in flight.
    Revising,
    /// parked for human/verifier adjudication (correlated-judge / Escalate).
    NeedsReview,
    /// verdict admit (terminal-good for the grade rail).
    Admitted,
    /// verdict reject; retained for DPO `rejected` + bad_patterns + judge audit.
    Rejected,
    /// projected to target template(s) per `CotPolicy`.
    Formatted,
    /// written into a versioned shard; `dataset_version` + Hub SHA attached.
    Exported,
    /// terminal-until-requeue; carries last error + attempt count.
    Error,
}
