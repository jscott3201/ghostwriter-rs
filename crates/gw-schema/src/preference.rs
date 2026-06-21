//! DPO-export shape (DATA-SCHEMA §6.3, B13). `reasoning` is an OPTIONAL SIBLING, NEVER
//! inlined into `content` (INVARIANT a).

use serde::{Deserialize, Serialize};

use crate::message::Message;

/// A `(chosen, rejected)` preference pair for best-of-k siblings sharing
/// `prompt_hash == sibling_group_id`. Produced by the `DatasetExporter` in DPO mode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreferenceRecord {
    /// Shared prompt turns (pairing key = `prompt_hash == sibling_group_id`).
    pub prompt: Vec<Message>,
    /// Admitted sibling.
    pub chosen: PreferenceSide,
    /// Retained sibling, ordered by the bias-corrected aggregate.
    pub rejected: PreferenceSide,
    /// `== sibling_group_id`; the pairing key.
    pub prompt_hash: String,
}

/// One side of a [`PreferenceRecord`]. `reasoning` is an OPTIONAL sibling, NEVER inlined.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreferenceSide {
    /// Clean final answer.
    pub content: String,
    /// OPTIONAL sibling CoT; NEVER inlined into `content`. (INVARIANT a)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    pub record_id: String,
    /// `judging.aggregate` (length-controlled + same-family-excluded; already bias-corrected).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aggregate: Option<f64>,
}
