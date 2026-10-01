//! Versioned, inspectable preference evidence. Preparation and hashing live outside this I/O-free
//! contract. Deserialization alone does not certify a pair; use the engine preparation boundary.
use serde::{Deserialize, Serialize};

use crate::{
    CotPolicy, ExecutionEvidence, Judging, Lifecycle, Message, Provenance, ReasoningQuality,
    TaskProvenance, Verification, VerificationContract,
};

/// Supported preference policy, evidence and pair version. Material floating-point numbers use
/// canonical `{"binary64":"0123456789abcdef"}` bit objects on this wire contract. They remain
/// typed `f64` values in Rust. This preserves exact evidence across JSON parsing without changing
/// ordinary training-record serialization.
pub const PREFERENCE_VERSION: u32 = 1;

/// The first supported source declares ranking by the records' stored judge aggregate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreferenceSource {
    /// Judge ranking is not independent ground truth or proof of decision execution lineage.
    JudgeScoreRanking,
}

/// The declared direction of the judge aggregate comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreferenceDirection {
    /// The chosen aggregate must exceed the rejected aggregate by the strict policy margin.
    HigherAggregateIsChosen,
}

/// Ties never establish direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreferenceTiePolicy {
    /// Reject ties and any gap at or below the configured margin.
    Reject,
}

/// The supported policy conservatively requires decisive grading and verification evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreferenceUncertaintyPolicy {
    /// Reject review intent, review verdicts, missing decisive counts and uncertain votes or axes.
    RequireDecisive,
}

/// An explicit rule; its margin has no implied empirical quality qualification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreferencePolicy {
    /// Must equal [`PREFERENCE_VERSION`].
    pub version: u32,
    /// Declared preference source.
    pub source: PreferenceSource,
    /// Explicit score ordering.
    pub direction: PreferenceDirection,
    /// Caller-declared immutable protocol revision; not proof it produced the grades.
    pub protocol_revision: String,
    /// Supervised or stripped reasoning. Masked is unqualified for DPO likelihoods.
    pub cot_policy: CotPolicy,
    /// Required verification reasoning policy, checked against persisted facts.
    pub cot_required: bool,
    /// Strict lower bound on chosen minus rejected aggregate, finite and in `[0,1]`.
    #[serde(with = "crate::preference_numbers::margin")]
    pub minimum_margin: f64,
    /// Explicit tie handling.
    pub ties: PreferenceTiePolicy,
    /// Explicit uncertainty handling.
    pub uncertainty: PreferenceUncertaintyPolicy,
}

/// Material source evidence, frozen separately from the content identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreferenceSourceSnapshot {
    /// Exact record identity; the two sides must differ.
    pub record_id: String,
    /// Exact training-area identity.
    pub training_area: String,
    /// Run, teacher and user-task declarations, including the run identity.
    pub provenance: Provenance,
    /// Fresh content hash, including reasoning.
    pub record_hash: String,
    /// Fresh grouping hash; does not prove complete prefix equality.
    pub prompt_hash: String,
    /// Fresh answer-content hash, excluding reasoning.
    pub completion_hash: String,
    /// Original conversation, preserving structured reasoning and metadata before projection.
    pub original_messages: Vec<Message>,
    /// Complete material grading evidence, including the actual aggregate used for ranking.
    #[serde(with = "crate::preference_numbers::judge")]
    pub judging: Judging,
    /// Applied authority, observations, legacy checks, and review evidence.
    #[serde(with = "crate::preference_numbers::verifier")]
    pub verification: Verification,
    /// Task authority contract; absent historical contracts cannot authorize preparation.
    #[serde(with = "crate::preference_numbers::task")]
    pub verification_contract: Option<VerificationContract>,
    /// Reviewed source/group/split lineage, if present.
    pub task_provenance: Option<TaskProvenance>,
    /// Candidate-bound execution report, if present.
    pub execution_evidence: Option<ExecutionEvidence>,
    /// Optional reasoning quality evidence that can affect ranking.
    #[serde(with = "crate::preference_numbers::reasoning")]
    pub reasoning_quality: Option<ReasoningQuality>,
    /// Selection state and history; an Admit grade alone does not prove selection.
    pub lifecycle: Lifecycle,
}

/// Caller-supplied assessment inputs. Preparation re-captures both snapshots from actual records.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreferenceAssessment {
    /// Must equal [`PREFERENCE_VERSION`].
    pub version: u32,
    /// Explicit direction, margin and reasoning policy.
    pub policy: PreferencePolicy,
    /// Expected chosen-side evidence and score.
    pub chosen: PreferenceSourceSnapshot,
    /// Expected rejected-side evidence and score.
    pub rejected: PreferenceSourceSnapshot,
}

/// Physical receipt/output and decision-execution provenance are not currently bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreferenceBindingStatus {
    /// No immutable binding exists in current source records.
    Unbound,
}

/// Current records cannot distinguish natural stop from length truncation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreferenceTermination {
    /// Terminal message shape was checked, but stop/length metadata is unavailable.
    Unknown,
}

/// Inspectable proof inputs and explicit qualification limits for a prepared pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreferenceEvidence {
    /// Exact verified assessment, including both identities and actual material snapshots.
    pub assessment: PreferenceAssessment,
    /// Complete equal prefix before reasoning projection, including earlier assistant turns.
    pub original_prefix: Vec<Message>,
    /// Domain-separated canonical hash of that original prefix.
    pub original_prefix_hash: String,
    /// Domain-separated hash of the assessment and original prefix.
    pub decision_evidence_hash: String,
    /// Selection qualification enforced by the shared selected-admission predicate.
    pub chosen_selected: bool,
    /// Chosen stop/length metadata is not captured by the source envelope.
    pub chosen_termination: PreferenceTermination,
    /// Rejected stop/length metadata is not captured by the source envelope.
    pub rejected_termination: PreferenceTermination,
    /// Physical receipts do not currently bind accepted output digests.
    pub receipt_output_binding: PreferenceBindingStatus,
    /// A declared protocol revision does not establish producer decision execution.
    pub decision_execution_binding: PreferenceBindingStatus,
}

/// Prepared TRL conversational preference arrays, with reasoning as a separate sibling field.
/// This is a pure preparation result, not a published or adapter-qualified DPO artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreferenceRecord {
    /// Must equal [`PREFERENCE_VERSION`].
    pub version: u32,
    /// Canonical identity covering projection, source evidence and policy.
    pub pair_id: String,
    /// Shared prompt after the requested reasoning policy.
    pub prompt: Vec<Message>,
    /// One nonempty terminal assistant message after policy.
    pub chosen: Vec<Message>,
    /// One distinct nonempty terminal assistant message after policy.
    pub rejected: Vec<Message>,
    /// Frozen evidence and explicit unavailable provenance.
    pub evidence: PreferenceEvidence,
}
