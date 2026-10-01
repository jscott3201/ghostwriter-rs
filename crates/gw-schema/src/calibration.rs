//! Strict supplied offline calibration evidence. These types provide no collection authority.
//! Numerical fields use exact finite binary64 objects; opaque request and response text is untouched.
use crate::CandidateBinding;
use serde::{Deserialize, Serialize};

/// Current offline evidence and snapshot version.
pub const CALIBRATION_VERSION: u32 = 1;
/// Group-equal squared error against one declared quality target, not a probability forecast.
pub const CALIBRATION_METHOD: &str = "group_mean_squared_score_error_v1";
/// Fixed exp backend and accumulation order. Refitting qualification is target/build-specific.
pub const CALIBRATION_NUMERICAL_RECIPE: &str =
    "libm-0.2.16-fixed-rust-exp-canonical-group-candidate-sequential-panel-sum-v1";

/// A single higher-is-better unit-interval reference target and measurement protocol.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationTarget {
    /// Target name; every row uses this exact target.
    pub name: String,
    /// Target semantics revision.
    pub version: String,
    /// Independent reference collection/adjudication protocol revision.
    pub protocol_revision: String,
    /// Only higher-is-better unit-interval targets are supported.
    pub semantics: CalibrationTargetSemantics,
}

/// Explicit target semantics; probability calibration is not implied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationTargetSemantics {
    /// A quality measurement bounded by zero and one, with larger values preferred.
    HigherIsBetterUnitInterval,
}

/// One full production projection and its independently recomputed identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationJudgeContract {
    /// Canonical production JSON retained as opaque text, never parsed and reserialized.
    pub projection_json: String,
    /// Domain-separated digest of this declaration, including exact sampling controls.
    pub identity: String,
    /// Explicit production temperature, including its saved bits.
    #[serde(with = "crate::finite_numbers::scalar")]
    pub temperature: f64,
    /// Optional production top-p, including its saved bits.
    #[serde(with = "crate::finite_numbers::optional")]
    pub top_p: Option<f64>,
}

/// Persisted description of a stable panel. A supplied instance cannot construct live authority.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationPanel {
    /// Panel declaration version.
    pub version: u32,
    /// Exact training area.
    pub training_area: String,
    /// Production candidate renderer contract.
    pub render_contract: String,
    /// Ordered full empty-candidate judge declarations, aligned with every observation vector.
    pub judges: Vec<CalibrationJudgeContract>,
}

/// Declared collection state, independent of whether the text parses successfully.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationCollection {
    /// The submitter supplies a response; this does not prove stream success or termination.
    Observed,
    /// No response was collected.
    Missing,
    /// The submitter declares collection failure.
    Failed,
    /// The collection outcome is unknown.
    Unknown,
}

/// Claimed interpreted verdict; checked against the shared production parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationVerdict {
    /// Production Accept, including the raw alias Admit.
    Accept,
    /// Production Revise.
    Revise,
    /// Production Reject.
    Reject,
    /// Production Uncertain, including unknown raw verdict tokens.
    Uncertain,
}

/// Exact concatenated response text. Arbitrary JSON-like text stays opaque.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationRawResponse {
    /// The production raw.response text.
    pub response: String,
}

/// A supplied observation bound to its candidate, stable column, actual request and payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationObservation {
    /// Exact candidate identity, including run and freshly recomputed content hashes.
    pub candidate: CandidateBinding,
    /// Identity of this position's empty-candidate declaration.
    pub column_identity: String,
    /// Full candidate-specific request projection rebuilt through the production builder.
    pub request: CalibrationJudgeContract,
    /// Supported production interpretation revision.
    pub interpretation_version: u32,
    /// Supplied collection state; never inferred from parsing.
    pub collection: CalibrationCollection,
    /// Exact response, or explicit absence.
    pub raw: Option<CalibrationRawResponse>,
    /// Digest of the exact optional response, independently recomputed.
    pub payload_identity: String,
    /// Claimed normalized score. This is compared exactly, never normalized a second time.
    #[serde(with = "crate::finite_numbers::optional")]
    pub score: Option<f64>,
    /// Claimed production verdict, or no interpreted result.
    pub verdict: Option<CalibrationVerdict>,
    /// Digest of all other observation fields, independently recomputed.
    pub identity: String,
}

/// Known reference quality or an explicit unknown; absence is never a zero target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum CalibrationLabel {
    /// A supplied finite unit-interval reference measurement.
    Known {
        /// Higher-is-better reference quality.
        #[serde(with = "crate::finite_numbers::scalar")]
        value: f64,
    },
    /// No reference quality has been established.
    Unknown {
        /// Supplied reason for the missing measurement.
        reason: String,
    },
}

/// One explicit corpus member and its globally scoped prompt group.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationRow {
    /// Full identity of the declared candidate.
    pub candidate: CandidateBinding,
    /// Globally scoped task/prompt group, not scoped to a run.
    pub prompt_group: String,
    /// Independent reference measurement under the envelope target/protocol.
    pub label: CalibrationLabel,
    /// Ordered observations, one per panel column. Missing cells make evidence incomplete.
    pub observations: Vec<CalibrationObservation>,
}

/// Only supplied, unverified runtime provenance is available in this software slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationRuntimeProvenance {
    /// Matching contracts establish applicability only; historical collection remains unverified.
    SuppliedUnverified,
}

/// Declared provenance. This does not establish independent tasks, blinding or truthful labels.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationProvenance {
    /// Source name or supplied artifact reference.
    pub source: String,
    /// Supplied digest of the reference collection artifact.
    pub reference_digest: String,
    /// Explicit declaration of independence/blinding and adjudication provenance.
    pub independence_blinding: String,
    /// Always supplied/unverified; no approved/verified state is supported.
    pub runtime: CalibrationRuntimeProvenance,
}

/// The assumed correlation policy actually sealed with this descriptive snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CalibrationCorrelation {
    /// A constant off-diagonal prior; no empirical estimator is activated.
    AssumedConstantRho {
        /// Finite prior obeying the production correlation policy.
        #[serde(with = "crate::finite_numbers::scalar")]
        rho: f64,
    },
}

/// Entire supplied evidence document. Fit and assessment use a common panel and target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationEvidence {
    /// Evidence wire contract version.
    pub version: u32,
    /// Explicit version of the globally scoped candidate-to-prompt-group map.
    pub group_map_version: u32,
    /// Fixed supported method name.
    pub method: String,
    /// Fixed supported numerical recipe.
    pub numerical_recipe: String,
    /// One reference target and protocol for every row.
    pub target: CalibrationTarget,
    /// Supplied provenance declaration.
    pub provenance: CalibrationProvenance,
    /// Full ordered stable panel declarations, compared against separately resolved live values.
    pub panel: CalibrationPanel,
    /// Digest of the full stable panel declaration.
    pub panel_identity: String,
    /// Finite nonnegative softmax coefficient; zero is an explicit equal-weight control.
    #[serde(with = "crate::finite_numbers::scalar")]
    pub beta: f64,
    /// Actual assumed prior, sealed only in the complete snapshot identity.
    pub correlation: CalibrationCorrelation,
    /// Fit corpus, canonicalized by global group and full candidate identity.
    pub fit: Vec<CalibrationRow>,
    /// Disjoint held-out corpus. Its labels never enter fitted weights or fit identity.
    pub assessment: Vec<CalibrationRow>,
}

/// The only result statuses available; none asserts empirical quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationStatus {
    /// Contradictory, stale, malformed or unsupported evidence.
    InvalidEvidence,
    /// The declared common population lacks usable labels or observations.
    IncompleteEvidence,
    /// Complete descriptive computation, with no empirical qualification or runtime authority.
    ComputedUnqualified,
}

/// Exact fitted values in explicit judge order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationFit {
    /// Digest of the fit method/target/panel/coefficient/membership evidence only.
    pub identity: String,
    /// Common complete candidate count.
    pub candidates: usize,
    /// Equal-weight prompt-group count.
    pub groups: usize,
    /// Group-equal squared score errors in panel order.
    #[serde(with = "crate::finite_numbers::vector")]
    pub losses: Vec<f64>,
    /// Strictly positive normalized weights in the same panel order.
    #[serde(with = "crate::finite_numbers::vector")]
    pub weights: Vec<f64>,
    /// Always zero: this method rejects incomplete populations instead of dropping rows.
    pub exclusions: usize,
}

/// Descriptive held-out results using frozen fitted weights, without confidence claims.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationAssessment {
    /// Complete held-out candidate count.
    pub candidates: usize,
    /// Equal-weight held-out group count.
    pub groups: usize,
    /// Individual held-out judge losses in panel order.
    #[serde(with = "crate::finite_numbers::vector")]
    pub losses: Vec<f64>,
    /// Group-equal squared error of each candidate's frozen-weight score.
    #[serde(with = "crate::finite_numbers::scalar")]
    pub weighted_score_loss: f64,
}

/// Sealed descriptive snapshot. Validate by exact refitting before trusting supplied results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationSnapshot {
    /// Snapshot wire version.
    pub version: u32,
    /// Must be ComputedUnqualified; other states cannot describe a complete snapshot.
    pub status: CalibrationStatus,
    /// Canonical full evidence, preserving exact numeric and opaque request/payload bytes.
    pub evidence: CalibrationEvidence,
    /// Fitted values whose identity excludes assessment inputs/results and assumed correlation.
    pub fit: CalibrationFit,
    /// Frozen-weight descriptive assessment.
    pub assessment: CalibrationAssessment,
    /// Digest of the entire snapshot except this field.
    pub identity: String,
}

/// Intake report; invalid/incomplete evidence never includes a usable snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationReport {
    /// Typed computation status.
    pub status: CalibrationStatus,
    /// Actionable evidence failures, empty for complete computation.
    pub reasons: Vec<String>,
    /// Present only after all declared cells are validated and computed.
    pub snapshot: Option<CalibrationSnapshot>,
}
