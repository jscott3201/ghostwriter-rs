//! Versioned, independently supplied outcome evidence for a frozen evaluation corpus.
//!
//! These declarations make provenance auditable; they cannot prove evaluator independence or
//! blinding. One envelope carries one metric and reference protocol for every outcome. Strict
//! decoding rejects per-candidate contract overrides and unsupported reference sources.

use gw_schema::TrainingRecord;
use serde::{Deserialize, Serialize};

use crate::{EvalError, Result};

/// The supported outcome-envelope version.
pub const OUTCOME_EVIDENCE_VERSION: u32 = 1;

/// Independent measurements and their exact, frozen evaluation population.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeEvidence {
    /// Format version; currently [`OUTCOME_EVIDENCE_VERSION`].
    pub version: u32,
    /// Every declared member must belong to this run.
    pub run_id: String,
    /// Every declared member must belong to this training area.
    pub training_area: String,
    /// One metric contract shared by every candidate outcome.
    pub metric: OutcomeMetric,
    /// One reference contract shared by every candidate outcome.
    pub provenance: OutcomeProvenance,
    /// The sampling assumption needed by the confidence bound; declared, not independently proven.
    pub sampling_assumption: SamplingAssumption,
    /// Exact frozen membership, including candidates whose outcome is unknown or missing.
    pub corpus: Vec<CandidateBinding>,
    /// Measurements joined to `corpus` by exact identity. Missing entries never shrink the corpus.
    pub outcomes: Vec<CandidateOutcome>,
}

impl OutcomeEvidence {
    /// Decode an envelope, rejecting unknown fields, duplicate fields and unsupported variants.
    ///
    /// # Errors
    /// Returns [`EvalError::OutcomeEvidenceParse`] on malformed JSON or an unsupported shape.
    /// Semantic validity is checked by [`crate::separation::analyze`].
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).map_err(EvalError::OutcomeEvidenceParse)
    }
}

/// A bounded outcome metric. Every known value must lie in `[0, 1]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeMetric {
    /// Nonempty metric name.
    pub name: String,
    /// Nonempty revision of the metric definition.
    pub version: String,
    /// Orientation used by the paired comparison.
    pub direction: MetricDirection,
}

/// Supported metric orientation. Convert lower-is-better measurements before supplying evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricDirection {
    /// Larger outcome values are better.
    HigherIsBetter,
}

/// The externally declared origin of the outcome labels.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeProvenance {
    /// Accepted independent-reference category; judge aggregates are not a reference source.
    pub source: ReferenceSource,
    /// Nonempty evaluator or adjudication protocol revision.
    pub protocol_revision: String,
    /// Digest of the frozen reference artifact. This declaration does not load that artifact.
    pub reference_artifact_digest: ReferenceDigest,
}

/// Accepted sources for independently supplied outcomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceSource {
    /// A deterministic task reference, such as frozen test answers.
    DeterministicTaskReference,
    /// Labels produced under a declared adjudication protocol.
    AdjudicatedReference,
}

/// Content digest identifying the reference artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceDigest {
    /// Digest algorithm.
    pub algorithm: DigestAlgorithm,
    /// Full 64-character lowercase hexadecimal digest.
    pub hex: String,
}

/// Supported 256-bit digest algorithms for reference declarations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DigestAlgorithm {
    /// SHA-256.
    Sha256,
    /// BLAKE3 with its default 256-bit output.
    Blake3,
}

/// Assumption under which the reported confidence bound applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingAssumption {
    /// Distinct prompt hashes represent independent draws from the declared evaluation population.
    IndependentPrompts,
}

/// Exact identity of one declared candidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateBinding {
    /// Stable record identity.
    pub record_id: String,
    /// Run identity, checked independently of the content hash.
    pub run_id: String,
    /// Training area, checked independently of the content hash.
    pub training_area: String,
    /// Full recomputed prompt hash.
    pub prompt_hash: String,
    /// Full recomputed [`gw_storage::record_hash`], covering the record's content projection.
    /// This hash excludes mutable lifecycle, judging and generation metadata by storage contract.
    pub record_hash: String,
}

impl CandidateBinding {
    /// Bind a record using freshly computed hashes, without trusting its stored hash fields.
    ///
    /// # Errors
    /// Returns [`EvalError::Storage`] if canonical content serialization fails.
    pub fn from_record(record: &TrainingRecord) -> Result<Self> {
        Ok(Self {
            record_id: record.record_id.clone(),
            run_id: record.provenance.run_id.clone(),
            training_area: record.training_area.clone(),
            prompt_hash: gw_storage::prompt_hash(&record.messages)?,
            record_hash: gw_storage::record_hash(record)?,
        })
    }
}

/// One candidate's independent outcome under the envelope's single metric/reference contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateOutcome {
    /// Exact candidate binding; must equal its declared corpus member.
    pub candidate: CandidateBinding,
    /// A known bounded measurement or explicit unknown.
    pub outcome: ReferenceOutcome,
}

/// Missing evidence is distinct from a known zero outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReferenceOutcome {
    /// A finite measurement in `[0, 1]`.
    Known {
        /// Independent outcome value.
        value: f64,
    },
    /// The candidate has no usable independent measurement.
    Unknown {
        /// Nonempty explanation for the unknown result.
        reason: String,
    },
}

/// Qualification settings for independent prompt-level paired comparisons.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeConfig {
    /// Minimum distinct fully evaluated prompts; must be positive. Default `30`.
    pub min_evaluated_prompts: usize,
    /// One-sided confidence level in `(0, 1)`. Default `0.95`.
    pub confidence_level: f64,
}

impl Default for OutcomeConfig {
    fn default() -> Self {
        Self {
            min_evaluated_prompts: 30,
            confidence_level: 0.95,
        }
    }
}

/// Result of evaluating the declared selection policy against independent outcomes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutcomeReport {
    /// Qualification or the class of reason preventing it.
    pub status: OutcomeStatus,
    /// Evaluated selection rule and population.
    pub policy: SelectionPolicy,
    /// Conservative one-sided bound used for the paired gaps in `[-1, 1]`.
    pub method: BoundMethod,
    /// Valid confidence level; `null` for invalid configuration.
    pub confidence_level: Option<f64>,
    /// Required independent prompt count.
    pub min_evaluated_prompts: usize,
    /// Declared run, if evidence was supplied.
    pub run_id: Option<String>,
    /// Declared area, if evidence was supplied.
    pub training_area: Option<String>,
    /// Declared metric, if evidence was supplied.
    pub metric: Option<OutcomeMetric>,
    /// Declared reference provenance; this is not proof of independence or blinding.
    pub provenance: Option<OutcomeProvenance>,
    /// Declared sampling assumption; repeated prompts never increase the sample count.
    pub sampling_assumption: Option<SamplingAssumption>,
    /// Exact corpus and measurement coverage.
    pub coverage: OutcomeCoverage,
    /// Numeric results; `null` when no complete, valid eligible corpus was evaluated.
    pub statistics: Option<OutcomeStatistics>,
    /// Typed reasons preventing qualification; empty only on qualification.
    pub reasons: Vec<OutcomeReason>,
}

/// Qualification states; invalid input is distinct from an unsuccessful valid analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeStatus {
    /// Enough distinct prompts and a strictly positive conservative lower bound.
    Qualified,
    /// Complete evidence, but too few prompts or no positive lower bound.
    Inconclusive,
    /// Absent, incomplete or unknown outcomes, or no eligible prompt population.
    InsufficientEvidence,
    /// Invalid configuration, identity, contract, value or arithmetic.
    InvalidEvidence,
}

/// The evaluated policy does not include the engine's full admission or revision rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionPolicy {
    /// Within all-pass prompt groups, choose the highest finite judge score, breaking ties by the
    /// lowest completion index. Evaluated candidates require distinct, present completion indices.
    JudgeArgmaxLowestCompletionIndexV1,
}

/// Confidence-bound method recorded in each report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundMethod {
    /// `mean_gap - sqrt(2 * ln(1 / alpha) / n)` for independent prompt gaps in `[-1, 1]`.
    OneSidedHoeffdingPairedGap,
}

/// Counts are actual observations; absent numerical comparisons live in `statistics: null`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutcomeCoverage {
    /// Records available after the caller's store filter.
    pub scanned_records: usize,
    /// Explicitly declared corpus members, including unknown or missing labels.
    pub declared_records: usize,
    /// Declared members found with matching identities in the supplied records.
    pub matched_records: usize,
    /// Declared members with a finite bounded known outcome.
    pub known_outcomes: usize,
    /// Declared members with an explicit unknown outcome.
    pub unknown_outcomes: usize,
    /// Declared members without any outcome entry.
    pub missing_outcomes: usize,
    /// Distinct all-pass prompts with at least two scored candidates in the declared corpus.
    pub eligible_prompts: usize,
    /// Distinct prompts actually compared; zero when the declared corpus is incomplete or invalid.
    pub evaluated_prompts: usize,
}

/// Independent outcome summaries with equal weight for each distinct prompt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutcomeStatistics {
    /// Mean independent outcome of the selected candidate.
    pub selector_mean: f64,
    /// Mean uniform-random expected outcome over the same candidates in each prompt.
    pub random_mean: f64,
    /// Mean of the paired prompt differences, each in `[-1, 1]`.
    pub mean_gap: f64,
    /// Conservative one-sided lower bound; it may lie below `-1`.
    pub lower_bound: f64,
}

/// The exact cause of invalid, insufficient or inconclusive evidence.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OutcomeReason {
    /// No independent evidence envelope was provided.
    MissingEvidence,
    /// Unsupported envelope version.
    UnsupportedVersion { version: u32 },
    /// An analysis setting is outside its permitted domain.
    InvalidConfiguration { field: String },
    /// A required metric/provenance/binding field is empty or malformed.
    InvalidContract { field: String },
    /// The declared corpus lists the same record more than once.
    DuplicateCorpusMember { record_id: String },
    /// The supplied record slice has ambiguous record identity.
    DuplicateRecord { record_id: String },
    /// A declared member is absent from the filtered input.
    MissingRecord { record_id: String },
    /// A run, area, prompt or content identity does not match.
    IdentityMismatch { record_id: String, field: String },
    /// Canonical record hashing failed.
    RecordHashFailed { record_id: String },
    /// Multiple outcome entries refer to one record, even if their values agree.
    DuplicateOutcome { record_id: String },
    /// An outcome refers to a candidate outside the declared corpus.
    UnexpectedOutcome { record_id: String },
    /// A declared member lacks a measurement entry.
    MissingOutcome { record_id: String },
    /// A declared member has an explicitly unknown measurement.
    UnknownOutcome { record_id: String },
    /// A known independent outcome is non-finite or outside `[0, 1]`.
    InvalidOutcome { record_id: String },
    /// A selector score is non-finite or outside `[0, 1]`.
    InvalidJudgeScore { record_id: String },
    /// An evaluated candidate lacks the index needed to reproduce the policy.
    MissingCompletionIndex { record_id: String },
    /// Two scored candidates in the same prompt have an ambiguous completion index.
    DuplicateCompletionIndex {
        prompt_hash: String,
        completion_index: u32,
    },
    /// No declared prompt meets the evaluated population's eligibility requirements.
    NoEligiblePrompts,
    /// Complete comparisons exist, but the required prompt count is not reached.
    TooFewEvaluatedPrompts { required: usize, observed: usize },
    /// The conservative confidence bound does not establish a positive outcome difference.
    NonPositiveLowerBound,
    /// A derived numerical result is not finite or exceeds its mathematical domain.
    InvalidArithmetic,
}
