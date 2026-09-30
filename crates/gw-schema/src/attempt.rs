//! Versioned evidence for physical model transmissions. No request or response text is stored.
use serde::{Deserialize, Serialize};

/// Cooperative capability declared by the actual injected client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AccountingCapability {
    /// Physical transmission coverage has not been established.
    #[default]
    Unknown,
    /// This implementation performs no model requests.
    NoModelRequests,
    /// Each physical transmission uses the v1 observer with required call context.
    PhysicalAttemptsV1,
}

/// Functional lane making a model request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptRole {
    /// Assistant generation.
    Teacher,
    /// Candidate grading.
    Judge,
    /// Diversity embeddings.
    Embedding,
}

/// Why the engine requested this model operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptPurpose {
    /// Initial assistant generation.
    Initial,
    /// Larger-token retry after truncated reasoning.
    TruncationRetry,
    /// Revised candidate generation.
    Revision,
    /// One panel member's grade.
    Grade,
    /// Candidate diversity gate, before teacher dispatch.
    CandidateQc,
    /// Add a newly admitted prompt to the diversity corpus.
    AdmittedPrior,
    /// Rebuild an admitted prompt's diversity vector on launch.
    ResumePrior,
}

/// Ownership and purpose supplied independently of serialized model requests.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttemptContext {
    /// Stable run identity.
    pub run_id: String,
    /// Fresh launch identity; it does not prove process liveness or death.
    pub launch_id: String,
    /// Shard when known, including before a record exists.
    pub shard: Option<i64>,
    /// Intended record identity; no record foreign key is required.
    pub record_id: Option<String>,
    /// Model lane.
    pub role: AttemptRole,
    /// Operation within that lane.
    pub purpose: AttemptPurpose,
}

/// Pre-send evidence committed immediately before one physical POST.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttemptIntent {
    /// Receipt contract version, currently 1.
    pub version: u32,
    /// Run/launch ownership and purpose.
    pub context: AttemptContext,
    /// BLAKE3 digest of the exact serialized request body.
    pub request_digest: String,
    /// Zero-based intentional provider retry ordinal for this logical call.
    pub retry_ordinal: u32,
    /// Requested model identifier; no revision is inferred.
    pub requested_model: String,
    /// Configured endpoint origin and path, with credentials/query/fragment removed.
    pub endpoint: String,
}

/// Optional reported cost. Invalid evidence is neither missing nor a known zero.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "status", content = "value", rename_all = "snake_case")]
pub enum ReportedCost {
    /// No cost field was reported.
    #[default]
    Missing,
    /// Finite, nonnegative provider-reported dollars, including zero.
    Known(f64),
    /// A cost field was present but could not establish valid dollars.
    Invalid,
}

/// Independently normalized cumulative usage and response identifiers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct AttemptMetadata {
    /// Cumulative prompt tokens when valid and present.
    pub prompt_tokens: Option<u64>,
    /// Cumulative completion tokens when valid and present.
    pub completion_tokens: Option<u64>,
    /// Cumulative total tokens when valid and present.
    pub total_tokens: Option<u64>,
    /// Cumulative reasoning tokens when valid and present.
    pub reasoning_tokens: Option<u64>,
    /// Optional provider-reported dollars.
    pub cost_usd: ReportedCost,
    /// Response or generation identifier reported by the backend.
    pub response_id: Option<String>,
    /// Resolved model identifier, without invented revision data.
    pub model: Option<String>,
    /// Reported upstream provider identifier.
    pub provider: Option<String>,
    /// Names of malformed metadata fields; no raw response is retained.
    pub invalid_fields: Vec<String>,
}

/// One ordered metadata persistence operation, including its durable validation result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttemptObservation {
    /// Zero-based sequence within one physical attempt; exact write retries reuse this value.
    pub sequence: u64,
    /// Cumulative fields supplied by this observation, without summing repeated measurements.
    pub metadata: AttemptMetadata,
    /// Contradictions introduced by this operation. Exact retries return the same result.
    pub conflicts: Vec<String>,
}

/// Observed end of the HTTP/SSE exchange, separate from output usability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransportOutcome {
    /// Successful HTTP response and complete body or SSE DONE sentinel.
    Complete,
    /// A non-success HTTP response was returned.
    HttpError,
    /// The connection or body failed before transport completion.
    Failed,
}

/// Terminal transport observation. Dropped requests have no settlement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportSettlement {
    /// How client-side consumption ended.
    pub outcome: TransportOutcome,
    /// HTTP status when response headers were observed.
    pub http_status: Option<u16>,
    /// Wall-clock milliseconds since durable intent, not GPU compute time.
    pub elapsed_ms: u64,
}

/// Higher-layer interpretation after transport observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputInterpretation {
    /// A teacher turn, grade, or vector decoded successfully.
    Accepted,
    /// Model output was malformed, empty, or otherwise unusable.
    Invalid,
    /// Reasoning or grading output reached its token cap.
    Truncated,
    /// Transport did not yield a complete interpretable response.
    Failed,
}

/// A durable physical-attempt receipt; missing settlement means unresolved execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttemptReceipt {
    /// Durable unique identity allocated before transmission.
    pub attempt_id: String,
    /// Immutable pre-send evidence.
    pub intent: AttemptIntent,
    /// Latest cumulative metadata; never sum snapshots from one attempt.
    pub metadata: AttemptMetadata,
    /// Ordered operations retained to distinguish fresh evidence from exact write retries.
    pub observations: Vec<AttemptObservation>,
    /// Contradictory field names; a conflicted receipt cannot establish complete accounting.
    pub conflicts: Vec<String>,
    /// Missing after process loss, cancellation by drop, or settlement failure.
    pub transport: Option<TransportSettlement>,
    /// Optional higher-level interpretation, separate from transport completion.
    pub interpretation: Option<OutputInterpretation>,
}

/// Evidence about execution before this run's first observed launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccountingHistory {
    /// Existing creation APIs cannot establish complete historical accounting.
    Unknown,
    /// Reserved for an atomic run-creation and coverage-registration boundary.
    RecordedFromCreation,
}

/// Coverage assessed from actual clients at one engine launch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchCoverage {
    /// Coverage contract version, currently 1.
    pub version: u32,
    /// Stable run identity.
    pub run_id: String,
    /// Fresh durable launch identity.
    pub launch_id: String,
    /// Historical completeness is independent of this launch's lane capabilities.
    pub history: AccountingHistory,
    /// Actual teacher client's capability.
    pub teacher: AccountingCapability,
    /// Actual judge client's capability.
    pub judge: AccountingCapability,
    /// Actual embedding client's capability.
    pub embedding: AccountingCapability,
}
