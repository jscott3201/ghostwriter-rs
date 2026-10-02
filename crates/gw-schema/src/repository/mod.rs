//! Portable declarations for repository episodes. No reference is opened and no execution is trusted.
use crate::{
    ContentDigest, Declaration, ExecutionEvidence, Message, ModelReference, NamespacedTaskId,
    ReviewedTaskRights, TaskSource, TaskSplit, VerificationOutcome,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

mod identity;
mod json;
mod validation;
pub use json::strict_repository_json;

/// Maximum complete request, artifact, or output frame, including a terminating newline.
pub const REPOSITORY_EPISODE_MAX_BYTES: usize = 32 * 1024 * 1024;
/// A protocol error with no caller-supplied content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RepositoryEpisodeError(pub(crate) &'static str);
impl RepositoryEpisodeError {
    /// Create a fixed protocol diagnostic; caller-supplied content cannot be borrowed here.
    #[must_use]
    pub const fn new(message: &'static str) -> Self {
        Self(message)
    }
}
impl std::fmt::Display for RepositoryEpisodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for RepositoryEpisodeError {}
type Result<T> = std::result::Result<T, RepositoryEpisodeError>;

/// Explicit algorithm of a declared immutable Git object ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepositoryRevisionAlgorithm {
    /// Forty lowercase hex digits.
    GitSha1,
    /// Sixty-four lowercase hex digits.
    GitSha256,
}
/// An immutable revision declaration, never a branch name or a fetched object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepositoryRevision {
    /// Git object hash algorithm.
    pub algorithm: RepositoryRevisionAlgorithm,
    /// Lowercase full object ID.
    pub hex: String,
}
/// Declared environment identity; no recipe is executed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepositoryEnvironment {
    /// Digest of producer-defined environment bytes.
    pub digest: ContentDigest,
    /// Explicit platform declaration.
    pub platform: String,
    /// Opaque recipe identity, not executable content.
    pub recipe: String,
}
/// Redacted obligations. Test contents and reference fixes have no fields in this contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepositoryPrivateContract {
    /// Digest of the private contract, without its contents.
    pub digest: ContentDigest,
    /// Exact externally declared required case IDs; order is retained.
    pub required_test_ids: Vec<String>,
}
/// A public repository task with explicit unknown immutable inputs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepositoryTask {
    /// Human/source label; distinct from the derived semantic identity.
    pub task_id: String,
    /// Existing source and citation declarations; blank revision leaves identity pending.
    pub source: TaskSource,
    /// Supplied rights declarations, excluded from task semantics.
    pub rights: ReviewedTaskRights,
    /// Declared corpus group, excluded from task semantics.
    pub group: NamespacedTaskId,
    /// Declared split, excluded from task semantics.
    pub split: TaskSplit,
    /// Public problem text, preserved exactly.
    pub problem: String,
    /// Credential-free opaque repository locator, never fetched.
    pub repository: ModelReference,
    /// Original upstream base, or explicitly unknown.
    pub upstream_base: Option<RepositoryRevision>,
    /// Prepared workspace base seen by the actor, which may differ after setup.
    pub actor_baseline: Option<RepositoryRevision>,
    /// Environment identity, or explicitly unknown.
    pub environment: Option<RepositoryEnvironment>,
    /// Redacted private obligations, or explicitly unknown.
    pub private_contract: Option<RepositoryPrivateContract>,
}
/// The only supported regular file modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RepositoryFileMode {
    /// Ordinary regular file.
    #[serde(rename = "100644")]
    Regular,
    /// Executable regular file.
    #[serde(rename = "100755")]
    Executable,
}
/// Exact state before or after a path change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RepositoryFileState {
    /// The path does not exist.
    Absent,
    /// Regular UTF-8 file. Newlines and lack of final newline are significant.
    Text {
        /// Exact UTF-8 text bytes; binary NUL content is unsupported.
        text: String,
        /// Supported regular-file mode.
        mode: RepositoryFileMode,
    },
}
/// A complete change to one relative path; rename is represented as deletion plus addition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepositoryFileChange {
    /// Portable relative UTF-8 path, validated without filesystem access.
    pub path: String,
    /// Exact baseline state.
    pub before: RepositoryFileState,
    /// Exact resulting state.
    pub after: RepositoryFileState,
}
/// Producer declaration about whole-workspace enumeration, not a local observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepositoryEnumeration {
    /// Producer declares all changes, including untracked files, were enumerated.
    Complete,
    /// Producer cannot supply a complete enumeration.
    Incomplete,
}
/// An unsupported change category, retained without pretending to have captured its bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepositoryUnsupportedChange {
    /// Binary file content.
    Binary,
    /// Symbolic link.
    Symlink,
    /// Nested repository or submodule.
    Submodule,
    /// Path is not valid UTF-8.
    NonUtf8Path,
}
/// Declared full regular-file delta. Unsupported changes prevent a candidate identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepositoryDelta {
    /// Whether the producer declares a complete enumeration.
    pub enumeration: RepositoryEnumeration,
    /// Changes sorted by exact path during capture; trajectory order is never sorted.
    pub changes: Vec<RepositoryFileChange>,
    /// Unsupported categories actually encountered; an empty set declares none.
    pub unsupported: Vec<RepositoryUnsupportedChange>,
}
/// Candidate material, independent of generation claims and evaluator declarations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepositoryCandidate {
    /// Explicit attempt identifier.
    pub attempt: String,
    /// Complete ordered canonical conversation, including reasoning and raw arguments.
    pub messages: Vec<Message>,
    /// Full tool definitions. Null means unavailable; empty means no tools declared.
    pub tools: Option<Vec<Value>>,
    /// Whether the producer declares the entire trajectory was captured.
    pub trajectory_complete: bool,
    /// Complete declared file delta relative to the prepared actor baseline.
    pub delta: RepositoryDelta,
}
/// Usage values are supplied claims. Unknown and known zero have different encodings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepositoryUsage {
    /// Input token count, if reported.
    pub input_tokens: Option<u64>,
    /// Output token count, if reported.
    pub output_tokens: Option<u64>,
    /// Reasoning token count, if reported.
    pub reasoning_tokens: Option<u64>,
    /// Finite nonnegative USD cost, if reported.
    pub cost_usd: Option<f64>,
}
/// Supplied generation details, without physical-attempt or loaded-model authority.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepositoryGeneration {
    /// Arbitrary producer model claims or explicit unknown.
    pub model: Declaration<Value>,
    /// Arbitrary producer serving claims or explicit unknown.
    pub serving: Declaration<Value>,
    /// Exact structured settings, including finite fractional values.
    pub settings: Value,
    /// Supplied usage, with absent values kept unknown.
    pub usage: RepositoryUsage,
}
/// Original publisher report plus an optional normalized diagnostic projection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepositoryPublisherReport {
    /// Publisher label; not a trust anchor.
    pub publisher: String,
    /// Opaque source reference, retained without opening it.
    pub source_ref: Option<String>,
    /// Original report JSON, including raw per-case statuses. No private-content detection is implied.
    pub raw: Value,
    /// Optional supplied projection for the existing factual declaration diagnostic.
    pub execution: Option<ExecutionEvidence>,
}
/// Version-one capture request; all embedded material is received as a declaration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepositoryEpisodeRequest {
    /// Only version 1 is supported.
    pub version: u32,
    /// Public task and provenance declarations.
    pub task: RepositoryTask,
    /// Candidate contents and completeness declarations.
    pub candidate: RepositoryCandidate,
    /// Generation and usage claims.
    pub generation: RepositoryGeneration,
    /// Attached original evaluation report, if any.
    pub report: Option<RepositoryPublisherReport>,
}
/// Independently recomputed identities and explicit reasons for an incomplete supported identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepositoryEpisodeIdentities {
    /// BLAKE3 task semantics identity, absent when immutable inputs are missing.
    pub task: Option<String>,
    /// BLAKE3 task/attempt/trajectory/tools/delta identity, absent when unsupported or incomplete.
    pub candidate: Option<String>,
    /// BLAKE3 identity of every canonical request declaration, including reports and usage.
    pub capture: String,
    /// Stable, non-content-bearing incompleteness reason codes.
    pub pending: Vec<String>,
}
/// Relationship of supplied report bindings to the candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepositoryReportBinding {
    /// No normalized report was supplied.
    Absent,
    /// Candidate/task identity is incomplete.
    Pending,
    /// Supplied task and attempt match but candidate identity differs.
    Stale,
    /// Supplied task or attempt differs.
    Foreign,
    /// All declared binding fields match.
    Matched,
}
/// Declaration-only diagnostic. This protocol cannot grant execution or training authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepositoryEpisodeAssessment {
    /// Report binding diagnostic, independently checked before interpreting reported outcomes.
    pub report_binding: RepositoryReportBinding,
    /// Existing factual interpretation of the supplied report only.
    pub declared_outcome: VerificationOutcome,
    /// Always Unknown; no local qualified execution occurs in this protocol.
    pub observed_execution: VerificationOutcome,
    /// Always false; capture and consistency checking are not training admission.
    pub training_eligible: bool,
}
/// Complete portable persistence boundary, with identities and diagnostics checked on verification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepositoryEpisodeArtifact {
    /// Only version 1 is supported.
    pub version: u32,
    /// Full received declarations in canonical form.
    pub request: RepositoryEpisodeRequest,
    /// Recomputed content identities.
    pub identities: RepositoryEpisodeIdentities,
    /// Recomputed declaration-only diagnostic.
    pub assessment: RepositoryEpisodeAssessment,
}
/// Content-free successful verification receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepositoryEpisodeReceipt {
    /// Only version 1 is supported.
    pub version: u32,
    /// Verified identities and pending reason codes.
    pub identities: RepositoryEpisodeIdentities,
    /// Recomputed declaration-only diagnostic.
    pub assessment: RepositoryEpisodeAssessment,
}
