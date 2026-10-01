//! Frozen descriptive results. Deserialization supplies no validation authority.
use crate::{ProtectedScreeningCoverage, ScreeningDeclaration, ScreeningRecordId, TaskSplit};
use serde::{Deserialize, Serialize};

/// Whether lexical work completed over all required supplied inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LexicalScreeningStatus {
    /// Complete declared coverage and no protected match. No semantic guarantee is implied.
    CompleteNoMatch,
    /// Complete declared coverage with protected matches quarantining existing groups.
    MatchQuarantined,
    /// Required coverage, lineage, supported payload or bounded work was incomplete.
    Incomplete,
}
/// Semantic screening has no implementation or empirical qualification in this lexical recipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticScreeningStatus {
    /// No semantic screening ran.
    NotRun,
}
/// Capture authority of this provider-free supplied-file API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScreeningPopulationCheck {
    /// Only the declared supplied population was checked; no current database query occurred.
    SuppliedFilesOnly,
}
/// Source-level coverage cannot establish the tokenizer/template's effective prompt separation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffectivePromptSeparation {
    /// Actual student tokenizer/template prompts were not checked across the full corpus.
    Unknown,
}
/// Exact representation scope covered by this source screening report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScreeningLexicalScope {
    /// Canonical source fields and shared renderer examples under the pinned export policy.
    CanonicalSourceAndPinnedExportPolicy,
}
/// A deterministic reason, using identifiers without protected text.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScreeningIssue {
    /// Stable machine-readable failure/limitation code.
    pub code: String,
    /// Record, set, field, or declaration identity; never protected payload text.
    pub subject: String,
}
/// Actual record/input binding, including declarations absent from the existing content hash.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScreeningInputBinding {
    /// Bound population member.
    pub record: ScreeningRecordId,
    /// Recomputed existing full content binding.
    pub record_hash: String,
    /// Exact exported typed conversation, task declarations/contract, area and verdict digest.
    /// Covers reasoning-detail metadata and tool turns; aggregate/cost/history remain excluded.
    pub export_projection_id: String,
    /// Actual message/tool shapes plus task/lineage/sibling/policy/eligibility binding;
    /// excludes publication history/timestamps.
    pub screening_input_id: String,
}
/// One connected component and supplied split decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenScreeningGroup {
    /// Stable component identity, retained on validated extensions.
    pub group_id: String,
    /// Canonically ordered member IDs, including excluded relatives.
    pub members: Vec<ScreeningRecordId>,
    /// Equal supplied split assignment, or None when missing/conflicting.
    pub split: Option<TaskSplit>,
    /// Protected or split/extension conflicts exclude the whole component.
    pub quarantined: bool,
    /// Canonical quarantine reasons.
    pub reasons: Vec<ScreeningIssue>,
}
/// One canonical grouping edge, without source text.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScreeningGroupEdge {
    /// Lexicographically smaller record identity.
    pub left: ScreeningRecordId,
    /// Lexicographically larger record identity.
    pub right: ScreeningRecordId,
    /// Declared task/source/parent/sibling, exact example, exact prompt, or lexical prompt.
    pub kind: String,
    /// Selected-target unit/segment match identities; empty for declared edges.
    pub evidence: Vec<LexicalScreeningEvidence>,
}
/// Independent segment-pair evidence; all qualifying n values are retained.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexicalScreeningEvidence {
    /// Stable source segment or complete-prompt unit identity.
    pub left: String,
    /// Stable protected/source segment or complete-prompt unit identity.
    pub right: String,
    /// None for exact whole structured prompts; Some(n) for a shingle set comparison.
    pub n: Option<u32>,
    /// Distinct shared n-token tuples; zero for a complete-prompt exact match.
    pub intersection: u64,
    /// Distinct union n-token tuples; zero for a complete-prompt exact match.
    pub union: u64,
}
/// A protected match quarantines the already connected component; it never creates an edge.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtectedScreeningMatch {
    /// Matched record, including excluded members.
    pub record: ScreeningRecordId,
    /// Matched canonical protected set.
    pub protected_set: String,
    /// Matched item identity.
    pub item_id: String,
    /// Canonical segment/prompt evidence.
    pub evidence: LexicalScreeningEvidence,
}
/// Text-free immutable protected input summary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtectedScreeningIdentity {
    /// Canonical set identity.
    pub canonical_id: String,
    /// Immutable declared revision.
    pub source_revision: String,
    /// Recomputed typed-content digest.
    pub content_digest: String,
    /// Full input identity, including actual contents and rights/coverage assertions.
    pub input_id: String,
    /// Whether rights, fields, languages, media and declared contents were complete.
    pub complete: bool,
    /// Supplied field/language/media declarations, without protected payload text.
    pub coverage: ProtectedScreeningCoverage,
    /// Whether a nonempty intended-use rights assertion was supplied; not legal certification.
    pub rights_declared: bool,
    /// Canonical supplied item IDs.
    pub item_ids: Vec<String>,
}
/// Available descriptive strata with unknown metadata explicitly represented by None.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScreeningRecordStratum {
    /// Captured record.
    pub record: ScreeningRecordId,
    /// Declared task label when available.
    pub task: Option<String>,
    /// Training area retained as a descriptive stratum, not a duplicate key.
    pub area: String,
    /// Reviewed domain, if present.
    pub domain: Option<String>,
    /// Reviewed difficulty label, if present.
    pub difficulty: Option<String>,
    /// Supplied teacher model identity.
    pub teacher: String,
    /// Supported normalized text token count, including independently retained reasoning fields;
    /// None when a resource failure prevented a complete count.
    pub tokens: Option<u64>,
}
/// Explicit reason a requested output member is excluded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScreeningExclusion {
    /// Requested member.
    pub record: ScreeningRecordId,
    /// Canonical reason codes, including incomplete plan, eligibility, split and quarantine.
    pub reasons: Vec<String>,
}
/// Descriptive report counts. A zero protected-match count certifies no semantic claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScreeningCounts {
    /// Every captured record in the declared runs, before eligibility filtering.
    pub population_records: u64,
    /// Connected components in the captured population.
    pub groups: u64,
    /// Components excluded by protected/split/extension conflicts.
    pub quarantined_groups: u64,
    /// Requested artifact members.
    pub requested_output_records: u64,
    /// Requested members meeting this source plan's eligibility rules.
    pub eligible_output_records: u64,
    /// Requested members with explicit exclusion reasons.
    pub excluded_output_records: u64,
    /// Supplied protected manifests, irrespective of completeness.
    pub supplied_protected_sets: u64,
    /// Canonical protected match evidence entries, including all qualifying n values.
    pub protected_matches: u64,
}
/// Content-addressed frozen plan and report. Validate against actual records/protected inputs;
/// parsing, self-reported statuses and digest strings do not confer authority.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenScreeningPlan {
    /// Supported plan version; must equal [`crate::SCREENING_PLAN_VERSION`].
    pub version: u32,
    /// Descriptive counts over this exact plan.
    pub counts: ScreeningCounts,
    /// Canonical supplied declaration, with exact policy float bits.
    pub declaration: ScreeningDeclaration,
    /// Canonical population IDs and per-record input bindings.
    pub population: Vec<ScreeningInputBinding>,
    /// Canonical union of text-field classes present in the entire captured population, including
    /// excluded records and top-level tool definitions. Every complete protected set must cover it.
    pub required_fields: Vec<crate::ScreeningField>,
    /// Identity over declaration and exact captured population inputs.
    pub screening_input_id: String,
    /// Policy identity independent of record/protected input content.
    pub policy_id: String,
    /// Full protected input identities, without protected text.
    pub protected_inputs: Vec<ProtectedScreeningIdentity>,
    /// Identity over all required/supplied protected input identities.
    pub protected_input_id: String,
    /// Canonical connected components.
    pub groups: Vec<FrozenScreeningGroup>,
    /// Canonical grouping/split identity independent of protected matching.
    pub grouping_id: String,
    /// All canonical grouping edges.
    pub edges: Vec<ScreeningGroupEdge>,
    /// All canonical protected matches; no protected content is copied.
    pub protected_matches: Vec<ProtectedScreeningMatch>,
    /// Completeness of descriptive lexical screening.
    pub lexical_status: LexicalScreeningStatus,
    /// Always NotRun in this version.
    pub semantic_status: SemanticScreeningStatus,
    /// Always SuppliedFilesOnly for this API.
    pub population_check: ScreeningPopulationCheck,
    /// Scope of the lexical result: canonical source segments and the pinned export policy.
    pub lexical_scope: ScreeningLexicalScope,
    /// Always Unknown: actual tokenizer/template prompt separation was not established.
    pub effective_prompt_separation: EffectivePromptSeparation,
    /// Missing evidence or unsupported resource/payload coverage.
    pub incomplete: Vec<ScreeningIssue>,
    /// Requested output members meeting supplied eligibility, Train and quarantine rules.
    pub eligible_output: Vec<ScreeningRecordId>,
    /// Requested members excluded by the report.
    pub exclusions: Vec<ScreeningExclusion>,
    /// Available task/domain/difficulty/teacher/length strata for every population member.
    pub strata: Vec<ScreeningRecordStratum>,
    /// Validated predecessor. Retains stable group assignments; chains are limited to 16 plans.
    pub previous: Option<Box<FrozenScreeningPlan>>,
    /// Complete report identity, excluding only this self-reference.
    pub plan_id: String,
}
