//! Supplied finite-corpus declarations for descriptive lexical screening. No collection or
//! semantic qualification is implied by these persisted, untrusted input types.
use crate::{CotPolicy, Message, MultiTurnLoss, TrlFormat};
use serde::{Deserialize, Serialize};

/// Supported frozen screening declaration and report version.
pub const SCREENING_VERSION: u32 = 1;
/// Exact normalization, shingle and grouping software recipe.
pub const LEXICAL_SCREEN_RECIPE: &str = "lexical-screen-v1";

/// Globally unambiguous record reference within a declared run corpus.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScreeningRecordId {
    /// Operator-declared run identity.
    pub run_id: String,
    /// Record identity within that run.
    pub record_id: String,
}

/// Explicit finite population scope. Preparation sorts IDs and rejects duplicates/empty IDs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeclaredRunSet {
    /// Every supplied record in these runs is captured before any eligibility filtering.
    pub run_ids: Vec<String>,
}

/// Source item revision expected by the operator, including all declared generated records.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedScreeningTask {
    /// Source namespace.
    pub namespace: String,
    /// Source item identifier (revisions of this item remain connected).
    pub item: String,
    /// Immutable supplied source revision.
    pub revision: String,
    /// Complete declared record set for this source revision in the bounded corpus.
    pub records: Vec<ScreeningRecordId>,
}

/// Explicit sibling membership; observed completion counts alone never assert completeness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeclaredScreeningSiblings {
    /// Sibling groups are scoped to this run.
    pub run_id: String,
    /// Must agree with each member's checked sibling/prompt identity.
    pub sibling_group_id: String,
    /// Complete declared membership, including excluded/rejected siblings.
    pub records: Vec<ScreeningRecordId>,
}

/// Selected artifact scope, distinct from the full screening population.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScreeningOutputScope {
    /// One output run; dependencies may belong to other declared runs.
    pub run_id: String,
    /// Requested output candidates. The report separately identifies eligible Train members.
    pub record_ids: Vec<String>,
}

/// Bounded work counters. Values may be lowered, but cannot exceed the v1 ceilings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScreeningLimits {
    /// At most 64 MiB of supported text across captured records and protected inputs.
    pub total_text_bytes: u64,
    /// At most 1 MiB in each text segment.
    pub segment_bytes: u64,
    /// At most 65,536 normalized tokens per segment.
    pub segment_tokens: u64,
    /// At most 100,000 segments.
    pub segments: u64,
    /// At most 2,000,000 stored distinct shingles, including contiguous-overlap gate sets.
    pub distinct_shingles: u64,
    /// At most 64,000,000 token positions across candidate shingle windows, including repeated
    /// windows. This bounds repeated-token work independently of distinct-set cardinality.
    pub shingle_token_work: u64,
    /// At most 1,000,000 candidate comparisons; structural unit checks also consume this budget.
    pub comparisons: u64,
}
impl Default for ScreeningLimits {
    fn default() -> Self {
        Self {
            total_text_bytes: 64 * 1024 * 1024,
            segment_bytes: 1024 * 1024,
            segment_tokens: 65_536,
            segments: 100_000,
            distinct_shingles: 2_000_000,
            shingle_token_work: 64_000_000,
            comparisons: 1_000_000,
        }
    }
}

/// Pinned software and export policies; thresholds are not measured recall guarantees.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScreeningPolicy {
    /// Must equal [`LEXICAL_SCREEN_RECIPE`].
    pub recipe: String,
    /// Inclusive shingle lengths; at least one length must qualify.
    pub ngram: [u32; 2],
    /// Required contiguous shared run; not an additional short-gram detector.
    pub min_overlap_tokens: u32,
    /// Inclusive finite threshold from zero to one, retaining exact binary64 bits.
    #[serde(with = "crate::finite_numbers::scalar")]
    pub jaccard_threshold: f64,
    /// Explicit operational limits, bounded by the v1 ceilings.
    pub limits: ScreeningLimits,
    /// Canonical emitted-example renderer.
    pub target: TrlFormat,
    /// Pinned reasoning treatment in emitted training examples.
    pub cot_policy: CotPolicy,
    /// Selected target units: all assistant prefixes or only the final target.
    pub multi_turn_loss: MultiTurnLoss,
    /// Required sets added to the canonical protected-set union.
    pub additional_protected_sets: Vec<String>,
    /// Operator-declared languages requiring protected coverage; no language detection is claimed.
    pub required_languages: Vec<String>,
}
impl Default for ScreeningPolicy {
    fn default() -> Self {
        Self {
            recipe: LEXICAL_SCREEN_RECIPE.into(),
            ngram: [8, 13],
            min_overlap_tokens: 5,
            jaccard_threshold: 0.8,
            limits: ScreeningLimits::default(),
            target: TrlFormat::OpenAiMessages,
            cot_policy: CotPolicy::Supervised,
            multi_turn_loss: MultiTurnLoss::AllAssistant,
            additional_protected_sets: vec![],
            required_languages: vec!["en".into()],
        }
    }
}

/// Complete supplied declaration. Missing task/source/parent/sibling evidence is incomplete.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScreeningDeclaration {
    /// Must equal [`SCREENING_VERSION`].
    pub version: u32,
    /// Finite run corpus; this file assertion does not prove current database membership.
    pub runs: DeclaredRunSet,
    /// Requested artifact selection, before eligibility/split/quarantine exclusions.
    pub output: ScreeningOutputScope,
    /// Pinned normalization, matching, rendering, supervision and resource rules.
    pub policy: ScreeningPolicy,
    /// Complete declared source-item/revision membership within this corpus.
    pub expected_tasks: Vec<ExpectedScreeningTask>,
    /// Complete declared best-of-k membership within this corpus.
    pub siblings: Vec<DeclaredScreeningSiblings>,
}

/// Distinct text-field coverage classes; segment boundaries remain independent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScreeningField {
    /// Plain or multipart message text.
    Content,
    /// Flat reasoning field.
    Reasoning,
    /// Plaintext or summary reasoning details.
    ReasoningDetail,
    /// Tool argument object keys, string leaves and retained raw arguments.
    ToolArguments,
    /// Tool/function names.
    ToolName,
    /// Tool result text.
    ToolResult,
    /// Tool definition object keys and string leaves.
    ToolDefinition,
}

/// Human-supplied assertion permitting the specific screening use; never legal certification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtectedScreeningRights {
    /// Reviewer identity.
    pub reviewer: String,
    /// Supporting references retained without fetching.
    pub evidence: Vec<String>,
    /// Explicit permission for the intended local screening use.
    pub screening_permitted: bool,
}

/// Asserted field/language/media completeness of one supplied protected revision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtectedScreeningCoverage {
    /// Whether the operator asserts that all required items are supplied.
    pub complete: bool,
    /// Declared covered languages.
    pub languages: Vec<String>,
    /// Supported media declaration; v1 can screen only "text".
    pub media: Vec<String>,
    /// Explicit field coverage, checked against the actual captured record segments.
    pub fields: Vec<ScreeningField>,
}

/// A supplied protected item. Its text is never copied into the emitted plan/report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtectedScreeningItem {
    /// Unique stable identity within its set.
    pub item_id: String,
    /// Declared language; no automatic language identification occurs.
    pub language: String,
    /// One complete structured prompt, including any prior assistant/tool history.
    #[serde(deserialize_with = "crate::screening_messages::deserialize")]
    pub prompt: Vec<Message>,
    /// Covered answer/reasoning/tool material, with original field boundaries.
    #[serde(deserialize_with = "crate::screening_messages::deserialize")]
    pub responses: Vec<Message>,
}

/// Strict, local protected revision. Digests are recomputed from the supplied typed contents.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtectedScreeningSet {
    /// Supported manifest version.
    pub version: u32,
    /// Canonical protected-set ID, including every required canonical union member.
    pub canonical_id: String,
    /// Immutable operator-supplied source revision.
    pub source_revision: String,
    /// BLAKE3 content digest computed by the protected-content helper over canonical item order.
    pub content_digest: String,
    /// Required intended-use rights declaration.
    pub rights: Option<ProtectedScreeningRights>,
    /// Required coverage declarations.
    pub coverage: ProtectedScreeningCoverage,
    /// Must equal the pinned normalization recipe.
    pub normalization: String,
    /// Actual protected contents; no automatic acquisition occurs.
    pub items: Vec<ProtectedScreeningItem>,
}
