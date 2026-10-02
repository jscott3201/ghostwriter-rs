//! Consumed prepared SFT inputs. Rust validates structure; the pinned Python consumer replays
//! official rendering/tokenization. Neither layer establishes eligible weights or learned benefit.
use crate::{
    CotPolicy, MultiTurnLoss, NamespacedTaskId, ReviewedTaskRights, ScreeningRecordId,
    SemanticTaskIdentity, TaskSplit,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Complete payload committed together with the original Parquet source by the build framing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedSftPayload {
    /// Supported payload version (one).
    pub version: u32,
    /// Captured source coordinate, checked against actual embedded Parquet verification.
    pub source: PreparedSftSourceSnapshot,
    /// Producer recipe, complete counts, rejections, and explicit qualification limits.
    pub manifest: PreparedSftManifest,
    /// Exact ordered examples, including every token, label, ownership field, and source binding.
    pub examples: Vec<PreparedSftExample>,
}

/// A source descriptor gains verification authority only by checking the embedded source bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedSftSourceSnapshot {
    /// Original logical Parquet artifact identity.
    pub artifact_id: String,
    /// Exact captured source size.
    pub byte_length: u64,
    /// Ordinary BLAKE3 of the exact source bytes.
    pub snapshot_blake3: String,
}

/// Complete producer accounting; input identity lives in the enclosing frame, not this manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedSftManifest {
    /// Supported manifest version (one).
    pub build_manifest_version: u32,
    /// Producer recipe reference; Python replay verifies its algorithm and installed source.
    pub recipe_id: String,
    /// Complete tokenizer/adapter/dependency/runtime and preparation-policy recipe.
    pub recipe: PreparedSftRecipe,
    /// Verified source rows available for preparation.
    pub source_record_count: u64,
    /// Source rows represented by at least one accepted example.
    pub accepted_source_record_count: u64,
    /// Selected assistant targets after record-level rejection.
    pub candidate_target_count: u64,
    /// Number of ordered examples.
    pub expanded_example_count: u64,
    /// Complete record/target rejection count.
    pub rejected_item_count: u64,
    /// Every declared record/target rejection, separately replayed by Python.
    pub rejections: Vec<PreparedSftRejection>,
    /// Rejections before target enumeration.
    pub rejected_record_count: u64,
    /// Enumerated targets rejected during preparation.
    pub rejected_target_count: u64,
    /// Source rows with no accepted example, including fully rejected targets.
    pub source_records_with_no_examples: u64,
    /// Exact ordered example references.
    pub example_ids: Vec<String>,
    /// Sum of all unshifted supervised labels.
    pub supervised_token_count: u64,
    /// Sum of masked context labels.
    pub context_token_count: u64,
    /// Sum of supervised labels surviving causal shifting.
    pub effective_shifted_supervised_token_count: u64,
    /// Sum of whole nonwhitespace answer tokens surviving causal shifting.
    pub effective_shifted_answer_token_count: u64,
    /// Explicit source-only screening scope and unavailable model/execution/decision evidence.
    pub qualification_limits: Value,
}

/// Producer recipe. Version two pins a named text profile's files, wrapper, and dependencies.
/// Historical version-one tokenizer declarations retain their original shape validation.
/// Python replay checks actual official rendering and tokenization; Rust does not implement BPE.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedSftRecipe {
    /// Supported preparation recipe version.
    pub version: u32,
    /// Installed adapter package version.
    pub adapter_version: String,
    /// Installed adapter source/package-policy identity.
    pub adapter_source_sha256: String,
    /// Explicit version-two text profile; absent only on historical version-one recipes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preparation_profile: Option<crate::PreparedSftProfile>,
    /// Complete qualified dependency versions.
    pub dependencies: BTreeMap<String, String>,
    /// Tokenizer repository/files/template identity, distinct from student weights.
    pub tokenizer: Value,
    /// Complete runtime wrapper, vocabulary, backend, and added-token declarations.
    pub tokenizer_policy: Value,
    /// Actual producer Python/platform recipe.
    pub runtime: Value,
    /// Declared tokenizer target repository/revision; it does not identify loaded weights.
    pub tokenizer_target: Value,
    /// Selected reasoning loss policy.
    pub cot_policy: CotPolicy,
    /// Selected assistant-turn policy.
    pub multi_turn_loss: MultiTurnLoss,
    /// Prefix or full-final layout, derived from the turn policy.
    pub layout: String,
    /// Maximum accepted unpadded sequence length; truncation is forbidden.
    pub max_length: u64,
    /// Python Unicode-codepoint offsets.
    pub offset_unit: String,
    /// Unshifted causal language-model labels.
    pub labels: String,
    /// Must be false; official rendering already supplies control tokens.
    pub add_special_tokens: bool,
    /// Must be false; overlength examples are explicit rejections.
    pub truncation: bool,
    /// Source-screening bindings, present exactly for a screened source artifact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub screening: Option<PreparedSftScreeningIdentity>,
}

/// Complete source-screening references checked against the captured artifact witness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedSftScreeningIdentity {
    /// Frozen screening plan identity.
    pub plan_id: String,
    /// Frozen lexical policy identity.
    pub policy_id: String,
    /// Captured screening input identity.
    pub screening_input_id: String,
    /// Protected input identity.
    pub protected_input_id: String,
    /// Connected-component grouping identity.
    pub grouping_id: String,
    /// Transaction-checked publication population identity.
    pub population_id: String,
}

/// Source coordinate and all original data used by this example's preparation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedSftExampleSource {
    /// Exact source record ID.
    pub record_id: String,
    /// Source record content hash.
    pub record_hash: String,
    /// Source prompt hash.
    pub prompt_hash: String,
    /// Exact original canonical messages JSON, with separate reasoning fields retained.
    pub messages_json: String,
    /// Exact original reviewed task block or explicit null.
    #[serde(deserialize_with = "required_option")]
    pub task_json: Option<String>,
    /// Exact version-four origin projection; absent only for historical source artifacts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin_json: Option<String>,
    /// Exact v5 tools payload: outer absence means a historical artifact; inner null means no tools.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "present_tools"
    )]
    pub tools_json: Option<Option<String>>,
    /// Logical source artifact ID.
    pub artifact_id: String,
    /// Source-record, declared-task-group, or screened-component grouping basis.
    pub group_kind: String,
    /// Producer group reference; screened groups must match actual frozen component IDs.
    pub group_id: String,
    /// Actual task declarations when present in the captured source row.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub declared_task: Option<PreparedSftDeclaredTask>,
    /// Complete component binding when the source artifact is screened.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub screening: Option<PreparedSftExampleScreening>,
}

/// Reviewed declarations copied from the actual original task projection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedSftDeclaredTask {
    /// Original recomputed semantic task identity.
    pub identity: SemanticTaskIdentity,
    /// Original declared corpus group.
    pub group: NamespacedTaskId,
    /// Original declared split; accepted training examples require Train.
    pub split: TaskSplit,
    /// Original reviewed rights assertions, without an independent eligibility claim.
    pub rights: ReviewedTaskRights,
}

/// Exact source witness coordinate for a prepared example.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedSftExampleScreening {
    /// Frozen plan identity.
    pub plan_id: String,
    /// Frozen policy identity.
    pub policy_id: String,
    /// Original screening input identity.
    pub screening_input_id: String,
    /// Original protected input identity.
    pub protected_input_id: String,
    /// Original grouping identity.
    pub grouping_id: String,
    /// Original population identity.
    pub population_id: String,
    /// Actual run/record coordinate.
    pub record: ScreeningRecordId,
    /// Actual frozen connected component.
    pub component_id: String,
    /// Exported source semantics bound by the witness.
    pub export_projection_id: String,
}

/// One whole record or selected assistant target rejected before training.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedSftRejection {
    /// Exact source record coordinate.
    pub record_id: String,
    /// Original assistant message index, or explicit null for record-level rejection.
    #[serde(deserialize_with = "required_option")]
    pub target_index: Option<u64>,
    /// Producer's concrete rejection reason; Python replay verifies the decision.
    pub reason: String,
}

/// One half-open ownership span in Unicode codepoints.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedSftSpan {
    /// Inclusive start.
    pub start: u64,
    /// Exclusive end.
    pub end: u64,
    /// Header, context, reasoning, wrapper, answer, end, or separator.
    pub kind: String,
    /// Whether labels owned by this span are supervised.
    pub supervised: bool,
    /// Source message index, retained through prefix expansion.
    pub message_index: u64,
}

/// All prepared features and audit evidence; no field is omitted from the enclosing build hash.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedSftExample {
    /// Complete unpadded IDs.
    pub input_ids: Vec<u32>,
    /// Unpadded attention, exactly one per token.
    pub attention_mask: Vec<u8>,
    /// Corresponding token ID or -100; unshifted.
    pub labels: Vec<i64>,
    /// Exact rendered training string.
    pub rendered: String,
    /// Complete character ownership ledger.
    pub spans: Vec<PreparedSftSpan>,
    /// Actual tokenizer offsets.
    pub offset_mapping: Vec<[u64; 2]>,
    /// Canonical-combining-sequence ownership offsets.
    pub ownership_offsets: Vec<[u64; 2]>,
    /// Sorted distinct ownership kinds for each token.
    pub token_kinds: Vec<Vec<String>>,
    /// Complete sorted whole-answer token indices that survive causal shifting.
    pub shifted_answer_token_indices: Vec<u64>,
    /// Exact original source and grouping bindings.
    pub source: PreparedSftExampleSource,
    /// Original assistant message index.
    pub target_index: u64,
    /// Producer example reference; checked by official Python replay.
    pub example_id: String,
    /// Count of unshifted supervised labels.
    pub supervised_tokens: u64,
    /// Count of masked context labels.
    pub context_tokens: u64,
    /// Count of supervised labels after excluding label zero for causal shifting.
    pub effective_shifted_supervised_tokens: u64,
}

fn required_option<'de, D: serde::Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> Result<Option<T>, D::Error> {
    Option::<T>::deserialize(deserializer)
}

fn present_tools<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Option<String>>, D::Error> {
    Option::<String>::deserialize(deserializer).map(Some)
}
