//! Versioned declared Gemma generation captures; saved fields never create model authority.
use crate::{CheckpointFile, CheckpointModelSummary};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Maximum bytes of one paired protocol message; stdout frames include their final newline.
pub const CODING_PAIR_MAX_BYTES: usize = 32 * 1024 * 1024;

/// Exactly one comparison side, always retained on every ordered output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodingModelSide {
    /// Fresh untouched approved starting state.
    Base,
    /// Fresh base plus measured final LoRA adapter from a live completion.
    Candidate,
}
/// Complete source identities used by the fresh Python controller; saved history is declared.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodingComparisonModels {
    /// Exact full producer bundle identity.
    pub completion_id: String,
    /// Original complete prepared training input.
    pub prepared_build_id: String,
    /// Complete measured frozen base inventory.
    pub base_model: CheckpointModelSummary,
    /// Complete measured final adapter inventory.
    pub final_adapter: CheckpointModelSummary,
    /// Exact observed producer recipe identity.
    pub producer_recipe_id: String,
    /// Approved release or explicitly owned reduced random fixture; never authority by itself.
    pub source_authorization: String,
    /// Actual producer implementation identity.
    pub training_source_sha256: String,
    /// Original unchanged preparation implementation identity.
    pub preparation_source_sha256: String,
    /// Complete measured producer file population.
    pub checkpoint_files: Vec<CheckpointFile>,
    /// Selected publisher model name.
    pub publisher_model: String,
    /// Selected immutable publisher revision.
    pub publisher_revision: String,
    /// Publisher-named earlier parent, still declared.
    pub declared_parent: String,
    /// Unknown publisher parent revision, always null in this recipe.
    pub parent_revision: Option<String>,
}
/// Separately versioned actual CPU generation recipe; preparation rendering remains unchanged.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GemmaCodingGenerationRecipe {
    /// Exactly one.
    pub version: u32,
    /// Installed comparison implementation identity.
    pub source_sha256: String,
    /// Exact qualified installed Python dependency closure.
    pub dependencies: BTreeMap<String, String>,
    /// Actual Python implementation/version, operating system and machine.
    pub runtime: BTreeMap<String, String>,
    /// Exact selected text profile.
    pub profile: String,
    /// Pinned complete official tokenizer file inventory and publisher declarations.
    pub tokenizer: serde_json::Value,
    /// Pinned backend, wrapper and complete added-token policy.
    pub tokenizer_policy: serde_json::Value,
    /// Identical system prompt for both sides, including exact original text.
    pub system_prompt: String,
    /// One fixed suffix bound for every item and side.
    pub max_new_tokens: usize,
    /// Reject oversized prompts; never truncate.
    pub max_prompt_tokens: usize,
    /// Always true for generation, separately from preparation.
    pub add_generation_prompt: bool,
    /// Always false.
    pub enable_thinking: bool,
    /// Always false; official historical reasoning handling applies.
    pub preserve_thinking: bool,
    /// Actual CPU device.
    pub device: String,
    /// Actual float32 parameter and compute profile.
    pub dtype: String,
    /// Actual eager attention implementation.
    pub attention: String,
    /// One CPU intra-operation thread.
    pub threads: u32,
    /// One model-owning process.
    pub processes: u32,
    /// Deterministic algorithms enabled with errors, not warnings.
    pub deterministic_algorithms: bool,
    /// Owned CPU seed.
    pub seed: u32,
    /// Always false; one unpadded task per call.
    pub padding: bool,
    /// Always false.
    pub truncation: bool,
    /// No supplied past keys or reused conversation state.
    pub fresh_cache: bool,
    /// Actual ordinary dynamic cache.
    pub cache: String,
    /// Always false.
    pub compile: bool,
    /// Canonical full effective Transformers config, including inactive defaults and float values.
    pub effective_config_json: String,
}
/// Exact official rendered prompt and decoder inputs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodingGenerationPrompt {
    /// Exact rendered text, without private oracle payload.
    pub rendered: String,
    /// Actual complete untruncated decoder input.
    pub input_ids: Vec<u32>,
    /// Exactly one for each unpadded prompt token.
    pub attention_mask: Vec<u32>,
}
/// Exact generation stopping evidence; correctness remains a separate native result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodingGenerationTermination {
    /// Observed final EOS ID 1.
    Eos,
    /// Observed final turn-end ID 106.
    TurnEnd,
    /// Observed unsupported tool handoff ID 50, retained in the body.
    ToolHandoff,
    /// No terminal and exactly the permitted number of suffix tokens.
    LengthLimit,
    /// No terminal and unexpectedly fewer tokens; Unknown.
    ShortReturn,
    /// Returned prompt prefix differs; Unknown and no suffix is invented.
    PrefixMismatch,
    /// Returned more tokens than permitted; Unknown.
    OverBound,
}
/// Deterministic plaintext representation status, without a correctness assertion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodingGeneratedRepresentation {
    /// Exact bounded nonempty module bytes, eligible for fresh native execution.
    Module,
    /// Observed unsupported controls or empty/oversized body; deterministic candidate failure.
    Unsupported,
    /// Generation contract was not completely observed; never scored as zero.
    Unknown,
}
/// Lossless token capture and exact decoded strings, verified again by Python token replay.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodingGenerationOutput {
    /// Entire returned decoder sequence, including the exact original prompt.
    pub sequence_ids: Vec<u32>,
    /// Every token after the exact prompt boundary.
    pub suffix_ids: Vec<u32>,
    /// Suffix with only one actually observed final 1/106 removed.
    pub body_ids: Vec<u32>,
    /// Unfiltered decode of the original generated suffix.
    pub original_suffix_text: String,
    /// Exact body decode, preserving ordinary Unicode and whitespace.
    pub body_text: String,
    /// Actually observed final 1, 106 or 50, otherwise null.
    pub terminal_id: Option<u32>,
    /// Independent fact: suffix length equals the configured bound.
    pub at_token_bound: bool,
    /// Observed termination classification.
    pub termination: CodingGenerationTermination,
    /// Native format gate before coding execution.
    pub representation: CodingGeneratedRepresentation,
}
/// One completed local model call, retaining every supplied and returned token.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapturedCodingGeneration {
    /// Actual complete decoder input.
    pub prompt: CodingGenerationPrompt,
    /// Actual complete decoder output.
    pub output: CodingGenerationOutput,
    /// Actual generated-length rule: prompt length plus suffix bound.
    pub effective_max_length: usize,
    /// Observed concrete returned cache class.
    pub cache_type: String,
}
/// One side of one task, including explicitly unobserved model failures.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodingGeneratedAnswer {
    /// Exact side, never inferred from row position alone.
    pub side: CodingModelSide,
    /// Exact accepted member.
    pub member_id: String,
    /// Derived complete base or base-plus-adapter identity.
    pub model_id: String,
    /// Present only when the full model call returned.
    pub generation: Option<CapturedCodingGeneration>,
    /// Bounded stable failure category, never arbitrary exception text or a saved pass flag.
    pub failure: Option<String>,
    /// Exact input captured before a failed model call; absent when no input was accepted.
    pub failed_prompt: Option<CodingGenerationPrompt>,
}
/// Honest training/held-out separation scope, bound to the actual verified prepared build.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodingComparisonSeparation {
    /// Registered reference Train population or explicit owned software fixture training.
    pub training_population: String,
    /// Every distinct accepted training member when the prepared build uses registered references.
    pub training_member_ids: Vec<String>,
    /// Declared source screening or not_run; no semantic decontamination claim.
    pub source_screening: String,
    /// Always not_run for this software route.
    pub semantic_screening: String,
    /// Ordered held-out members whose prompts could not be rendered under this recipe.
    pub unrendered_member_ids: Vec<String>,
    /// Successfully rendered prompts were checked disjoint; incomplete scope is explicit.
    pub effective_prompt_separation: String,
}
/// Complete ordered generated pair request. Saved requests convey declarations only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodingPairRequest {
    /// Exactly one.
    pub version: u32,
    /// Exact redacted held-out population captured by the native controller.
    pub population_id: String,
    /// Captured producer and both fresh model bindings.
    pub models: CodingComparisonModels,
    /// One identical protocol for both sides and every task.
    pub recipe: GemmaCodingGenerationRecipe,
    /// Actual training-population and rendered separation checks, with retained scope limits.
    pub separation: CodingComparisonSeparation,
    /// Base then Candidate for each population member in accepted order.
    pub rows: Vec<CodingGeneratedAnswer>,
}
