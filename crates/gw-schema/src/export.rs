//! Export-time contracts: [`CotPolicy`], [`TrlFormat`], and the [`ExportManifest`]
//! (DATA-SCHEMA §3.1/§3.2/§4.3, B9).

use serde::{Deserialize, Serialize};

/// Whether `reasoning` enters the loss region on export (DATA-SCHEMA §3.1). An export
/// parameter (per-`training_area`, recorded in the manifest), NOT a record mutation — the
/// same admitted record can be exported `Supervised` for one build and `Stripped` for another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CotPolicy {
    /// Render `reasoning` into the supervised (loss) region — CoT-SFT. Default for CoT corpora.
    #[default]
    Supervised,
    /// Render `reasoning` but mask it out of loss (present at render time, label -100).
    Masked,
    /// Drop `reasoning` entirely; render answer-only (empty thought wrapper for Gemma-4).
    Stripped,
}

/// The export-target template enum (DATA-SCHEMA §3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrlFormat {
    /// byte-exact, via the pinned `chat_template.jinja` (the initial target).
    Gemma4,
    /// `<|im_start|>{role}\n…<|im_end|>`.
    #[serde(rename = "chatml")]
    ChatML,
    /// `{conversations:[{from,value}]}`.
    ShareGpt,
    /// `{messages:[…]}` conversational (sft_modal.py ingest shape).
    OpenAiMessages,
    /// gpt-oss channels (thinking→analysis, content→final).
    Harmony,
    /// `{prompt:[…], completion:[…]}`.
    TrlPromptCompletion,
}

/// Which assistant turns enter the loss region on export (B9). DEFAULT [`AllAssistant`]
/// pins the de-facto prefix-delta masker (DATA-SCHEMA §3.4).
///
/// [`AllAssistant`]: MultiTurnLoss::AllAssistant
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MultiTurnLoss {
    #[default]
    AllAssistant,
    FinalTurnOnly,
}

/// RESERVED for B1 (Phase 1) corpus-diversity stats.
///
// TODO(B1): corpus diversity stats — defined and filled in Phase 1 (taxonomy coverage,
// cluster counts, embedding-spread metrics). Kept as a forward-compatible placeholder so
// `ExportManifest` compiles with a None-defaulted `diversity` field; NOT defined in Phase 0.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct CorpusDiversityStats {}

/// Records an export build so it is reproducible (B9, promoted prose → serde struct).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExportManifest {
    pub target: TrlFormat,
    pub cot_policy: CotPolicy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset_version: Option<semver::Version>,
    /// None until `push_to_hub` has actually run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hub_commit_sha: Option<String>,
    pub n_records: u64,
    pub n_admitted: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decontam_index_id: Option<String>,
    pub build_inputs_hash: String,
    /// Which assistant turns enter the loss region on export. DEFAULT `AllAssistant`.
    #[serde(default)]
    pub multi_turn_loss: MultiTurnLoss,
    /// RESERVED for B1 (Phase 1) corpus-diversity stats; None-defaulted for forward
    /// compatibility. [`CorpusDiversityStats`] is NOT defined in Phase 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diversity: Option<CorpusDiversityStats>,
}
