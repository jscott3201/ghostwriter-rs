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

/// The READER contract version of an exported shard's Parquet column set — i.e. what a consumer
/// may assume about the columns, NOT the record schema inside them.
///
/// This exists because the exported column set is a public, versioned artifact: a shard written
/// years ago must remain interpretable, so a column change is a versioned transition rather than a
/// silent overwrite. One column, one meaning, no drifting duplicates:
///
/// - [`RoleContentText`](ExportSchemaVersion::RoleContentText) (v1) — `messages_json` held one
///   `{role, content}` object per turn, with `content` flattened to a STRING, plus a parallel
///   `reasoning_json` column carrying `messages[i].reasoning`. Lossy: `content: null` collapsed to
///   `""`, `Content::Parts` was re-encoded as a JSON string inside a string, and `tool_calls`,
///   `name` and the `tool_call_id` RESULT LINK were DROPPED entirely. Readable only by a
///   text-only consumer that needs nothing but role + flat body text.
/// - [`CanonicalMessages`](ExportSchemaVersion::CanonicalMessages) (v2, current) — a SINGLE
///   `messages_json` column holding the canonical `Message[]` JSON in conversation order, with
///   every structural field preserved (content variant incl. `null`, `reasoning`,
///   `reasoning_details`, `tool_calls` incl. retained `raw_arguments`, `tool_call_id`, `name`).
///   `reasoning_json` is REMOVED: it duplicated `messages[i].reasoning` in a second column that
///   could only agree with the first by index, so a reorder or a partial rewrite silently forked
///   the two. `gw_storage::export_parquet` writes this version; see that function's docs for a
///   worked read-back example.
///
/// v1 → v2 is an intentional BREAK for any consumer reading `reasoning_json` or parsing
/// `messages_json` as `Vec<{role, content}>`. It is recorded in
/// [`ExportManifest::column_schema_version`], so a reader can branch on the version instead of
/// guessing from the columns present.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExportSchemaVersion {
    /// v1 — lossy `{role, content}` projection + parallel `reasoning_json`. Historical shards only.
    RoleContentText = 1,
    /// v2 — the current, lossless single `messages_json` column of canonical `Message[]` JSON.
    CanonicalMessages = 2,
}

impl ExportSchemaVersion {
    /// The version this build WRITES. Exported shards always carry it in the manifest.
    pub const CURRENT: Self = Self::CanonicalMessages;
}

impl Default for ExportSchemaVersion {
    /// A manifest with NO `column_schema_version` key predates the field, so it is read as the
    /// shape that was actually being written at the time: v1. Defaulting to the current version
    /// instead would make a historical shard claim a contract it never had.
    fn default() -> Self {
        Self::RoleContentText
    }
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
    /// The column/reader contract of the shard this manifest describes. A manifest written before
    /// the field existed omits the key and reads back as
    /// [`ExportSchemaVersion::RoleContentText`], which is what those shards actually contained.
    #[serde(default)]
    pub column_schema_version: ExportSchemaVersion,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A manifest written before `column_schema_version` existed omits the key. It must read back
    /// as the version those shards ACTUALLY carried (v1, the lossy role+content projection) —
    /// defaulting to the current version would let a reader apply the v2 contract to a v1 file.
    #[test]
    fn legacy_manifest_without_version_reads_as_v1() {
        let legacy = r#"{
          "target": "chatml",
          "cot_policy": "supervised",
          "n_records": 3,
          "n_admitted": 2,
          "build_inputs_hash": "abc"
        }"#;
        let m: ExportManifest = serde_json::from_str(legacy).unwrap();
        assert_eq!(
            m.column_schema_version,
            ExportSchemaVersion::RoleContentText
        );
        assert_ne!(
            m.column_schema_version,
            ExportSchemaVersion::CURRENT,
            "an absent key must never claim the current contract"
        );
    }

    /// The version round-trips as a stable, human-readable token a non-Rust consumer can branch on.
    #[test]
    fn current_version_serializes_as_a_named_token() {
        assert_eq!(
            serde_json::to_string(&ExportSchemaVersion::CURRENT).unwrap(),
            "\"canonical_messages\""
        );
        let m = ExportManifest {
            column_schema_version: ExportSchemaVersion::CURRENT,
            target: TrlFormat::Gemma4,
            cot_policy: CotPolicy::Supervised,
            dataset_version: None,
            hub_commit_sha: None,
            n_records: 1,
            n_admitted: 1,
            decontam_index_id: None,
            build_inputs_hash: "h".into(),
            multi_turn_loss: MultiTurnLoss::default(),
            diversity: None,
        };
        let s = serde_json::to_string(&m).unwrap();
        assert!(
            s.contains(r#""column_schema_version":"canonical_messages""#),
            "{s}"
        );
        assert_eq!(serde_json::from_str::<ExportManifest>(&s).unwrap(), m);
    }

    /// The discriminants are the wire numbers a reader may compare against, and CURRENT is the
    /// newer one — a version that never moves is not a version.
    #[test]
    fn discriminants_are_ordered_and_current_is_the_newest() {
        assert_eq!(
            ExportSchemaVersion::RoleContentText as u8,
            1,
            "v1 is the historical lossy shape"
        );
        assert_eq!(ExportSchemaVersion::CanonicalMessages as u8, 2);
        assert_eq!(
            ExportSchemaVersion::CURRENT,
            ExportSchemaVersion::CanonicalMessages
        );
        assert!(ExportSchemaVersion::CURRENT > ExportSchemaVersion::RoleContentText);
    }
}
