//! The `TrainingRecord` envelope — the canonical on-disk record (DATA-SCHEMA §1.11/§1.12).
//!
//! The envelope is a rich superset; export projects it DOWN to a trainer's columns at the
//! last moment (never discard provenance to match a trainer's shape).

use serde::{Deserialize, Serialize};

use crate::cost::Cost;
use crate::generation::Generation;
use crate::hashes::Hashes;
use crate::judging::Judging;
use crate::lifecycle::Lifecycle;
use crate::message::Message;
use crate::provenance::Provenance;
use crate::reasoning_quality::ReasoningQuality;
use crate::verification::Verification;

/// The canonical training-record envelope. Required fields (JSON-Schema §1.12): `record_id`,
/// `schema_version`, `training_area`, `messages`, `provenance`, `generation`, `lifecycle`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingRecord {
    /// ULID / UUIDv7 (time-sortable).
    pub record_id: String,
    /// Pinned `"1.0.0"` for v1.
    pub schema_version: semver::Version,
    /// Semver of the dataset build; None until a build assigns one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset_version: Option<semver::Version>,
    /// Declarative bundle name (e.g. `"rust-async"`).
    pub training_area: String,
    #[serde(default)]
    pub tags: Vec<String>,

    /// Clean content + first-class reasoning, per turn.
    pub messages: Vec<Message>,
    /// JSON-schema tool defs, or None.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,

    pub provenance: Provenance,
    pub generation: Generation,
    #[serde(default)]
    pub verification: Verification,
    #[serde(default)]
    pub judging: Judging,
    /// OPTIONAL per-step CoT verdict block (§1.13). Soft — drives ranking + a revise-band
    /// trigger only; the hard floor is OFF by default. Absent on records not step-graded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_quality: Option<ReasoningQuality>,
    pub lifecycle: Lifecycle,
    #[serde(default)]
    pub hashes: Hashes,
    #[serde(default)]
    pub cost: Cost,
}
