//! `hashes{}` — exact + near-dup signatures (DATA-SCHEMA §1.9).

use serde::{Deserialize, Serialize};

/// BLAKE3 (+ MinHash) signatures. Canonicalization for hashing sorts keys, normalizes
/// whitespace, and EXCLUDES volatile fields (`record_id`, timestamps, `cost`,
/// `lifecycle.history`). `completion_hash` deliberately excludes `reasoning`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct Hashes {
    /// blake3 of non-assistant prompt messages (sibling grouping only, not full-prefix proof).
    #[serde(default)]
    pub prompt_hash: String,
    /// blake3 of assistant content (EXCLUDES reasoning, so same answer / different CoT collapses).
    #[serde(default)]
    pub completion_hash: String,
    /// blake3 of the canonical full record (exact dedup key).
    #[serde(default)]
    pub record_hash: String,
    /// 64..256 MinHash perms for near-dup LSH (§5.2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minhash: Option<Vec<u64>>,
}
