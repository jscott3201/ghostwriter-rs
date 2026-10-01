//! Shared exact candidate identities. Hashing remains in gw-storage.
use serde::{Deserialize, Serialize};

/// Exact identity of one declared candidate.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateBinding {
    /// Stable record identity.
    pub record_id: String,
    /// Run identity, checked independently of the content hash.
    pub run_id: String,
    /// Training area, checked independently of the content hash.
    pub training_area: String,
    /// Full recomputed prompt hash.
    pub prompt_hash: String,
    /// Full recomputed content hash, using the storage record-content projection.
    /// This hash excludes mutable lifecycle, judging and generation metadata by storage contract.
    pub record_hash: String,
}
