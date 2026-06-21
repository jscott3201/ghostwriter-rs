//! `cost{}` — token accounting (DATA-SCHEMA §1.10).

use serde::{Deserialize, Serialize};

/// Per-record token + spend accounting. `reasoning_tokens > 0` is part of the Verify gate
/// (§1.6); the engine cost meter sums `usd` across in-flight records to gate a run against a
/// budget cap.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Cost {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    /// OpenRouter bills these as OUTPUT tokens.
    #[serde(default)]
    pub reasoning_tokens: u32,
    #[serde(default)]
    pub usd: f64,
    #[serde(default)]
    pub latency_ms: u32,
}
