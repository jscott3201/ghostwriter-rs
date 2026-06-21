//! `DecontamConfig` — canonical decontamination config (DATA-SCHEMA §5.3, REMEDIATION ITEM 4).
//!
//! Per-area `protected_benchmarks` are UNIONED onto the canonical default union below, never
//! replacing it. The single-int `decontam_ngram` is replaced by an `[min, max]` range.

use serde::{Deserialize, Serialize};

/// The canonical default benchmark union — always decontaminated; per-area
/// `protected_benchmarks` are unioned on top (DATA-SCHEMA §5.3, CONFIG §7.2).
pub const CANONICAL_PROTECTED_BENCHMARKS: [&str; 10] = [
    "gsm8k",
    "math_500",
    "aime_2024",
    "aime_2025",
    "mmlu",
    "mmlu_stem",
    "gpqa",
    "humaneval",
    "livecodebench",
    "ifeval",
];

/// Decontamination thresholds + protected sets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecontamConfig {
    /// Protected benchmark/eval sets (the canonical default union is always included).
    #[serde(default)]
    pub protected_benchmarks: Vec<String>,
    /// `[min, max]` token n-gram overlap range. Default `[8, 13]`.
    pub decontam_ngram: [u32; 2],
    /// Filter overlaps shorter than this many tokens. Default 5.
    pub min_overlap_tokens: u32,
    /// Jaccard contamination threshold. Default 0.80.
    pub jaccard_threshold: f64,
    /// Optional embedding-similarity decontam path. Default false.
    pub embedding_decontam: bool,
    /// Cosine threshold for the embedding path. Default 0.86.
    pub embedding_cosine_threshold: f64,
}

impl Default for DecontamConfig {
    fn default() -> Self {
        Self {
            protected_benchmarks: Vec::new(),
            decontam_ngram: [8, 13],
            min_overlap_tokens: 5,
            jaccard_threshold: 0.80,
            embedding_decontam: false,
            embedding_cosine_threshold: 0.86,
        }
    }
}
