//! `RatingRecord{}` — relational Glicko-2 reputation for the judge + teacher arms
//! (DATA-SCHEMA §1.13b, B10).
//!
//! Persisted relationally (one row per `(subject, model_slug, training_area)`), keyed
//! `(teacher_slug, training_area)` for the teacher arm from the start so the v1 single-area
//! case is a no-op and no v2 migration is needed. The Glicko-2/Brier/log-score compute
//! (`update_glicko2`) is harness-side Rust; this schema only stores its state.

use serde::{Deserialize, Serialize};

/// Per-`(subject, model_slug, training_area)` reputation state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RatingRecord {
    /// `Judge | Teacher` discriminant.
    pub subject: RatingSubject,
    pub model_slug: String,
    /// `Some` for teacher ratings (area-keyed); `None` acceptable for global judge ratings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_area: Option<String>,
    /// Glicko-2 rating `r`.
    pub glicko_r: f64,
    /// Glicko-2 rating deviation `RD`.
    pub glicko_rd: f64,
    /// Glicko-2 volatility `σ`.
    pub glicko_sigma: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_brier: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_log_score: Option<f64>,
    pub n_updates: u64,
}

/// Which reputation arm a [`RatingRecord`] belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RatingSubject {
    Judge,
    Teacher,
}
