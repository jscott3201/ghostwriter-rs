//! Descriptive judge-score/verifier diagnostics and an optional independent outcome analysis.
//!
//! Judge-score spread measures only score variation. It cannot establish selector quality: a
//! maximum is necessarily at least its own group's mean. Qualification requires the separately
//! supplied [`OutcomeEvidence`], evaluated on the exact frozen corpus under its declared sampling
//! and reference contracts. Even qualification does not establish student learning benefit.

use std::collections::BTreeMap;

use gw_schema::TrainingRecord;
use gw_storage::{RecordFilter, Store};
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::outcome_analysis::analyze_outcomes;
use crate::outcomes::{
    OutcomeConfig, OutcomeEvidence, OutcomeReason, OutcomeReport, OutcomeStatus,
};

/// Independent-outcome settings and descriptive verifier-mixedness thresholds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeparationConfig {
    /// Descriptive warning threshold for mixed verifier groups; never a qualification gate.
    pub min_decidable_groups: usize,
    /// Descriptive warning floor in `[0, 1]`; never a qualification gate.
    pub min_decidable_fraction: f64,
    /// Qualification settings applied only to independent outcome evidence.
    pub outcomes: OutcomeConfig,
}

impl Default for SeparationConfig {
    fn default() -> Self {
        Self {
            min_decidable_groups: 10,
            min_decidable_fraction: 0.05,
            outcomes: OutcomeConfig::default(),
        }
    }
}

/// Descriptive score diagnostics and the separately qualified independent outcome comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeparationReport {
    /// Verifier mixedness and judge-score variation over the supplied records.
    pub diagnostics: SeparationDiagnostics,
    /// Independent outcomes for the exact declared corpus, with coverage and typed reasons.
    pub outcome_evaluation: OutcomeReport,
}

impl SeparationReport {
    /// Whether the independent outcome analysis qualifies its declared corpus and policy.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.outcome_evaluation.status == OutcomeStatus::Qualified
    }
}

/// Descriptive measurements only. None of these fields can establish independent quality.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeparationDiagnostics {
    /// Distinct prompt groups within each run and area, including singletons.
    pub n_groups: usize,
    /// Groups with only one record.
    pub n_singletons: usize,
    /// Multi-candidate groups where every verifier pass bit is true.
    pub n_allpass: usize,
    /// Multi-candidate groups where every verifier pass bit is false.
    pub n_allfail: usize,
    /// Multi-candidate groups with both pass and fail bits.
    pub n_mixed: usize,
    /// Mixed groups divided by all multi-candidate groups; `null` when none exist.
    pub decidable_fraction: Option<f64>,
    /// All-pass groups with at least two finite bounded judge scores.
    pub n_score_groups: usize,
    /// Score variation for those groups; `null` when no groups are eligible.
    pub judge_scores: Option<JudgeScoreDiagnostics>,
    /// Present judge scores excluded because they are non-finite or outside `[0, 1]`.
    pub n_invalid_judge_scores: usize,
    /// Descriptive mixedness warning; `null` for an invalid diagnostic threshold.
    pub low_decidability: Option<bool>,
}

/// Judge-score summaries, explicitly separated from independent outcome measurements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JudgeScoreDiagnostics {
    /// Mean group maximum judge score.
    pub mean_group_max: f64,
    /// Mean of the per-group mean judge scores, with equal group weight.
    pub mean_group_mean: f64,
    /// Mean over the flat pool of scored candidates, with equal candidate weight.
    pub pooled_mean: f64,
    /// Mean `maximum - group_mean`; nonnegative by construction, not a quality signal.
    pub mean_max_minus_mean: f64,
}

struct Group {
    passed: Vec<bool>,
    scores: Vec<f64>,
}

fn diagnostics(records: &[TrainingRecord], cfg: &SeparationConfig) -> SeparationDiagnostics {
    let mut groups: BTreeMap<(&str, &str, &str), Group> = BTreeMap::new();
    let mut n_invalid_judge_scores = 0;
    for record in records {
        let group = groups
            .entry((
                &record.provenance.run_id,
                &record.training_area,
                &record.hashes.prompt_hash,
            ))
            .or_insert_with(|| Group {
                passed: Vec::new(),
                scores: Vec::new(),
            });
        group.passed.push(record.verification.all_passed);
        if let Some(score) = record.judging.aggregate {
            if score.is_finite() && (0.0..=1.0).contains(&score) {
                group.scores.push(score);
            } else {
                n_invalid_judge_scores += 1;
            }
        }
    }
    let mut report = SeparationDiagnostics {
        n_groups: groups.len(),
        n_singletons: 0,
        n_allpass: 0,
        n_allfail: 0,
        n_mixed: 0,
        decidable_fraction: None,
        n_score_groups: 0,
        judge_scores: None,
        n_invalid_judge_scores,
        low_decidability: None,
    };
    let mut maxima = Vec::new();
    let mut means = Vec::new();
    let mut pool = Vec::new();
    for group in groups.values() {
        if group.passed.len() < 2 {
            report.n_singletons += 1;
        } else if group.passed.iter().all(|&passed| passed) {
            report.n_allpass += 1;
            if group.scores.len() >= 2 {
                maxima.push(group.scores.iter().copied().fold(0.0, f64::max));
                means.push(mean(&group.scores));
                pool.extend_from_slice(&group.scores);
            }
        } else if group.passed.iter().all(|&passed| !passed) {
            report.n_allfail += 1;
        } else {
            report.n_mixed += 1;
        }
    }
    let multi = report.n_allpass + report.n_allfail + report.n_mixed;
    report.decidable_fraction = (multi > 0).then(|| report.n_mixed as f64 / multi as f64);
    report.n_score_groups = maxima.len();
    if !maxima.is_empty() {
        let gaps: Vec<_> = maxima
            .iter()
            .zip(&means)
            .map(|(max, mean)| max - mean)
            .collect();
        report.judge_scores = Some(JudgeScoreDiagnostics {
            mean_group_max: mean(&maxima),
            mean_group_mean: mean(&means),
            pooled_mean: mean(&pool),
            mean_max_minus_mean: mean(&gaps),
        });
    }
    if valid_fraction(cfg.min_decidable_fraction) {
        report.low_decidability = Some(
            report.n_mixed < cfg.min_decidable_groups
                || report.decidable_fraction.unwrap_or(0.0) < cfg.min_decidable_fraction,
        );
    }
    report
}

fn valid_fraction(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Analyze persisted record facts and optional independent evidence without I/O.
///
/// Diagnostics describe all supplied records. Outcome analysis joins only the envelope's explicit
/// frozen membership against those records, recomputes candidate hashes, and rejects missing or
/// conflicting identities. Unlisted records do not enter the frozen random-control population.
/// Invalid evidence/configuration is returned as a typed report, never a fabricated numerical result.
#[must_use]
pub fn analyze(
    records: &[TrainingRecord],
    cfg: &SeparationConfig,
    outcomes: Option<&OutcomeEvidence>,
) -> SeparationReport {
    let diagnostics = diagnostics(records, cfg);
    let mut outcome_evaluation = analyze_outcomes(records, outcomes, &cfg.outcomes);
    if !valid_fraction(cfg.min_decidable_fraction) {
        outcome_evaluation.status = OutcomeStatus::InvalidEvidence;
        outcome_evaluation.statistics = None;
        outcome_evaluation.coverage.evaluated_prompts = 0;
        outcome_evaluation
            .reasons
            .push(OutcomeReason::InvalidConfiguration {
                field: "min_decidable_fraction".into(),
            });
    }
    outcome_evaluation.reasons.sort();
    outcome_evaluation.reasons.dedup();
    SeparationReport {
        diagnostics,
        outcome_evaluation,
    }
}

/// Scan a store using the caller's filter and perform the pure analysis.
///
/// An evidence member excluded by the filter is an invalid missing record, not silently dropped.
///
/// # Errors
/// Returns [`crate::EvalError::Storage`] if the scan fails. Evidence invalidity is in the report.
pub async fn analyze_store(
    store: &Store,
    filter: &RecordFilter,
    cfg: &SeparationConfig,
    outcomes: Option<&OutcomeEvidence>,
) -> Result<SeparationReport> {
    let records = store.scan(filter).await?;
    Ok(analyze(&records, cfg, outcomes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{group, scored_sibling};

    #[test]
    fn positive_score_spread_has_no_independent_qualification() {
        let records = group("prompt", true, &[0.9, 0.5, 0.1]);
        let cfg = SeparationConfig {
            min_decidable_groups: 0,
            min_decidable_fraction: 0.0,
            ..Default::default()
        };
        let report = analyze(&records, &cfg, None);
        let scores = report.diagnostics.judge_scores.unwrap();
        assert!((scores.mean_max_minus_mean - 0.4).abs() < 1e-12);
        assert_eq!(
            report.outcome_evaluation.status,
            OutcomeStatus::InsufficientEvidence
        );
        assert_eq!(
            report.outcome_evaluation.reasons,
            [OutcomeReason::MissingEvidence]
        );
        assert!(report.outcome_evaluation.statistics.is_none());
    }

    #[test]
    fn verifier_mixedness_is_descriptive_and_excludes_singletons() {
        let records = [
            scored_sibling("mixed", true, None),
            scored_sibling("mixed", false, None),
            scored_sibling("pass", true, Some(0.7)),
            scored_sibling("pass", true, Some(0.7)),
            scored_sibling("fail", false, None),
            scored_sibling("fail", false, None),
            scored_sibling("solo", true, Some(0.9)),
        ];
        let report = analyze(&records, &SeparationConfig::default(), None);
        let d = report.diagnostics;
        assert_eq!(
            (
                d.n_groups,
                d.n_singletons,
                d.n_allpass,
                d.n_allfail,
                d.n_mixed
            ),
            (4, 1, 1, 1, 1)
        );
        assert_eq!(d.decidable_fraction, Some(1.0 / 3.0));
        assert_eq!(d.low_decidability, Some(true));
        assert_eq!(d.judge_scores.unwrap().mean_max_minus_mean, 0.0);
    }

    #[test]
    fn invalid_scores_are_counted_and_do_not_poison_diagnostics() {
        let records = group(
            "prompt",
            true,
            &[0.9, 0.5, f64::NAN, f64::INFINITY, f64::MAX, -0.1],
        );
        let report = analyze(&records, &SeparationConfig::default(), None);
        assert_eq!(report.diagnostics.n_invalid_judge_scores, 4);
        assert_eq!(report.diagnostics.n_score_groups, 1);
        assert!(
            serde_json::to_value(report).unwrap()["outcome_evaluation"]["statistics"].is_null()
        );
    }

    #[test]
    fn empty_corpus_has_null_numeric_comparisons() {
        let report = analyze(&[], &SeparationConfig::default(), None);
        assert!(!report.passed());
        assert!(report.diagnostics.decidable_fraction.is_none());
        assert!(report.diagnostics.judge_scores.is_none());
        assert!(report.outcome_evaluation.statistics.is_none());
    }
}
