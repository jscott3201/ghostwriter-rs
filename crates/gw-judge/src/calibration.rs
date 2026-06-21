//! Calibration weights from per-judge reputation (JUDGE-DESIGN §5.3 / §5.4).
//!
//! Each judge's influence on the consensus is its **track-record calibration**, NOT its stated
//! confidence (`Grade.confidence` is audit-only — §4.4: "calibration not confidence"). The weight
//! derives from the judge's [`RatingRecord`]: its mean Brier score and mean log score on items
//! where ground truth later became available (verifier-decided,
//! human-audited). Both are strictly proper scoring rules where **lower is better**, so a
//! well-calibrated judge weighs more and a confidently-wrong judge's influence decays.
//!
//! The weights are a panel-local softmax over `-(beta·brier + gamma·logscore)`, matching the §5.3
//! pseudocode. Softmax keeps them strictly positive and summing to 1, which is exactly what the
//! design-effect `(Σw)²/(wᵀRw)` wants (a uniform vector then gives `n_eff = k`).
//!
//! **Cold-start default (documented):** a judge with NO rating history — or a panel with no
//! ratings at all — gets a **flat / uniform** weight (it is *explored*, not *trusted*; mirrors a
//! high Glicko-2 RD, §5.4). [`uniform_weights`] is the explicit default the panel falls back to,
//! and [`calibration_weights`] returns uniform when every judge is cold.

use gw_schema::{RatingRecord, RatingSubject};

/// Default Brier coefficient `beta` in the calibration softmax (§5.3). Brier ∈ `[0, 2]`; a
/// moderate slope so a clearly better-calibrated judge measurably out-weighs a worse one without a
/// single judge dominating the panel.
pub const DEFAULT_BETA: f64 = 2.0;

/// Default log-score coefficient `gamma` in the calibration softmax (§5.3). The log score punishes
/// confident errors hardest, so it carries the same nominal slope as Brier.
pub const DEFAULT_GAMMA: f64 = 1.0;

/// Tunable coefficients for the calibration softmax. [`Default`] is `(DEFAULT_BETA, DEFAULT_GAMMA)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibrationParams {
    /// Brier coefficient `beta`.
    pub beta: f64,
    /// Log-score coefficient `gamma`.
    pub gamma: f64,
}

impl Default for CalibrationParams {
    fn default() -> Self {
        Self {
            beta: DEFAULT_BETA,
            gamma: DEFAULT_GAMMA,
        }
    }
}

/// A flat weight vector of length `k` (every judge weight `1.0`). The cold-start / no-history
/// default: explored, not trusted. With this vector the design effect gives `n_eff = k` for
/// independent judges, so a fresh panel is treated as fully independent until reputation accrues.
#[must_use]
pub fn uniform_weights(k: usize) -> Vec<f64> {
    vec![1.0; k]
}

/// `true` when a [`RatingRecord`] carries USABLE calibration history: at least one update AND at
/// least one of the two scoring metrics recorded. A just-seeded row (`n_updates == 0`) or one with
/// neither metric is NOT usable — it must not be allowed to shadow a proven rating (V3).
fn rating_is_usable(rating: &RatingRecord) -> bool {
    rating.n_updates > 0 && (rating.mean_brier.is_some() || rating.mean_log_score.is_some())
}

/// Look up the [`RatingRecord`] for a judge `model_slug` among `ratings`, matching only
/// `RatingSubject::Judge` rows. Judge ratings may be global (`training_area == None`) or
/// area-scoped.
///
/// An area-scoped match for `area` is preferred — but ONLY when it is USABLE
/// ([`rating_is_usable`]). A just-seeded area row (`n_updates == 0`, or no metrics) must NOT shadow
/// a proven global rating and demote a known-good judge to the cold-start prior (V3); so an
/// unusable area row is recorded as a last-resort fallback and the scan continues for a usable
/// global row, preferring: usable area > usable global > any area > any global.
fn find_judge_rating<'a>(
    ratings: &'a [RatingRecord],
    slug: &str,
    area: Option<&str>,
) -> Option<&'a RatingRecord> {
    let mut usable_global: Option<&RatingRecord> = None;
    let mut any_area: Option<&RatingRecord> = None;
    let mut any_global: Option<&RatingRecord> = None;
    for r in ratings {
        if r.subject != RatingSubject::Judge || r.model_slug != slug {
            continue;
        }
        let is_area_match = matches!((&r.training_area, area), (Some(a), Some(want)) if a == want);
        if is_area_match {
            if rating_is_usable(r) {
                return Some(r); // usable area-scoped rating: the best signal
            }
            any_area = any_area.or(Some(r));
        } else if r.training_area.is_none() {
            if rating_is_usable(r) {
                usable_global = usable_global.or(Some(r));
            }
            any_global = any_global.or(Some(r));
        }
    }
    // Prefer a usable global rating over an unusable area row; fall back to any row so the
    // caller can still see the record exists (penalty() then re-checks usability).
    usable_global.or(any_area).or(any_global)
}

/// The penalty `beta·brier + gamma·logscore` for one judge, or `None` when the judge has no usable
/// calibration history ([`rating_is_usable`]). An unusable row (`n_updates == 0`, or no metrics)
/// yields `None`, so an unproven judge cannot out-weigh a proven one — it takes the neutral prior.
fn penalty(rating: &RatingRecord, params: CalibrationParams) -> Option<f64> {
    if !rating_is_usable(rating) {
        return None;
    }
    Some(
        params.beta * rating.mean_brier.unwrap_or(0.0)
            + params.gamma * rating.mean_log_score.unwrap_or(0.0),
    )
}

/// Per-judge calibration weights for `slugs`, derived from `ratings` (JUDGE-DESIGN §5.3).
///
/// For each judge we look up its [`RatingRecord`] (area-scoped preferred, else global) and form the
/// penalty `beta·brier + gamma·logscore`; the weights are the panel-local softmax over the negated
/// penalties, so a lower-penalty (better-calibrated) judge weighs more. Judges with no history get
/// the panel mean penalty (a neutral prior), so an unproven judge neither dominates nor is starved.
///
/// **The documented default:** if NO judge in the panel has any calibration history, this returns
/// [`uniform_weights`] — the cold-start flat prior. Returned weights are strictly positive and sum
/// to `1.0` (a softmax), which is what the design effect [`effective_n`](crate::effective_n)
/// expects. Returns an empty vector for an empty `slugs`.
#[must_use]
pub fn calibration_weights(
    slugs: &[String],
    ratings: &[RatingRecord],
    area: Option<&str>,
    params: CalibrationParams,
) -> Vec<f64> {
    if slugs.is_empty() {
        return Vec::new();
    }

    // Per-judge penalty (None = cold). The penalty cache lets us compute the neutral prior (the
    // mean over judges that DO have history) for the cold ones.
    let penalties: Vec<Option<f64>> = slugs
        .iter()
        .map(|slug| find_judge_rating(ratings, slug, area).and_then(|r| penalty(r, params)))
        .collect();

    // No judge has history at all → the documented cold-start uniform default.
    let known: Vec<f64> = penalties.iter().filter_map(|p| *p).collect();
    if known.is_empty() {
        return normalize(&uniform_weights(slugs.len()));
    }

    // Cold judges take the panel mean penalty (a neutral prior — explored, not starved).
    let prior = known.iter().sum::<f64>() / known.len() as f64;
    let filled: Vec<f64> = penalties.iter().map(|p| p.unwrap_or(prior)).collect();

    softmax_neg(&filled)
}

/// Numerically-stable softmax over the NEGATED penalties: `wᵢ ∝ exp(-(pᵢ - min_p))`. Subtracting
/// the min penalty (so the smallest penalty maps to `exp(0) = 1`) avoids overflow while preserving
/// the ratios. The result is strictly positive and sums to 1.
fn softmax_neg(penalties: &[f64]) -> Vec<f64> {
    // Lower penalty → higher weight, so shift by the MIN penalty.
    let min_p = penalties.iter().copied().fold(f64::INFINITY, f64::min);
    let exps: Vec<f64> = penalties.iter().map(|p| (-(p - min_p)).exp()).collect();
    normalize(&exps)
}

/// Normalize a non-negative vector to sum to 1. A zero-sum vector falls back to uniform so the
/// design effect never divides by zero.
fn normalize(v: &[f64]) -> Vec<f64> {
    let sum: f64 = v.iter().sum();
    if sum <= 0.0 {
        return vec![1.0 / v.len() as f64; v.len()];
    }
    v.iter().map(|x| x / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn judge_rating(
        slug: &str,
        area: Option<&str>,
        brier: Option<f64>,
        logsc: Option<f64>,
        n: u64,
    ) -> RatingRecord {
        RatingRecord {
            subject: RatingSubject::Judge,
            model_slug: slug.to_string(),
            training_area: area.map(str::to_string),
            glicko_r: 1500.0,
            glicko_rd: 350.0,
            glicko_sigma: 0.06,
            mean_brier: brier,
            mean_log_score: logsc,
            n_updates: n,
        }
    }

    #[test]
    fn uniform_when_no_ratings_exist() {
        let slugs = vec!["a".into(), "b".into(), "c".into()];
        let w = calibration_weights(&slugs, &[], None, CalibrationParams::default());
        assert_eq!(w.len(), 3);
        for wi in &w {
            assert!((wi - 1.0 / 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn cold_panel_returns_documented_uniform_default() {
        // Ratings exist but all have n_updates == 0 → still cold → uniform.
        let slugs = vec!["a".into(), "b".into()];
        let ratings = vec![
            judge_rating("a", None, None, None, 0),
            judge_rating("b", None, Some(0.1), Some(0.2), 0),
        ];
        let w = calibration_weights(&slugs, &ratings, None, CalibrationParams::default());
        assert!((w[0] - 0.5).abs() < 1e-12);
        assert!((w[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn better_calibrated_judge_weighs_more() {
        let slugs = vec!["good".into(), "bad".into()];
        let ratings = vec![
            judge_rating("good", None, Some(0.05), Some(0.1), 50),
            judge_rating("bad", None, Some(0.45), Some(0.9), 50),
        ];
        let w = calibration_weights(&slugs, &ratings, None, CalibrationParams::default());
        assert!(
            w[0] > w[1],
            "low-Brier judge must out-weigh high-Brier: {w:?}"
        );
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn area_scoped_rating_preferred_over_global() {
        // Judge "j" has a BAD global rating but a GREAT math_cot-scoped rating; the area-scoped one
        // must win, so "j" out-weighs an always-bad "k" when grading in math_cot.
        let ratings = vec![
            judge_rating("j", None, Some(0.4), Some(0.4), 10), // global: bad
            judge_rating("j", Some("math_cot"), Some(0.02), Some(0.02), 10), // area: great
            judge_rating("k", Some("math_cot"), Some(0.4), Some(0.4), 10),
        ];
        let two: Vec<String> = vec!["j".into(), "k".into()];
        let w = calibration_weights(
            &two,
            &ratings,
            Some("math_cot"),
            CalibrationParams::default(),
        );
        assert!(w[0] > w[1], "area-scoped good rating must dominate: {w:?}");
    }

    #[test]
    fn just_seeded_area_row_does_not_shadow_proven_global() {
        // V3: judge "j" has a PROVEN global rating (n_updates=200, low Brier) and a just-seeded
        // area row (n_updates=0, no metrics). The unusable area row must NOT shadow the global one
        // and demote "j" to the cold-start prior — "j" keeps its proven weight against a bad "k".
        let ratings = vec![
            judge_rating("j", None, Some(0.03), Some(0.03), 200), // global: proven, great
            judge_rating("j", Some("math_cot"), None, None, 0),   // area: just seeded, unusable
            judge_rating("k", Some("math_cot"), Some(0.45), Some(0.9), 50), // proven, bad
        ];
        let two: Vec<String> = vec!["j".into(), "k".into()];
        let w = calibration_weights(
            &two,
            &ratings,
            Some("math_cot"),
            CalibrationParams::default(),
        );
        assert!(
            w[0] > w[1],
            "proven global rating must survive an unusable area row: {w:?}"
        );

        // Direct lookup: the proven global row is selected, not the just-seeded area row.
        let found = find_judge_rating(&ratings, "j", Some("math_cot")).unwrap();
        assert_eq!(found.n_updates, 200);
        assert_eq!(found.training_area, None);
    }

    #[test]
    fn cold_judge_gets_neutral_prior_not_dominance() {
        // One proven mediocre judge, one cold judge: the cold judge takes the panel mean penalty,
        // so it does NOT out-weigh the proven one by being unrated.
        let slugs = vec!["proven".into(), "cold".into()];
        let ratings = vec![judge_rating("proven", None, Some(0.2), Some(0.3), 30)];
        let w = calibration_weights(&slugs, &ratings, None, CalibrationParams::default());
        // The cold judge inherits the proven judge's penalty → equal weights here.
        assert!(
            (w[0] - w[1]).abs() < 1e-9,
            "cold judge must not dominate: {w:?}"
        );
    }

    #[test]
    fn weights_are_strictly_positive_and_normalized() {
        let slugs = vec!["a".into(), "b".into(), "c".into()];
        let ratings = vec![
            judge_rating("a", None, Some(0.01), Some(0.01), 5),
            judge_rating("b", None, Some(0.5), Some(1.5), 5),
            judge_rating("c", None, Some(0.25), Some(0.5), 5),
        ];
        let w = calibration_weights(&slugs, &ratings, None, CalibrationParams::default());
        assert!(w.iter().all(|x| *x > 0.0));
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn empty_slugs_gives_empty_weights() {
        assert!(calibration_weights(&[], &[], None, CalibrationParams::default()).is_empty());
    }

    #[test]
    fn teacher_ratings_are_ignored() {
        let slugs = vec!["j".into()];
        let teacher = RatingRecord {
            subject: RatingSubject::Teacher,
            ..judge_rating("j", None, Some(0.01), Some(0.01), 100)
        };
        // Only a teacher row exists for this slug → treated as cold → uniform.
        let w = calibration_weights(&slugs, &[teacher], None, CalibrationParams::default());
        assert!((w[0] - 1.0).abs() < 1e-12);
    }
}
