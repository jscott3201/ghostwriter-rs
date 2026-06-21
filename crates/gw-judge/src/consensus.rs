//! The consensus math (JUDGE-DESIGN §5.5) — the heart of `gw-judge`.
//!
//! INVARIANT-f: panel `aggregate` is **NEVER a plain mean or majority vote**. It is the
//! calibration-weighted, correlation-adjusted consensus. The single load-bearing number is the
//! **weighted correlation design-effect**:
//!
//! ```text
//! n_eff = (Σ wᵢ)² / (wᵀ R w)
//! ```
//!
//! where `w` is the per-judge calibration weight vector (`calibration.rs`) and `R` is the
//! pairwise inter-judge correlation matrix. This single formula unifies BOTH failure modes:
//!
//! - **Weight concentration** — when `R = I` it reduces to the plain Kish design effect
//!   `(Σwᵢ)² / Σwᵢ²`, which is `≤ k` and `== k` only when all weights are equal.
//! - **Inter-judge correlation** — off-diagonal `R_ij > 0` inflates the denominator `wᵀ R w`, so
//!   a panel whose judges agree by correlation (not by independent observation) deflates toward 1.
//!
//! Two bounds are MANDATORY (prior spec reviews caught their absence as a real bug):
//!
//! 1. **Clip negative off-diagonals to 0** before the guard. Raw Phi/Pearson correlation CAN be
//!    negative (anti-correlated judges); anti-correlation is treated as INDEPENDENCE, not
//!    bonus-independence beyond `k`. Clipping keeps `wᵀ R w ≥ Σwᵢ²`, so `n_eff ≤ Kish ≤ k`.
//! 2. **Cap `n_eff ≤ k`** (belt-and-suspenders) after the division, so a numerical wobble can
//!    never report more effective votes than there are judges.
//!
//! The diagonal of `R` is always 1 (a judge is perfectly self-correlated). Equally-weighted,
//! perfectly-correlated judges (`R` all-ones) collapse to `n_eff ≈ 1`; equally-weighted,
//! independent judges (`R = I`) give `n_eff ≈ k`. Those two extremes are the correlation guard's
//! contract and are asserted in the unit tests.

use crate::error::{JudgeError, Result};

/// A square pairwise inter-judge correlation matrix, row-major `k × k`, with `diag == 1`.
///
/// Off-diagonals may arrive negative (raw Phi/Pearson over anti-correlated judges); the design
/// effect clips them to `0` internally so anti-correlation is treated as independence, never as
/// independence *beyond* `k`. Construct via [`CorrelationMatrix::identity`] (the `R = I`
/// independent-judges case), [`CorrelationMatrix::uniform_offdiagonal`] (every pair shares one
/// `rho` — the cold-start prior), or [`CorrelationMatrix::from_rows`] (an estimated matrix).
#[derive(Debug, Clone, PartialEq)]
pub struct CorrelationMatrix {
    /// Row-major `k × k` entries.
    rows: Vec<Vec<f64>>,
}

impl CorrelationMatrix {
    /// The identity matrix — `k` perfectly *independent* judges (`R = I`). With equal weights this
    /// yields `n_eff == k`.
    #[must_use]
    pub fn identity(k: usize) -> Self {
        let mut rows = vec![vec![0.0; k]; k];
        for (i, row) in rows.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        Self { rows }
    }

    /// Every off-diagonal pair shares the same correlation `rho`; the diagonal is `1`. Used both
    /// for the perfectly-correlated extreme (`rho = 1` → `n_eff ≈ 1`) and as the cold-start prior
    /// (`rho ≈ 0.7`) when there is too little co-judging history to estimate a real matrix — an
    /// unproven judge cannot inflate independence (mirrors a high Glicko-2 sigma, §5.4).
    #[must_use]
    pub fn uniform_offdiagonal(k: usize, rho: f64) -> Self {
        let mut rows = vec![vec![rho; k]; k];
        for (i, row) in rows.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        Self { rows }
    }

    /// Build from explicit rows (an estimated `k × k` matrix). The caller supplies the raw
    /// correlation estimates; clipping and the diagonal-is-1 normalization happen at design-effect
    /// time, so a slightly-off diagonal or a negative off-diagonal is tolerated here.
    ///
    /// # Errors
    /// Returns [`JudgeError::Invariant`] if `rows` is not square.
    pub fn from_rows(rows: Vec<Vec<f64>>) -> Result<Self> {
        let k = rows.len();
        if rows.iter().any(|r| r.len() != k) {
            return Err(JudgeError::Invariant(format!(
                "correlation matrix must be square; got {k} rows with mismatched widths"
            )));
        }
        Ok(Self { rows })
    }

    /// Panel size `k`.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.rows.len()
    }

    /// `true` when this is (numerically) the identity matrix — every off-diagonal is `0` and the
    /// diagonal is `1`. Used by the grader to detect a caller that passed `R = I` for a `k > 1`
    /// panel, which silently degrades the correlation guard to Kish-only (V8 observability).
    #[must_use]
    pub fn is_identity(&self) -> bool {
        let k = self.rows.len();
        for (i, row) in self.rows.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                let want = if i == j { 1.0 } else { 0.0 };
                if (v - want).abs() > 1e-12 {
                    return false;
                }
            }
        }
        k > 0
    }

    /// The principal submatrix over `indices` (in the given order). Used to recompute the design
    /// effect over a DECISIVE subset of judges after `Uncertain` grades are excluded (V2): dropping
    /// a judge must also drop its row+column so the correlation guard reflects the judges that
    /// actually voted. Out-of-range indices are skipped.
    #[must_use]
    pub fn submatrix(&self, indices: &[usize]) -> CorrelationMatrix {
        let k = self.rows.len();
        let kept: Vec<usize> = indices.iter().copied().filter(|&i| i < k).collect();
        let rows = kept
            .iter()
            .map(|&i| kept.iter().map(|&j| self.rows[i][j]).collect())
            .collect();
        CorrelationMatrix { rows }
    }

    /// Read entry `(i, j)`.
    fn get(&self, i: usize, j: usize) -> f64 {
        self.rows[i][j]
    }
}

/// The effective number of *independent* judges — the weighted correlation design-effect
/// `n_eff = (Σ wᵢ)² / (wᵀ R w)`, clamped to `[~0, k]`.
///
/// `weights` are the per-judge calibration weights (`calibration.rs`); `r` is the `k × k`
/// correlation matrix. Off-diagonal entries are clipped to `0` and the diagonal forced to `1`
/// before the quadratic form, which guarantees `wᵀ R w ≥ Σ wᵢ²` and therefore `n_eff ≤ Kish ≤ k`
/// — the bound the whole guard rests on. The result is additionally `min(k, …)`-clamped so a
/// numerical wobble can never exceed `k`.
///
/// # Errors
/// - [`JudgeError::EmptyPanel`] if `weights` is empty.
/// - [`JudgeError::Invariant`] if `weights.len() != r.dim()`, a weight is non-finite or negative,
///   all weights are zero (the denominator would be `0`), or `wᵀ R w` is non-positive.
pub fn effective_n(weights: &[f64], r: &CorrelationMatrix) -> Result<f64> {
    let k = weights.len();
    if k == 0 {
        return Err(JudgeError::EmptyPanel(
            "effective_n requires at least one judge weight".into(),
        ));
    }
    if r.dim() != k {
        return Err(JudgeError::Invariant(format!(
            "weight vector length {k} != correlation matrix dim {}",
            r.dim()
        )));
    }
    if let Some(bad) = weights.iter().find(|w| !w.is_finite() || **w < 0.0) {
        return Err(JudgeError::Invariant(format!(
            "calibration weight must be finite and non-negative; got {bad}"
        )));
    }

    let sum_w: f64 = weights.iter().sum();
    if sum_w <= 0.0 {
        return Err(JudgeError::Invariant(
            "calibration weights sum to zero; design effect is undefined".into(),
        ));
    }

    // The quadratic form wᵀ R w with off-diagonals clipped to max(0, R_ij) and the diagonal
    // forced to 1. Clipping is what enforces n_eff ≤ k: it keeps every cross term non-negative,
    // so wᵀ R w ≥ Σ wᵢ² (the Kish denominator), hence the ratio is ≤ Kish ≤ k.
    let mut wtrw = 0.0;
    for i in 0..k {
        for j in 0..k {
            let r_ij = if i == j { 1.0 } else { r.get(i, j).max(0.0) };
            wtrw += weights[i] * r_ij * weights[j];
        }
    }
    if wtrw <= 0.0 {
        return Err(JudgeError::Invariant(
            "wᵀRw is non-positive; design effect is undefined".into(),
        ));
    }

    let n_eff = (sum_w * sum_w) / wtrw;
    // Belt-and-suspenders cap: never report more effective votes than judges.
    Ok(n_eff.min(k as f64))
}

/// The plain Kish effective sample size `(Σ wᵢ)² / Σ wᵢ²` — the `R = I` special case of
/// [`effective_n`], measuring weight CONCENTRATION only. Exposed for audit + tests: it is the
/// upper bound the full correlation-adjusted `n_eff` must never exceed.
///
/// # Errors
/// Returns [`JudgeError::EmptyPanel`] / [`JudgeError::Invariant`] under the same conditions as
/// [`effective_n`] (it is `effective_n` with an identity matrix).
pub fn kish_effective_n(weights: &[f64]) -> Result<f64> {
    let r = CorrelationMatrix::identity(weights.len().max(1));
    effective_n(weights, &r)
}

/// The calibration-weighted consensus aggregate in `[0, 1]` — the weighted MEAN of per-judge
/// scores, `Σ wᵢ sᵢ / Σ wᵢ`.
///
/// This is the score that feeds the accept-threshold. It is **NOT a plain mean**: a high-weight
/// judge moves it measurably more than a low-weight judge, so a pure-mean implementation produces
/// a different (and demonstrably wrong) number — the property the INVARIANT-f test pins. The
/// weighting is the *only* place per-judge reputation enters the admitted score; the
/// correlation-adjusted [`effective_n`] gates whether that score may be trusted at all (the
/// escalate path), but does not itself rescale the aggregate.
///
/// # Errors
/// - [`JudgeError::EmptyPanel`] if either slice is empty.
/// - [`JudgeError::Invariant`] if the lengths differ, a value is non-finite, a WEIGHT is negative
///   (V5: mirrors [`effective_n`], which rejects the same vector — a negative weight is a config
///   bug and would otherwise let one judge subtract from the consensus), or the weights sum to
///   zero. The returned aggregate is clamped into `[0, 1]` to defend the threshold comparison.
pub fn weighted_aggregate(scores: &[f64], weights: &[f64]) -> Result<f64> {
    if scores.is_empty() || weights.is_empty() {
        return Err(JudgeError::EmptyPanel(
            "weighted_aggregate requires at least one judge".into(),
        ));
    }
    if scores.len() != weights.len() {
        return Err(JudgeError::Invariant(format!(
            "score/weight length mismatch: {} scores vs {} weights",
            scores.len(),
            weights.len()
        )));
    }
    if let Some(bad) = scores.iter().chain(weights).find(|v| !v.is_finite()) {
        return Err(JudgeError::Invariant(format!(
            "score/weight must be finite; got {bad}"
        )));
    }
    // V5: reject negative weights, exactly as effective_n does for the same `weights` vector, so the
    // two consumers of `weights` agree and a negative weight can never silently flip the consensus.
    if let Some(bad) = weights.iter().find(|w| **w < 0.0) {
        return Err(JudgeError::Invariant(format!(
            "calibration weight must be non-negative; got {bad}"
        )));
    }
    let sum_w: f64 = weights.iter().sum();
    if sum_w <= 0.0 {
        return Err(JudgeError::Invariant(
            "weights sum to zero; weighted aggregate is undefined".into(),
        ));
    }
    let dot: f64 = scores.iter().zip(weights).map(|(s, w)| s * w).sum();
    // Clamp into [0,1]: a well-formed call already lands there, but this defends the downstream
    // threshold band against a score slightly outside [0,1] slipping through.
    Ok((dot / sum_w).clamp(0.0, 1.0))
}

/// Inter-judge **agreement** as `1 - variance` of the per-judge scores in `[0, 1]` (higher =
/// tighter agreement). Persisted into `Judging.agreement`. This is a descriptive spread statistic
/// for audit and the cascade-signature monitor — it is NOT the aggregation rule (that is
/// [`weighted_aggregate`] gated by [`effective_n`]); a tight agreement among CORRELATED judges is
/// exactly the false-unanimity `effective_n` exists to discount.
///
/// # Errors
/// Returns [`JudgeError::EmptyPanel`] if `scores` is empty.
pub fn agreement(scores: &[f64]) -> Result<f64> {
    if scores.is_empty() {
        return Err(JudgeError::EmptyPanel(
            "agreement requires at least one judge".into(),
        ));
    }
    let n = scores.len() as f64;
    let mean = scores.iter().sum::<f64>() / n;
    let var = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n;
    Ok((1.0 - var).clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mandatory test 1: n_eff ≈ k for k INDEPENDENT, equally-weighted judges.
    #[test]
    fn neff_equals_k_for_independent_equal_weight() {
        for k in [2usize, 3, 4, 6] {
            let w = vec![1.0; k];
            let r = CorrelationMatrix::identity(k);
            let n_eff = effective_n(&w, &r).unwrap();
            assert!(
                (n_eff - k as f64).abs() < 1e-9,
                "independent equal-weight panel of {k} must give n_eff≈{k}, got {n_eff}"
            );
        }
    }

    /// Mandatory test 2: n_eff ≈ 1 for k PERFECTLY-CORRELATED, equally-weighted judges. This is the
    /// correlation guard actually working — a Kish-only n_eff would wrongly report k here.
    #[test]
    fn neff_collapses_to_one_for_perfectly_correlated() {
        for k in [2usize, 3, 4, 6] {
            let w = vec![1.0; k];
            let r = CorrelationMatrix::uniform_offdiagonal(k, 1.0);
            let n_eff = effective_n(&w, &r).unwrap();
            assert!(
                (n_eff - 1.0).abs() < 1e-9,
                "perfectly-correlated equal-weight panel of {k} must collapse to n_eff≈1, got {n_eff}"
            );
            // And Kish alone would have reported k — proving the guard is doing real work.
            let kish = kish_effective_n(&w).unwrap();
            assert!((kish - k as f64).abs() < 1e-9);
        }
    }

    /// Mandatory test 3: n_eff ≤ k ALWAYS, including a negative-off-diagonal input (clipping holds
    /// the bound — anti-correlation is treated as independence, never bonus-independence).
    #[test]
    fn neff_never_exceeds_k_even_with_negative_offdiagonal() {
        let k = 4;
        let w = vec![1.0, 0.5, 2.0, 0.25];
        // Strongly anti-correlated off-diagonals: raw (Σw)²/(wᵀRw) without clipping would blow up
        // past k. Clipping each to 0 caps it at the independent value.
        let r = CorrelationMatrix::uniform_offdiagonal(k, -0.9);
        let n_eff = effective_n(&w, &r).unwrap();
        assert!(
            n_eff <= k as f64 + 1e-9,
            "n_eff must be ≤ k even with negative off-diagonals; got {n_eff}"
        );
        // Clipping the -0.9 off-diagonals to 0 makes this identical to the Kish (R=I) value.
        let kish = kish_effective_n(&w).unwrap();
        assert!((n_eff - kish).abs() < 1e-9);
    }

    #[test]
    fn neff_partial_correlation_is_between_one_and_k() {
        let k = 4;
        let w = vec![1.0; k];
        let r = CorrelationMatrix::uniform_offdiagonal(k, 0.5);
        let n_eff = effective_n(&w, &r).unwrap();
        assert!(n_eff > 1.0 && n_eff < k as f64, "got {n_eff}");
    }

    /// Mandatory test 4: calibration-weighted aggregate ≠ plain mean — a high-weight judge moves it
    /// measurably. A pure-mean implementation would fail this exact assertion.
    #[test]
    fn weighted_aggregate_is_not_a_plain_mean() {
        // Two judges: a low scorer with high weight, a high scorer with low weight.
        let scores = vec![0.2, 0.9];
        let weights = vec![3.0, 1.0];
        let agg = weighted_aggregate(&scores, &weights).unwrap();
        let plain_mean = (0.2 + 0.9) / 2.0;
        // The weighted aggregate is pulled toward the high-weight low scorer: 0.2*3+0.9*1 / 4 = 0.375.
        assert!((agg - 0.375).abs() < 1e-9, "got {agg}");
        assert!(
            (agg - plain_mean).abs() > 0.1,
            "weighted aggregate {agg} must differ from the plain mean {plain_mean}"
        );
    }

    #[test]
    fn weighted_aggregate_equals_mean_only_when_weights_equal() {
        let scores = vec![0.2, 0.9, 0.5];
        let weights = vec![1.0, 1.0, 1.0];
        let agg = weighted_aggregate(&scores, &weights).unwrap();
        let mean = scores.iter().sum::<f64>() / 3.0;
        assert!((agg - mean).abs() < 1e-12);
    }

    /// V5: a negative weight is rejected (mirroring effective_n on the same vector), and the output
    /// is clamped into [0,1] so an out-of-range score can never slip into the threshold band.
    #[test]
    fn weighted_aggregate_rejects_negative_weight_and_clamps_output() {
        // A negative weight on the same vector effective_n rejects must be an Invariant error here.
        assert!(matches!(
            weighted_aggregate(&[0.5, 0.5], &[1.0, -0.5]),
            Err(JudgeError::Invariant(_))
        ));
        // An out-of-[0,1] score is clamped (defensive): scores at 2.0 with equal weights → 1.0.
        let agg = weighted_aggregate(&[2.0, 2.0], &[1.0, 1.0]).unwrap();
        assert!((agg - 1.0).abs() < 1e-12);
        // effective_n rejects the identical negative-weight vector — the two consumers agree.
        let r = CorrelationMatrix::identity(2);
        assert!(matches!(
            effective_n(&[1.0, -0.5], &r),
            Err(JudgeError::Invariant(_))
        ));
    }

    #[test]
    fn empty_panel_is_an_error_not_nan() {
        assert!(matches!(
            effective_n(&[], &CorrelationMatrix::identity(0)),
            Err(JudgeError::EmptyPanel(_))
        ));
        assert!(matches!(
            weighted_aggregate(&[], &[]),
            Err(JudgeError::EmptyPanel(_))
        ));
        assert!(matches!(agreement(&[]), Err(JudgeError::EmptyPanel(_))));
    }

    #[test]
    fn mismatched_lengths_are_rejected() {
        let w = vec![1.0, 1.0];
        let r = CorrelationMatrix::identity(3);
        assert!(matches!(effective_n(&w, &r), Err(JudgeError::Invariant(_))));
        assert!(matches!(
            weighted_aggregate(&[0.5, 0.5, 0.5], &[1.0, 1.0]),
            Err(JudgeError::Invariant(_))
        ));
    }

    #[test]
    fn non_square_correlation_matrix_is_rejected() {
        assert!(matches!(
            CorrelationMatrix::from_rows(vec![vec![1.0, 0.0], vec![0.0]]),
            Err(JudgeError::Invariant(_))
        ));
    }

    #[test]
    fn weight_concentration_alone_deflates_neff_below_k() {
        // Independent judges (R=I) but very concentrated weights: Kish must be < k.
        let w = vec![10.0, 1.0, 1.0, 1.0];
        let r = CorrelationMatrix::identity(4);
        let n_eff = effective_n(&w, &r).unwrap();
        assert!(n_eff < 4.0 && n_eff > 1.0, "got {n_eff}");
        // It equals the Kish value because R=I.
        assert!((n_eff - kish_effective_n(&w).unwrap()).abs() < 1e-12);
    }

    #[test]
    fn correlation_and_concentration_compound() {
        // Both concentrated AND correlated → penalised for the compound effect, not just the larger.
        let w = vec![5.0, 1.0, 1.0, 1.0];
        let independent = effective_n(&w, &CorrelationMatrix::identity(4)).unwrap();
        let correlated = effective_n(&w, &CorrelationMatrix::uniform_offdiagonal(4, 0.5)).unwrap();
        assert!(
            correlated < independent,
            "adding correlation must further deflate n_eff: {correlated} !< {independent}"
        );
    }

    #[test]
    fn agreement_is_one_when_all_scores_equal() {
        assert!((agreement(&[0.7, 0.7, 0.7]).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn agreement_drops_with_spread() {
        let tight = agreement(&[0.5, 0.55, 0.45]).unwrap();
        let wide = agreement(&[0.0, 1.0, 0.5]).unwrap();
        assert!(wide < tight);
    }

    #[test]
    fn non_finite_weight_is_rejected() {
        let r = CorrelationMatrix::identity(2);
        assert!(matches!(
            effective_n(&[1.0, f64::NAN], &r),
            Err(JudgeError::Invariant(_))
        ));
        assert!(matches!(
            effective_n(&[1.0, -1.0], &r),
            Err(JudgeError::Invariant(_))
        ));
    }

    #[test]
    fn is_identity_detects_identity_and_non_identity() {
        // V8: the helper that drives the grader's "correlation guard inert" warning.
        assert!(CorrelationMatrix::identity(3).is_identity());
        assert!(CorrelationMatrix::identity(1).is_identity());
        assert!(!CorrelationMatrix::uniform_offdiagonal(3, 0.5).is_identity());
        assert!(!CorrelationMatrix::uniform_offdiagonal(2, 1.0).is_identity());
        // A 0-dim matrix is not "identity" (no panel to guard).
        assert!(!CorrelationMatrix::identity(0).is_identity());
    }

    #[test]
    fn submatrix_extracts_principal_block_and_preserves_neff() {
        // V2: dropping a judge must drop its row+column. A 3x3 correlated matrix sliced to its 2
        // independent... here build a matrix where rows {0,2} are independent of each other but
        // row 1 is correlated, and confirm the 2x2 submatrix over {0,2} is the identity.
        let r = CorrelationMatrix::from_rows(vec![
            vec![1.0, 0.8, 0.0],
            vec![0.8, 1.0, 0.8],
            vec![0.0, 0.8, 1.0],
        ])
        .unwrap();
        let sub = r.submatrix(&[0, 2]);
        assert_eq!(sub.dim(), 2);
        assert!(sub.is_identity(), "rows {{0,2}} are mutually independent");
        // n_eff over the 2 independent decisive judges is ~2.
        let n = effective_n(&[1.0, 1.0], &sub).unwrap();
        assert!((n - 2.0).abs() < 1e-9);
    }

    #[test]
    fn submatrix_skips_out_of_range_indices() {
        let r = CorrelationMatrix::identity(2);
        let sub = r.submatrix(&[0, 5]); // 5 is out of range, dropped
        assert_eq!(sub.dim(), 1);
    }
}
