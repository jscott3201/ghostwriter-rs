//! Pure validation of the resolved equal-weight, constant-correlation admission contract.

use gw_schema::AdmissionIntent;

use crate::{AreaThresholds, CorrelationMatrix, JudgeError, Result};

/// Conservative cold-start correlation assumption; this is not measured judge calibration.
pub const DEFAULT_CORRELATION_RHO: f64 = 0.7;

/// Assessment of whether any decisive subset can clear both effective-count safeguards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PanelAssessment {
    /// First decisive count that can satisfy both floors, or `None` for an unattainable review-only
    /// configuration. Uncertain votes are excluded, so this count may be smaller than the panel.
    pub feasible_decisive_count: Option<usize>,
}

impl AreaThresholds {
    /// Validate finite, ordered score bands and effective-count floors.
    ///
    /// # Errors
    /// Returns a configuration error for out-of-domain or non-finite settings.
    pub fn validate(&self) -> Result<()> {
        if !self.reject_below.is_finite()
            || !self.accept_threshold.is_finite()
            || self.reject_below < 0.0
            || self.reject_below > self.accept_threshold
            || self.accept_threshold > 1.0
        {
            return Err(JudgeError::Invariant("panel preflight: score bands require finite 0 <= reject_below <= accept_threshold <= 1".into()));
        }
        if !self.min_n_eff.is_finite() || self.min_n_eff < 0.0 {
            return Err(JudgeError::Invariant(
                "panel preflight: min_n_eff must be finite and nonnegative".into(),
            ));
        }
        if !self.min_n_eff_ratio.is_finite() || !(0.0..=1.0).contains(&self.min_n_eff_ratio) {
            return Err(JudgeError::Invariant(
                "panel preflight: min_n_eff_ratio must be finite and within [0,1]".into(),
            ));
        }
        Ok(())
    }
}

/// Validate the engine's cold-start correlation prior without building a full panel matrix.
///
/// # Errors
/// Requires finite `0 <= rho <= 1`; multi-judge panels additionally require a positive prior
/// that is numerically nonidentity under the consensus matrix's existing policy.
pub fn validate_correlation_prior(judge_count: usize, rho: f64) -> Result<()> {
    if !rho.is_finite() || !(0.0..=1.0).contains(&rho) {
        return Err(JudgeError::Invariant(
            "panel preflight: correlation_rho must be finite and within [0,1]".into(),
        ));
    }
    if judge_count > 1 && CorrelationMatrix::uniform_offdiagonal(2, rho).is_identity() {
        return Err(JudgeError::Invariant(
            "panel preflight: multi-judge correlation prior must be positive and nonidentity"
                .into(),
        ));
    }
    Ok(())
}

/// Assess the resolved generation panel before any provider spending.
///
/// The cold start uses equal weights and a constant assumed `rho`. Every possible nonempty
/// decisive count is considered: `n_eff = d / (1 + (d - 1) * rho)`. This assessment establishes
/// mathematical attainability under that assumption, not empirical calibration or future acceptance.
/// Review-only intent bypasses only attainability; numeric domains and a nonempty panel still apply.
///
/// # Errors
/// Rejects empty panels, invalid numeric settings, and unattainable automatic admission.
pub fn assess_panel(
    judge_count: usize,
    thresholds: AreaThresholds,
    rho: f64,
    intent: AdmissionIntent,
) -> Result<PanelAssessment> {
    thresholds.validate()?;
    validate_correlation_prior(judge_count, rho)?;
    if judge_count == 0 {
        return Err(JudgeError::EmptyPanel("panel preflight: generation requires at least one judge, including review-only collection".into()));
    }
    let feasible_decisive_count = (1..=judge_count).find(|&count| {
        let d = count as f64;
        let n_eff = d / (1.0 + (d - 1.0) * rho);
        n_eff >= thresholds.min_n_eff && n_eff / d >= thresholds.min_n_eff_ratio
    });
    if intent == AdmissionIntent::Automatic && feasible_decisive_count.is_none() {
        return Err(JudgeError::Invariant(format!(
            "panel preflight: automatic admission is unattainable for {judge_count} judges with correlation_rho={rho}, min_n_eff={}, min_n_eff_ratio={}; configure an attainable panel assumption or explicitly select review_only",
            thresholds.min_n_eff, thresholds.min_n_eff_ratio
        )));
    }
    Ok(PanelAssessment {
        feasible_decisive_count,
    })
}

/// Check typed persisted evidence before choosing the denominator for rederivation. Historical
/// rows without a decisive count retain the conservative full-panel denominator.
pub(crate) fn persisted_decisive_count(judging: &gw_schema::Judging) -> Result<usize> {
    let Some(count) = judging.decisive_count else {
        if judging.n_eff.is_some_and(|n| !n.is_finite() || n < 0.0) {
            return Err(JudgeError::Invariant(
                "persisted n_eff must be finite and nonnegative".into(),
            ));
        }
        return Ok(judging.panel.len().max(1));
    };
    if count > judging.panel.len() {
        return Err(JudgeError::Invariant(
            "persisted decisive_count exceeds panel length".into(),
        ));
    }
    if count == 0 {
        if judging.aggregate.is_some()
            || judging.n_eff.is_some()
            || judging.verdict != Some(gw_schema::Verdict::Reject)
        {
            return Err(JudgeError::Invariant("zero decisive_count requires an all-uncertain rejection with no aggregate or n_eff".into()));
        }
    } else if judging.aggregate.is_some()
        && !judging
            .n_eff
            .is_some_and(|n| n.is_finite() && n > 0.0 && n <= count as f64 + 1e-12)
    {
        return Err(JudgeError::Invariant(
            "persisted decisive_count requires finite 0 < n_eff <= decisive_count".into(),
        ));
    }
    Ok(count)
}
