//! Pure offline fitting and exact replay verification. Runtime consensus never consumes this output.
use crate::ResolvedCalibrationPanel;
use gw_schema::*;

/// Validate and fit supplied observations without provider calls or I/O.
/// Complete evidence produces only a descriptive ComputedUnqualified snapshot.
/// `records` is a lookup pool with unique `(run_id, record_id)` pairs. Evidence fit/assessment rows
/// define the entire measured population; unreferenced pool records are neither evaluated nor
/// bound in the snapshot.
#[must_use]
pub fn fit_calibration(
    panel: &ResolvedCalibrationPanel,
    records: &[TrainingRecord],
    evidence: &CalibrationEvidence,
) -> CalibrationReport {
    let canonical = crate::calibration_intake::canonical_evidence(evidence);
    let intake = crate::calibration_intake::validate(panel, records, &canonical);
    if !intake.invalid.is_empty() {
        return report(CalibrationStatus::InvalidEvidence, intake.invalid);
    }
    if !intake.incomplete.is_empty() {
        return report(CalibrationStatus::IncompleteEvidence, intake.incomplete);
    }
    if !cfg!(any(target_arch = "aarch64", target_arch = "x86_64")) {
        return report(CalibrationStatus::InvalidEvidence, vec!["numerical recipe supports fixed Rust exp dispatch only on aarch64/x86_64; this target is unsupported".into()]);
    }
    match compute(canonical, panel.len()) {
        Ok(snapshot) => CalibrationReport {
            status: CalibrationStatus::ComputedUnqualified,
            reasons: vec![],
            snapshot: Some(snapshot),
        },
        Err(error) => report(CalibrationStatus::InvalidEvidence, vec![error.to_string()]),
    }
}

fn report(status: CalibrationStatus, reasons: Vec<String>) -> CalibrationReport {
    CalibrationReport {
        status,
        reasons,
        snapshot: None,
    }
}

/// Refit and compare every saved bit, identity and result; no epsilon repair is accepted.
///
/// # Errors
/// Rejects invalid/incomplete evidence, modified output, or any refitting mismatch.
pub fn verify_calibration_snapshot(
    panel: &ResolvedCalibrationPanel,
    records: &[TrainingRecord],
    snapshot: &CalibrationSnapshot,
) -> crate::Result<()> {
    let report = fit_calibration(panel, records, &snapshot.evidence);
    let expected = report.snapshot.ok_or_else(|| {
        crate::JudgeError::Invariant("snapshot evidence did not compute completely".into())
    })?;
    if serde_json::to_vec(&expected).map_err(crate::calibration_contract::serialization)?
        != serde_json::to_vec(snapshot).map_err(crate::calibration_contract::serialization)?
    {
        return Err(crate::JudgeError::Invariant(
            "snapshot exact refit mismatch".into(),
        ));
    }
    Ok(())
}

fn compute(evidence: CalibrationEvidence, columns: usize) -> crate::Result<CalibrationSnapshot> {
    let (fit_losses, _, groups) = losses(&evidence.fit, columns, None);
    let weights = weights(&fit_losses, evidence.beta)?;
    let fit = CalibrationFit {
        identity: crate::calibration_intake::fit_identity(&evidence)?,
        candidates: evidence.fit.len(),
        groups,
        losses: fit_losses,
        weights,
        exclusions: 0,
    };
    let (assessment_losses, weighted_score_loss, groups) =
        losses(&evidence.assessment, columns, Some(&fit.weights));
    let assessment = CalibrationAssessment {
        candidates: evidence.assessment.len(),
        groups,
        losses: assessment_losses,
        weighted_score_loss,
    };
    if fit
        .losses
        .iter()
        .chain(&assessment.losses)
        .any(|value| !value.is_finite())
        || !assessment.weighted_score_loss.is_finite()
    {
        return Err(crate::JudgeError::Invariant(
            "nonfinite descriptive score loss".into(),
        ));
    }
    let mut snapshot = CalibrationSnapshot {
        version: CALIBRATION_VERSION,
        status: CalibrationStatus::ComputedUnqualified,
        evidence,
        fit,
        assessment,
        identity: String::new(),
    };
    snapshot.identity =
        crate::calibration_contract::identity("calibration-snapshot-v1", &snapshot)?;
    Ok(snapshot)
}

/// Input rows are already validated and sorted by (global group, full candidate identity).
/// Every reduction is sequential binary64, including the explicit panel-aligned weighted sum.
fn losses(
    rows: &[CalibrationRow],
    columns: usize,
    weights: Option<&[f64]>,
) -> (Vec<f64>, f64, usize) {
    let mut totals = vec![0.0; columns];
    let mut weighted_total = 0.0;
    let mut groups = 0;
    let mut start = 0;
    while start < rows.len() {
        let mut end = start + 1;
        while end < rows.len() && rows[end].prompt_group == rows[start].prompt_group {
            end += 1;
        }
        let mut group_losses = vec![0.0; columns];
        let mut group_weighted_loss = 0.0;
        for row in &rows[start..end] {
            let CalibrationLabel::Known { value: label } = row.label else {
                unreachable!("validated label")
            };
            let mut weighted_score = 0.0;
            for (column, cell) in row.observations.iter().enumerate() {
                let score = cell.score.expect("validated common observation");
                let error = score - label;
                group_losses[column] += error * error;
                if let Some(weights) = weights {
                    weighted_score += weights[column] * score;
                }
            }
            if weights.is_some() {
                let error = weighted_score - label;
                group_weighted_loss += error * error;
            }
        }
        let count = (end - start) as f64;
        for (total, group) in totals.iter_mut().zip(group_losses) {
            *total += group / count;
        }
        weighted_total += group_weighted_loss / count;
        groups += 1;
        start = end;
    }
    for total in &mut totals {
        *total /= groups as f64;
    }
    (totals, weighted_total / groups as f64, groups)
}

fn weights(losses: &[f64], beta: f64) -> crate::Result<Vec<f64>> {
    let minimum = losses.iter().copied().fold(f64::INFINITY, f64::min);
    let mut values = Vec::with_capacity(losses.len());
    let mut sum = 0.0;
    for loss in losses {
        if !loss.is_finite() || !(0.0..=1.0).contains(loss) {
            return Err(crate::JudgeError::Invariant(
                "score losses must be finite and in [0,1]".into(),
            ));
        }
        let value = libm::exp(-beta * (loss - minimum));
        if !value.is_finite() || value <= 0.0 {
            return Err(crate::JudgeError::Invariant(
                "weight exponential underflow or nonfinite value; no fallback is permitted".into(),
            ));
        }
        values.push(value);
        sum += value;
    }
    for value in &mut values {
        *value /= sum;
        if !value.is_finite() || *value <= 0.0 {
            return Err(crate::JudgeError::Invariant(
                "normalized weight must be finite and strictly positive".into(),
            ));
        }
    }
    Ok(values)
}
