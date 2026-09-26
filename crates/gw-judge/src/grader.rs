//! The `HybridGrader` — the two-rail composition (ARCHITECTURE §5 wiring; JUDGE-DESIGN §1.3, §8).
//!
//! Order of operations:
//! 1. **Verifier rail** (rail 1, `verifier.rs`) where ground truth exists — a hard `Reject`
//!    short-circuits the WHOLE grade (no judge tokens spent). The verifier is **authoritative**:
//!    where `Verification.all_passed == false` the record is REJECTED regardless of any panel
//!    score (`verifier_authoritative` enforced in `HybridGrader::grade`).
//! 2. **JudgePanel rail** (rail 2, `panel.rs` + the never-re-spend `cache.rs`) on the remainder /
//!    as advisory — a blind sealed parallel pass.
//! 3. **consensus** (`consensus.rs`): calibration-weighted aggregate gated by the
//!    correlation-adjusted `n_eff`; `n_eff/k < min_n_eff_ratio` (or `< min_n_eff`) ESCALATES
//!    rather than trusting a correlated panel.
//! 4. **threshold** → `Accept | Revise | Reject` from the per-area bands, with the
//!    `threshold_at_decision` STORED so admission re-derives under a new threshold WITHOUT
//!    re-judging ([`rederive_verdict`]).
//!
//! [`HybridGrader::grade`] populates a [`GradeOutcome`] carrying both the [`Decision`] and the
//! [`Judging`] block to persist.

use gw_schema::{Judging, Verification};

use crate::calibration::{CalibrationParams, calibration_weights};
use crate::consensus::{CorrelationMatrix, agreement, effective_n, weighted_aggregate};
use crate::decision::{Decision, DecisionReason, EscalateTo, Verdict};
use crate::error::{JudgeError, Result};
use crate::panel::Grade;
use crate::verifier::{VerifierGrade, verifier_reject_decision};
use gw_schema::RatingRecord;

/// The per-area admission thresholds (JUDGE-DESIGN §10). The aggregate is compared against these
/// bands AFTER the correlation guard. `accept_threshold` and `reject_below` define the three bands;
/// `min_n_eff_ratio` / `min_n_eff` are the correlation-guard floors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AreaThresholds {
    /// Admit at or above this score (length-controlled, calibration-weighted). Default `0.80`.
    pub accept_threshold: f64,
    /// Reject below this score. The `[reject_below, accept_threshold)` window is the revise band.
    pub reject_below: f64,
    /// Escalate if `n_eff/k < this` (default `0.50`).
    pub min_n_eff_ratio: f64,
    /// Absolute `n_eff` floor; escalate if `n_eff < this` (default `1.5`).
    pub min_n_eff: f64,
}

impl Default for AreaThresholds {
    fn default() -> Self {
        Self {
            accept_threshold: 0.80,
            reject_below: 0.60,
            min_n_eff_ratio: 0.50,
            min_n_eff: 1.5,
        }
    }
}

impl AreaThresholds {
    /// Map an aggregate score to the band verdict, IGNORING the correlation guard (that runs first
    /// in the `HybridGrader::decide` step). `>= accept_threshold` ⇒ Accept, `< reject_below` ⇒
    /// Reject, the window between ⇒ Revise. This is the re-derivation core: stored aggregate + a
    /// (possibly new) threshold → verdict, no re-judging.
    #[must_use]
    pub fn band(&self, aggregate: f64) -> Decision {
        if aggregate >= self.accept_threshold {
            Decision::Accept {
                reason: DecisionReason::AboveThreshold,
            }
        } else if aggregate < self.reject_below {
            Decision::Reject {
                reason: DecisionReason::BelowThreshold,
            }
        } else {
            Decision::Revise {
                reason: DecisionReason::ReviseBand,
            }
        }
    }
}

/// The full outcome of grading one candidate: the panel-level [`Decision`] plus the [`Judging`]
/// block to persist (panel votes, aggregate, agreement, n_eff, verdict, verdict_reason,
/// threshold_at_decision) and the [`Verification`] block from rail 1.
#[derive(Debug, Clone, PartialEq)]
pub struct GradeOutcome {
    /// The panel-level decision (maps to the lifecycle).
    pub decision: Decision,
    /// The persisted judging block (INVARIANT-f: aggregate is the weighted consensus, never a mean).
    pub judging: Judging,
    /// The persisted verification block from the deterministic rail.
    pub verification: Verification,
}

/// The two-rail grader. Holds the area thresholds + calibration params; the verifier grade and
/// panel grades are produced by the caller (the verifier rail is pure; the panel goes through the
/// cache) and handed to [`grade`](HybridGrader::grade), keeping this composition step free of I/O.
#[derive(Debug, Clone)]
pub struct HybridGrader {
    /// Per-area admission bands + correlation-guard floors.
    pub thresholds: AreaThresholds,
    /// Calibration softmax coefficients.
    pub calibration: CalibrationParams,
}

impl HybridGrader {
    /// A grader with the given area thresholds and default calibration params.
    #[must_use]
    pub fn new(thresholds: AreaThresholds) -> Self {
        Self {
            thresholds,
            calibration: CalibrationParams::default(),
        }
    }

    /// Grade one candidate from an ALREADY-COMPUTED verifier grade (rail 1) and panel grades (rail
    /// 2), plus the judge `ratings` (for calibration weights) and an inter-judge correlation matrix
    /// `r` (estimated upstream; cold-start callers pass a conservative prior).
    ///
    /// This is the pure composition step — it spends nothing. The verifier authoritative gate fires
    /// FIRST: a hard verifier reject sinks the record regardless of the panel. Otherwise the panel
    /// consensus is computed and gated by `n_eff`, then thresholded.
    ///
    /// # Errors
    /// - [`JudgeError::EmptyPanel`] if there is no verifier hard-reject AND no panel grades (nothing
    ///   could decide the record).
    /// - [`JudgeError::Invariant`] if the correlation matrix dimension does not match the panel.
    pub fn grade(
        &self,
        verifier: Option<&VerifierGrade>,
        panel: &[Grade],
        ratings: &[RatingRecord],
        area: Option<&str>,
        r: &CorrelationMatrix,
    ) -> Result<GradeOutcome> {
        // 1. Verifier authoritative hard gate — rejects regardless of the panel (INVARIANT).
        if let Some(v) = verifier
            && v.is_hard_reject()
        {
            let decision = verifier_reject_decision();
            let judging = Judging {
                panel: panel.iter().map(Grade::to_vote).collect(),
                aggregate: None,
                agreement: None,
                n_eff: None,
                verdict: decision.to_schema_verdict(),
                verdict_reason: Some(decision.reason().as_str().to_string()),
                threshold_at_decision: Some(self.thresholds.accept_threshold),
            };
            return Ok(GradeOutcome {
                decision,
                judging,
                verification: v.verification.clone(),
            });
        }

        let verification = verifier.map(|v| v.verification.clone()).unwrap_or_default();

        // 1b. A deterministic axis that could not decide BLOCKS admission and is NOT handed to the
        // panel: a panel score cannot stand in for ground truth the deterministic rail could not
        // obtain, so a high-scoring panel must not turn "unproven" into an admit. The record is held
        // for review. This is distinct from (and strictly softer than) the hard reject above, which
        // means a failure WAS proven. `aggregate: None` so `rederive_verdict` reproduces this
        // Escalate on the reconcile edge instead of re-banding a panel score into an admit.
        if let Some(v) = verifier
            && v.blocks_admission()
        {
            let decision = Decision::Escalate {
                to: EscalateTo::Verifier,
                reason: DecisionReason::VerifierUndecided,
            };
            let judging = Judging {
                panel: panel.iter().map(Grade::to_vote).collect(),
                aggregate: None,
                agreement: None,
                n_eff: None,
                verdict: decision.to_schema_verdict(),
                verdict_reason: Some(decision.reason().as_str().to_string()),
                threshold_at_decision: Some(self.thresholds.accept_threshold),
            };
            return Ok(GradeOutcome {
                decision,
                judging,
                verification,
            });
        }

        // The verifier passed (or is absent). The panel decides the remainder.
        if panel.is_empty() {
            return Err(JudgeError::EmptyPanel(
                "no verifier hard-reject and no panel grades; nothing can decide the record".into(),
            ));
        }

        let (decision, judging) = self.decide(panel, ratings, area, r)?;
        Ok(GradeOutcome {
            decision,
            judging,
            verification,
        })
    }

    /// Run the panel consensus + correlation guard + threshold, producing the panel-level decision
    /// and the [`Judging`] block. The aggregate is the calibration-weighted consensus
    /// ([`weighted_aggregate`]), NEVER a plain mean; `n_eff` is the correlation-adjusted design
    /// effect. When `n_eff/k < min_n_eff_ratio` or `n_eff < min_n_eff`, the panel is too correlated
    /// to trust → [`Decision::Escalate`].
    fn decide(
        &self,
        panel: &[Grade],
        ratings: &[RatingRecord],
        area: Option<&str>,
        r: &CorrelationMatrix,
    ) -> Result<(Decision, Judging)> {
        let k = panel.len();
        let threshold = self.thresholds.accept_threshold;
        let persisted_votes: Vec<_> = panel.iter().map(Grade::to_vote).collect();

        // V8 observability: a k>1 panel graded against an identity R has its correlation guard
        // silently degraded to Kish-only (weight-concentration only). The engine must supply the
        // estimated R (or the cold-start prior); warn loudly when it did not.
        if k > 1 && r.is_identity() {
            tracing::warn!(
                panel_size = k,
                "correlation guard inert: identity R for a k>1 panel — n_eff degrades to Kish-only \
                 (engine should supply the estimated inter-judge R or the cold-start prior)"
            );
        }

        // V6 (Phase-0): the per-judge `dimensions` B7 reject axes (`over_refusal`,
        // `groundedness_sycophancy`) are RECORDED on the votes but NOT read here — their §5.8
        // soft-minority-veto enforcement is a tracked follow-up. `decide` thresholds the aggregate
        // only; it does not yet gate on a dimension axis.

        // V2: EXCLUDE Uncertain grades from the aggregate, weights, and the design effect. An
        // Uncertain grade carries no accept/reject signal (panel.rs contract), so letting its 0.0
        // score enter the weighted aggregate would silently deflate a clean accept.
        let decisive: Vec<usize> = panel
            .iter()
            .enumerate()
            .filter(|(_, g)| g.verdict != Verdict::Uncertain)
            .map(|(i, _)| i)
            .collect();

        // Whole panel Uncertain ⇒ no decisive signal ⇒ conservative tie REJECT. Persist
        // aggregate=None (V4) so a round-trip rederive_verdict reproduces this Reject instead of
        // re-banding a stale aggregate into an Admit.
        if decisive.is_empty() {
            let decision = Decision::Reject {
                reason: DecisionReason::ConservativeTie,
            };
            let judging = Judging {
                panel: persisted_votes,
                aggregate: None,
                agreement: agreement(&panel.iter().map(|g| g.score).collect::<Vec<_>>()).ok(),
                n_eff: None,
                verdict: decision.to_schema_verdict(),
                verdict_reason: Some(decision.reason().as_str().to_string()),
                threshold_at_decision: Some(threshold),
            };
            return Ok((decision, judging));
        }

        // Compute the consensus over the DECISIVE subset only.
        let slugs: Vec<String> = decisive
            .iter()
            .map(|&i| panel[i].judge_model.clone())
            .collect();
        let scores: Vec<f64> = decisive.iter().map(|&i| panel[i].score).collect();
        let weights = calibration_weights(&slugs, ratings, area, self.calibration);
        if weights.len() != decisive.len() {
            return Err(JudgeError::Invariant(format!(
                "calibration produced {} weights for {} decisive judges",
                weights.len(),
                decisive.len()
            )));
        }
        let r_decisive = r.submatrix(&decisive);

        // INVARIANT-f: the aggregate is the calibration-weighted consensus, not a mean/majority.
        let aggregate = weighted_aggregate(&scores, &weights)?;
        let agree = agreement(&scores)?;
        let n_eff = effective_n(&weights, &r_decisive)?;

        // 2. Correlation guard (§5.5): a too-correlated (or too-thin, after excluding Uncertain)
        // decisive panel cannot be trusted → escalate. The ratio is over the DECISIVE count, so a
        // partial-Uncertain panel that leaves too few decisive judges naturally escalates here.
        let kd = decisive.len();
        let ratio = n_eff / kd as f64;
        let decision =
            if ratio < self.thresholds.min_n_eff_ratio || n_eff < self.thresholds.min_n_eff {
                Decision::Escalate {
                    to: EscalateTo::Human,
                    reason: DecisionReason::CorrelatedJudges,
                }
            } else {
                // 3. Threshold the trusted aggregate into the band verdict.
                self.thresholds.band(aggregate)
            };

        let judging = Judging {
            panel: persisted_votes,
            aggregate: Some(aggregate),
            agreement: Some(agree),
            n_eff: Some(n_eff),
            verdict: decision.to_schema_verdict(),
            verdict_reason: Some(decision.reason().as_str().to_string()),
            threshold_at_decision: Some(threshold),
        };
        Ok((decision, judging))
    }
}

/// Re-derive the admission [`Decision`] from an ALREADY-PERSISTED [`Judging`] block under a (possibly
/// NEW) `thresholds`, WITHOUT re-judging (the re-derivable-admission invariant; ARCHITECTURE §5).
///
/// Uses the stored `aggregate` and `n_eff`: it re-applies the same correlation guard and band logic
/// the live grade used, so the same stored panel under two different accept-thresholds yields two
/// different verdicts with zero model calls. A `NeedsReview` / escalate that was forced by the
/// correlation guard stays escalated (the guard does not depend on the threshold). A block with no
/// stored aggregate (e.g. a verifier hard-reject) re-derives to the verifier reject it recorded.
///
/// # Errors
/// Returns [`JudgeError::Invariant`] if the block carries neither an aggregate nor a persisted
/// verdict (it was never graded).
pub fn rederive_verdict(judging: &Judging, thresholds: AreaThresholds) -> Result<Decision> {
    // A block with no aggregate (a verifier hard-reject, or an all-Uncertain conservative tie)
    // re-derives to its recorded verdict — there is no aggregate to threshold. The stored
    // verdict_reason is preserved so a ConservativeTie is not re-attributed as a VerifierReject.
    let Some(aggregate) = judging.aggregate else {
        // A block with no aggregate carries its terminal verdict; the recorded reason is preserved so
        // it is not re-attributed (a ConservativeTie must not become a VerifierReject, and a
        // record held for an undecidable deterministic axis must not become "correlated judges").
        let recorded = if judging.verdict_reason.as_deref()
            == Some(DecisionReason::ConservativeTie.as_str())
        {
            DecisionReason::ConservativeTie
        } else if judging.verdict_reason.as_deref()
            == Some(DecisionReason::VerifierUndecided.as_str())
        {
            DecisionReason::VerifierUndecided
        } else {
            DecisionReason::VerifierReject
        };
        return match judging.verdict {
            Some(gw_schema::Verdict::Reject) => Ok(Decision::Reject { reason: recorded }),
            Some(gw_schema::Verdict::NeedsReview) => Ok(Decision::Escalate {
                to: match recorded {
                    // The panel was never consulted; the deterministic rail could not decide.
                    DecisionReason::VerifierUndecided => EscalateTo::Verifier,
                    _ => EscalateTo::Human,
                },
                reason: recorded,
            }),
            Some(gw_schema::Verdict::Admit) | None => Err(JudgeError::Invariant(
                "judging block has no aggregate to re-derive a verdict from".into(),
            )),
        };
    };

    // Re-apply the correlation guard from the stored n_eff (the guard is threshold-independent).
    if let Some(n_eff) = judging.n_eff {
        let k = judging.panel.len().max(1) as f64;
        if n_eff / k < thresholds.min_n_eff_ratio || n_eff < thresholds.min_n_eff {
            return Ok(Decision::Escalate {
                to: EscalateTo::Human,
                reason: DecisionReason::CorrelatedJudges,
            });
        }
    }
    Ok(thresholds.band(aggregate))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::Verdict;
    use gw_schema::Verdict as SchemaVerdict;

    fn grade(slug: &str, score: f64, verdict: Verdict) -> Grade {
        Grade {
            judge_model: slug.into(),
            score,
            verdict,
            confidence: 0.5,
            meta_prediction: None,
            dimensions: None,
            rationale: None,
            raw: serde_json::Value::Null,
            temperature: 0.0,
            top_p: None,
            seed: None,
            rubric_id: None,
        }
    }

    fn verifier_pass() -> VerifierGrade {
        VerifierGrade {
            verdict: Verdict::Accept,
            verification: Verification {
                all_passed: true,
                ..Default::default()
            },
        }
    }

    fn verifier_fail() -> VerifierGrade {
        VerifierGrade {
            verdict: Verdict::Reject,
            verification: Verification {
                all_passed: false,
                ..Default::default()
            },
        }
    }

    /// A deterministic axis that could not decide BLOCKS admission: the panel is not a substitute
    /// for ground truth the deterministic rail could not obtain, so a glowing panel must escalate
    /// to review rather than admit. And the persisted block carries no aggregate, so the reconcile
    /// edge re-derives the same hold instead of re-banding the score into an admit.
    #[test]
    fn verifier_undecided_escalates_and_never_admits_on_a_high_panel() {
        let verifier = VerifierGrade {
            verdict: Verdict::Accept,
            verification: Verification {
                checks: vec![],
                all_passed: true,
                needs_review: Some("execution_evidence: undecided".into()),
            },
        };
        let panel = vec![
            grade("a", 0.99, Verdict::Accept),
            grade("b", 0.99, Verdict::Accept),
            grade("c", 0.99, Verdict::Accept),
        ];
        let r = crate::consensus::CorrelationMatrix::uniform_offdiagonal(3, 0.7);
        let grader = HybridGrader::new(AreaThresholds::default());
        let out = grader
            .grade(Some(&verifier), &panel, &[], Some("math"), &r)
            .unwrap();
        assert!(
            matches!(out.decision, Decision::Escalate { .. }),
            "an undecidable deterministic axis must escalate, got {:?}",
            out.decision
        );
        assert_eq!(out.judging.verdict, Some(SchemaVerdict::NeedsReview));
        assert_eq!(
            out.judging.aggregate, None,
            "no aggregate is persisted, so reconcile cannot re-band the score into an admit"
        );
        // The reconcile edge reproduces the same hold (and the same reason).
        let rederived = rederive_verdict(&out.judging, AreaThresholds::default()).unwrap();
        assert_eq!(rederived, out.decision);
    }

    /// A record with no execution axis is untouched: `needs_review` unset means the panel decides.
    #[test]
    fn a_plain_verifier_pass_still_hands_the_record_to_the_panel() {
        let panel = vec![
            grade("a", 0.9, Verdict::Accept),
            grade("b", 0.85, Verdict::Accept),
            grade("c", 0.88, Verdict::Accept),
        ];
        let r = CorrelationMatrix::identity(3);
        let grader = HybridGrader::new(AreaThresholds::default());
        let out = grader
            .grade(Some(&verifier_pass()), &panel, &[], Some("math"), &r)
            .unwrap();
        assert!(matches!(out.decision, Decision::Accept { .. }));
    }

    /// Mandatory test 5: a verifier REJECT sinks the record even with a high panel score.
    #[test]
    fn verifier_reject_sinks_high_panel_score() {
        let grader = HybridGrader::new(AreaThresholds::default());
        // A unanimous, glowing panel — but the verifier hard-failed.
        let panel = vec![
            grade("a", 1.0, Verdict::Accept),
            grade("b", 1.0, Verdict::Accept),
            grade("c", 1.0, Verdict::Accept),
        ];
        let r = CorrelationMatrix::identity(3);
        let out = grader
            .grade(Some(&verifier_fail()), &panel, &[], None, &r)
            .unwrap();
        assert!(matches!(out.decision, Decision::Reject { .. }));
        assert_eq!(out.judging.verdict, Some(SchemaVerdict::Reject));
        assert_eq!(
            out.judging.verdict_reason.as_deref(),
            Some("verifier_reject")
        );
        // The panel votes are still recorded (audit), but the aggregate was never trusted.
        assert_eq!(out.judging.panel.len(), 3);
        assert!(out.judging.aggregate.is_none());
    }

    #[test]
    fn independent_high_panel_admits() {
        let grader = HybridGrader::new(AreaThresholds::default());
        let panel = vec![
            grade("a", 0.9, Verdict::Accept),
            grade("b", 0.85, Verdict::Accept),
            grade("c", 0.88, Verdict::Accept),
        ];
        let r = CorrelationMatrix::identity(3);
        let out = grader
            .grade(Some(&verifier_pass()), &panel, &[], None, &r)
            .unwrap();
        assert!(matches!(out.decision, Decision::Accept { .. }));
        assert_eq!(out.judging.verdict, Some(SchemaVerdict::Admit));
        // n_eff ≈ k for independent equal-weight judges.
        assert!((out.judging.n_eff.unwrap() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn correlated_panel_escalates_even_when_high() {
        let grader = HybridGrader::new(AreaThresholds::default());
        let panel = vec![
            grade("a", 0.95, Verdict::Accept),
            grade("b", 0.95, Verdict::Accept),
            grade("c", 0.95, Verdict::Accept),
        ];
        // Perfectly correlated → n_eff ≈ 1 < min_n_eff (1.5) → escalate, do NOT admit.
        let r = CorrelationMatrix::uniform_offdiagonal(3, 1.0);
        let out = grader
            .grade(Some(&verifier_pass()), &panel, &[], None, &r)
            .unwrap();
        assert!(matches!(
            out.decision,
            Decision::Escalate {
                reason: DecisionReason::CorrelatedJudges,
                ..
            }
        ));
        assert_eq!(out.judging.verdict, Some(SchemaVerdict::NeedsReview));
    }

    #[test]
    fn below_threshold_rejects() {
        let grader = HybridGrader::new(AreaThresholds::default());
        let panel = vec![
            grade("a", 0.3, Verdict::Reject),
            grade("b", 0.4, Verdict::Reject),
            grade("c", 0.2, Verdict::Reject),
        ];
        let r = CorrelationMatrix::identity(3);
        let out = grader.grade(None, &panel, &[], None, &r).unwrap();
        assert!(matches!(out.decision, Decision::Reject { .. }));
    }

    #[test]
    fn revise_band_routes_to_revise() {
        let grader = HybridGrader::new(AreaThresholds::default());
        let panel = vec![
            grade("a", 0.7, Verdict::Revise),
            grade("b", 0.65, Verdict::Revise),
            grade("c", 0.72, Verdict::Revise),
        ];
        let r = CorrelationMatrix::identity(3);
        let out = grader.grade(None, &panel, &[], None, &r).unwrap();
        assert!(matches!(out.decision, Decision::Revise { .. }));
        // Revise is transient: no terminal verdict persisted.
        assert_eq!(out.judging.verdict, None);
    }

    #[test]
    fn all_uncertain_panel_conservatively_rejects() {
        let grader = HybridGrader::new(AreaThresholds::default());
        // Scores happen to sit high, but every verdict is Uncertain → conservative reject.
        let panel = vec![
            grade("a", 0.9, Verdict::Uncertain),
            grade("b", 0.9, Verdict::Uncertain),
            grade("c", 0.9, Verdict::Uncertain),
        ];
        let r = CorrelationMatrix::identity(3);
        let out = grader.grade(None, &panel, &[], None, &r).unwrap();
        assert!(matches!(
            out.decision,
            Decision::Reject {
                reason: DecisionReason::ConservativeTie
            }
        ));
        // V4: the conservative-tie persists aggregate=None so a round-trip reproduces the Reject.
        assert_eq!(out.judging.aggregate, None);
        assert_eq!(out.judging.n_eff, None);
    }

    /// V2: one Uncertain grade in a 3-judge panel does NOT drag the aggregate — the two decisive
    /// 0.9s still Accept (a pre-fix impl let the Uncertain's 0.0 deflate 0.9,0.9,0.0 → 0.60 → Revise).
    #[test]
    fn one_uncertain_does_not_drag_the_aggregate() {
        let grader = HybridGrader::new(AreaThresholds::default());
        let panel = vec![
            grade("a", 0.9, Verdict::Accept),
            grade("b", 0.9, Verdict::Accept),
            grade("c", 0.0, Verdict::Uncertain),
        ];
        let r = CorrelationMatrix::identity(3);
        let out = grader.grade(None, &panel, &[], None, &r).unwrap();
        assert!(
            matches!(out.decision, Decision::Accept { .. }),
            "decisive 0.9s must still Accept, got {:?}",
            out.decision
        );
        // The aggregate is over the two DECISIVE judges only (0.9), not dragged to 0.60 by the 0.0.
        assert!((out.judging.aggregate.unwrap() - 0.9).abs() < 1e-9);
        // n_eff is over the 2 decisive judges.
        assert!((out.judging.n_eff.unwrap() - 2.0).abs() < 1e-9);
        // All three votes are still persisted for audit.
        assert_eq!(out.judging.panel.len(), 3);
    }

    /// V2: when excluding Uncertain leaves too few decisive judges to trust, the panel escalates
    /// (the correlation/min_n_eff guard over the decisive subset fires).
    #[test]
    fn partial_uncertain_escalates_when_too_few_decisive() {
        let grader = HybridGrader::new(AreaThresholds::default());
        // Only ONE decisive judge survives → n_eff = 1 < min_n_eff (1.5) → escalate.
        let panel = vec![
            grade("a", 0.95, Verdict::Accept),
            grade("b", 0.0, Verdict::Uncertain),
            grade("c", 0.0, Verdict::Uncertain),
        ];
        let r = CorrelationMatrix::identity(3);
        let out = grader.grade(None, &panel, &[], None, &r).unwrap();
        assert!(matches!(
            out.decision,
            Decision::Escalate {
                reason: DecisionReason::CorrelatedJudges,
                ..
            }
        ));
    }

    /// V4: a round-trip rederive of the all-Uncertain block at the ORIGINAL thresholds reproduces
    /// Reject (a pre-fix impl persisted aggregate=Some(0.9) and re-banded it to Admit).
    #[test]
    fn conservative_tie_survives_round_trip() {
        let grader = HybridGrader::new(AreaThresholds::default());
        let panel = vec![
            grade("a", 0.9, Verdict::Uncertain),
            grade("b", 0.9, Verdict::Uncertain),
        ];
        let r = CorrelationMatrix::identity(2);
        let out = grader.grade(None, &panel, &[], None, &r).unwrap();
        assert!(matches!(out.decision, Decision::Reject { .. }));
        // Re-derive at the SAME thresholds — must stay Reject, not flip to Admit.
        let red = rederive_verdict(&out.judging, AreaThresholds::default()).unwrap();
        assert!(
            matches!(red, Decision::Reject { .. }),
            "conservative-tie Reject must round-trip, got {red:?}"
        );
    }

    #[test]
    fn no_verifier_reject_and_empty_panel_errors() {
        let grader = HybridGrader::new(AreaThresholds::default());
        let r = CorrelationMatrix::identity(0);
        let err = grader.grade(None, &[], &[], None, &r).unwrap_err();
        assert!(matches!(err, JudgeError::EmptyPanel(_)));
    }

    /// Mandatory test 7: the SAME stored panel + two thresholds → two verdicts, no re-judging.
    #[test]
    fn threshold_rederivation_changes_verdict_without_rejudging() {
        let grader = HybridGrader::new(AreaThresholds::default());
        let panel = vec![
            grade("a", 0.72, Verdict::Revise),
            grade("b", 0.74, Verdict::Revise),
            grade("c", 0.73, Verdict::Revise),
        ];
        let r = CorrelationMatrix::identity(3);
        let out = grader.grade(None, &panel, &[], None, &r).unwrap();
        // At the default 0.80 accept threshold this aggregate (~0.73) is a Revise.
        assert!(matches!(out.decision, Decision::Revise { .. }));

        // Re-derive under a LOWER accept threshold (0.70): now it admits — no model call.
        let lenient = AreaThresholds {
            accept_threshold: 0.70,
            reject_below: 0.50,
            ..AreaThresholds::default()
        };
        let redrived = rederive_verdict(&out.judging, lenient).unwrap();
        assert!(matches!(redrived, Decision::Accept { .. }));

        // Re-derive under a STRICTER reject_below (0.75): now it rejects.
        let strict = AreaThresholds {
            accept_threshold: 0.90,
            reject_below: 0.75,
            ..AreaThresholds::default()
        };
        let redrived2 = rederive_verdict(&out.judging, strict).unwrap();
        assert!(matches!(redrived2, Decision::Reject { .. }));
    }

    #[test]
    fn rederive_keeps_correlated_escalation() {
        let grader = HybridGrader::new(AreaThresholds::default());
        let panel = vec![
            grade("a", 0.95, Verdict::Accept),
            grade("b", 0.95, Verdict::Accept),
        ];
        let r = CorrelationMatrix::uniform_offdiagonal(2, 1.0);
        let out = grader.grade(None, &panel, &[], None, &r).unwrap();
        assert!(matches!(out.decision, Decision::Escalate { .. }));
        // Even under a very lenient threshold, a correlated panel stays escalated.
        let lenient = AreaThresholds {
            accept_threshold: 0.10,
            reject_below: 0.05,
            ..AreaThresholds::default()
        };
        let red = rederive_verdict(&out.judging, lenient).unwrap();
        assert!(matches!(red, Decision::Escalate { .. }));
    }

    #[test]
    fn rederive_verifier_reject_stays_rejected() {
        let grader = HybridGrader::new(AreaThresholds::default());
        let panel = vec![grade("a", 1.0, Verdict::Accept)];
        let r = CorrelationMatrix::identity(1);
        let out = grader
            .grade(Some(&verifier_fail()), &panel, &[], None, &r)
            .unwrap();
        let red = rederive_verdict(&out.judging, AreaThresholds::default()).unwrap();
        assert!(matches!(red, Decision::Reject { .. }));
    }
}
