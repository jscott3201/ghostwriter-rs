//! Pure grading helpers bridging the persisted envelope and the `gw-judge` rail types.
//!
//! These functions carry NO side effects — they re-derive `gw-judge` decisions from a persisted
//! [`TrainingRecord`] envelope, and they build the load-bearing inter-judge correlation matrix. They
//! live apart from `crate::step` so the R-prior invariant has one obvious home and its own tests.
//!
//! ## The R-prior invariant (LOAD-BEARING — never identity for k > 1)
//!
//! [`correlation_prior`] builds `CorrelationMatrix::uniform_offdiagonal(k, rho)` and ASSERTS the
//! result is non-identity for a `k > 1` panel, returning [`EngineError::Invariant`] otherwise. The
//! gw-judge review flagged that passing `CorrelationMatrix::identity` to a `k > 1` grade silently
//! degrades the correlation guard to Kish-only (weight-concentration only, blind to inter-judge
//! correlation). The engine is the one place that constructs this matrix, so the assertion lives here
//! — a misconfigured `rho` of exactly `0.0` (which would make `uniform_offdiagonal` numerically
//! identity) is caught LOUD rather than silently degrading every admission.

use gw_judge::{
    CorrelationMatrix, Decision, Verdict as JudgeVerdict, VerifierGrade, rederive_verdict,
};
use gw_schema::{TrainingRecord, Verification};

use crate::clients::AreaConfig;
use crate::error::{EngineError, Result};

/// Build the inter-judge correlation matrix for a panel of `k` judges from the cold-start prior
/// `rho`, ENFORCING the never-identity invariant for `k > 1`.
///
/// For `k <= 1` there is no pair to correlate, so the (degenerate) 1×1 or 0×0 matrix is returned as
/// is — the correlation guard is inert at `k <= 1` by construction. For `k > 1` the matrix is
/// `CorrelationMatrix::uniform_offdiagonal(k, rho)`; if that comes out numerically IDENTITY (a `rho`
/// of `0.0`, or non-finite), this returns [`EngineError::Invariant`] — the engine must never hand the
/// grader an identity R for a real panel.
///
/// # Errors
/// Returns [`EngineError::Invariant`] when `k > 1` and the resulting matrix is identity (a `rho` that
/// silently disables the correlation guard).
pub fn correlation_prior(k: usize, rho: f64) -> Result<CorrelationMatrix> {
    if k <= 1 {
        return Ok(CorrelationMatrix::uniform_offdiagonal(k, rho));
    }
    if !rho.is_finite() || rho <= 0.0 {
        return Err(EngineError::Invariant(format!(
            "correlation prior rho={rho} would yield an identity R for a k={k} panel — the \
             correlation guard would silently degrade to Kish-only; rho must be a positive \
             cold-start prior (~0.7)"
        )));
    }
    let r = CorrelationMatrix::uniform_offdiagonal(k, rho);
    if r.is_identity() {
        return Err(EngineError::Invariant(format!(
            "built an identity correlation matrix for a k={k} panel (rho={rho}); refusing to \
             degrade the correlation guard to Kish-only"
        )));
    }
    Ok(r)
}

/// Reconstruct a [`VerifierGrade`] from the persisted `verification` block on a record (pure).
///
/// The verifier rail already ran (the record is at/after `Verified`), so this re-derives the grade
/// from the stored block WITHOUT re-running the rail: `all_passed == false` is a hard reject;
/// otherwise an `Accept` (the panel decides the remainder). This keeps the `Verified → Judged` edge
/// honoring the hard gate without recomputing the deterministic checks.
#[must_use]
pub fn verifier_grade_from_verification(rec: &TrainingRecord, _area: &AreaConfig) -> VerifierGrade {
    let verification: Verification = rec.verification.clone();
    let verdict = if verification.all_passed {
        JudgeVerdict::Accept
    } else {
        JudgeVerdict::Reject
    };
    VerifierGrade {
        verdict,
        verification,
    }
}

/// Re-derive the panel-level [`Decision`] from a record's persisted `Judging` block at the AREA's live
/// thresholds (the `Judged → {Admitted|Rejected|Revising|NeedsReview}` reconciliation), WITHOUT
/// re-judging. Delegates to `gw-judge`'s [`rederive_verdict`], which re-applies the same correlation
/// guard + band logic the live grade used.
///
/// The reconcile edge runs with the area config in hand, so it re-derives at the area's FULL
/// thresholds (`accept_threshold`, `reject_below`, AND the `min_n_eff` / `min_n_eff_ratio` correlation
/// floors) — not just the stored `threshold_at_decision`. The persisted block stores only the accept
/// threshold, so re-deriving from the record alone would lose the n_eff floors and could flip a live
/// escalate into an admit; passing the area thresholds keeps the reconcile faithful to the grade.
///
/// # Errors
/// Returns [`EngineError::Judge`] if the `Judging` block carries neither an aggregate nor a persisted
/// verdict (it was never graded — a programmer error reaching reconcile too early).
pub fn decision_from_judging(rec: &TrainingRecord, area: &AreaConfig) -> Result<Decision> {
    Ok(rederive_verdict(&rec.judging, area.thresholds)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gw_judge::{AreaThresholds, Grade, HybridGrader};
    use gw_schema::Verdict as SchemaVerdict;

    #[test]
    fn correlation_prior_is_non_identity_for_k_gt_1() {
        let r = correlation_prior(3, 0.7).unwrap();
        assert!(!r.is_identity(), "k>1 prior must NOT be identity");
        assert_eq!(r.dim(), 3);
    }

    #[test]
    fn correlation_prior_rejects_zero_rho_for_k_gt_1() {
        // rho=0.0 would make uniform_offdiagonal numerically identity → fail loud.
        let err = correlation_prior(4, 0.0).unwrap_err();
        assert!(matches!(err, EngineError::Invariant(_)));
        assert!(err.to_string().contains("Kish-only"));
    }

    #[test]
    fn correlation_prior_rejects_negative_and_nan_rho() {
        assert!(matches!(
            correlation_prior(2, -0.1).unwrap_err(),
            EngineError::Invariant(_)
        ));
        assert!(matches!(
            correlation_prior(2, f64::NAN).unwrap_err(),
            EngineError::Invariant(_)
        ));
    }

    #[test]
    fn correlation_prior_is_inert_at_k_le_1() {
        // k<=1: no pair to correlate, the degenerate matrix is fine (guard inert by construction).
        assert!(correlation_prior(1, 0.0).is_ok());
        assert!(correlation_prior(0, 0.0).is_ok());
    }

    #[test]
    fn verifier_grade_reflects_all_passed() {
        let mut rec = sample_record();
        rec.verification = Verification {
            all_passed: true,
            ..Default::default()
        };
        let g = verifier_grade_from_verification(&rec, &area());
        assert!(!g.is_hard_reject());

        rec.verification.all_passed = false;
        let g = verifier_grade_from_verification(&rec, &area());
        assert!(g.is_hard_reject());
    }

    #[test]
    fn decision_from_judging_reproduces_admit() {
        // Grade a clean panel with the NON-IDENTITY cold-start prior, persist the judging block, then
        // re-derive the decision. With rho=0.7 on a 3-judge panel n_eff≈1.25 < the default min_n_eff
        // (1.5), so the default thresholds ESCALATE a correlated panel — the correlation guard working
        // as designed. To exercise the admit round-trip we relax the n_eff floor (a real area whose
        // panel is trusted at this size would configure these), keeping the prior NON-identity.
        let thresholds = AreaThresholds {
            min_n_eff: 1.0,
            min_n_eff_ratio: 0.3,
            ..AreaThresholds::default()
        };
        let area = AreaConfig::new("math", "m", vec![], "r").with_thresholds(thresholds);
        let grader = HybridGrader::new(thresholds);
        let panel = vec![grade("a", 0.9), grade("b", 0.88), grade("c", 0.91)];
        let r = correlation_prior(3, 0.7).unwrap();
        assert!(
            !r.is_identity(),
            "the prior handed to the grader must be non-identity"
        );
        let out = grader.grade(None, &panel, &[], Some("math"), &r).unwrap();
        assert_eq!(out.judging.verdict, Some(SchemaVerdict::Admit));

        let mut rec = sample_record();
        rec.judging = out.judging;
        // Reconcile re-derives at the AREA's full thresholds (incl. the relaxed n_eff floor), so the
        // round-trip reproduces the live Admit rather than re-escalating under the default floor.
        let decision = decision_from_judging(&rec, &area).unwrap();
        assert!(matches!(decision, Decision::Accept { .. }));
    }

    #[test]
    fn correlated_prior_escalates_a_small_panel_under_default_floors() {
        // The flip side: under the DEFAULT thresholds, the cold-start rho=0.7 prior makes a 3-judge
        // panel escalate (n_eff≈1.25 < 1.5) — proving the engine is NOT silently passing identity R
        // (which would give n_eff≈3 and admit). This is the V8 guard the R-prior wiring protects.
        let grader = HybridGrader::new(AreaThresholds::default());
        let panel = vec![grade("a", 0.95), grade("b", 0.95), grade("c", 0.95)];
        let r = correlation_prior(3, 0.7).unwrap();
        let out = grader.grade(None, &panel, &[], Some("math"), &r).unwrap();
        assert_eq!(
            out.judging.verdict,
            Some(SchemaVerdict::NeedsReview),
            "rho=0.7 on a 3-panel must escalate under default floors (identity R would admit)"
        );
    }

    fn area() -> AreaConfig {
        AreaConfig::new("math", "m", vec![], "r")
    }

    fn grade(slug: &str, score: f64) -> Grade {
        Grade {
            judge_model: slug.into(),
            score,
            verdict: JudgeVerdict::Accept,
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

    fn sample_record() -> TrainingRecord {
        use gw_schema::{Content, Generation, Message, Provenance, Role, TeacherRef};
        TrainingRecord {
            record_id: "rec-1".into(),
            schema_version: semver::Version::new(1, 0, 0),
            dataset_version: None,
            training_area: "math".into(),
            tags: vec![],
            messages: vec![Message {
                role: Role::Assistant,
                content: Content::Text("42".into()),
                reasoning: Some("work".into()),
                reasoning_details: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            tools: None,
            provenance: Provenance {
                run_id: "run-1".into(),
                parent_ids: vec![],
                teacher: TeacherRef {
                    provider: "openrouter".into(),
                    slug: "m".into(),
                    served_by: None,
                    model_card_revision: None,
                },
                user_synth_model: None,
                user_turn_kind: None,
                in_scope_safe: None,
                judge_models: vec![],
                harness_version: "0.1.0".into(),
                git_commit: None,
            },
            generation: Generation::default(),
            verification_contract: None,
            execution_evidence: None,
            verification: Verification::default(),
            judging: Default::default(),
            reasoning_quality: None,
            lifecycle: Default::default(),
            hashes: Default::default(),
            cost: Default::default(),
        }
    }
}
