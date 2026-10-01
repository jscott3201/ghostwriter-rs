//! Grading reconstruction and executable verification preflight.
//!
//! Decisions are re-derived from persisted facts without model or oracle calls. Run preflight reads
//! stored envelopes before dispatch. The correlation prior has one shared implementation here.
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
use gw_schema::{TrainingRecord, Verification, VerificationPolicy};

use crate::clients::AreaConfig;
use crate::error::{EngineError, Result};

/// Build the inter-judge correlation matrix for a panel of `k` judges from the cold-start prior
/// `rho`, ENFORCING the never-identity invariant for `k > 1`.
///
/// Requires finite `0 <= rho <= 1`. For `k <= 1` there is no pair to correlate. For a multi-judge
/// panel the prior must also be positive and numerically nonidentity, using the same validation
/// as admission preflight.
///
/// # Errors
/// Returns [`EngineError::Invariant`] for invalid correlation settings.
pub fn correlation_prior(k: usize, rho: f64) -> Result<CorrelationMatrix> {
    gw_judge::validate_correlation_prior(k, rho)
        .map_err(|error| EngineError::Invariant(error.to_string()))?;
    Ok(CorrelationMatrix::uniform_offdiagonal(k, rho))
}

/// Reject unsupported executable records without changing their historical envelope.
///
/// # Errors
/// Rejects missing task policy, missing/unsupported facts after verification, or applied policies
/// that disagree with the task contract or the area's reasoning requirement.
pub fn validate_record_verification(rec: &TrainingRecord, area: &AreaConfig) -> Result<()> {
    validate_record_verification_for_reasoning(rec, area.cot_required)
}

/// Shared task-authority validation using the only area setting that affects verification.
fn validate_record_verification_for_reasoning(
    rec: &TrainingRecord,
    cot_required: bool,
) -> Result<()> {
    let contract = rec.verification_contract.as_ref().ok_or_else(|| {
        EngineError::Invariant(
            "record lacks task verification policy; historical records are inspect/export only"
                .into(),
        )
    })?;
    contract
        .validate()
        .map_err(|reason| EngineError::Invariant(reason.into()))?;
    if let Some(task) = &rec.task_provenance {
        let prompt = rec.messages.first().ok_or_else(|| {
            EngineError::Invariant("numeric task record lacks its user prompt".into())
        })?;
        task.validate_for(prompt, contract)
            .map_err(|reason| EngineError::Invariant(reason.into()))?;
    }
    if let Some(facts) = rec.verification.interpretation.as_ref() {
        facts
            .gate()
            .map_err(|reason| EngineError::Invariant(reason.into()))?;
        let reasoning_policy = if cot_required {
            VerificationPolicy::Authoritative
        } else {
            VerificationPolicy::Absent
        };
        if facts.reasoning.policy != reasoning_policy {
            return Err(EngineError::Invariant(
                "persisted reasoning policy disagrees with configured CoT requirement".into(),
            ));
        }
        if Some(facts.answer.policy) != contract.answer_policy
            || Some(facts.execution.policy) != contract.execution_policy
        {
            return Err(EngineError::Invariant(
                "persisted verification policy disagrees with task contract".into(),
            ));
        }
    } else if !matches!(
        rec.lifecycle.state,
        gw_schema::LifecycleState::Seeded
            | gw_schema::LifecycleState::UserSynthesized
            | gw_schema::LifecycleState::AssistantGenerated
            | gw_schema::LifecycleState::Error
    ) {
        return Err(EngineError::Invariant("record lacks supported verification interpretation; historical records are inspect/export only".into()));
    }
    Ok(())
}

/// Check every stored run record before dispatching model work, including other shards.
///
/// # Errors
/// Rejects unsupported historical records, applied policies inconsistent with the task/area, or a
/// storage failure.
pub async fn validate_run_verification(
    store: &gw_storage::Store,
    run_id: &str,
    area: &AreaConfig,
) -> Result<()> {
    for rec in store
        .scan(&gw_storage::RecordFilter::new().run_id(run_id))
        .await?
    {
        validate_record_verification(&rec, area)?;
    }
    Ok(())
}

/// Reconstruct admission from versioned factual observations and applied policy, without I/O.
/// Historical booleans never supply current facts.
///
/// # Errors
/// Rejects missing or unsupported interpretation and inconsistent policies.
pub fn verifier_grade_from_verification(
    rec: &TrainingRecord,
    area: &AreaConfig,
) -> Result<VerifierGrade> {
    verifier_grade_for_reasoning(rec, area.cot_required)
}

/// Reconstruct the same authority gate for pure callers that carry a frozen reasoning policy.
pub(crate) fn verifier_grade_for_reasoning(
    rec: &TrainingRecord,
    cot_required: bool,
) -> Result<VerifierGrade> {
    validate_record_verification_for_reasoning(rec, cot_required)?;
    let mut verification: Verification = rec.verification.clone();
    let facts = verification.interpretation.as_ref().ok_or_else(|| {
        EngineError::Invariant("record has no completed verification interpretation".into())
    })?;
    (verification.all_passed, verification.needs_review) = facts
        .gate()
        .map_err(|reason| EngineError::Invariant(reason.into()))?;
    let verdict = if !verification.all_passed {
        JudgeVerdict::Reject
    } else if verification.needs_review.is_some() {
        JudgeVerdict::Uncertain
    } else {
        JudgeVerdict::Accept
    };
    Ok(VerifierGrade {
        verdict,
        verification,
    })
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
    let verifier = verifier_grade_from_verification(rec, area)?;
    if verifier.is_hard_reject() || verifier.blocks_admission() {
        return Ok(gw_judge::HybridGrader::new(area.thresholds)
            .with_admission_intent(area.intent_for(rec))
            .grade(
                Some(&verifier),
                &[],
                &[],
                Some(&area.training_area),
                &CorrelationMatrix::identity(0),
            )?
            .decision);
    }
    Ok(
        rederive_verdict(&rec.judging, area.thresholds)?
            .with_admission_intent(area.intent_for(rec)),
    )
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
        assert!(err.to_string().contains("nonidentity"));
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
    fn verifier_grade_uses_facts_instead_of_historical_booleans() {
        let mut rec = sample_record();
        rec.verification.all_passed = false;
        assert!(
            !verifier_grade_from_verification(&rec, &area())
                .unwrap()
                .is_hard_reject()
        );
        rec.verification
            .interpretation
            .as_mut()
            .unwrap()
            .answer
            .observation
            .as_mut()
            .unwrap()
            .outcome = gw_schema::VerificationOutcome::Fail;
        rec.verification.all_passed = true;
        assert!(
            verifier_grade_from_verification(&rec, &area())
                .unwrap()
                .is_hard_reject()
        );
        rec.verification.interpretation = None;
        assert!(verifier_grade_from_verification(&rec, &area()).is_err());
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
        let area = AreaConfig::new("math", "m", vec![], "r")
            .with_cot_required(false)
            .with_thresholds(thresholds);
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
        AreaConfig::new("math", "m", vec![], "r").with_cot_required(false)
    }

    fn grade(slug: &str, score: f64) -> Grade {
        Grade {
            effective_contract: Some(
                gw_judge::EffectiveJudgeContract::json_score(&gw_judge::build_judge_request(
                    &gw_judge::PanelJudge::new(slug, "family"),
                    "rubric",
                    "candidate",
                ))
                .unwrap(),
            ),
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
        let mut rec = TrainingRecord {
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
            origin: gw_schema::RecordOrigin::Generated(Box::new(gw_schema::GeneratedOrigin {
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
            })),
            task_provenance: None,
            verification_contract: None,
            execution_evidence: None,
            verification: Verification::default(),
            judging: Default::default(),
            reasoning_quality: None,
            lifecycle: Default::default(),
            hashes: Default::default(),
            cost: Default::default(),
        };
        rec.verification_contract = Some(gw_schema::VerificationContract {
            answer_policy: Some(gw_schema::VerificationPolicy::Authoritative),
            execution_policy: Some(gw_schema::VerificationPolicy::Absent),
            required_tests: vec![],
            kind: gw_schema::VerificationKind::NumericMatch,
            oracle: gw_schema::Oracle::Literal {
                expected: "42".into(),
            },
            numeric: Some(gw_schema::NumericComparison::default()),
        });
        rec.verification = gw_judge::run_verifier(
            &gw_judge::VerifierInput {
                messages: &rec.messages,
                reasoning_tokens: 0,
                cot_required: false,
                contract: rec.verification_contract.as_ref(),
                execution_evidence: None,
                evidence_key: Default::default(),
            },
            &gw_judge::NullSandboxOracle,
        )
        .unwrap()
        .verification;
        rec.lifecycle.state = gw_schema::LifecycleState::Verified;
        rec
    }
}
