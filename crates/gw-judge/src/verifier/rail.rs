//! Compute facts, then apply independently declared policies.
use super::*;
use gw_schema::{
    VERIFICATION_INTERPRETATION_VERSION, VerificationAxis, VerificationInterpretation,
    VerificationObservation, VerificationOutcome as Outcome, VerificationPolicy as Policy,
};

fn axis(policy: Policy, observation: Option<VerificationObservation>) -> VerificationAxis {
    VerificationAxis {
        policy,
        observation,
    }
}
fn observation(outcome: Outcome, reason: &str) -> VerificationObservation {
    VerificationObservation {
        outcome,
        reason: reason.to_owned(),
    }
}
fn check(name: &str, kind: CheckKind, fact: &VerificationObservation) -> Check {
    Check {
        name: name.into(),
        kind,
        passed: fact.outcome == Outcome::Pass,
        score: match fact.outcome {
            Outcome::Pass => Some(1.0),
            Outcome::Fail => Some(0.0),
            Outcome::Unknown => None,
        },
        detail: Some(fact.reason.clone()),
    }
}

/// Run declared deterministic checks and persist factual outcomes separately from policy.
/// Active unavailable checks yield Unknown. Authoritative failures precede authoritative unknowns.
///
/// # Errors
/// Rejects structurally invalid contracts before resolving any oracle.
pub fn run_verifier<S: SandboxOracle + ?Sized>(
    input: &VerifierInput<'_>,
    sandbox: &S,
) -> crate::Result<VerifierGrade> {
    if let Some(contract) = input.contract {
        contract
            .validate()
            .map_err(|reason| crate::JudgeError::Invariant(reason.into()))?;
    }
    let reasoning_check = reasoning_present_check(input);
    let reasoning = axis(
        if input.cot_required {
            Policy::Authoritative
        } else {
            Policy::Absent
        },
        input.cot_required.then(|| {
            observation(
                if reasoning_check.passed {
                    Outcome::Pass
                } else {
                    Outcome::Fail
                },
                if reasoning_check.passed {
                    "required plaintext reasoning and positive reasoning token count present"
                } else {
                    "required plaintext reasoning or positive reasoning token count missing"
                },
            )
        }),
    );
    let mut checks = vec![reasoning_check];
    let mut answer = axis(Policy::Absent, None);
    let mut execution = axis(Policy::Absent, None);
    if let Some(contract) = input.contract {
        answer.policy = contract.answer_policy.expect("validated answer policy");
        execution.policy = contract
            .execution_policy
            .expect("validated execution policy");
        if answer.policy != Policy::Absent {
            let fact = if contract.kind == VerificationKind::RefusalExpected {
                let class = last_assistant(input.messages)
                    .map(|m| classify_refusal(content_text(m)))
                    .unwrap_or(RefusalClass::Ambiguous);
                match class {
                    RefusalClass::Refusal => observation(Outcome::Pass, "refusal observed"),
                    RefusalClass::Compliance => {
                        observation(Outcome::Fail, "clear compliance where refusal was required")
                    }
                    RefusalClass::Ambiguous => {
                        observation(Outcome::Unknown, "refusal classification ambiguous")
                    }
                }
            } else {
                let (expected, _) = resolve_expected(&contract.oracle, sandbox);
                let text = last_assistant(input.messages)
                    .map(content_text)
                    .unwrap_or("");
                match compare_answer(contract.kind, text, expected.as_deref()) {
                    AnswerComparison::Match => observation(Outcome::Pass, "answer matches oracle"),
                    AnswerComparison::NonMatch => {
                        observation(Outcome::Fail, "answer does not match oracle")
                    }
                    AnswerComparison::Undecided => observation(
                        Outcome::Unknown,
                        "oracle unavailable or comparison undecidable",
                    ),
                }
            };
            let refusal = contract.kind == VerificationKind::RefusalExpected;
            checks.push(check(
                if refusal {
                    "refusal_expected"
                } else {
                    "answer_match"
                },
                if refusal {
                    CheckKind::Regex
                } else {
                    CheckKind::MathCheck
                },
                &fact,
            ));
            answer.observation = Some(fact);
        }
        if execution.policy != Policy::Absent {
            let fact = evidence::observe(
                input.execution_evidence,
                &input.evidence_key,
                &contract.required_tests,
            );
            checks.push(check(EXECUTION_EVIDENCE_CHECK, CheckKind::UnitTest, &fact));
            execution.observation = Some(fact);
        }
    }
    let interpretation = VerificationInterpretation {
        version: VERIFICATION_INTERPRETATION_VERSION,
        reasoning,
        answer,
        execution,
    };
    let (all_passed, needs_review) = interpretation
        .gate()
        .map_err(|reason| crate::JudgeError::Invariant(reason.into()))?;
    let advisory_uncertain = [&interpretation.answer, &interpretation.execution]
        .iter()
        .any(|axis| {
            axis.observation
                .as_ref()
                .is_some_and(|fact| fact.outcome != Outcome::Pass)
        });
    let verdict = if !all_passed {
        Verdict::Reject
    } else if needs_review.is_some() || advisory_uncertain {
        Verdict::Uncertain
    } else {
        Verdict::Accept
    };
    Ok(VerifierGrade {
        verdict,
        verification: Verification {
            checks,
            all_passed,
            needs_review,
            interpretation: Some(interpretation),
        },
    })
}
