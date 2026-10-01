//! One shared pure numeric evaluator for admission observations and fresh external rewards.
use gw_schema::{
    NUMERIC_REWARD_VERSION, NumericComparison, NumericRewardBatch, NumericRewardBatchReport,
    NumericRewardResult, NumericTaskOracle, RewardSnapshotIdentity, RewardTermination,
    VerificationOutcome, parse_finite_decimal,
};
use std::collections::HashMap;

/// Evaluate only the supplied assistant content under explicit numeric semantics.
/// No prompt, reasoning, teacher result, execution report, or admission boolean is consulted.
/// Missing/unparseable evidence is Unknown; a successfully parsed disagreement is factual Fail.
#[must_use]
pub fn evaluate_numeric_answer(
    content: &str,
    expected: Option<&str>,
    settings: &NumericComparison,
) -> VerificationOutcome {
    let Some(actual) = settings.extract(content).and_then(parse_finite_decimal) else {
        return VerificationOutcome::Unknown;
    };
    let Some(expected) = expected.and_then(parse_finite_decimal) else {
        return VerificationOutcome::Unknown;
    };
    let Ok(bound) = settings.bound(expected) else {
        return VerificationOutcome::Unknown;
    };
    // Preserve the existing symmetric rounded binary64 distance. Finite-operand subtraction
    // can overflow to infinity, which exceeds every valid finite bound.
    if (actual - expected).abs() <= bound {
        VerificationOutcome::Pass
    } else {
        VerificationOutcome::Fail
    }
}

/// Validate the complete batch before evaluating its fresh contents in submitted order.
/// Factual Unknown is retained with no reward; an external trainer decides how to abort its batch.
///
/// # Errors
/// Rejects any unsupported or stale artifact, attempt, token-policy, or completion binding.
pub fn evaluate_numeric_reward_batch(
    request: &NumericRewardBatch,
) -> Result<Vec<NumericRewardResult>, &'static str> {
    request.validate()?;
    let tasks: HashMap<_, _> = request
        .artifact
        .tasks
        .iter()
        .map(|task| (&task.task.task_id, &task.task.verification))
        .collect();
    request
        .items
        .iter()
        .map(|item| {
            let verification = tasks
                .get(&item.binding.task_id)
                .ok_or("reward task is absent from corpus")?;
            let NumericTaskOracle::Literal { expected } = &verification.oracle;
            let outcome = evaluate_numeric_answer(
                &item.completion.text,
                Some(expected),
                &verification.numeric,
            );
            let reward = match outcome {
                VerificationOutcome::Pass => Some(1.0),
                VerificationOutcome::Fail => Some(0.0),
                VerificationOutcome::Unknown => None,
            };
            let termination = if item.completion.token_ids.last()
                == Some(&request.completion_policy.eos_token_id)
            {
                RewardTermination::ObservedEos
            } else {
                RewardTermination::Unknown
            };
            Ok(NumericRewardResult {
                binding: item.binding.clone(),
                outcome,
                reward,
                termination,
            })
        })
        .collect()
}

/// Evaluate one complete strict JSON request and bind the complete response to its captured bytes.
/// Parsing and all computation remain pure; the CLI owns stdin/stdout and the external timeout.
///
/// # Errors
/// Rejects malformed wire input or any invalid binding before constructing a response.
pub fn evaluate_numeric_reward_json(bytes: &[u8]) -> Result<NumericRewardBatchReport, String> {
    let request = NumericRewardBatch::from_json(bytes)?;
    let results = evaluate_numeric_reward_batch(&request).map_err(str::to_owned)?;
    Ok(NumericRewardBatchReport {
        report_version: NUMERIC_REWARD_VERSION,
        request: RewardSnapshotIdentity::for_bytes(bytes),
        completion_policy_id: request
            .completion_policy
            .identity()
            .map_err(str::to_owned)?,
        mask_truncated_completions: request.mask_truncated_completions,
        results,
    })
}
