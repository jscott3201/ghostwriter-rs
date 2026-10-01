//! Exact tolerance encoding scoped to the new reward snapshot. Remote derives preserve strict
//! streaming deserialization and leave reviewed task inputs and historical identities unchanged.
use crate::{
    NamespacedTaskId, NumericComparison, NumericExtraction, NumericTaskKind, NumericTaskOracle,
    NumericTolerance, ReviewedNumericTask, ReviewedNumericVerification, ReviewedTaskRights,
    TaskObservations, TaskPrompt, TaskSource, TaskSplit, VerificationPolicy,
};
use serde::{Deserialize, Serialize};

/// Reward-only wire view of a reviewed task; all nonnumeric fields keep their original contract.
#[derive(Serialize, Deserialize)]
#[serde(remote = "ReviewedNumericTask", deny_unknown_fields)]
pub(super) struct Task {
    task_id: String,
    source: TaskSource,
    rights: ReviewedTaskRights,
    group: NamespacedTaskId,
    split: TaskSplit,
    prompt: TaskPrompt,
    #[serde(with = "Verification")]
    verification: ReviewedNumericVerification,
    observations: TaskObservations,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "ReviewedNumericVerification", deny_unknown_fields)]
struct Verification {
    kind: NumericTaskKind,
    oracle: NumericTaskOracle,
    #[serde(with = "Comparison")]
    numeric: NumericComparison,
    answer_policy: VerificationPolicy,
    execution_policy: VerificationPolicy,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "NumericComparison", deny_unknown_fields)]
struct Comparison {
    extraction: NumericExtraction,
    #[serde(with = "Tolerance")]
    tolerance: NumericTolerance,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "NumericTolerance", deny_unknown_fields)]
struct Tolerance {
    #[serde(with = "crate::finite_numbers::scalar")]
    absolute: f64,
    #[serde(with = "crate::finite_numbers::scalar")]
    relative: f64,
}
