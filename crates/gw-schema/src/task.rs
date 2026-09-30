//! Reviewed numeric task input. Declarations record human review; they do not certify rights,
//! difficulty, safety, or cross-corpus decontamination. References are retained without fetching.
use crate::{
    NumericComparison, Oracle, VerificationContract, VerificationKind, VerificationPolicy,
};
use serde::{Deserialize, Serialize};

/// Supported strict task document version.
pub const TASK_DOCUMENT_VERSION: u32 = 1;

/// One ordered input document. Unknown fields at every level are rejected.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericTaskDocument {
    /// Must equal [`TASK_DOCUMENT_VERSION`].
    pub version: u32,
    /// Reviewed tasks in source order, before stable round-robin sharding.
    pub tasks: Vec<ReviewedNumericTask>,
}

/// One reviewed numeric task with a precomputed literal answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReviewedNumericTask {
    /// Stable label, unique within this document. This is not a trusted content digest.
    pub task_id: String,
    /// Actual source identity, immutable revision, and citation.
    pub source: TaskSource,
    /// Reviewed rights assertions, retained verbatim.
    pub rights: ReviewedTaskRights,
    /// Corpus group identity, independent of prompt-hash sibling groups.
    pub group: NamespacedTaskId,
    /// Declared split assignment; excluded from semantic task identity.
    pub split: TaskSplit,
    /// One user-text message. Other roles and multimodal/conversation structures are unsupported.
    pub prompt: TaskPrompt,
    /// Literal numeric answer, extraction, tolerance, and explicit axis policies.
    pub verification: ReviewedNumericVerification,
    /// Reviewed domain, difficulty, and pre-generation QC observations.
    pub observations: TaskObservations,
}

/// Source identity and citation. Revision is a reviewed immutable identifier, not a fetched ref.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskSource {
    /// Source owner or corpus namespace.
    pub namespace: String,
    /// Item identifier within that namespace.
    pub item: String,
    /// Immutable source revision asserted by the reviewer.
    pub revision: String,
    /// Citation/reference recorded without I/O.
    pub citation: String,
}

/// A namespace-qualified group or split-manifest identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NamespacedTaskId {
    /// Owning namespace.
    pub namespace: String,
    /// Identifier within the namespace.
    pub id: String,
}

/// The basis a reviewer recorded for permitted use. The harness does not assess legal validity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskRightsBasis {
    /// Reviewer asserts ownership of the source material.
    Owned,
    /// Reviewer asserts public-domain status.
    PublicDomain,
    /// Reviewer cites an applicable license.
    License,
    /// Reviewer cites explicit permission.
    Permission,
}

/// A use the reviewer declares permitted. Recording a use is not legal qualification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskPermittedUse {
    /// Model training or preparation of training data.
    Training,
    /// Model evaluation.
    Evaluation,
    /// Redistribution of the source or derived data under the reviewed terms.
    Redistribution,
}

/// Rights evidence and the human assertion it supports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReviewedTaskRights {
    /// Reviewed legal/ownership basis.
    pub basis: TaskRightsBasis,
    /// Nonempty references or evidence descriptions. These are not fetched.
    pub evidence: Vec<String>,
    /// Reviewer identity supplied by the operator.
    pub reviewer: String,
    /// Nonempty, unique reviewed uses.
    pub permitted_uses: Vec<TaskPermittedUse>,
}

/// A task's declared dataset role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskSplitRole {
    /// Training split.
    Train,
    /// Validation split.
    Validation,
    /// Held-out test split.
    Test,
}

/// Explicit split identity and assignment; no cross-corpus qualification is implied.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskSplit {
    /// Namespace-qualified split manifest.
    pub manifest: NamespacedTaskId,
    /// Immutable declared manifest revision.
    pub revision: String,
    /// Assigned role for this entire declared corpus group.
    pub role: TaskSplitRole,
}

/// The only prompt structure supported by the first task family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case", deny_unknown_fields)]
pub enum TaskPrompt {
    /// One nonempty user text turn.
    User {
        /// Prompt retained exactly, including meaningful whitespace.
        content: String,
    },
}
impl TaskPrompt {
    /// Exact user text, without trimming or flattening.
    #[must_use]
    pub fn text(&self) -> &str {
        let Self::User { content } = self;
        content
    }
}

/// Only numeric tasks are accepted by the reviewed task intake.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NumericTaskKind {
    /// Strict finite numeric comparison.
    NumericMatch,
}

/// Only precomputed literal answers are accepted by reviewed numeric task intake.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "oracle", rename_all = "snake_case", deny_unknown_fields)]
pub enum NumericTaskOracle {
    /// A decimal/scientific string, validated before any dispatch.
    Literal {
        /// Strict finite numeric token. Strings preserve overflow/underflow detection.
        expected: String,
    },
}

/// Strict task-file subset of the runtime verification contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReviewedNumericVerification {
    /// The supported numeric task family.
    pub kind: NumericTaskKind,
    /// Precomputed literal answer; executable oracles are unsupported here.
    pub oracle: NumericTaskOracle,
    /// Persisted extraction and tolerance settings.
    pub numeric: NumericComparison,
    /// How the numeric observation affects admission.
    pub answer_policy: VerificationPolicy,
    /// Must be absent for this family, which supplies no execution evidence contract.
    pub execution_policy: VerificationPolicy,
}
impl ReviewedNumericVerification {
    /// Materialize the sole runtime answer contract without changing its declared semantics.
    #[must_use]
    pub fn contract(&self) -> VerificationContract {
        let NumericTaskOracle::Literal { expected } = &self.oracle;
        VerificationContract {
            answer_policy: Some(self.answer_policy),
            execution_policy: Some(self.execution_policy),
            required_tests: vec![],
            kind: VerificationKind::NumericMatch,
            oracle: Oracle::Literal {
                expected: expected.clone(),
            },
            numeric: Some(self.numeric.clone()),
        }
    }
}

/// Reviewed difficulty observation; not a measured calibration score.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskDifficulty {
    /// Operator's difficulty band.
    pub label: String,
    /// Explanation/evidence for the asserted band.
    pub basis: String,
}

/// Reviewed pre-generation QC declarations. Runtime embedding dedup remains a separate check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReviewedTaskQc {
    /// Reviewer asserts that the prompt is answerable.
    pub answerable: bool,
    /// Reviewer asserts the declared difficulty is appropriate.
    pub difficulty_targeted: bool,
    /// Reviewer asserts the task is within the intended scope and safety policy.
    pub in_scope: bool,
    /// Reviewer identity supplied by the operator.
    pub reviewer: String,
    /// Nonempty supporting descriptions/references; not fetched or independently certified.
    pub evidence: Vec<String>,
}

/// Domain, difficulty, and QC assertions supplied with a reviewed task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskObservations {
    /// Domain label for downstream analysis.
    pub domain: String,
    /// Reviewed difficulty band and basis.
    pub difficulty: TaskDifficulty,
    /// Reviewed QC declarations supplied to the normal generation gate.
    pub qc: ReviewedTaskQc,
}
