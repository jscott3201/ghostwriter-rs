//! Captured coding inputs and explicitly declared saved reports. Digests are not authentication.
use super::runtime::CodingRuntimeIdentity;
use anyhow::ensure;
use gw_schema::{
    CodingSuiteBinding, CodingTaskDocument, ExecutionOutcome, ReviewedCodingTask, TaskProvenance,
    TestStatus, VerificationInterpretation, coding_digest, strict_coding_json,
};
use serde::{Deserialize, Serialize};

/// Captured task and exact module bytes; no later path read can substitute either input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapturedCodingInput {
    /// Complete reviewed task, including private oracles. This is an evaluator artifact, not SFT.
    pub task: ReviewedCodingTask,
    /// Exact UTF-8 module content; no trimming, fence stripping, or extraction.
    pub code: String,
    /// Derived exact code identity.
    pub code_id: String,
    /// Derived public suite binding.
    pub suite: CodingSuiteBinding,
    /// Derived task provenance with redacted semantics.
    pub provenance: TaskProvenance,
    /// Identity of complete task declarations plus exact candidate bytes.
    pub input_id: String,
}
impl CapturedCodingInput {
    /// Validate the complete task document, select one label, and capture the candidate once.
    ///
    /// # Errors
    /// Rejects invalid/ambiguous documents, absent labels, or oversized candidate modules.
    pub fn new(document: &CodingTaskDocument, label: &str, code: String) -> anyhow::Result<Self> {
        document.validate().map_err(anyhow::Error::msg)?;
        let task = document
            .tasks
            .iter()
            .find(|task| task.task_id == label)
            .ok_or_else(|| anyhow::anyhow!("coding task label is absent"))?
            .clone();
        let provenance = TaskProvenance::from_coding_task(&task).map_err(anyhow::Error::msg)?;
        let suite = task.suite_binding();
        let code_id = coding_digest("ghostwriter.coding-module.v1", code.as_bytes());
        let input_id = digest("ghostwriter.coding-input.v1", &(&task, &code));
        let captured = Self {
            task,
            code,
            code_id,
            suite,
            provenance,
            input_id,
        };
        captured.validate()?;
        Ok(captured)
    }
    pub(super) fn validate(&self) -> anyhow::Result<()> {
        self.task.validate().map_err(anyhow::Error::msg)?;
        ensure!(
            serde_json::to_vec(&self.task)?.len() <= 1024 * 1024,
            "captured coding task exceeds 1 MiB"
        );
        ensure!(
            !self.code.is_empty() && self.code.len() <= 65_536,
            "candidate must contain 1..65536 UTF-8 bytes"
        );
        ensure!(
            self.code_id == coding_digest("ghostwriter.coding-module.v1", self.code.as_bytes())
                && self.suite == self.task.suite_binding()
                && self.provenance
                    == TaskProvenance::from_coding_task(&self.task).map_err(anyhow::Error::msg)?
                && self.input_id
                    == digest("ghostwriter.coding-input.v1", &(&self.task, &self.code)),
            "coding capture identity or suite binding mismatch"
        );
        ensure!(
            serde_json::to_vec(self)?.len() <= 1024 * 1024,
            "complete coding capture exceeds 1 MiB"
        );
        Ok(())
    }
}

/// Stable externally observed case reason, independent of raw candidate stdout/stderr.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodingCaseReason {
    /// Exact externally held expected result matched.
    Matched,
    /// A supported result differed from the expected value.
    WrongResult,
    /// Output was absent, malformed, unsupported, duplicated, or accompanied by stderr.
    InvalidOutput,
    /// The isolated candidate exited unsuccessfully, including syntax/exception/resource failure.
    CandidateExit,
    /// External per-case wall limit ended the whole container.
    WallLimit,
    /// Bounded stdout/stderr capture ended the whole container.
    OutputLimit,
    /// Caller cancellation or whole-evaluation deadline prevented a complete observation.
    Cancelled,
    /// Docker/runtime/cleanup acknowledgment was unavailable or uncertain.
    Infrastructure,
    /// No invocation was dispatched after cancellation or infrastructure uncertainty.
    NotRun,
}
/// One observed or explicitly unobserved case. Private inputs and expected values are absent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodingCaseObservation {
    /// Exact task-derived case identity, in complete suite order.
    pub case_id: String,
    /// Pass/fail only for an observed candidate; skipped means no completed result.
    pub status: TestStatus,
    /// Stable external reason.
    pub reason: CodingCaseReason,
    /// Canonical typed result identity when parsing succeeded; never an expected-value field.
    pub result_id: Option<String>,
    /// Actual candidate exec exit, absent when externally interrupted or never started.
    pub exit_code: Option<i32>,
    /// Whether the owned container, clients, and readers were settled with acknowledged absence.
    pub settled: bool,
    /// Actual ephemeral container identity, distinct from stable replay semantics.
    pub container_id: Option<String>,
    /// Monotonic elapsed time; excluded from stable replay comparison.
    pub elapsed_ms: u64,
}
impl CodingCaseObservation {
    pub(super) fn not_run(case_id: String) -> Self {
        Self {
            case_id,
            status: TestStatus::Skipped,
            reason: CodingCaseReason::NotRun,
            result_id: None,
            exit_code: None,
            settled: true,
            container_id: None,
            elapsed_ms: 0,
        }
    }
    pub(super) fn validate(&self) -> anyhow::Result<()> {
        use CodingCaseReason::*;
        let expected = match self.reason {
            Matched => TestStatus::Passed,
            WrongResult | InvalidOutput | CandidateExit | WallLimit | OutputLimit => {
                TestStatus::Failed
            }
            Cancelled | Infrastructure | NotRun => TestStatus::Skipped,
        };
        ensure!(
            self.status == expected,
            "declared coding status contradicts its reason"
        );
        if matches!(self.reason, Matched | WrongResult) {
            ensure!(
                self.result_id.is_some()
                    && self.exit_code == Some(0)
                    && self.container_id.is_some(),
                "declared coding result lacks successful execution details"
            );
        }
        if self.reason == NotRun {
            ensure!(
                self.container_id.is_none()
                    && self.exit_code.is_none()
                    && self.result_id.is_none()
                    && self.elapsed_ms == 0,
                "unexecuted case claims execution details"
            );
        }
        for id in [&self.result_id, &self.container_id].into_iter().flatten() {
            ensure!(hash_valid(id), "malformed coding observation identity");
        }
        Ok(())
    }
    pub(super) fn stable(&self) -> impl Serialize + '_ {
        (
            &self.case_id,
            self.status,
            self.reason,
            &self.result_id,
            self.exit_code,
            self.settled,
        )
    }
}

/// Saved report fields are declarations. Only a fresh opaque runtime observation enters the new
/// native consumer; reading this structure never establishes historical execution authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodingReport {
    /// Fresh invocation identity; independent of input and prior saved claims.
    pub run_id: String,
    /// Externally derived three-valued outcome.
    pub outcome: ExecutionOutcome,
    /// Every required case exactly once, including explicitly unexecuted cases.
    pub cases: Vec<CodingCaseObservation>,
    /// Native verifier's factual interpretation; no quality-panel admission is asserted.
    pub native_verification: VerificationInterpretation,
}

/// Self-contained saved coding evaluation. Includes private oracles for explicit local replay;
/// use the redacted task contract for training exports instead.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodingArtifact {
    /// Strict saved artifact version.
    pub version: u32,
    /// Whole serialized declaration identity, not a producer signature.
    pub artifact_id: String,
    /// Captured actual task/module input.
    pub input: CapturedCodingInput,
    /// Declared exact runtime, checked against current qualification before re-execution.
    pub runtime: CodingRuntimeIdentity,
    /// Declared prior observed report, verified by fresh execution during replay.
    pub report: CodingReport,
    /// Previous declaration compared by this fresh run, when explicit replay was requested.
    pub replayed_declaration_id: Option<String>,
}
impl CodingArtifact {
    /// Read and strictly validate saved declarations before any Docker access.
    ///
    /// # Errors
    /// Rejects malformed/unknown fields, stale bindings, impossible coverage, or inconsistent flags.
    pub fn from_json(bytes: &[u8]) -> anyhow::Result<Self> {
        ensure!(
            bytes.len() <= 2 * 1024 * 1024,
            "coding artifact exceeds 2 MiB"
        );
        let value = strict_coding_json(bytes).map_err(anyhow::Error::msg)?;
        let artifact: Self = serde_json::from_value(value.clone())?;
        // Also rejects ignored nested fields/defaulted omissions in reused legacy schema types.
        ensure!(
            value == serde_json::to_value(&artifact)?,
            "coding artifact contains unknown or omitted fields"
        );
        artifact.validate()?;
        Ok(artifact)
    }
    pub(super) fn seal(&mut self) {
        self.artifact_id = self.identity();
    }
    fn identity(&self) -> String {
        digest(
            "ghostwriter.coding-artifact.v1",
            &(
                self.version,
                &self.input,
                &self.runtime,
                &self.report,
                &self.replayed_declaration_id,
            ),
        )
    }
    pub(super) fn validate(&self) -> anyhow::Result<()> {
        ensure!(self.version == 1, "unsupported coding artifact version");
        self.input.validate()?;
        ensure!(
            self.runtime == CodingRuntimeIdentity::expected(),
            "stale or unsupported coding runtime identity"
        );
        ensure!(
            self.report.run_id.len() == 32
                && self.report.run_id.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid coding run identity"
        );
        ensure!(
            self.report
                .cases
                .iter()
                .map(|case| &case.case_id)
                .eq(self.input.suite.case_ids.iter()),
            "coding report has missing, duplicate, reordered, or foreign cases"
        );
        for case in &self.report.cases {
            case.validate()?;
        }
        ensure!(
            self.report.outcome == outcome(&self.report.cases),
            "declared coding outcome contradicts exact coverage"
        );
        let native =
            super::consume::interpret(&self.input, &self.report.run_id, &self.report.cases);
        ensure!(
            self.report.native_verification == native,
            "declared native coding verification is inconsistent"
        );
        if let Some(id) = &self.replayed_declaration_id {
            ensure!(hash_valid(id), "invalid predecessor declaration identity");
        }
        ensure!(
            self.artifact_id == self.identity(),
            "coding artifact declaration digest mismatch"
        );
        Ok(())
    }
    pub(super) fn stable_matches(&self, other: &Self) -> bool {
        self.input == other.input
            && self.runtime == other.runtime
            && self.report.outcome == other.report.outcome
            && digest(
                "ghostwriter.coding-stable-report.v1",
                &self
                    .report
                    .cases
                    .iter()
                    .map(CodingCaseObservation::stable)
                    .collect::<Vec<_>>(),
            ) == digest(
                "ghostwriter.coding-stable-report.v1",
                &other
                    .report
                    .cases
                    .iter()
                    .map(CodingCaseObservation::stable)
                    .collect::<Vec<_>>(),
            )
            && self.report.native_verification == other.report.native_verification
    }
}
pub(super) fn outcome(cases: &[CodingCaseObservation]) -> ExecutionOutcome {
    if cases
        .iter()
        .any(|case| !case.settled || case.status == TestStatus::Skipped)
    {
        ExecutionOutcome::Unknown
    } else if cases.iter().any(|case| case.status == TestStatus::Failed) {
        ExecutionOutcome::Failed
    } else {
        ExecutionOutcome::Passed
    }
}
pub(super) fn digest<T: Serialize>(domain: &str, value: &T) -> String {
    let canonical = serde_json::to_value(value).expect("coding wire values");
    coding_digest(domain, &serde_json::to_vec(&canonical).expect("JSON value"))
}
fn hash_valid(id: &str) -> bool {
    id.len() == 64
        && id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
