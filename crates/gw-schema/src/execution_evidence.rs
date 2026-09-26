//! `execution_evidence{}` — PRECOMPUTED ground truth from an external evaluator (DATA-SCHEMA §1.6
//! extension; the `Verification` block's `execution_evidence` check).
//!
//! This crate performs no I/O and owns no executor. An evidence record is the *result* an
//! out-of-process evaluator already computed (it ran the candidate's code, the test suite, the
//! tool) — the harness never re-executes anything to produce one. The verifier ADAPTS a carried
//! record into a deterministic [`Check`](crate::Check); it does not re-run the work.
//!
//! ## Why the outcome is carried, not recomputed
//!
//! Three things make a self-reported "passed" insufficient on its own, and the type carries the
//! corroborating detail so the verifier can cross-check rather than trust:
//!
//! 1. **Binding** ([`EvidenceBinding`]) — which task, attempt, and patch the run was executed
//!    against. Evidence computed for a different candidate is not evidence for this one.
//! 2. **The required tests** ([`ExecutionEvidence::required_tests`]) — what the area declares MUST
//!    pass. A green suite that never ran the required test is not a pass.
//! 3. **The per-case detail** ([`ExecutionEvidence::cases`]) — what each test actually reported.
//!
//! The adapter accepts a `Passed` claim only when the detail corroborates it (every required node
//! reported `Passed`, no reported error, and a zero exit). An uncorroborated claim is treated as
//! `Unknown` — never as a pass. A report the adapter cannot read (no cases, duplicate case keys, a
//! missing exit code) is `Unknown` too, so an unreadable evaluator result can never admit a record.
//!
//! ## The vocabulary is deliberately three-valued
//!
//! `Unknown` is a first-class outcome, not an error. It is what an interrupted run, an
//! infrastructure fault, a malformed report, or a missing required-test contract all reduce to, and
//! it routes the record to human/verifier review — it never rejects (the work may still be fine)
//! and never admits (nothing proved it).

use serde::{Deserialize, Serialize};

/// The terminal outcome an external evaluator reported for a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionOutcome {
    /// The evaluator reported the run succeeded.
    Passed,
    /// The evaluator reported a definite failure (a failing required test, a nonzero exit, a
    /// reported suite error). Authoritative: sinks the record regardless of any panel score.
    Failed,
    /// The evaluator could not decide (interrupted, infrastructure fault, unreadable report, no
    /// required-test contract). Routes to review; never admits and never rejects.
    Unknown,
}

/// What one test node reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TestStatus {
    /// The node ran and passed.
    Passed,
    /// The node ran and failed (or errored).
    Failed,
    /// The node did not run. A SKIPPED required test is NOT a pass — skipping is a way of not
    /// proving the work.
    Skipped,
}

/// One reported test node: the evaluator's stable node key (`classname::name`, or whatever key the
/// area's evaluator uses — it must match the strings in [`ExecutionEvidence::required_tests`]) and
/// the status it reported.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TestCase {
    /// The node key. Compared verbatim against [`ExecutionEvidence::required_tests`].
    pub node: String,
    /// What the evaluator reported for this node.
    pub status: TestStatus,
}

/// Which task, attempt, and patch an evaluation run was executed against.
///
/// The harness owns all three keys, so a run's evidence can be proven to describe THIS candidate:
/// `task` is the run, `attempt` is the record (a retry attempt is a distinct record), and
/// `patch_hash` is a content hash of the produced candidate. A mismatch is never followed — see
/// [`gw_judge`](https://docs.rs/gw-judge)'s evidence adapter for the per-key consequence.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct EvidenceBinding {
    /// The run the evaluation belongs to (the envelope's `provenance.run_id`).
    #[serde(default)]
    pub task: String,
    /// The attempt the evaluation belongs to (the envelope's `record_id`; a retry attempt is a
    /// distinct record id).
    #[serde(default)]
    pub attempt: String,
    /// A content hash of the exact candidate the evaluation ran. A candidate whose content changed
    /// no longer matches the run, so the evidence is stale and is never followed.
    #[serde(default)]
    pub patch_hash: String,
}

/// A precomputed execution verdict, carried on the record envelope so the deterministic verifier can
/// reach a decision on a trajectory whose correctness is only knowable by running it.
///
/// The adapter is deliberately CONSERVATIVE: it re-derives an outcome from the structured detail
/// and takes the more conservative of that derivation and the reported
/// [`ExecutionOutcome`]. `Unknown` is the conservative default — it outranks both others, so an
/// outcome can never be softened into a pass.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionEvidence {
    /// The outcome the evaluator reported.
    pub outcome: ExecutionOutcome,
    /// The nodes this area declares REQUIRED. A pass requires EVERY one of them to be reported
    /// `Passed`; a missing or skipped required node is a `Failed`. Empty ⇒ nothing was required, so
    /// nothing can be proven — `Unknown`.
    #[serde(default)]
    pub required_tests: Vec<String>,
    /// What each node reported. Absent or empty ⇒ the report is unreadable ⇒ `Unknown`.
    #[serde(default)]
    pub cases: Vec<TestCase>,
    /// The run's exit code. `None` ⇒ no usable exit status ⇒ `Unknown`. Any nonzero ⇒ `Failed`.
    #[serde(default)]
    pub exit_code: Option<i32>,
    /// Report-level errors the evaluator surfaced (a suite that could not complete, a collection
    /// error). Any entry ⇒ `Failed`, even when individual nodes look green.
    #[serde(default)]
    pub errors: Vec<String>,
    /// An opaque reference to the raw report the evaluator produced (a locator the operator can
    /// audit), carried verbatim and never interpreted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_ref: Option<String>,
    /// Which task/attempt/patch this run was executed against.
    pub binding: EvidenceBinding,
}
