//! The precomputed-execution adapter: turn a carried [`ExecutionEvidence`] into a deterministic
//! [`Check`] plus the rail-level consequence, WITHOUT executing anything.
//!
//! The engine never runs the candidate — an external evaluator already did, and its report is
//! carried on the envelope ([`ExecutionEvidence`]). This module is the only place that decides what
//! that report means, so the "unknown is conservative, a failure is authoritative" rule has one
//! obvious home.
//!
//! ## The three outcomes and what each one is allowed to do
//!
//! - [`EvidenceVerdict::Failed`] — a definite failure. The check FAILS, `all_passed` goes false, and
//!   the rail returns `Reject`. This is the verifier-first hard gate: a panel score, however high,
//!   cannot reach the record (the engine short-circuits before any judge call is spent).
//! - [`EvidenceVerdict::Unknown`] — nothing proved the work. The check passes inertly (a failure was
//!   not proven, so the record is not rejected) but the rail raises
//!   [`VerifierGrade::blocks_admission`](super::VerifierGrade::blocks_admission), so the record goes
//!   to `NeedsReview` instead of being admitted — or outvoted by — the panel. The assistant's own
//!   summary of its work is never a substitute for evidence, so an unevidenced "it passes" is
//!   Unknown, not Passed.
//! - [`EvidenceVerdict::Passed`] — the report is both reported green AND corroborated by its own
//!   detail. The check passes and the rail hands the remainder to the panel.
//!
//! ## The precedence rule (why a claim is cross-checked at all)
//!
//! An [`ExecutionEvidence`] carries both a reported [`ExecutionOutcome`] and the structured detail
//! behind it. The adapter re-derives an outcome from that detail and keeps the MORE CONSERVATIVE of
//! the two, where `Unknown` outranks `Failed` outranks `Passed`. So:
//!
//! - a `Passed` claim with no corroborating detail is downgraded to `Unknown` (a bare claim is not
//!   evidence);
//! - a `Passed` claim contradicted by its own detail (a failing node, a reported suite error, a
//!   missing required node) is downgraded to `Failed`;
//! - a `Failed` report is never softened — an authoritative failure stands even if the detail the
//!   adapter can read looks green;
//! - an `Unknown` report is never hardened — an interrupted or infrastructure-faulted run stays
//!   `Unknown` whatever its partial detail shows.
//!
//! ## Binding: a report describes one candidate only
//!
//! The binding keys are checked BEFORE the outcome, because evidence computed for a different
//! candidate is not evidence for this one. The two mismatches are deliberately NOT the same
//! consequence:
//!
//! - a **task / attempt** mismatch is a definite [`EvidenceVerdict::Failed`] — the report belongs to
//!   a different record, so applying it would be applying another candidate's result;
//! - a **patch-hash** mismatch is [`EvidenceVerdict::Unknown`] — the same attempt, but the content
//!   moved under it. The result is stale rather than wrong, and a stale result is never followed,
//!   so the record goes to review instead of being sunk on evidence that no longer describes it.

use gw_schema::{
    Check, CheckKind, EvidenceBinding, ExecutionEvidence, ExecutionOutcome, TestStatus,
};

use super::{EXECUTION_EVIDENCE_CHECK, VerifierInput};

/// The outcome of adapting one [`ExecutionEvidence`] into the verifier rail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceVerdict {
    /// The report proves the candidate failed. Hard-fails the check (authoritative reject).
    Failed,
    /// The report proves nothing. The check passes inertly, but admission is blocked.
    Unknown,
    /// The report proves the candidate passed, and its own detail corroborates it.
    Passed,
}

impl EvidenceVerdict {
    /// The more conservative of two outcomes, where `Unknown` (nothing proven) outranks `Failed` (a
    /// proven failure) and `Failed` outranks `Passed`.
    const fn more_conservative(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unknown, _) | (_, Self::Unknown) => Self::Unknown,
            (Self::Failed, _) | (_, Self::Failed) => Self::Failed,
            (Self::Passed, Self::Passed) => Self::Passed,
        }
    }
}

/// Re-derive an outcome from the report's structured detail alone (the reported
/// [`ExecutionOutcome`] deliberately plays no part).
///
/// The order is load-bearing and mirrors the required-test contract this adapter implements: an
/// absent required-test contract, an absent exit code, and an unreadable report are all `Unknown`
/// and outrank everything else, so a report that proves nothing can never be read as a failure by
/// accident; a reported suite error and a failing node are `Failed`; and a pass requires a zero
/// exit AND every required node reported `Passed` AND at least one node actually having passed.
fn derive_from_detail(e: &ExecutionEvidence) -> EvidenceVerdict {
    // Nothing was required, so nothing can be proven. A run with no declared required tests is not
    // evidence of correctness.
    if e.required_tests.is_empty() {
        return EvidenceVerdict::Unknown;
    }
    // No usable exit status: the run never reported how it terminated.
    let Some(exit) = e.exit_code else {
        return EvidenceVerdict::Unknown;
    };
    // A report-level error (the suite could not complete) fails the run even if nodes look green.
    if !e.errors.is_empty() {
        return EvidenceVerdict::Failed;
    }
    // An unreadable report (no nodes at all) proves nothing.
    if e.cases.is_empty() {
        return EvidenceVerdict::Unknown;
    }
    // Ambiguous report: the same node reported twice could carry contradictory statuses, and there is
    // no way to tell which is authoritative.
    let mut seen: Vec<&str> = Vec::with_capacity(e.cases.len());
    for c in &e.cases {
        if seen.contains(&c.node.as_str()) {
            return EvidenceVerdict::Unknown;
        }
        seen.push(&c.node);
    }
    // A nonzero exit is a failure regardless of what the nodes claim.
    if exit != 0 {
        return EvidenceVerdict::Failed;
    }
    // A node that failed or errored fails the run.
    if e.cases.iter().any(|c| c.status == TestStatus::Failed) {
        return EvidenceVerdict::Failed;
    }
    // Every REQUIRED node must be present AND passed. A missing or skipped required node is a
    // failure, not an unknown: the area declared that this test MUST pass, and it demonstrably did
    // not. (A required test that was never collected is the clearest way to fake a green run.)
    let passed_nodes: Vec<&str> = e
        .cases
        .iter()
        .filter(|c| c.status == TestStatus::Passed)
        .map(|c| c.node.as_str())
        .collect();
    if !e
        .required_tests
        .iter()
        .all(|req| passed_nodes.contains(&req.as_str()))
    {
        return EvidenceVerdict::Failed;
    }
    // Defensible: if nothing passed, there is nothing to admit.
    if passed_nodes.is_empty() {
        return EvidenceVerdict::Unknown;
    }
    EvidenceVerdict::Passed
}

/// Classify one carried [`ExecutionEvidence`] against the key of the candidate it is being applied
/// to. Binding first (an off-candidate report is never interpreted), then the conservative
/// cross-check of the reported outcome against the report's own detail.
#[must_use]
pub fn classify(e: &ExecutionEvidence, key: &EvidenceBinding) -> EvidenceVerdict {
    // Bound to a different task or attempt: this report describes ANOTHER candidate. Applying it
    // would be applying someone else's result, which is a definite failure of the evidence, not a
    // stale one.
    if e.binding.task != key.task || e.binding.attempt != key.attempt {
        return EvidenceVerdict::Failed;
    }
    // Bound to the same attempt but the content moved: the report is stale. Never followed, and not
    // attributed as a failure of the current content.
    if e.binding.patch_hash != key.patch_hash {
        return EvidenceVerdict::Unknown;
    }
    let reported = match e.outcome {
        ExecutionOutcome::Passed => EvidenceVerdict::Passed,
        ExecutionOutcome::Failed => EvidenceVerdict::Failed,
        ExecutionOutcome::Unknown => EvidenceVerdict::Unknown,
    };
    reported.more_conservative(derive_from_detail(e))
}

/// A human-readable reason for a non-passing verdict, folded into [`Check::detail`].
fn detail_for(e: &ExecutionEvidence, key: &EvidenceBinding, verdict: EvidenceVerdict) -> String {
    let tail = e
        .source_ref
        .as_deref()
        .map(|r| format!(" (source: {r})"))
        .unwrap_or_default();
    match verdict {
        EvidenceVerdict::Passed => format!("execution evidence passed{}", tail),
        EvidenceVerdict::Failed => {
            let reason = if e.binding.task != key.task || e.binding.attempt != key.attempt {
                format!(
                    "bound to task={:?}/attempt={:?}, not this candidate (task={:?}/attempt={:?})",
                    e.binding.task, e.binding.attempt, key.task, key.attempt
                )
            } else {
                let required = e.required_tests.len();
                let ran = e.cases.len();
                format!(
                    "failed: required={required}, reported={ran}, exit={:?}, errors={}, failing_nodes={}",
                    e.exit_code,
                    e.errors.len(),
                    e.cases
                        .iter()
                        .filter(|c| c.status == TestStatus::Failed)
                        .count()
                )
            };
            format!("execution evidence {reason}{tail}")
        }
        EvidenceVerdict::Unknown => {
            let reason = if e.binding.patch_hash != key.patch_hash {
                "stale: evidence patch_hash does not match this candidate's content".to_string()
            } else {
                format!(
                    "undecided: required={}, reported={}, exit={:?}, errors={}, outcome={:?}",
                    e.required_tests.len(),
                    e.cases.len(),
                    e.exit_code,
                    e.errors.len(),
                    e.outcome
                )
            };
            format!("execution evidence {reason} — routed to review, never admitted{tail}")
        }
    }
}

/// Adapt the input's carried [`ExecutionEvidence`] into a [`Check`] plus the rail-level consequence
/// `(check, verdict, admission_blocked)`.
///
/// Returns `None` when the input carries NO evidence: an area with no execution axis has no
/// execution check, exactly as a `None` contract contributes no answer check. Absent evidence is
/// not a failure and not a block — it simply is not this rail's business.
#[must_use]
pub fn execution_evidence_check(
    input: &VerifierInput<'_>,
) -> Option<(Check, EvidenceVerdict, bool)> {
    let evidence = input.execution_evidence?;
    let key = input.evidence_key.clone();
    let verdict = classify(evidence, &key);
    let passed = verdict != EvidenceVerdict::Failed;
    // Only an undecidable report blocks admission. A proven failure hard-rejects on `passed`
    // instead, and a proven pass hands the remainder to the panel.
    let admission_blocked = verdict == EvidenceVerdict::Unknown;
    let check = Check {
        name: EXECUTION_EVIDENCE_CHECK.into(),
        kind: CheckKind::UnitTest,
        passed,
        score: match verdict {
            EvidenceVerdict::Passed => Some(1.0),
            EvidenceVerdict::Failed => Some(0.0),
            EvidenceVerdict::Unknown => None,
        },
        detail: Some(detail_for(evidence, &key, verdict)),
    };
    Some((check, verdict, admission_blocked))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The required-test node key used throughout: an explicit `classname::name` string.
    const NODE: &str = "tests.test_demo::test_behavior";

    fn key() -> EvidenceBinding {
        EvidenceBinding {
            task: "run-1".into(),
            attempt: "rec-1".into(),
            patch_hash: "patch-1".into(),
        }
    }

    /// A `Passed` report whose detail is exactly the shape the required-test contract describes:
    /// the required node ran and passed, and the run exited zero.
    fn passing_evidence() -> ExecutionEvidence {
        ExecutionEvidence {
            outcome: ExecutionOutcome::Passed,
            required_tests: vec![NODE.into()],
            cases: vec![TestCase {
                node: NODE.into(),
                status: TestStatus::Passed,
            }],
            exit_code: Some(0),
            errors: vec![],
            source_ref: None,
            binding: key(),
        }
    }

    /// The ported required-test contract cases. Each builds the STRUCTURED report an external
    /// evaluator would carry for that situation and asserts the adapted outcome — a pass needs the
    /// required node green AND a zero exit AND no reported error; a missing, skipped, or unreadable
    /// proof is never a pass.
    #[test]
    fn required_test_passing_is_a_pass() {
        assert_eq!(
            classify(&passing_evidence(), &key()),
            EvidenceVerdict::Passed
        );
    }

    #[test]
    fn nonzero_exit_is_a_failure_not_a_pass() {
        let mut e = passing_evidence();
        e.exit_code = Some(1);
        e.outcome = ExecutionOutcome::Failed;
        assert_eq!(classify(&e, &key()), EvidenceVerdict::Failed);
    }

    #[test]
    fn unreadable_or_empty_report_is_unknown() {
        // A report with no nodes at all (absent / malformed / not a test report) proves nothing, and
        // so does one carrying no exit status. Neither may be read as a pass OR a failure.
        for mutate in [
            (|e: &mut ExecutionEvidence| e.cases.clear()) as fn(&mut ExecutionEvidence),
            |e: &mut ExecutionEvidence| e.exit_code = None,
            |e: &mut ExecutionEvidence| e.cases[0].node = "other::test".into(),
        ] {
            let mut e = passing_evidence();
            e.outcome = ExecutionOutcome::Unknown;
            mutate(&mut e);
            let v = classify(&e, &key());
            assert_eq!(v, EvidenceVerdict::Unknown, "expected Unknown, got {v:?}");
        }
    }

    #[test]
    fn skipped_required_test_is_a_failure() {
        // Skipping a test the area declared REQUIRED is a failure, not an unknown: the required
        // proof was not produced.
        let mut e = passing_evidence();
        e.cases[0].status = TestStatus::Skipped;
        e.outcome = ExecutionOutcome::Failed;
        assert_eq!(classify(&e, &key()), EvidenceVerdict::Failed);
    }

    #[test]
    fn missing_required_test_is_a_failure() {
        // The suite is green, but the required node was never collected — a fake green run.
        let mut e = passing_evidence();
        e.required_tests = vec!["other::test".into()];
        e.outcome = ExecutionOutcome::Failed;
        assert_eq!(classify(&e, &key()), EvidenceVerdict::Failed);
    }

    #[test]
    fn failing_node_fails_even_with_a_zero_exit() {
        let mut e = passing_evidence();
        e.cases[0].status = TestStatus::Failed;
        assert_eq!(
            classify(&e, &key()),
            EvidenceVerdict::Failed,
            "a claimed pass contradicted by its own failing node must fail"
        );
    }

    #[test]
    fn an_infrastructure_fault_stays_unknown_whatever_the_detail_shows() {
        // An interrupted / infrastructure-faulted run reported Unknown stays Unknown even though a
        // nonzero exit alone would otherwise read as a failure. An unproven failure must not be
        // attributed to the candidate.
        let mut e = passing_evidence();
        e.outcome = ExecutionOutcome::Unknown;
        e.exit_code = Some(1);
        assert_eq!(classify(&e, &key()), EvidenceVerdict::Unknown);
    }

    #[test]
    fn a_reported_suite_error_fails_the_run() {
        // The evaluator surfaced a report-level error: the run is failed regardless of the nodes.
        let mut e = passing_evidence();
        e.errors = vec!["collection error: could not load fixture".into()];
        assert_eq!(classify(&e, &key()), EvidenceVerdict::Failed);
    }

    #[test]
    fn an_absent_required_test_contract_is_unknown() {
        // Nothing was declared required, so no report can prove the work — never a pass.
        let mut e = passing_evidence();
        e.required_tests.clear();
        assert_eq!(classify(&e, &key()), EvidenceVerdict::Unknown);
    }

    #[test]
    fn a_duplicate_node_key_makes_the_report_unreadable() {
        // The same node reported twice could carry contradictory statuses; there is no way to tell
        // which is authoritative, so the report proves nothing.
        let mut e = passing_evidence();
        e.cases.push(TestCase {
            node: NODE.into(),
            status: TestStatus::Failed,
        });
        assert_eq!(classify(&e, &key()), EvidenceVerdict::Unknown);
    }

    #[test]
    fn a_reported_failure_is_never_softened_by_its_detail() {
        // The evaluator observed a failure the adapter cannot see (it was not in the report). The
        // authoritative failure stands.
        let mut e = passing_evidence();
        e.outcome = ExecutionOutcome::Failed;
        assert_eq!(classify(&e, &key()), EvidenceVerdict::Failed);
    }

    #[test]
    fn a_bare_pass_claim_with_no_detail_is_unknown_not_a_pass() {
        // The anti-fabrication case: an unevidenced claim of success is not evidence of success.
        let mut e = passing_evidence();
        e.cases.clear();
        assert_eq!(classify(&e, &key()), EvidenceVerdict::Unknown);
    }

    #[test]
    fn another_tasks_or_attempts_report_is_a_failure() {
        let mut e = passing_evidence();
        e.binding.attempt = "rec-other".into();
        assert_eq!(classify(&e, &key()), EvidenceVerdict::Failed);
        let mut e = passing_evidence();
        e.binding.task = "run-other".into();
        assert_eq!(classify(&e, &key()), EvidenceVerdict::Failed);
    }

    #[test]
    fn a_moved_patch_is_stale_unknown_never_followed() {
        // Same attempt, different content: the report is stale, so it is never followed — and it is
        // not attributed as a failure of the current content either.
        let mut e = passing_evidence();
        e.binding.patch_hash = "patch-older".into();
        assert_eq!(classify(&e, &key()), EvidenceVerdict::Unknown);
        // Staleness is checked before the outcome, so even a reported failure of the OLDER patch
        // is not attributed to the current content.
        e.outcome = ExecutionOutcome::Failed;
        assert_eq!(classify(&e, &key()), EvidenceVerdict::Unknown);
    }

    /// Binding is checked BEFORE the report is interpreted, so an off-candidate report can never be
    /// honoured even when it is otherwise a fully corroborated pass.
    #[test]
    fn binding_is_checked_before_the_outcome() {
        let mut e = passing_evidence();
        e.binding.attempt = "rec-other".into();
        e.outcome = ExecutionOutcome::Failed;
        assert_eq!(classify(&e, &key()), EvidenceVerdict::Failed);
    }

    // --- the rail-level consequences, exercised through `run_verifier` itself ---

    use gw_schema::{Content, Message, ReasoningDetail, Role, TestCase};

    use crate::decision::Verdict as RailVerdict;
    use crate::verifier::{NullSandboxOracle, VerifierGrade, run_verifier};

    /// A completed assistant turn with a plaintext CoT detail, so the reasoning-present gate passes
    /// and the execution axis is what is under test.
    fn assistant() -> Message {
        Message {
            role: Role::Assistant,
            content: Content::Text("here is the patch".into()),
            reasoning: Some("worked through it".into()),
            reasoning_details: Some(vec![ReasoningDetail::Text {
                text: "worked through it".into(),
                signature: None,
                id: None,
                format: None,
                index: 0,
            }]),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    fn rail_input<'a>(
        messages: &'a [Message],
        e: Option<&'a ExecutionEvidence>,
    ) -> VerifierInput<'a> {
        VerifierInput {
            messages,
            reasoning_tokens: 50,
            cot_required: true,
            contract: None,
            rule_only_authoritative: false,
            execution_evidence: e,
            evidence_key: key(),
        }
    }

    fn check_present(g: &VerifierGrade) -> bool {
        g.verification
            .checks
            .iter()
            .any(|c| c.name == EXECUTION_EVIDENCE_CHECK)
    }

    /// A proven execution failure is the verifier-first hard gate: `Reject` with `all_passed ==
    /// false`, so no panel score can ever reach the record.
    #[test]
    fn a_proven_execution_failure_is_an_authoritative_reject() {
        let msgs = vec![assistant()];
        let mut e = passing_evidence();
        e.outcome = ExecutionOutcome::Failed;
        e.cases[0].status = TestStatus::Failed;
        let g = run_verifier(&rail_input(&msgs, Some(&e)), &NullSandboxOracle);
        assert_eq!(g.verdict, RailVerdict::Reject);
        assert!(g.is_hard_reject());
        assert!(check_present(&g));
    }

    /// An undecidable report does NOT reject (nothing was proven to have failed) but it DOES block
    /// admission, so the record is held for review rather than sent to judge rescue.
    #[test]
    fn an_undecidable_report_blocks_admission_without_rejecting() {
        let msgs = vec![assistant()];
        let mut e = passing_evidence();
        e.outcome = ExecutionOutcome::Unknown;
        e.exit_code = None;
        let g = run_verifier(&rail_input(&msgs, Some(&e)), &NullSandboxOracle);
        assert_eq!(g.verdict, RailVerdict::Uncertain);
        assert!(
            !g.is_hard_reject(),
            "an undecidable report must not sink the record"
        );
        assert!(g.blocks_admission());
        assert!(g.verification.needs_review.is_some());
    }

    /// A corroborated pass contributes a passing check and hands the remainder to the panel.
    #[test]
    fn a_corroborated_pass_hands_the_remainder_to_the_panel() {
        let msgs = vec![assistant()];
        let e = passing_evidence();
        let g = run_verifier(&rail_input(&msgs, Some(&e)), &NullSandboxOracle);
        assert_eq!(g.verdict, RailVerdict::Accept);
        assert!(!g.blocks_admission());
        assert!(g.verification.all_passed);
        assert!(check_present(&g));
    }

    /// An area with no execution axis is entirely unaffected: no check, no block, no behaviour
    /// change. This is what keeps the pre-evidence pipeline intact.
    #[test]
    fn an_absent_report_leaves_the_execution_axis_inert() {
        let msgs = vec![assistant()];
        let g = run_verifier(&rail_input(&msgs, None), &NullSandboxOracle);
        assert_eq!(g.verdict, RailVerdict::Accept);
        assert!(!g.blocks_admission());
        assert!(!check_present(&g), "no evidence, no check");
    }

    /// The execution axis composes with the reasoning gate: a failing CoT gate still rejects, and a
    /// passing one does not rescue a proven execution failure.
    #[test]
    fn the_execution_axis_composes_with_the_reasoning_gate() {
        let mut no_cot = assistant();
        no_cot.reasoning = None;
        no_cot.reasoning_details = None;
        let msgs = vec![no_cot];
        let passing = passing_evidence();
        let mut input = rail_input(&msgs, Some(&passing));
        input.reasoning_tokens = 0;
        assert_eq!(
            run_verifier(&input, &NullSandboxOracle).verdict,
            RailVerdict::Reject
        );
    }
}
