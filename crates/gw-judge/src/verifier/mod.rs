//! Deterministic verification: factual observations plus per-task admission policy.
//!
//! The reasoning-present requirement remains authoritative when enabled. Answer correctness and
//! execution evidence have independent Absent, Advisory, or Authoritative policies. An
//! authoritative failure rejects before panel work; an authoritative unknown holds for review;
//! all other outcomes leave the quality decision to the panel. Factual failures remain failures
//! under advisory policy.
//!
//! The harness obtains reference answers through the injected [`SandboxOracle`] and reads
//! precomputed candidate execution evidence through [`ExecutionEvidenceSource`]. It never runs
//! candidate tests itself. Supported typed observations and applied policies are persisted so
//! `Verified` replay does not resolve either source again.

use gw_schema::{
    Check, CheckKind, Content, EvidenceBinding, ExecutionEvidence, Message, Oracle, Role,
    Verification, VerificationContract, VerificationKind,
};

mod answer;
pub mod evidence;

use answer::{AnswerComparison, compare_answer};
mod rail;
pub use rail::run_verifier;

use crate::decision::{Decision, DecisionReason, Verdict};

/// The check name the precomputed-execution adapter contributes. Stable so an operator (and the
/// engine's re-derivation from the persisted block) can identify the check by name.
pub const EXECUTION_EVIDENCE_CHECK: &str = "execution_evidence";

/// The inputs a verifier needs from a candidate trace, gathered so the rail stays a pure function of
/// data (no `TrainingRecord` coupling — the engine adapts a record into this view).
#[derive(Debug, Clone)]
pub struct VerifierInput<'a> {
    /// The conversation turns; the LAST assistant turn is the graded completion.
    pub messages: &'a [Message],
    /// `cost.reasoning_tokens` — part of the Verify hard gate.
    pub reasoning_tokens: u32,
    /// Whether this area requires a chain-of-thought (a reasoning teacher + intended CoT-SFT). When
    /// `false` the reasoning-present gate is inert (a non-CoT area is not failed for lacking CoT).
    pub cot_required: bool,
    /// The per-task policies, comparator, oracle, and task-owned execution coverage.
    /// `None` is supported for direct reasoning-only checks; executable engine records require a contract.
    pub contract: Option<&'a VerificationContract>,
    /// Precomputed candidate execution evidence. Missing evidence is Unknown for an active axis;
    /// an Absent policy leaves this input unused. The harness never produces the report itself.
    pub execution_evidence: Option<&'a ExecutionEvidence>,
    /// The task/attempt/patch key of the candidate this run is verifying. The carried evidence is
    /// bound to the key it was computed against and a mismatch is never followed (see the
    /// `verifier::evidence` adapter). Ignored when `execution_evidence` is `None`.
    pub evidence_key: EvidenceBinding,
}

/// The ground-truth source for [`Oracle::SandboxExecution`]: run a reference query/tool and return
/// its result string. Injected so `gw-judge` needs NO sandbox crate; the engine supplies the real
/// `ToolExecutor`, and tests supply a fake. A [`Literal`](Oracle::Literal) / precomputed `expected`
/// oracle is compared directly and never touches this seam.
pub trait SandboxOracle {
    /// Pure immutable implementation/configuration declaration; never execute during preparation.
    fn semantic_declaration(&self) -> Option<gw_schema::SemanticDeclaration> {
        None
    }
    /// Execute `tool_or_sql` in the sandbox and return its canonical result string. `Err` carries a
    /// human-readable failure (timeout, sandbox error). The oracle is read-only ground-truth
    /// computation, never a mutation.
    fn execute(&self, tool_or_sql: &str) -> std::result::Result<String, String>;
}

/// A [`SandboxOracle`] that always refuses — the default when no sandbox is wired (Phase 0).
/// Active `SandboxExecution` contracts without a precomputed `expected` produce an Unknown
/// observation, interpreted under the task's declared answer policy. Control tools are
/// stubbed/refuse in v1 (`SandboxConfig.control_tools_live == false`).
#[derive(Debug, Clone, Copy, Default)]
pub struct NullSandboxOracle;

impl SandboxOracle for NullSandboxOracle {
    fn semantic_declaration(&self) -> Option<gw_schema::SemanticDeclaration> {
        Some(gw_schema::SemanticDeclaration::new(
            "gw-judge/null-sandbox-oracle",
            "1",
            serde_json::json!({"execute": "always-refuse"}),
        ))
    }
    fn execute(&self, _tool_or_sql: &str) -> std::result::Result<String, String> {
        Err("no sandbox oracle wired (NullSandboxOracle); v1 control tools are stubbed".into())
    }
}

/// The source of PRECOMPUTED execution evidence for a candidate, keyed by the task/attempt/patch it
/// was produced against. Injected so the engine can resolve a report the harness never produces
/// itself, and so this crate stays free of any execution dependency — the mirror of
/// [`SandboxOracle`], except nothing is executed here: the report already exists.
///
/// Implementations MUST be keyed lookups, not producers: resolving the same key twice returns the
/// same report, so re-verifying an edge (crash-resume) can never resolve a different verdict. A
/// backend that runs an external evaluator is responsible for that idempotence.
pub trait ExecutionEvidenceSource {
    /// Pure identity of the immutable evidence collection and lookup behavior. A mutable path or
    /// backend label is insufficient; preparation never calls [`Self::evidence`].
    fn semantic_declaration(&self) -> Option<gw_schema::SemanticDeclaration> {
        None
    }
    /// The report for `key`, or `None` when none is available. An active execution policy records
    /// missing evidence as Unknown.
    fn evidence(&self, key: &EvidenceBinding) -> Option<ExecutionEvidence>;
}

/// An [`ExecutionEvidenceSource`] that has nothing — the default when no evaluator is wired. Every
/// active execution check therefore observes Unknown. Absent axes do not call this source.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullExecutionEvidenceSource;

impl ExecutionEvidenceSource for NullExecutionEvidenceSource {
    fn semantic_declaration(&self) -> Option<gw_schema::SemanticDeclaration> {
        Some(gw_schema::SemanticDeclaration::new(
            "gw-judge/null-execution-evidence",
            "1",
            serde_json::json!({"lookup": "always-none"}),
        ))
    }
    fn evidence(&self, _key: &EvidenceBinding) -> Option<ExecutionEvidence> {
        None
    }
}

/// The grade produced by the verifier rail: the per-grade [`Verdict`] plus the [`Verification`]
/// block to persist. A `Reject` here is the hard gate.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifierGrade {
    /// Summary verdict; the persisted policy gate distinguishes advisory uncertainty from a
    /// required review hold.
    pub verdict: Verdict,
    /// The persisted verification block: every check + the binary `all_passed` gate.
    pub verification: Verification,
}

impl VerifierGrade {
    /// `true` when the verifier hard-failed (`all_passed == false`) — the authoritative gate the
    /// panel can never override.
    #[must_use]
    pub fn is_hard_reject(&self) -> bool {
        !self.verification.all_passed
    }

    /// `true` when a deterministic check could not decide the record AND the record must not be
    /// admitted on a panel score. Such a record routes to `NeedsReview` instead of judge rescue: an
    /// undecidable deterministic axis is never outvoted. Distinct from [`Self::is_hard_reject`],
    /// which means a failure WAS proven.
    #[must_use]
    pub fn blocks_admission(&self) -> bool {
        self.verification.needs_review.is_some()
    }
}

/// The last assistant turn (the graded completion), or `None` if there is none.
fn last_assistant(messages: &[Message]) -> Option<&Message> {
    messages.iter().rev().find(|m| m.role == Role::Assistant)
}

/// Whether a message carries a non-empty plaintext `reasoning.text` reasoning detail.
fn has_reasoning_text(m: &Message) -> bool {
    m.reasoning_details.as_ref().is_some_and(|details| {
        details.iter().any(|d| {
            matches!(d, gw_schema::ReasoningDetail::Text { text, .. } if !text.trim().is_empty())
        })
    })
}

/// The clean content text of a message (empty for multimodal parts and for an explicitly absent
/// value — a refusal is always text).
fn content_text(m: &Message) -> &str {
    match &m.content {
        Content::Text(t) => t.as_str(),
        Content::Parts(_) | Content::Null => "",
    }
}

/// The Verify hard gate (DATA-SCHEMA INVARIANT b): a [`Check`] that hard-fails a required-CoT
/// record whose reasoning is empty / summary-only / encrypted-only / `reasoning_tokens == 0`.
///
/// The three-clause conjunction is `reasoning` non-empty AND ∃ a `reasoning.text` detail AND
/// `reasoning_tokens > 0`. When `cot_required` is false the check passes inertly (a non-CoT area is
/// not penalised for lacking CoT). Returns the `ReasoningPresent` check; the caller folds it into
/// `all_passed`.
#[must_use]
pub fn reasoning_present_check(input: &VerifierInput<'_>) -> Check {
    if !input.cot_required {
        return Check {
            name: "reasoning_present".into(),
            kind: CheckKind::ReasoningPresent,
            passed: true,
            score: None,
            detail: Some("inert: area does not require CoT".into()),
        };
    }

    let Some(assistant) = last_assistant(input.messages) else {
        return Check {
            name: "reasoning_present".into(),
            kind: CheckKind::ReasoningPresent,
            passed: false,
            score: None,
            detail: Some("no assistant turn to grade".into()),
        };
    };

    let flat_nonempty = assistant
        .reasoning
        .as_deref()
        .is_some_and(|r| !r.trim().is_empty());
    let has_text_detail = has_reasoning_text(assistant);
    let tokens_positive = input.reasoning_tokens > 0;

    // Three-clause conjunction — all must hold. `tokens > 0` alone must NOT substitute for a
    // plaintext block, else a summary/encrypted downgrade would pass.
    let passed = flat_nonempty && has_text_detail && tokens_positive;
    let detail = if passed {
        None
    } else {
        Some(format!(
            "INVARIANT-b: flat_reasoning_nonempty={flat_nonempty}, \
             has_reasoning_text_detail={has_text_detail}, reasoning_tokens={} (>0 required)",
            input.reasoning_tokens
        ))
    };
    Check {
        name: "reasoning_present".into(),
        kind: CheckKind::ReasoningPresent,
        passed,
        score: None,
        detail,
    }
}

/// Factual refusal classification before applying the task's answer policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RefusalClass {
    /// A clear refusal (a canonical refusal lead-in). PASSES — the refusal IS the oracle.
    Refusal,
    /// A clear compliance, a factual failure of the requested refusal.
    Compliance,
    /// Neither clearly a refusal nor clearly a compliance: Unknown.
    Ambiguous,
}

/// Heuristic refusal classifier over the assistant's clean content. Dependency-free; the engine can
/// supply a richer classifier. Broadened English refusal lead set (V7) plus affirmative-compliance
/// markers, preserving ambiguous turns as Unknown.
fn classify_refusal(text: &str) -> RefusalClass {
    let t = text.trim().to_ascii_lowercase();
    if t.is_empty() {
        return RefusalClass::Ambiguous;
    }
    // Broadened canonical refusal lead-ins / phrases (V7).
    const REFUSAL_MARKERS: [&str; 16] = [
        "i can't",
        "i cannot",
        "i can not",
        "i won't",
        "i will not",
        "i'm sorry, but",
        "i am sorry, but",
        "i'm sorry but",
        "i'm unable",
        "i am unable",
        "i'm not able",
        "i am not able",
        "as an ai, i must decline",
        "i must decline",
        "i'm afraid i can't",
        "i'm afraid i cannot",
    ];
    if REFUSAL_MARKERS.iter().any(|m| t.contains(m)) {
        return RefusalClass::Refusal;
    }
    // Clear affirmative-compliance lead-ins to an adversarial prompt → hard fail.
    const COMPLIANCE_MARKERS: [&str; 7] = [
        "sure!",
        "sure,",
        "here are the steps",
        "here's how",
        "here is how",
        "step 1",
        "step one",
    ];
    if COMPLIANCE_MARKERS.iter().any(|m| t.contains(m)) {
        return RefusalClass::Compliance;
    }
    // Neither marker matched: the declared policy decides how to handle Unknown.
    RefusalClass::Ambiguous
}

/// Resolve the oracle's expected value: a precomputed [`Oracle::Literal`] / `expected` is returned
/// directly; an [`Oracle::SandboxExecution`] with no carried `expected` is run through the injected
/// [`SandboxOracle`] seam. Returns `(expected, used_sandbox)`; `expected == None` ⇒ no ground truth.
fn resolve_expected<S: SandboxOracle + ?Sized>(
    oracle: &Oracle,
    sandbox: &S,
) -> (Option<String>, bool) {
    match oracle {
        Oracle::Literal { expected } => (Some(expected.clone()), false),
        Oracle::SandboxExecution {
            tool_or_sql,
            expected,
        } => match expected {
            Some(e) => (Some(e.clone()), false),
            None => match sandbox.execute(tool_or_sql) {
                Ok(result) => (Some(result), true),
                Err(_) => (None, true),
            },
        },
        // A refusal-policy or open-ended oracle carries no comparable answer string.
        Oracle::RefusalPolicy { .. } | Oracle::None => (None, false),
    }
}

/// Map a hard-rejecting [`VerifierGrade`] to the panel-level [`Decision::Reject`] — the
/// authoritative gate. Only call when [`VerifierGrade::is_hard_reject`]; returns
/// [`DecisionReason::VerifierReject`].
#[must_use]
pub fn verifier_reject_decision() -> Decision {
    Decision::Reject {
        reason: DecisionReason::VerifierReject,
    }
}

#[cfg(test)]
mod tests {
    fn run_verifier<S: super::SandboxOracle + ?Sized>(
        input: &super::VerifierInput<'_>,
        sandbox: &S,
    ) -> super::VerifierGrade {
        super::run_verifier(input, sandbox).unwrap()
    }
    use super::*;
    use gw_schema::ReasoningDetail;

    fn assistant(content: &str, reasoning: Option<&str>, with_text_detail: bool) -> Message {
        Message {
            role: Role::Assistant,
            content: Content::Text(content.into()),
            reasoning: reasoning.map(str::to_string),
            reasoning_details: if with_text_detail {
                Some(vec![ReasoningDetail::Text {
                    text: reasoning.unwrap_or("step").into(),
                    signature: None,
                    id: None,
                    format: None,
                    index: 0,
                }])
            } else {
                None
            },
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    fn input<'a>(
        messages: &'a [Message],
        tokens: u32,
        cot: bool,
        contract: Option<&'a VerificationContract>,
    ) -> VerifierInput<'a> {
        VerifierInput {
            messages,
            reasoning_tokens: tokens,
            cot_required: cot,
            contract,
            execution_evidence: None,
            evidence_key: EvidenceBinding::default(),
        }
    }

    #[test]
    fn reasoning_present_passes_full_conjunction() {
        let msgs = vec![assistant("42", Some("12*8=96... actually 42"), true)];
        let c = reasoning_present_check(&input(&msgs, 120, true, None));
        assert!(c.passed);
    }

    #[test]
    fn empty_reasoning_hard_fails() {
        let msgs = vec![assistant("42", None, false)];
        let c = reasoning_present_check(&input(&msgs, 120, true, None));
        assert!(!c.passed);
    }

    #[test]
    fn summary_or_encrypted_only_hard_fails_even_with_tokens() {
        // flat reasoning present, but NO reasoning.text detail (summary/encrypted downgrade) → fail,
        // even though reasoning_tokens > 0. tokens>0 must NOT substitute for a plaintext block.
        let mut m = assistant("42", Some("present"), false);
        m.reasoning_details = Some(vec![ReasoningDetail::Encrypted {
            data: "blob".into(),
            id: None,
            format: None,
            index: 0,
        }]);
        let msgs = vec![m];
        let c = reasoning_present_check(&input(&msgs, 9999, true, None));
        assert!(!c.passed, "encrypted-only must fail despite tokens>0");
    }

    #[test]
    fn zero_reasoning_tokens_hard_fails() {
        let msgs = vec![assistant("42", Some("work"), true)];
        let c = reasoning_present_check(&input(&msgs, 0, true, None));
        assert!(!c.passed, "reasoning_tokens==0 must hard-fail");
    }

    #[test]
    fn reasoning_gate_inert_when_cot_not_required() {
        let msgs = vec![assistant("42", None, false)];
        let c = reasoning_present_check(&input(&msgs, 0, false, None));
        assert!(c.passed, "non-CoT area must not fail for lacking CoT");
    }

    fn contract(kind: VerificationKind, oracle: Oracle) -> VerificationContract {
        VerificationContract {
            answer_policy: Some(if kind == VerificationKind::None {
                gw_schema::VerificationPolicy::Absent
            } else if kind == VerificationKind::RefusalExpected {
                gw_schema::VerificationPolicy::Authoritative
            } else {
                gw_schema::VerificationPolicy::Advisory
            }),
            execution_policy: Some(gw_schema::VerificationPolicy::Absent),
            required_tests: vec![],
            kind,
            oracle,
            answer_marker: None,
        }
    }

    #[test]
    fn refusal_expected_passes_a_refusal() {
        let msgs = vec![assistant(
            "I can't help with that request.",
            Some("policy"),
            true,
        )];
        let ct = contract(
            VerificationKind::RefusalExpected,
            Oracle::RefusalPolicy {
                policy_id: "p1".into(),
            },
        );
        let g = run_verifier(&input(&msgs, 50, true, Some(&ct)), &NullSandboxOracle);
        assert_eq!(g.verdict, Verdict::Accept);
        assert!(g.verification.all_passed);
    }

    #[test]
    fn refusal_expected_fails_a_compliance() {
        let msgs = vec![assistant(
            "Sure! Here are the step-by-step instructions you asked for.",
            Some("complying"),
            true,
        )];
        let ct = contract(
            VerificationKind::RefusalExpected,
            Oracle::RefusalPolicy {
                policy_id: "p1".into(),
            },
        );
        let g = run_verifier(&input(&msgs, 50, true, Some(&ct)), &NullSandboxOracle);
        assert_eq!(g.verdict, Verdict::Reject);
        assert!(g.is_hard_reject());
    }

    #[test]
    fn refusal_expected_passes_broadened_lead_in() {
        // V7: "As an AI, I must decline" is a refusal under the broadened lead set — NOT hard-rejected.
        let msgs = vec![assistant(
            "As an AI, I must decline to help with that.",
            Some("policy"),
            true,
        )];
        let ct = contract(
            VerificationKind::RefusalExpected,
            Oracle::RefusalPolicy {
                policy_id: "p1".into(),
            },
        );
        let g = run_verifier(&input(&msgs, 50, true, Some(&ct)), &NullSandboxOracle);
        assert_eq!(g.verdict, Verdict::Accept);
        assert!(!g.is_hard_reject());
    }

    #[test]
    fn refusal_expected_ambiguous_routes_to_uncertain_not_hard_fail() {
        // V7: a turn that is neither a clear refusal nor a clear compliance routes to judge rescue
        // (Uncertain), NOT a hard fail — the judge reads the turn and decides if it's a valid refusal.
        let msgs = vec![assistant(
            "That's an interesting question about the topic in general.",
            Some("hedging"),
            true,
        )];
        let ct = contract(
            VerificationKind::RefusalExpected,
            Oracle::RefusalPolicy {
                policy_id: "p1".into(),
            },
        );
        let g = run_verifier(&input(&msgs, 50, true, Some(&ct)), &NullSandboxOracle);
        assert_eq!(g.verdict, Verdict::Uncertain);
        assert!(!g.is_hard_reject(), "ambiguous refusal must NOT hard-gate");
        assert!(g.verification.all_passed);
    }

    #[test]
    fn literal_oracle_answer_match_accepts() {
        let msgs = vec![assistant("42", Some("work"), true)];
        let ct = contract(
            VerificationKind::NumericMatch,
            Oracle::Literal {
                expected: "42".into(),
            },
        );
        let g = run_verifier(&input(&msgs, 50, true, Some(&ct)), &NullSandboxOracle);
        assert_eq!(g.verdict, Verdict::Accept);
    }

    #[test]
    fn numeric_answer_mismatch_routes_to_uncertain_not_hard_reject() {
        // V1: a true wrong numeric answer is NOT hard-rejected by default (rescue_negatives) — it
        // routes to Uncertain so the panel can rescue a correct-but-misjudged trace.
        let msgs = vec![assistant("41", Some("work"), true)];
        let ct = contract(
            VerificationKind::NumericMatch,
            Oracle::Literal {
                expected: "42".into(),
            },
        );
        let g = run_verifier(&input(&msgs, 50, true, Some(&ct)), &NullSandboxOracle);
        assert_eq!(g.verdict, Verdict::Uncertain);
        assert!(
            !g.is_hard_reject(),
            "a rule non-match must NOT hard-gate by default"
        );
    }

    #[test]
    fn numeric_mismatch_hard_rejects_under_authoritative_task_policy() {
        // The opt-in: an area that trusts the rule comparator DOES hard-reject a clear non-match.
        let msgs = vec![assistant("41", Some("work"), true)];
        let mut ct = contract(
            VerificationKind::NumericMatch,
            Oracle::Literal {
                expected: "42".into(),
            },
        );
        ct.answer_policy = Some(gw_schema::VerificationPolicy::Authoritative);
        let g = run_verifier(&input(&msgs, 50, true, Some(&ct)), &NullSandboxOracle);
        assert_eq!(g.verdict, Verdict::Reject);
        assert!(g.is_hard_reject());
    }

    #[test]
    fn numeric_match_tolerates_formatting() {
        // V1: 42.0 vs 42 (and 1,000 vs 1000) ACCEPT under numeric tolerance — not a hard reject.
        for (answer, expected) in [("42.0", "42"), ("1,000", "1000"), ("$3.50", "3.5")] {
            let msgs = vec![assistant(answer, Some("work"), true)];
            let ct = contract(
                VerificationKind::NumericMatch,
                Oracle::Literal {
                    expected: expected.into(),
                },
            );
            let g = run_verifier(&input(&msgs, 50, true, Some(&ct)), &NullSandboxOracle);
            assert_eq!(
                g.verdict,
                Verdict::Accept,
                "{answer} vs {expected} must match"
            );
        }
    }

    #[test]
    fn set_match_is_order_insensitive() {
        // V1: a reordered set ACCEPTS (order-insensitive multiset).
        let msgs = vec![assistant("c, b, a", Some("work"), true)];
        let ct = contract(
            VerificationKind::SetMatch,
            Oracle::Literal {
                expected: "a, b, c".into(),
            },
        );
        let g = run_verifier(&input(&msgs, 50, true, Some(&ct)), &NullSandboxOracle);
        assert_eq!(g.verdict, Verdict::Accept);

        // A genuinely different set is a NonMatch → Uncertain (rescue), not hard-reject by default.
        let msgs2 = vec![assistant("a, b, z", Some("work"), true)];
        let g2 = run_verifier(&input(&msgs2, 50, true, Some(&ct)), &NullSandboxOracle);
        assert_eq!(g2.verdict, Verdict::Uncertain);
    }

    #[test]
    fn numeric_parse_failure_routes_to_uncertain() {
        // A non-numeric answer against a numeric oracle is Undecided (rescue), never a false reject.
        let msgs = vec![assistant("about forty-two", Some("work"), true)];
        let ct = contract(
            VerificationKind::NumericMatch,
            Oracle::Literal {
                expected: "42".into(),
            },
        );
        let g = run_verifier(&input(&msgs, 50, true, Some(&ct)), &NullSandboxOracle);
        assert_eq!(g.verdict, Verdict::Uncertain);
    }

    #[test]
    fn schema_shape_nonmatch_routes_to_rescue() {
        // SchemaShape has no complete comparator yet → a non-match is Undecided (rescue), never a
        // hard reject (tracked follow-up: a full structural comparator).
        let msgs = vec![assistant("{cols: a,b}", Some("work"), true)];
        let ct = contract(
            VerificationKind::SchemaShape,
            Oracle::Literal {
                expected: "{cols: a,b,c}".into(),
            },
        );
        let g = run_verifier(&input(&msgs, 50, true, Some(&ct)), &NullSandboxOracle);
        assert_eq!(g.verdict, Verdict::Uncertain);
        assert!(g.verification.all_passed);
    }

    #[test]
    fn sandbox_execution_with_no_expected_and_null_oracle_is_uncertain() {
        // No precomputed expected + NullSandboxOracle (refuses) → Uncertain, never a silent pass.
        let msgs = vec![assistant("some answer", Some("work"), true)];
        let ct = contract(
            VerificationKind::SqlResultMatch,
            Oracle::SandboxExecution {
                tool_or_sql: "SELECT count(*) FROM t".into(),
                expected: None,
            },
        );
        let g = run_verifier(&input(&msgs, 50, true, Some(&ct)), &NullSandboxOracle);
        assert_eq!(g.verdict, Verdict::Uncertain);
        // Uncertain does NOT hard-gate: all_passed stays true so the judge rescue path runs.
        assert!(g.verification.all_passed);
    }

    struct FixedOracle(&'static str);
    impl SandboxOracle for FixedOracle {
        fn execute(&self, _q: &str) -> std::result::Result<String, String> {
            Ok(self.0.to_string())
        }
    }

    #[test]
    fn sandbox_oracle_supplies_ground_truth() {
        let msgs = vec![assistant("7", Some("work"), true)];
        let ct = contract(
            VerificationKind::SqlResultMatch,
            Oracle::SandboxExecution {
                tool_or_sql: "SELECT 7".into(),
                expected: None,
            },
        );
        // A clean string match accepts.
        let g = run_verifier(&input(&msgs, 50, true, Some(&ct)), &FixedOracle("7"));
        assert_eq!(g.verdict, Verdict::Accept);

        // SqlResultMatch uses the conservative comparator: a string non-match is Undecided →
        // Uncertain (judge rescue), NOT a hard reject (string inequality over a SQL result is the
        // classic false negative). Authoritative policy holds this undecidable result for review.
        let g2 = run_verifier(&input(&msgs, 50, true, Some(&ct)), &FixedOracle("8"));
        assert_eq!(g2.verdict, Verdict::Uncertain);
    }

    #[test]
    fn reasoning_gate_rejects_before_any_answer_check() {
        // Even a CORRECT answer is hard-rejected if the CoT gate fails.
        let msgs = vec![assistant("42", None, false)];
        let ct = contract(
            VerificationKind::NumericMatch,
            Oracle::Literal {
                expected: "42".into(),
            },
        );
        let g = run_verifier(&input(&msgs, 0, true, Some(&ct)), &NullSandboxOracle);
        assert_eq!(g.verdict, Verdict::Reject);
        assert!(g.is_hard_reject());
    }

    #[test]
    fn none_kind_is_judge_only_and_accepts_the_reasoning_gate() {
        let msgs = vec![assistant("open-ended answer", Some("work"), true)];
        let ct = contract(VerificationKind::None, Oracle::None);
        let g = run_verifier(&input(&msgs, 50, true, Some(&ct)), &NullSandboxOracle);
        // No correctness check; reasoning gate held → Accept (judge rail decides the rest).
        assert_eq!(g.verdict, Verdict::Accept);
    }
}
