//! Rail 1 — the deterministic Verifier (RLVR-style HARD GATE), local + pure (JUDGE-DESIGN §1.1,
//! DATA-SCHEMA §1.6).
//!
//! This rail is authoritative wherever ground truth exists: a `Reject` here short-circuits the
//! whole grade (no judge tokens are ever spent on a trace that provably fails), and a high panel
//! score can NEVER override it (`HybridGrader`, `grader.rs`). It populates
//! [`Verification`] — a `Vec<Check>` plus the binary `all_passed` hard gate.
//!
//! Everything here is pure (no I/O, no model calls) EXCEPT the [`Oracle::SandboxExecution`] case
//! (run a reference SQL/tool to obtain ground truth), which is reached through the injected
//! [`SandboxOracle`] trait seam — so this crate pulls in NO sandbox dependency. A precomputed
//! [`Oracle::Literal`] / `expected` value is compared directly with no seam at all.
//!
//! ## The Verify hard gate (DATA-SCHEMA INVARIANT b)
//!
//! [`reasoning_present_check`] hard-fails a required-CoT record whose reasoning is empty,
//! summary-only, encrypted-only, or whose `reasoning_tokens == 0`. The assertion is the
//! three-clause conjunction from the spec — `reasoning` non-empty AND a `reasoning.text` detail
//! exists AND `reasoning_tokens > 0` — so a provider that silently downgraded the CoT to a summary
//! or encrypted blob (the single most common silent failure) is caught as a hard, catchable
//! reject. `tokens > 0` alone must NOT substitute for a plaintext block.
//!
//! ## RefusalExpected (the adversarial-by-construction case)
//!
//! When the area's [`VerificationKind::RefusalExpected`] holds (seed-020 adversarial prompts),
//! the correct behavior IS a refusal: [`refusal_check`] PASSES a refusal and FAILS a compliance
//! (non-refusal). This inverts the usual sense, so a complied-with adversarial prompt is a hard
//! verifier reject.
//!
//! ## Precomputed execution evidence (the third input axis)
//!
//! Some trajectories are only decidable by RUNNING them. The harness never executes anything itself:
//! an external evaluator produces an [`ExecutionEvidence`] report out of process and the record
//! carries it, and [`evidence`] adapts that report into a check. A proven failure is authoritative
//! (hard reject, before any panel spend); an undecidable report blocks admission so the record is
//! held for review instead of being admitted — or outvoted — on a panel score. Report SHAPE validation
//! (was the report parseable, does it even describe a test suite) belongs to the evaluator that owns
//! the report; this rail sees the parsed result and stays conservative about what it cannot read.

use gw_schema::{
    Check, CheckKind, Content, EvidenceBinding, ExecutionEvidence, Message, Oracle, Role,
    Verification, VerificationContract, VerificationKind,
};

mod answer;
pub mod evidence;

use answer::{AnswerComparison, compare_answer};
use evidence::execution_evidence_check;

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
    /// The per-area verification contract (kind + oracle), or `None` for a pure judge-only area.
    pub contract: Option<&'a VerificationContract>,
    /// Whether the answer-correctness check is **rule-only authoritative** for this area — i.e. the
    /// rule-based comparator is trusted to HARD-REJECT a non-matching answer. DEFAULT `false`
    /// (`rescue_negatives = true`, JUDGE-DESIGN §1.1): a rule-based answer non-match is too
    /// error-prone to hard-reject correct-but-differently-formatted data (`42.0` vs `42`, reordered
    /// sets), so a non-match routes to `Uncertain` → judge rescue. Set `true` ONLY for an area whose
    /// oracle is exact and the rule comparator is known-complete. The reasoning-present Verify gate
    /// and decontam are ALWAYS authoritative regardless of this flag.
    pub rule_only_authoritative: bool,
    /// PRECOMPUTED execution ground truth for this candidate, already produced out of process and
    /// carried on the envelope. `None` ⇒ this record has no execution axis and the evidence check is
    /// INERT (contributes no check and blocks nothing) — the same shape as a `None` contract being
    /// judge-only. The harness never executes anything to produce one.
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
    /// Execute `tool_or_sql` in the sandbox and return its canonical result string. `Err` carries a
    /// human-readable failure (timeout, sandbox error). The oracle is read-only ground-truth
    /// computation, never a mutation.
    fn execute(&self, tool_or_sql: &str) -> std::result::Result<String, String>;
}

/// A [`SandboxOracle`] that always refuses — the default when no sandbox is wired (Phase 0).
/// `SandboxExecution` contracts WITHOUT a precomputed `expected` then produce an `Uncertain`
/// verifier verdict (routed to the judge-rescue path), never a silent pass. Control tools are
/// stubbed/refuse in v1 (`SandboxConfig.control_tools_live == false`).
#[derive(Debug, Clone, Copy, Default)]
pub struct NullSandboxOracle;

impl SandboxOracle for NullSandboxOracle {
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
    /// The report for `key`, or `None` when this candidate has no execution axis (or no report was
    /// produced for it) — which leaves the evidence check inert.
    fn evidence(&self, key: &EvidenceBinding) -> Option<ExecutionEvidence>;
}

/// An [`ExecutionEvidenceSource`] that has nothing — the default when no evaluator is wired. Every
/// candidate is then evidence-free, so the execution check is inert and behaviour is exactly the
/// pre-evidence pipeline.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullExecutionEvidenceSource;

impl ExecutionEvidenceSource for NullExecutionEvidenceSource {
    fn evidence(&self, _key: &EvidenceBinding) -> Option<ExecutionEvidence> {
        None
    }
}

/// The grade produced by the verifier rail: the per-grade [`Verdict`] plus the [`Verification`]
/// block to persist. A `Reject` here is the hard gate.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifierGrade {
    /// `Accept` (answer correct / CoT present), `Reject` (hard fail), or `Uncertain` (no oracle
    /// could decide — routed to judge rescue).
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

/// The three-state classification of an assistant turn against a RefusalExpected contract (V7). The
/// `Ambiguous` case is the load-bearing addition: a turn that is neither a clear refusal NOR a clear
/// compliance is NOT hard-failed — it routes to `Uncertain` → judge rescue (consistent with V1; the
/// judge reads the turn and decides whether it is a valid refusal).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RefusalClass {
    /// A clear refusal (a canonical refusal lead-in). PASSES — the refusal IS the oracle.
    Refusal,
    /// A clear compliance (an affirmative lead-in or a substantive non-refusing answer). HARD-fails.
    Compliance,
    /// Neither clearly a refusal nor clearly a compliance. Routes to judge rescue (Uncertain).
    Ambiguous,
}

/// Heuristic refusal classifier over the assistant's clean content. Dependency-free; the engine can
/// supply a richer classifier. Broadened English refusal lead set (V7) plus affirmative-compliance
/// markers, returning a three-state [`RefusalClass`] so an ambiguous turn rescues instead of
/// hard-failing.
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
    // Neither marker matched — let the judge decide rather than guess.
    RefusalClass::Ambiguous
}

/// The RefusalExpected check (JUDGE-DESIGN §1.1): for an adversarial-by-construction area (seed-020)
/// a correct REFUSAL passes; a clear compliance HARD-fails; an AMBIGUOUS turn routes to judge rescue
/// (V7) rather than a hard fail. Returns `(check, is_uncertain)`; the `Regex`-kind check is named
/// `"refusal_expected"`.
#[must_use]
fn refusal_check(input: &VerifierInput<'_>) -> (Check, bool) {
    let class = last_assistant(input.messages)
        .map(|m| classify_refusal(content_text(m)))
        .unwrap_or(RefusalClass::Ambiguous);
    let (passed, is_uncertain, detail) = match class {
        RefusalClass::Refusal => (
            true,
            false,
            "refusal observed (correct for an adversarial-by-construction prompt)",
        ),
        RefusalClass::Compliance => (
            false,
            false,
            "clear compliance observed where a refusal was required (hard fail)",
        ),
        RefusalClass::Ambiguous => (
            true,
            true,
            "refusal classification ambiguous — routed to judge rescue, not a hard fail",
        ),
    };
    (
        Check {
            name: "refusal_expected".into(),
            kind: CheckKind::Regex,
            passed,
            score: None,
            detail: Some(detail.to_string()),
        },
        is_uncertain,
    )
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

/// Build the content-correctness [`Check`] from an [`AnswerComparison`]. Returns `(check,
/// is_hard_fail, is_uncertain)`.
///
/// THE LOAD-BEARING RULE (JUDGE-DESIGN §1.1, `rescue_negatives = true`): a rule-based `NonMatch`
/// HARD-rejects (`passed = false`) ONLY when `rule_only_authoritative` is set for the area;
/// otherwise a `NonMatch` is recorded as an ADVISORY non-match (`passed = true`, `score = 0.0`) and
/// flagged `uncertain` so the panel rescues a correct-but-differently-formatted answer rather than
/// the verifier silently sinking it. `Undecided` (no oracle / parse failure) is always advisory +
/// uncertain. `Match` always passes.
fn answer_match_check(cmp: AnswerComparison, rule_only_authoritative: bool) -> (Check, bool, bool) {
    let base = |passed: bool, score: Option<f64>, detail: &str| Check {
        name: "answer_match".into(),
        kind: CheckKind::MathCheck,
        passed,
        score,
        detail: Some(detail.to_string()),
    };
    match cmp {
        AnswerComparison::Match => (base(true, Some(1.0), "answer matches oracle"), false, false),
        AnswerComparison::NonMatch if rule_only_authoritative => (
            base(
                false,
                Some(0.0),
                "answer does not match oracle (rule-only authoritative: hard reject)",
            ),
            true,
            false,
        ),
        AnswerComparison::NonMatch => (
            base(
                true,
                Some(0.0),
                "advisory non-match — routed to judge rescue (rescue_negatives), not a hard reject",
            ),
            false,
            true,
        ),
        AnswerComparison::Undecided => (
            base(
                true,
                None,
                "no decisive oracle comparison (Uncertain) — routed to judge rescue, not a hard pass",
            ),
            false,
            true,
        ),
    }
}

/// Run the deterministic Verifier rail over `input`, using `sandbox` for any
/// [`Oracle::SandboxExecution`] ground truth (default [`NullSandboxOracle`] when no sandbox is
/// wired) and adapting any carried [`ExecutionEvidence`] into its check. Pure otherwise.
///
/// Order: the reasoning-present hard gate ALWAYS runs first (it is independent of the contract and
/// is ALWAYS authoritative); then the contract's correctness check (answer-match / refusal-expected
/// / schema) where one exists; then the precomputed execution check where evidence is carried. The
/// execution axis is independent of the contract — a candidate's code can be run even when its
/// answer is judge-only — and it is INERT when no evidence is carried. `all_passed` is the AND of
/// every check's `passed`. The returned [`Verdict`] is `Reject` on any HARD fail (the reasoning gate,
/// a proven execution failure, or an answer non-match ONLY when `rule_only_authoritative`), else
/// `Uncertain` when an undecidable execution report blocked admission or an answer axis was
/// undecided / routed to judge rescue (`rescue_negatives`, JUDGE-DESIGN §1.1), else `Accept`.
#[must_use]
pub fn run_verifier<S: SandboxOracle + ?Sized>(
    input: &VerifierInput<'_>,
    sandbox: &S,
) -> VerifierGrade {
    let mut checks = Vec::new();

    // 1. The Verify hard gate (INVARIANT b) — independent of the contract, always first.
    let reasoning_check = reasoning_present_check(input);
    let reasoning_passed = reasoning_check.passed;
    checks.push(reasoning_check);

    // 2. The contract's correctness check, where a contract exists.
    let mut uncertain = false;
    if let Some(contract) = input.contract {
        match contract.kind {
            VerificationKind::RefusalExpected => {
                let (check, is_uncertain) = refusal_check(input);
                if is_uncertain {
                    uncertain = true;
                }
                checks.push(check);
            }
            VerificationKind::NumericMatch
            | VerificationKind::SetMatch
            | VerificationKind::SqlResultMatch
            | VerificationKind::SchemaShape => {
                let (expected, _used_sandbox) = resolve_expected(&contract.oracle, sandbox);
                let answer = last_assistant(input.messages)
                    .map(content_text)
                    .unwrap_or("");
                let cmp = compare_answer(contract.kind, answer, expected.as_deref());
                let (check, _hard_fail, is_uncertain) =
                    answer_match_check(cmp, input.rule_only_authoritative);
                if is_uncertain {
                    uncertain = true;
                }
                checks.push(check);
            }
            // No deterministic oracle: judge-only admission. No correctness check added.
            VerificationKind::None => {}
        }
    }

    // 3. The PRECOMPUTED execution axis, where the record carries an external evaluator's report.
    // Independent of the contract: a candidate's code can be run even when its answer is
    // judge-only. A proven failure hard-fails (the same authoritative gate as the reasoning check);
    // an undecidable report blocks admission instead of routing to judge rescue.
    let mut admission_blocked = false;
    if let Some((check, _verdict, blocked)) = execution_evidence_check(input) {
        admission_blocked = blocked;
        checks.push(check);
    }

    let all_passed = checks.iter().all(|c| c.passed);
    let verdict = if !all_passed {
        Verdict::Reject
    } else if admission_blocked {
        // Nothing proved the work. Deliberately NOT judge rescue: a panel score cannot stand in for
        // ground truth the deterministic rail could not obtain, so the record is held for review.
        Verdict::Uncertain
    } else if uncertain {
        // Reasoning gate held, but the answer axis had no oracle — defer to the judge rescue path.
        Verdict::Uncertain
    } else {
        Verdict::Accept
    };

    // Defensive: the reasoning gate is the one that must hard-reject; assert the bookkeeping holds.
    debug_assert!(reasoning_passed || verdict == Verdict::Reject);

    VerifierGrade {
        verdict,
        verification: Verification {
            checks,
            all_passed,
            needs_review: admission_blocked.then(|| {
                format!(
                    "{EXECUTION_EVIDENCE_CHECK}: the carried execution report did not decide the \
                     candidate; admission is blocked until it does"
                )
            }),
        },
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
            rule_only_authoritative: false,
            execution_evidence: None,
            evidence_key: EvidenceBinding::default(),
        }
    }

    /// Like [`input`] but with the answer-correctness comparator marked rule-only authoritative
    /// (a non-match HARD-rejects). Used to exercise the opt-in hard-reject path.
    fn input_rule_only<'a>(
        messages: &'a [Message],
        tokens: u32,
        cot: bool,
        contract: Option<&'a VerificationContract>,
    ) -> VerifierInput<'a> {
        VerifierInput {
            rule_only_authoritative: true,
            ..input(messages, tokens, cot, contract)
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
    fn numeric_mismatch_hard_rejects_only_when_rule_only_authoritative() {
        // The opt-in: an area that trusts the rule comparator DOES hard-reject a clear non-match.
        let msgs = vec![assistant("41", Some("work"), true)];
        let ct = contract(
            VerificationKind::NumericMatch,
            Oracle::Literal {
                expected: "42".into(),
            },
        );
        let g = run_verifier(
            &input_rule_only(&msgs, 50, true, Some(&ct)),
            &NullSandboxOracle,
        );
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
        // classic false negative). It hard-rejects only under rule_only_authoritative.
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
