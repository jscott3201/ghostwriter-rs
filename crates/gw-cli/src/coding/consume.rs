//! Native verifier adapter. Public positive consumption requires the opaque observed runtime type.
use super::artifact::{CapturedCodingInput, CodingCaseObservation, outcome};
use gw_judge::{NullSandboxOracle, VerifierInput, run_verifier};
use gw_schema::{
    Content, EvidenceBinding, ExecutionEvidence, ExecutionOutcome, Message, Role, TestCase,
    VerificationInterpretation,
};

// Internal pure consistency derivation is also used to reject contradictory saved declarations.
// It does not authenticate them and is not a public deserialized-evidence consumer.
pub(super) fn interpret(
    input: &CapturedCodingInput,
    run_id: &str,
    cases: &[CodingCaseObservation],
) -> VerificationInterpretation {
    let outcome = outcome(cases);
    let binding = EvidenceBinding {
        task: input.provenance.identity.digest.clone(),
        attempt: run_id.into(),
        patch_hash: input.code_id.clone(),
    };
    let evidence = ExecutionEvidence {
        outcome,
        required_tests: input.suite.case_ids.clone(),
        cases: cases
            .iter()
            .map(|case| TestCase {
                node: case.case_id.clone(),
                status: case.status,
            })
            .collect(),
        exit_code: match outcome {
            ExecutionOutcome::Passed => Some(0),
            ExecutionOutcome::Failed => Some(1),
            ExecutionOutcome::Unknown => None,
        },
        errors: vec![],
        source_ref: None,
        binding: binding.clone(),
    };
    let messages = [
        input.task.prompt(),
        Message {
            role: Role::Assistant,
            content: Content::Text(input.code.clone()),
            reasoning: None,
            reasoning_details: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
    ];
    let contract = input.task.contract();
    let grade = run_verifier(
        &VerifierInput {
            messages: &messages,
            reasoning_tokens: 0,
            cot_required: false,
            contract: Some(&contract),
            execution_evidence: Some(&evidence),
            evidence_key: binding,
        },
        &NullSandboxOracle,
    )
    .expect("validated coding contract");
    grade
        .verification
        .interpretation
        .expect("native verifier always emits the current interpretation")
}
