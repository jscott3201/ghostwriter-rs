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
    interpret_parts(
        &input.provenance.identity.digest,
        &input.task.contract(),
        input.task.prompt(),
        &input.code,
        &input.code_id,
        run_id,
        cases,
    )
}
pub(super) fn interpret_public(
    member: &gw_schema::CodingPopulationMember,
    code: &str,
    run_id: &str,
    cases: &[CodingCaseObservation],
) -> VerificationInterpretation {
    let prompt = Message {
        role: Role::User,
        content: Content::Text(member.prompt.clone()),
        reasoning: None,
        reasoning_details: None,
        tool_calls: None,
        tool_call_id: None,
        name: None,
    };
    interpret_parts(
        &member.provenance.identity.digest,
        &member.contract(),
        prompt,
        code,
        &gw_schema::coding_digest("ghostwriter.coding-module.v1", code.as_bytes()),
        run_id,
        cases,
    )
}
fn interpret_parts(
    task_id: &str,
    contract: &gw_schema::VerificationContract,
    prompt: Message,
    code: &str,
    code_id: &str,
    run_id: &str,
    cases: &[CodingCaseObservation],
) -> VerificationInterpretation {
    let outcome = outcome(cases);
    let binding = EvidenceBinding {
        task: task_id.into(),
        attempt: run_id.into(),
        patch_hash: code_id.into(),
    };
    let evidence = ExecutionEvidence {
        outcome,
        required_tests: contract.required_tests.clone(),
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
        prompt,
        Message {
            role: Role::Assistant,
            content: Content::Text(code.into()),
            reasoning: None,
            reasoning_details: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
    ];
    let grade = run_verifier(
        &VerifierInput {
            messages: &messages,
            reasoning_tokens: 0,
            cot_required: false,
            contract: Some(contract),
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
