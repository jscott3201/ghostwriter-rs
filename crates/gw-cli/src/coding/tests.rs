//! Declared integrity and real owned-runtime acceptance, with no model/provider activity.
use super::*;
use gw_schema::{CodingTaskDocument, ExecutionOutcome, VerificationOutcome};

fn input(task: &str, file: &str) -> CapturedCodingInput {
    let document = CodingTaskDocument::from_json(include_bytes!(
        "../../../../examples/reviewed-coding-tasks.json"
    ))
    .unwrap();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/coding")
        .join(file);
    CapturedCodingInput::new(&document, task, std::fs::read_to_string(path).unwrap()).unwrap()
}
fn declared(input: CapturedCodingInput) -> CodingArtifact {
    let cases = input
        .suite
        .case_ids
        .iter()
        .map(|case_id| CodingCaseObservation {
            case_id: case_id.clone(),
            status: TestStatus::Passed,
            reason: CodingCaseReason::Matched,
            result_id: Some("0".repeat(64)),
            exit_code: Some(0),
            settled: true,
            container_id: Some("1".repeat(64)),
            elapsed_ms: 1,
        })
        .collect::<Vec<_>>();
    let run_id = "0".repeat(32);
    let native_verification = consume::interpret(&input, &run_id, &cases);
    let mut artifact = CodingArtifact {
        version: 1,
        artifact_id: String::new(),
        input,
        runtime: CodingRuntimeIdentity::expected(),
        report: CodingReport {
            run_id,
            outcome: ExecutionOutcome::Passed,
            cases,
            native_verification,
        },
        replayed_declaration_id: None,
    };
    artifact.seal();
    artifact
}

#[test]
fn saved_declarations_reject_stale_bindings_coverage_and_unknown_fields() {
    let original = declared(input("merge-closed", "merge_closed.correct.py"));
    // Internal consistency is deliberately not an authentication claim.
    CodingArtifact::from_json(&serde_json::to_vec(&original).unwrap()).unwrap();
    for scenario in 0..8 {
        let mut changed = original.clone();
        match scenario {
            0 => changed.input.code.push_str("\n# replaced"),
            1 => changed.input.task.train_cases[0].expected = CodingValue::Boolean(true),
            2 => changed.runtime.image.push_str("-stale"),
            3 => {
                changed.report.cases.pop();
            }
            4 => changed.report.cases.push(changed.report.cases[0].clone()),
            5 => {
                changed.report.cases[0].status = TestStatus::Skipped;
                changed.report.cases[0].reason = CodingCaseReason::Cancelled;
            }
            6 => changed.runtime.wrapper_id = "3".repeat(64),
            _ => changed.report.cases[0].settled = false,
        }
        changed.seal(); // Even a matching recomputed checksum cannot repair a false binding.
        assert!(
            CodingArtifact::from_json(&serde_json::to_vec(&changed).unwrap()).is_err(),
            "scenario {scenario}"
        );
    }
    let mut unknown = serde_json::to_value(&original).unwrap();
    unknown["report"]["native_verification"]["execution"]["trusted_producer"] = true.into();
    assert!(CodingArtifact::from_json(&serde_json::to_vec(&unknown).unwrap()).is_err());
}

#[tokio::test]
#[ignore = "requires cached qualified local Docker; no pulls"]
async fn correct_wrong_and_syntax_fixtures_reach_native_execution() {
    for (task, file, want) in [
        (
            "merge-closed",
            "merge_closed.correct.py",
            ExecutionOutcome::Passed,
        ),
        (
            "merge-closed",
            "merge_closed.wrong.py",
            ExecutionOutcome::Failed,
        ),
        ("runs", "runs.correct.py", ExecutionOutcome::Passed),
        ("runs", "runs.wrong.py", ExecutionOutcome::Failed),
        (
            "common-prefix",
            "common_prefix.correct.py",
            ExecutionOutcome::Passed,
        ),
        (
            "common-prefix",
            "common_prefix.wrong.py",
            ExecutionOutcome::Failed,
        ),
        ("merge-closed", "syntax_error.py", ExecutionOutcome::Failed),
    ] {
        let observed = observe_coding(input(task, file), CancellationToken::new())
            .await
            .unwrap()
            .consume();
        assert_eq!(
            observed.report.outcome, want,
            "{file}: {:?}",
            observed.report.cases
        );
        assert_eq!(
            observed
                .report
                .native_verification
                .execution
                .observation
                .as_ref()
                .unwrap()
                .outcome,
            if want == ExecutionOutcome::Passed {
                VerificationOutcome::Pass
            } else {
                VerificationOutcome::Fail
            }
        );
        assert!(observed.report.cases.iter().all(|case| case.settled));
        CodingArtifact::from_json(&serde_json::to_vec(&observed).unwrap()).unwrap();
    }
}

#[tokio::test]
#[ignore = "requires cached qualified local Docker; no pulls"]
async fn fresh_replay_matches_real_success_and_rejects_forged_success() {
    let initial = observe_coding(
        input("merge-closed", "merge_closed.correct.py"),
        CancellationToken::new(),
    )
    .await
    .unwrap()
    .consume();
    let saved = CodingArtifact::from_json(&serde_json::to_vec(&initial).unwrap()).unwrap();
    let fresh = replay_coding(saved, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(
        fresh.replayed_declaration_id.as_ref(),
        Some(&initial.artifact_id)
    );
    assert_ne!(fresh.report.run_id, initial.report.run_id);
    assert_ne!(
        fresh.report.cases[0].container_id,
        initial.report.cases[0].container_id
    );
    assert!(fresh.stable_matches(&initial));
    let mut forged = declared(input("merge-closed", "merge_closed.wrong.py"));
    // A malicious caller can copy plausible output identities too; no claim bypasses execution.
    for (fake, real) in forged.report.cases.iter_mut().zip(&initial.report.cases) {
        fake.result_id.clone_from(&real.result_id);
    }
    forged.seal();
    let saved = CodingArtifact::from_json(&serde_json::to_vec(&forged).unwrap()).unwrap();
    let error = replay_coding(saved, CancellationToken::new())
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("fresh coding observation differs")
    );
}

#[tokio::test]
#[ignore = "requires cached qualified local Docker; no pulls"]
async fn replacing_input_paths_cannot_change_captured_code_or_cases() {
    let directory = std::env::temp_dir().join(format!("gw-coding-capture-{}", nonce().unwrap()));
    std::fs::create_dir(&directory).unwrap();
    let path = directory.join("candidate.py");
    let task_path = directory.join("tasks.json");
    let original = input("merge-closed", "merge_closed.correct.py");
    std::fs::write(&path, &original.code).unwrap();
    std::fs::write(
        &task_path,
        include_bytes!("../../../../examples/reviewed-coding-tasks.json"),
    )
    .unwrap();
    let document = CodingTaskDocument::from_json(&std::fs::read(&task_path).unwrap()).unwrap();
    let captured = CapturedCodingInput::new(
        &document,
        "merge-closed",
        std::fs::read_to_string(&path).unwrap(),
    )
    .unwrap();
    std::fs::write(&path, "this replacement is not Python").unwrap();
    std::fs::write(&task_path, "{}").unwrap();
    let actual = observe_coding(captured, CancellationToken::new())
        .await
        .unwrap()
        .consume();
    assert_eq!(actual.report.outcome, ExecutionOutcome::Passed);
    assert_eq!(actual.input, original);
    std::fs::remove_dir_all(directory).unwrap();
}
