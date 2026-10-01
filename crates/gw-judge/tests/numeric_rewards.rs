//! Literal conformance oracles and complete fresh-completion bindings, without model execution.
use gw_judge::{
    evaluate_numeric_answer, evaluate_numeric_reward_batch, evaluate_numeric_reward_json,
};
use gw_schema::{
    NumericComparison, NumericExtraction, NumericRewardArtifact, NumericRewardBatch,
    NumericRewardBinding, NumericRewardInput, NumericTaskDocument, NumericTolerance, RewardAttempt,
    RewardCompletion, RewardCompletionPolicy, RewardControlToken, RewardDecodePolicy,
    RewardSnapshotIdentity, RewardTermination, VerificationOutcome,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Vectors {
    cases: Vec<Vector>,
}
#[derive(Deserialize)]
struct Vector {
    name: String,
    expected: Option<String>,
    content: String,
    outcome: VerificationOutcome,
    absolute: f64,
    relative: f64,
    extraction: NumericExtraction,
}

#[test]
fn independent_literal_numeric_vectors_cover_grammar_rounding_and_extraction() {
    let vectors: Vectors =
        serde_json::from_str(include_str!("fixtures/numeric-reward-vectors.json")).unwrap();
    assert_eq!(vectors.cases.len(), 35);
    for vector in vectors.cases {
        let settings = NumericComparison {
            extraction: vector.extraction,
            tolerance: NumericTolerance {
                absolute: vector.absolute,
                relative: vector.relative,
            },
        };
        assert_eq!(
            evaluate_numeric_answer(&vector.content, vector.expected.as_deref(), &settings),
            vector.outcome,
            "{}",
            vector.name
        );
    }
}

fn batch() -> NumericRewardBatch {
    let artifact = NumericRewardArtifact::from_documents(vec![
        NumericTaskDocument::from_json(include_str!(
            "../../../examples/reviewed-numeric-tasks.json"
        ))
        .unwrap(),
    ])
    .unwrap();
    let completion_policy = RewardCompletionPolicy {
        version: 1,
        tokenizer_id: "a".repeat(64),
        decode: RewardDecodePolicy::PlainAssistantSkipSpecialNoCleanupV1,
        vocab_size: 100,
        eos_token_id: 99,
        control_tokens: vec![
            RewardControlToken {
                id: 98,
                literal: "<think>".into(),
            },
            RewardControlToken {
                id: 99,
                literal: "<eos>".into(),
            },
        ],
    };
    let items = [
        (0, "FINAL: 5", vec![1, 99]),
        (0, "FINAL: 4", vec![2]),
        (1, "3", vec![3, 99]),
        (1, "ambiguous", vec![4]),
    ]
    .into_iter()
    .enumerate()
    .map(|(position, (task, text, token_ids))| {
        let task = &artifact.tasks[task];
        let completion = RewardCompletion {
            token_ids,
            text: text.into(),
        };
        NumericRewardInput {
            binding: NumericRewardBinding {
                artifact_id: artifact.artifact_id.clone(),
                task_id: task.task.task_id.clone(),
                semantic_task_digest: task.task_identity.digest.clone(),
                reward_contract_id: artifact.reward_contract_id.clone(),
                attempt: RewardAttempt {
                    callback_run_id: "b".repeat(32),
                    batch_sequence: 7,
                    position: position as u64,
                },
                completion_digest: completion_policy.completion_digest(&completion).unwrap(),
            },
            completion,
        }
    })
    .collect();
    NumericRewardBatch {
        version: 1,
        artifact,
        completion_policy,
        mask_truncated_completions: false,
        items,
    }
}

#[test]
fn fresh_batch_preserves_order_repeats_unknown_and_independent_termination() {
    let batch = batch();
    let results = evaluate_numeric_reward_batch(&batch).unwrap();
    assert_eq!(
        results.iter().map(|r| r.reward).collect::<Vec<_>>(),
        vec![Some(1.0), Some(0.0), Some(1.0), None]
    );
    assert_eq!(results[3].outcome, VerificationOutcome::Unknown);
    assert_eq!(results[1].termination, RewardTermination::Unknown);
    assert_eq!(results[0].termination, RewardTermination::ObservedEos);
    for (result, input) in results.iter().zip(&batch.items) {
        assert_eq!(result.binding, input.binding);
    }
    let raw = serde_json::to_vec(&batch).unwrap();
    let report = evaluate_numeric_reward_json(&raw).unwrap();
    assert_eq!(report.request, RewardSnapshotIdentity::for_bytes(&raw));
    assert_eq!(report.results, results);
    assert!(!report.mask_truncated_completions);
}

#[test]
fn any_late_binding_failure_rejects_complete_batch() {
    let base = batch();
    let raw = serde_json::to_value(&base).unwrap();
    for (pointer, replacement) in [
        (
            "/items/3/binding/artifact_id",
            serde_json::json!("0".repeat(64)),
        ),
        (
            "/items/3/binding/task_id",
            serde_json::json!("addition-001"),
        ),
        (
            "/items/3/binding/semantic_task_digest",
            serde_json::json!("0".repeat(64)),
        ),
        (
            "/items/3/binding/reward_contract_id",
            serde_json::json!("0".repeat(64)),
        ),
        ("/items/3/binding/attempt/position", serde_json::json!(0)),
        (
            "/items/3/binding/attempt/batch_sequence",
            serde_json::json!(8),
        ),
        (
            "/items/3/binding/attempt/callback_run_id",
            serde_json::json!("c".repeat(32)),
        ),
        (
            "/items/3/binding/completion_digest",
            serde_json::json!("0".repeat(64)),
        ),
        ("/items/3/completion/text", serde_json::json!("3")),
        ("/items/3/completion/token_ids/0", serde_json::json!(5)),
    ] {
        let mut changed = raw.clone();
        *changed.pointer_mut(pointer).unwrap() = replacement;
        assert!(
            evaluate_numeric_reward_json(changed.to_string().as_bytes()).is_err(),
            "{pointer}"
        );
    }
    let mut reordered = base.clone();
    reordered.items.swap(0, 1);
    assert!(evaluate_numeric_reward_batch(&reordered).is_err());
}

#[test]
fn raw_batch_duplicate_unknown_and_wrong_scalar_types_are_rejected() {
    let raw = serde_json::to_string(&batch()).unwrap();
    for (needle, replacement) in [
        ("\"version\":1", "\"version\":1,\"version\":1"),
        (
            "\"batch_sequence\":7",
            "\"batch_sequence\":7,\"batch_sequence\":7",
        ),
        ("\"position\":0", "\"position\":false"),
        ("\"position\":0", "\"position\":0.0"),
        (
            "\"mask_truncated_completions\":false",
            "\"mask_truncated_completions\":0",
        ),
        ("\"token_ids\":[1,99]", "\"token_ids\":[true,99]"),
        (
            "\"token_ids\":[1,99]",
            "\"token_ids\":[1,99],\"reasoning\":\"FINAL: 5\"",
        ),
    ] {
        assert!(raw.contains(needle));
        assert!(evaluate_numeric_reward_json(raw.replace(needle, replacement).as_bytes()).is_err());
    }
}

#[test]
fn raw_controls_cannot_be_hidden_by_skip_special_decoding() {
    let policy = batch().completion_policy;
    for ids in [vec![98, 1, 99], vec![99, 1], vec![99, 99], vec![100]] {
        assert!(
            policy
                .validate_completion(&RewardCompletion {
                    token_ids: ids,
                    text: "5".into()
                })
                .is_err()
        );
    }
    assert!(
        policy
            .validate_completion(&RewardCompletion {
                token_ids: vec![1],
                text: "<think>5".into()
            })
            .is_err()
    );
}

#[test]
fn batch_replay_preserves_exact_finite_tolerances_and_verification_identity() {
    let mut request = batch();
    let mut tasks: Vec<_> = request
        .artifact
        .tasks
        .iter()
        .map(|entry| entry.task.clone())
        .collect();
    tasks[0].verification.numeric.tolerance.absolute = f64::from_bits(0x3ff8_9d89_d89d_89d9);
    tasks[0].verification.numeric.tolerance.relative = -0.0;
    request.artifact =
        NumericRewardArtifact::from_documents(vec![NumericTaskDocument { version: 1, tasks }])
            .unwrap();
    for item in &mut request.items {
        item.binding
            .artifact_id
            .clone_from(&request.artifact.artifact_id);
        let task = request
            .artifact
            .tasks
            .iter()
            .find(|task| task.task.task_id == item.binding.task_id)
            .unwrap();
        item.binding
            .semantic_task_digest
            .clone_from(&task.task_identity.digest);
    }
    let raw = serde_json::to_vec(&request).unwrap();
    let restored = NumericRewardBatch::from_json(&raw).unwrap();
    assert_eq!(serde_json::to_vec(&restored).unwrap(), raw);
    let tolerance = &restored.artifact.tasks[0]
        .task
        .verification
        .numeric
        .tolerance;
    assert_eq!(tolerance.absolute.to_bits(), 0x3ff8_9d89_d89d_89d9);
    assert_eq!(tolerance.relative.to_bits(), (-0.0_f64).to_bits());
    let report = evaluate_numeric_reward_json(&raw).unwrap();
    assert_eq!(
        report.results.iter().map(|r| r.reward).collect::<Vec<_>>(),
        vec![Some(1.0), Some(1.0), Some(1.0), None]
    );
    assert_eq!(report.request, RewardSnapshotIdentity::for_bytes(&raw));
}
