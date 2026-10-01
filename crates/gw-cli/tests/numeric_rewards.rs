//! Portable numeric tasks and rewards cross only provider-free captured-byte command boundaries.
mod common;
use serde_json::{Value, json};
use std::io::Write;
use std::process::{Command, Output, Stdio};

const TASKS: &str = include_str!("../../../examples/reviewed-numeric-tasks.json");

fn command(args: &[&str], input: Option<&[u8]>) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_gw"))
        .env_clear()
        .env("RUST_LOG", "trace")
        .env("GW_MODEL_API_KEY_ENV", "invalid-secret-reference")
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    if let Some(input) = input {
        child.stdin.as_mut().unwrap().write_all(input).unwrap();
    }
    drop(child.stdin.take());
    child.wait_with_output().unwrap()
}

#[test]
fn export_projects_declared_train_tasks_once_and_verifies_captured_bytes() {
    let path = common::unique_temp_path("reward-tasks.json");
    let mut document: Value = serde_json::from_str(TASKS).unwrap();
    document["tasks"][1]["split"]["role"] = json!("test");
    std::fs::write(&path, document.to_string()).unwrap();
    let output = command(
        &["reward", "export", "--tasks", path.to_str().unwrap()],
        None,
    );
    std::fs::remove_file(path).unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let artifact: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(artifact["tasks"].as_array().unwrap().len(), 1);
    assert_eq!(artifact["tasks"][0]["task"]["task_id"], "addition-001");
    assert_eq!(artifact["tasks"][0]["task"]["split"]["role"], "train");
    assert_eq!(
        artifact["tasks"][0]["task"]["verification"]["oracle"]["expected"],
        "5"
    );
    let verified = command(&["reward", "verify", "--stdin"], Some(&output.stdout));
    assert!(
        verified.status.success(),
        "{}",
        String::from_utf8_lossy(&verified.stderr)
    );
    let report: Value = serde_json::from_slice(&verified.stdout).unwrap();
    assert_eq!(report["artifact_id"], artifact["artifact_id"]);
    assert_eq!(report["task_count"], 1);
    assert_eq!(report["snapshot"]["byte_length"], output.stdout.len());
}

#[test]
fn duplicate_documents_or_invalid_held_out_tasks_publish_no_artifact() {
    let path = common::unique_temp_path("reward-duplicate.json");
    std::fs::write(&path, TASKS).unwrap();
    let path_arg = path.to_str().unwrap();
    let duplicate = command(
        &["reward", "export", "--tasks", path_arg, "--tasks", path_arg],
        None,
    );
    assert!(!duplicate.status.success());
    assert!(duplicate.stdout.is_empty());
    let mut document: Value = serde_json::from_str(TASKS).unwrap();
    document["tasks"][1]["split"]["role"] = json!("test");
    document["tasks"][1]["verification"]["oracle"]["expected"] = json!("NaN");
    std::fs::write(&path, document.to_string()).unwrap();
    let invalid = command(&["reward", "export", "--tasks", path_arg], None);
    std::fs::remove_file(path).unwrap();
    assert!(!invalid.status.success());
    assert!(invalid.stdout.is_empty());
}

fn batch() -> Value {
    use gw_schema::{
        NumericRewardArtifact, NumericTaskDocument, RewardCompletion, RewardCompletionPolicy,
    };
    let artifact =
        NumericRewardArtifact::from_documents(vec![NumericTaskDocument::from_json(TASKS).unwrap()])
            .unwrap();
    let policy: RewardCompletionPolicy = serde_json::from_value(json!({
        "version": 1, "tokenizer_id": "a".repeat(64), "decode": "plain_assistant_skip_special_no_cleanup_v1",
        "vocab_size": 100, "eos_token_id": 99, "control_tokens": [{"id": 99, "literal": "<eos>"}],
    })).unwrap();
    let items: Vec<_> = ["FINAL: 5", "FINAL: 4", "undecidable"].iter().enumerate().map(|(position, text)| {
        let completion = RewardCompletion { token_ids: vec![1, 99], text: (*text).into() };
        json!({"binding": {
            "artifact_id": artifact.artifact_id, "task_id": artifact.tasks[0].task.task_id,
            "semantic_task_digest": artifact.tasks[0].task_identity.digest,
            "reward_contract_id": artifact.reward_contract_id,
            "attempt": {"callback_run_id": "b".repeat(32), "batch_sequence": 0, "position": position},
            "completion_digest": policy.completion_digest(&completion).unwrap(),
        }, "completion": completion})
    }).collect();
    json!({"version": 1, "artifact": artifact, "completion_policy": policy,
        "mask_truncated_completions": true, "items": items})
}

#[test]
fn stdin_evaluation_retains_unknown_and_rejects_late_errors_without_partial_stdout() {
    let request = batch();
    let raw = request.to_string();
    let output = command(&["reward", "evaluate", "--stdin"], Some(raw.as_bytes()));
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["results"].as_array().unwrap().len(), 3);
    assert_eq!(report["results"][0]["outcome"], "pass");
    assert_eq!(report["results"][0]["reward"], 1.0);
    assert_eq!(report["results"][1]["outcome"], "fail");
    assert_eq!(report["results"][1]["reward"], 0.0);
    assert_eq!(report["results"][2]["outcome"], "unknown");
    assert!(report["results"][2]["reward"].is_null());
    assert_eq!(
        report["request"],
        serde_json::to_value(gw_schema::RewardSnapshotIdentity::for_bytes(raw.as_bytes())).unwrap()
    );
    let mut changed = request.clone();
    changed["items"][2]["binding"]["completion_digest"] = json!("0".repeat(64));
    let unknown_field = raw.replacen("\"version\":1", "\"version\":1,\"teacher\":true", 1);
    let duplicate = raw.replacen("\"position\":2", "\"position\":2,\"position\":2", 1);
    for invalid in [
        changed.to_string(),
        raw[..raw.len() - 2].into(),
        unknown_field,
        duplicate,
    ] {
        let output = command(&["reward", "evaluate", "--stdin"], Some(invalid.as_bytes()));
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
    }
}
