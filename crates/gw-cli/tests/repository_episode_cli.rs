//! Real stdin/stdout controls, without credentials, config, a database or external execution.
use gw_judge::capture_repository_episode;
use gw_schema::REPOSITORY_EPISODE_MAX_BYTES;
use serde_json::{Value, json};
use std::io::Write;
use std::process::{Command, Output, Stdio};
const FIXTURE: &[u8] = include_bytes!("fixtures/repository-episode-request.json");
fn command(verb: &str, bytes: Vec<u8>) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_gw"))
        .args(["artifact", verb, "--stdin"])
        .env_clear()
        .env("RUST_LOG", "trace")
        .env("GW_CONFIG", "/does/not/exist/secret-config")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // A reader stops at its byte limit. A writer may then receive BrokenPipe; no restart is needed.
    let mut stdin = child.stdin.take().unwrap();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&bytes);
    });
    let output = child.wait_with_output().unwrap();
    writer.join().unwrap();
    output
}
fn wire(value: &impl serde::Serialize) -> Vec<u8> {
    serde_json::to_vec(value).unwrap()
}
#[test]
fn repository_cli_captures_and_verifies_one_complete_artifact_without_credentials() {
    let output = command("capture-repository", FIXTURE.to_vec());
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    assert_eq!(output.stdout.last(), Some(&b'\n'));
    let artifact: Value = serde_json::from_slice(&output.stdout).unwrap();
    let verified = command("verify-repository", output.stdout);
    assert!(
        verified.status.success(),
        "{}",
        String::from_utf8_lossy(&verified.stderr)
    );
    assert!(verified.stderr.is_empty());
    let receipt: Value = serde_json::from_slice(&verified.stdout).unwrap();
    assert_eq!(receipt["identities"], artifact["identities"]);
    assert_eq!(receipt["assessment"]["observed_execution"], "unknown");
    assert_eq!(receipt["assessment"]["training_eligible"], false);
    assert!(receipt.get("request").is_none());
}
#[test]
fn repository_cli_rejects_invalid_input_without_partial_stdout_or_sensitive_diagnostics() {
    let mut unknown: Value = serde_json::from_slice(FIXTURE).unwrap();
    unknown["private_test_contents"] = json!("DO_NOT_ECHO_PRIVATE");
    let mut invalid_number: Value = serde_json::from_slice(FIXTURE).unwrap();
    invalid_number["generation"]["usage"]["input_tokens"] = json!("DO_NOT_ECHO_PRIVATE");
    let mut unknown_enum: Value = serde_json::from_slice(FIXTURE).unwrap();
    unknown_enum["task"]["upstream_base"]["algorithm"] = json!("DO_NOT_ECHO_PRIVATE");
    let mut unknown_name: Value = serde_json::from_slice(FIXTURE).unwrap();
    unknown_name["candidate"]["DO_NOT_ECHO_PRIVATE"] = json!(true);
    let mut invalid_path: Value = serde_json::from_slice(FIXTURE).unwrap();
    invalid_path["candidate"]["delta"]["changes"][0]["path"] = json!("../DO_NOT_ECHO_PRIVATE");
    let duplicate =
        String::from_utf8(FIXTURE.to_vec())
            .unwrap()
            .replacen("{", "{\"version\":1,", 1);
    for bytes in [
        b"DO_NOT_ECHO_PRIVATE".to_vec(),
        wire(&unknown),
        wire(&invalid_number),
        wire(&unknown_enum),
        wire(&unknown_name),
        wire(&invalid_path),
        duplicate.into_bytes(),
        vec![b' '; REPOSITORY_EPISODE_MAX_BYTES + 1],
    ] {
        let output = command("capture-repository", bytes);
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
        let diagnostic = String::from_utf8_lossy(&output.stderr);
        assert!(!diagnostic.contains("DO_NOT_ECHO_PRIVATE"));
        assert!(!diagnostic.contains("private_test_contents"));
        assert!(!diagnostic.contains("API_KEY"));
    }
}
#[test]
fn repository_cli_checks_full_output_bound_before_stdout() {
    let mut value: Value = serde_json::from_slice(FIXTURE).unwrap();
    value["candidate"]["messages"][4]["content"] = json!("");
    let base = wire(&value).len();
    value["candidate"]["messages"][4]["content"] =
        json!("x".repeat(REPOSITORY_EPISODE_MAX_BYTES - base));
    let input = wire(&value);
    assert_eq!(input.len(), REPOSITORY_EPISODE_MAX_BYTES);
    let output = command("capture-repository", input);
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("output exceeds"));
}
#[test]
fn repository_cli_verification_rejects_hash_authority_and_duplicate_tampering() {
    let mut artifact = serde_json::to_value(capture_repository_episode(FIXTURE).unwrap()).unwrap();
    artifact["assessment"]["training_eligible"] = json!(true);
    let original = serde_json::to_string(&capture_repository_episode(FIXTURE).unwrap()).unwrap();
    let duplicate = original.replacen("{", "{\"version\":1,", 1);
    for bytes in [wire(&artifact), duplicate.into_bytes()] {
        let output = command("verify-repository", bytes);
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
    }
}
#[test]
fn repository_cli_requires_explicit_stdin_and_rejects_paths() {
    for verb in ["capture-repository", "verify-repository"] {
        for args in [
            vec!["artifact", verb],
            vec!["artifact", verb, "--stdin", "file.json"],
        ] {
            let output = Command::new(env!("CARGO_BIN_EXE_gw"))
                .args(args)
                .output()
                .unwrap();
            assert_eq!(output.status.code(), Some(2));
            assert!(output.stdout.is_empty());
        }
    }
}
