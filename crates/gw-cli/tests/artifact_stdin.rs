//! The external adapter must verify the same snapshot it consumes, without a path reopen.
use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn stdin_verification_is_provider_free_and_never_accepts_malformed_bytes() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_gw"))
        .args(["artifact", "verify", "--stdin"])
        .env_remove("MODEL_API_KEY")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"not parquet")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(!String::from_utf8_lossy(&output.stderr).contains("API_KEY"));
}

#[test]
fn stdin_report_binds_the_exact_bytes_and_ignores_trace_logging_configuration() {
    let bytes = include_bytes!("../../../adapters/trl/tests/fixtures/v3-text.parquet");
    let expected = gw_storage::verify_artifact_snapshot(bytes.to_vec()).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_gw"))
        .args(["artifact", "verify", "--stdin"])
        .env_remove("MODEL_API_KEY")
        .env("RUST_LOG", "trace")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(bytes).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    let actual: gw_storage::ArtifactSnapshotReport =
        serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(actual, expected);
    assert!(output.stderr.is_empty());
}

#[test]
fn stdin_mode_is_explicit_and_paths_are_not_accepted() {
    for args in [
        vec!["artifact", "verify"],
        vec!["artifact", "verify", "--stdin", "file.parquet"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_gw"))
            .args(args)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(output.stdout.is_empty());
    }
}

#[test]
fn prepared_stdin_verifies_everything_before_emitting_a_separate_receipt() {
    let accepted = include_bytes!("../../../adapters/trl/tests/fixtures/prepared-all.gwsft");
    for valid in [true, false] {
        let mut data = accepted.to_vec();
        if !valid {
            data.push(b'!');
        }
        let mut child = Command::new(env!("CARGO_BIN_EXE_gw"))
            .args(["artifact", "verify-prepared", "--stdin"])
            .env_remove("MODEL_API_KEY")
            .env("RUST_LOG", "trace")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.take().unwrap().write_all(&data).unwrap();
        let output = child.wait_with_output().unwrap();
        assert_eq!(output.status.success(), valid);
        if valid {
            let report: gw_storage::PreparedSftVerificationReport =
                serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(
                report,
                *gw_storage::verify_prepared_sft_snapshot(data)
                    .unwrap()
                    .report()
            );
            assert_eq!(report.example_count, 4);
            assert_eq!(report.tokenizer_replay, "not_run");
            assert!(output.stderr.is_empty());
        } else {
            assert!(output.stdout.is_empty());
        }
    }
}

#[test]
fn prepared_stdin_requires_explicit_mode_and_rejects_paths() {
    for args in [
        vec!["artifact", "verify-prepared"],
        vec!["artifact", "verify-prepared", "--stdin", "input.gwsft"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_gw"))
            .args(args)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(output.stdout.is_empty());
    }
}
