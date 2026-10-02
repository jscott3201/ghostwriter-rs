//! Consumed regressions for portable paths, one numeric interpretation, and the complete wire bound.
use gw_judge::{capture_repository_episode, verify_repository_episode};
use gw_schema::{REPOSITORY_EPISODE_MAX_BYTES, strict_repository_json};
use serde_json::{Value, json};
use std::io::Write;
use std::process::{Command, Output, Stdio};
const FIXTURE: &[u8] = include_bytes!("fixtures/repository-episode-request.json");
const MARKER: &str = "REPOSITORY_NUMBER_PLACEHOLDER";
fn request() -> Value {
    serde_json::from_slice(FIXTURE).unwrap()
}
fn wire(value: &impl serde::Serialize) -> Vec<u8> {
    serde_json::to_vec(value).unwrap()
}
fn raw_number(value: &Value, number: &str) -> Vec<u8> {
    let raw = serde_json::to_string(value).unwrap();
    assert_eq!(raw.matches(MARKER).count(), 1);
    raw.replace(&format!("\"{MARKER}\""), number).into_bytes()
}
fn command(verb: &str, bytes: Vec<u8>) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_gw"))
        .args(["artifact", verb, "--stdin"])
        .env_clear()
        .env("RUST_LOG", "trace")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&bytes);
    });
    let output = child.wait_with_output().unwrap();
    writer.join().unwrap();
    output
}
#[test]
fn repository_cli_rejects_superscript_device_names_in_every_component() {
    for prefix in ["COM", "LPT", "com", "lpt"] {
        for digit in ["¹", "²", "³"] {
            for path in [
                format!("{prefix}{digit}"),
                format!("src/{prefix}{digit}.py"),
                format!("{prefix}{digit}/child.txt"),
            ] {
                let mut input = request();
                input["candidate"]["delta"]["changes"][0]["path"] = json!(path);
                let output = command("capture-repository", wire(&input));
                assert_eq!(output.status.code(), Some(1), "reserved path {path}");
                assert!(output.stdout.is_empty());
                assert!(!String::from_utf8_lossy(&output.stderr).contains(&path));
            }
        }
    }
    // Similar ordinary names remain available; no compatibility normalization is performed.
    for path in ["COM10.txt", "LPT0", "COM¹extra", "src/myLPT².py"] {
        let mut input = request();
        input["candidate"]["delta"]["changes"][0]["path"] = json!(path);
        assert!(capture_repository_episode(&wire(&input)).is_ok());
    }
}
#[test]
fn repository_capture_and_verification_share_the_finite_numeric_boundary() {
    let mut input = request();
    input["generation"]["settings"]["boundary"] = json!(MARKER);
    for (raw, expected) in [
        ("1.7976931348623157e308", f64::MAX),
        ("1.7976931348623158e308", f64::MAX),
        ("-1.7976931348623158e308", -f64::MAX),
        ("3e-324", f64::from_bits(1)),
        ("-3e-324", -f64::from_bits(1)),
        ("5e-324", f64::from_bits(1)),
        ("0e9999", 0.0),
        ("-0e9999", -0.0),
        ("0.0", 0.0),
        ("-0", -0.0),
        ("-0.0", -0.0),
    ] {
        let bytes = raw_number(&input, raw);
        let artifact =
            capture_repository_episode(&bytes).unwrap_or_else(|error| panic!("{raw}: {error}"));
        assert_eq!(
            artifact.request.generation.settings["boundary"]
                .as_f64()
                .unwrap()
                .to_bits(),
            expected.to_bits(),
            "{raw}"
        );
        let mut saved = serde_json::to_value(&artifact).unwrap();
        saved["request"]["generation"]["settings"]["boundary"] = json!(MARKER);
        let receipt = verify_repository_episode(&raw_number(&saved, raw)).unwrap();
        assert_eq!(receipt.identities, artifact.identities, "{raw}");
        assert_eq!(
            verify_repository_episode(&wire(&artifact))
                .unwrap()
                .identities,
            artifact.identities
        );
    }
    let mut saved = serde_json::to_value(capture_repository_episode(FIXTURE).unwrap()).unwrap();
    saved["request"]["generation"]["settings"]["boundary"] = json!(MARKER);
    for raw in [
        "1.7976931348623159e308",
        "-1.7976931348623159e308",
        "1e309",
        "2e-324",
        "-2e-324",
        "1e-9999",
    ] {
        assert!(
            capture_repository_episode(&raw_number(&input, raw)).is_err(),
            "{raw}"
        );
        let error = verify_repository_episode(&raw_number(&saved, raw)).unwrap_err();
        assert_eq!(error.to_string(), "unsupported repository number", "{raw}");
    }
}
#[test]
fn repository_cli_accepts_finite_maximum_spelling_in_capture_and_saved_artifact() {
    let mut input = request();
    input["generation"]["settings"]["boundary"] = json!(MARKER);
    let output = command(
        "capture-repository",
        raw_number(&input, "1.7976931348623158e308"),
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    let mut artifact: Value = serde_json::from_slice(&output.stdout).unwrap();
    artifact["request"]["generation"]["settings"]["boundary"] = json!(MARKER);
    let verified = command(
        "verify-repository",
        raw_number(&artifact, "1.7976931348623158e308"),
    );
    assert!(
        verified.status.success(),
        "{}",
        String::from_utf8_lossy(&verified.stderr)
    );
    assert!(verified.stderr.is_empty());
}
#[test]
fn repository_numeric_structure_still_rejects_bad_grammar_and_duplicates() {
    for raw in [
        "00",
        "-00",
        "00.1",
        "1.",
        "1.e2",
        "1e",
        "1e+",
        "1e--2",
        "1..2",
        "--1",
        "-",
        "+1",
        ".1",
        "1true",
        "[1 2]",
        r#"{"x":1.7976931348623158e308,"x":0}"#,
    ] {
        assert!(strict_repository_json(raw.as_bytes()).is_err(), "{raw}");
    }
    let raw = br#"{"text":"1.7976931348623158e308 01 1e999 \"123\" \\ -0","value":-0,"nested":[true,null,{"value":0.125}]}"#;
    let parsed = strict_repository_json(raw).unwrap();
    assert_eq!(
        parsed["text"],
        "1.7976931348623158e308 01 1e999 \"123\" \\ -0"
    );
    assert_eq!(
        parsed["value"].as_f64().unwrap().to_bits(),
        (-0.0f64).to_bits()
    );
    assert_eq!(parsed["nested"][2]["value"], json!(0.125));
}
fn request_for_output_size(size: usize) -> Vec<u8> {
    let mut input = request();
    input["candidate"]["messages"][4]["content"] = json!("");
    let small = capture_repository_episode(&wire(&input)).unwrap();
    let overhead = wire(&small).len() + 1;
    input["candidate"]["messages"][4]["content"] = json!("x".repeat(size - overhead));
    let bytes = wire(&input);
    assert!(bytes.len() <= REPOSITORY_EPISODE_MAX_BYTES);
    bytes
}
#[test]
fn repository_library_rejects_a_complete_artifact_over_the_wire_bound() {
    let input = request_for_output_size(REPOSITORY_EPISODE_MAX_BYTES + 1);
    assert!(
        capture_repository_episode(&input).is_err(),
        "library returned an artifact its verifier cannot consume with the newline allowance"
    );
}
#[test]
fn repository_library_accepts_and_verifies_the_exact_complete_wire_bound() {
    let input = request_for_output_size(REPOSITORY_EPISODE_MAX_BYTES);
    let artifact = capture_repository_episode(&input).unwrap();
    let mut saved = wire(&artifact);
    saved.push(b'\n');
    assert_eq!(saved.len(), REPOSITORY_EPISODE_MAX_BYTES);
    assert_eq!(
        verify_repository_episode(&saved).unwrap().identities,
        artifact.identities
    );
}
