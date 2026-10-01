//! Actual owned saved functions evaluated through the qualified local container recipe.
use std::path::Path;
use std::process::Command;

#[test]
#[ignore = "requires the cached qualified local Docker image; no pulls or remote daemon"]
fn saved_correct_function_reaches_native_observed_evaluation() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output = std::env::temp_dir().join(format!("gw-coding-red-{}.json", std::process::id()));
    let result = Command::new(env!("CARGO_BIN_EXE_gw"))
        .args(["eval", "coding", "--tasks"])
        .arg(root.join("examples/reviewed-coding-tasks.json"))
        .args(["--task", "merge-closed", "--candidate"])
        .arg(root.join("examples/coding/merge_closed.correct.py"))
        .arg("--output")
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
    assert_eq!(report["report"]["outcome"], "passed");
    assert_eq!(
        report["report"]["native_verification"]["execution"]["observation"]["outcome"],
        "pass"
    );
    std::fs::remove_file(output).unwrap();
}
