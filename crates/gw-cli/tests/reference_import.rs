//! Complete owned synthetic population through real capture, registration, runtime and publication.
#[path = "../../gw-schema/tests/reference_support/mod.rs"]
mod support;
use gw_schema::*;
use std::{
    path::{Path, PathBuf},
    process::Command,
};

struct Directory(PathBuf);
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn source(directory: &Path) {
    let mut capture = support::capture();
    let catalogue: ReferenceCatalogue = serde_json::from_str(&capture.catalogue).unwrap();
    let mut documents: Vec<CodingTaskDocument> = capture
        .task_documents
        .iter()
        .map(|s| CodingTaskDocument::from_json(s.as_bytes()).unwrap())
        .collect();
    for (index, member) in catalogue.members.iter().enumerate() {
        let task = documents[member.task_document]
            .tasks
            .iter_mut()
            .find(|t| t.task_id == member.task_id)
            .unwrap();
        task.description =
            format!("Owned synthetic reference control {index}: return the integer zero.");
        task.function.entry_point = "probe".into();
        task.function.parameters.clear();
        task.visible_examples.clear();
        let case = CodingCase {
            label: "private-zero".into(),
            arguments: vec![],
            expected: CodingValue::Integer(0),
        };
        task.train_cases.clear();
        task.protected_cases.clear();
        if index < 64 {
            task.train_cases.push(case);
        } else {
            task.protected_cases.push(case);
        }
        capture.modules[index] =
            format!("# Owned synthetic member {index}\ndef probe():\n    return 0\n");
        let mut review: ReferenceReview = serde_json::from_str(&capture.reviews[index]).unwrap();
        review.task_digest = reference_task_digest(task);
        review.reference_code_id = coding_digest(
            "ghostwriter.coding-module.v1",
            capture.modules[index].as_bytes(),
        );
        capture.reviews[index] = serde_json::to_string(&review).unwrap();
        std::fs::write(directory.join(&member.module_path), &capture.modules[index]).unwrap();
        std::fs::write(directory.join(&member.review_path), &capture.reviews[index]).unwrap();
    }
    capture.task_documents = documents
        .iter()
        .map(|d| serde_json::to_string(d).unwrap())
        .collect();
    capture.validate().unwrap();
    for (name, text) in catalogue.task_documents.iter().zip(&capture.task_documents) {
        std::fs::write(directory.join(name), text).unwrap();
    }
    std::fs::write(directory.join("catalogue.json"), &capture.catalogue).unwrap();
}
fn invoke(dir: &Path, command: &str, more: &[&str]) -> serde_json::Value {
    let mut process = Command::new(env!("CARGO_BIN_EXE_gw"));
    process
        .args(["reference", command, "--db"])
        .arg(dir.join("private.sqlite"));
    if command != "export" {
        process.arg("--catalogue").arg(dir.join("catalogue.json"));
    }
    let result = process.args(more).output().unwrap();
    assert!(
        result.status.success(),
        "{}: {}",
        command,
        String::from_utf8_lossy(&result.stderr)
    );
    serde_json::from_slice(&result.stdout).unwrap()
}
#[test]
#[ignore = "requires cached qualified local Docker; complete 112-member owned population; no pulls"]
fn reference_full_population_cached_runtime_roundtrip() {
    let dir = Directory(
        std::env::temp_dir().join(format!("gw-reference-population-{}", std::process::id())),
    );
    std::fs::create_dir(&dir.0).unwrap();
    source(&dir.0);
    let registration = invoke(&dir.0, "register", &[]);
    assert_eq!(registration["fresh_execution"], false);
    let imported = invoke(&dir.0, "import", &[]);
    assert_eq!(imported["status"], "committed");
    assert_eq!(imported["fresh_execution"], true);
    assert_eq!(imported["training_records"], 64);
    assert_eq!(imported["private_held_out_members"], 48);
    let batch = imported["batch_id"].as_str().unwrap();
    let output = dir.0.join("train.parquet");
    let artifact = invoke(
        &dir.0,
        "export",
        &["--batch-id", batch, "--out", output.to_str().unwrap()],
    );
    assert_eq!(
        artifact["manifest"]["column_schema_version"],
        "record_origins"
    );
    assert_eq!(artifact["manifest"]["n_admitted"], 64);
    let reused = invoke(&dir.0, "import", &[]);
    assert_eq!(reused["status"], "already_committed");
    assert_eq!(reused["fresh_execution"], false);
    assert!(
        reused["records"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["state"] == "exported")
    );
    if let Some(destination) = std::env::var_os("GW_REFERENCE_TEST_FIXTURE_OUT") {
        std::fs::copy(&output, destination).unwrap();
    }
    println!(
        "Complete synthetic runtime evidence: 112 fresh native passes, 64 Train rows, 48 private heldouts, v4 export and historical Exported-state reuse."
    );
}
