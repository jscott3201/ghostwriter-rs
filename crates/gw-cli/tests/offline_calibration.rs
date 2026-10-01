//! Process contract for supplied offline judge evidence; no provider credentials are needed.

use std::process::Command;

#[test]
fn offline_calibration_command_is_available_without_credentials() {
    let output = Command::new(env!("CARGO_BIN_EXE_gw"))
        .args(["eval", "fit-calibration", "--help"])
        .env_remove("MODEL_API_KEY")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let help = String::from_utf8(output.stdout).unwrap();
    for flag in ["--config", "--records", "--evidence"] {
        assert!(help.contains(flag), "missing {flag}: {help}");
    }
}

#[path = "../../gw-judge/tests/calibration_support/mod.rs"]
mod calibration_support;
mod common;

struct Files {
    records: std::path::PathBuf,
    evidence: std::path::PathBuf,
    config: std::path::PathBuf,
}
impl Files {
    fn new(fixture: &calibration_support::Fixture) -> Self {
        let files = Self {
            records: common::unique_temp_path("calibration-records.json"),
            evidence: common::unique_temp_path("calibration-evidence.json"),
            config: common::unique_temp_path("calibration-panel.toml"),
        };
        std::fs::write(
            &files.records,
            serde_json::to_vec(&fixture.records).unwrap(),
        )
        .unwrap();
        // Deliberately infeasible automatic-admission floor. Offline fitting resolves actual
        // request contracts without performing dispatch/admission feasibility checks.
        std::fs::write(
            &files.config,
            r#"
[area]
training_area = "synthetic-area"
rubric = "synthetic rubric"
[area.thresholds]
min_n_eff = 999.0
[[area.judges]]
slug = "judge-a"
family = "a"
[[area.judges]]
slug = "judge-b"
family = "b"
"#,
        )
        .unwrap();
        files
    }
    fn run(&self, bytes: &[u8]) -> std::process::Output {
        std::fs::write(&self.evidence, bytes).unwrap();
        let mut command = Command::new(env!("CARGO_BIN_EXE_gw"));
        command
            .args(["eval", "fit-calibration", "--config"])
            .arg(&self.config)
            .arg("--records")
            .arg(&self.records)
            .arg("--evidence")
            .arg(&self.evidence)
            .env_remove("MODEL_API_KEY");
        for (key, _) in std::env::vars_os() {
            if key.to_string_lossy().starts_with("GW_") {
                command.env_remove(key);
            }
        }
        command.output().unwrap()
    }
}
impl Drop for Files {
    fn drop(&mut self) {
        for path in [&self.records, &self.evidence, &self.config] {
            let _ = std::fs::remove_file(path);
        }
    }
}

#[test]
fn complete_supplied_fixture_emits_only_unqualified_snapshot_without_credentials() {
    let fixture = calibration_support::Fixture::new();
    let files = Files::new(&fixture);
    let output = files.run(&serde_json::to_vec(&fixture.evidence).unwrap());
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let report: gw_schema::CalibrationReport = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        report.status,
        gw_schema::CalibrationStatus::ComputedUnqualified
    );
    let snapshot = report.snapshot.unwrap();
    assert_eq!(snapshot.fit.losses, vec![0.5, 0.125]);
    assert_ne!(snapshot.fit.identity, snapshot.identity);
    gw_judge::verify_calibration_snapshot(&fixture.panel, &fixture.records, &snapshot).unwrap();
}

#[test]
fn invalid_and_incomplete_evidence_have_distinct_process_status_without_snapshots() {
    let mut fixture = calibration_support::Fixture::new();
    let files = Files::new(&fixture);
    fixture.evidence.fit[0].observations.pop();
    let output = files.run(&serde_json::to_vec(&fixture.evidence).unwrap());
    assert_eq!(output.status.code(), Some(2));
    let report: gw_schema::CalibrationReport = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        report.status,
        gw_schema::CalibrationStatus::IncompleteEvidence
    );
    assert!(report.snapshot.is_none());
    fixture.evidence.fit[0].candidate.record_hash = "stale".into();
    let output = files.run(&serde_json::to_vec(&fixture.evidence).unwrap());
    assert_eq!(output.status.code(), Some(1));
    let report: gw_schema::CalibrationReport = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report.status, gw_schema::CalibrationStatus::InvalidEvidence);
    assert!(report.snapshot.is_none());
}

#[test]
fn malformed_wire_and_verified_flags_fail_without_any_snapshot_output() {
    let fixture = calibration_support::Fixture::new();
    let files = Files::new(&fixture);
    for mutate in [
        |value: &mut serde_json::Value| value["beta"] = serde_json::json!(0.5),
        |value: &mut serde_json::Value| {
            value["provenance"]["runtime"] = serde_json::json!("verified")
        },
    ] {
        let mut value = serde_json::to_value(&fixture.evidence).unwrap();
        mutate(&mut value);
        let output = files.run(&serde_json::to_vec(&value).unwrap());
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
    }
}
