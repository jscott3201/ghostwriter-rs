//! End-to-end process exit-code tests for the opt-in `gw eval --check` contract.

mod common;

use std::ffi::{OsStr, OsString};
use std::path::Path;
use std::process::{Command, Output};

use gw_schema::Verdict;

use common::{cleanup_db, record, seed_store, unique_temp_path, write_eval_results};

fn run_gw<I, S>(args: I) -> Output
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    Command::new(env!("CARGO_BIN_EXE_gw"))
        .args(args)
        .output()
        .expect("run gw binary")
}

fn assert_exit(output: &Output, expected: i32) {
    assert_eq!(
        output.status.code(),
        Some(expected),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn promote_args(baseline: &Path, candidate: &Path, drift_exit: i32, check: bool) -> Vec<OsString> {
    let mut args = vec![
        OsString::from("eval"),
        OsString::from("promote"),
        OsString::from("--baseline"),
        baseline.as_os_str().to_owned(),
        OsString::from("--candidate"),
        candidate.as_os_str().to_owned(),
        OsString::from("--drift-exit"),
        OsString::from(drift_exit.to_string()),
    ];
    if check {
        args.push(OsString::from("--check"));
    }
    args
}

#[test]
fn promote_process_exit_codes_follow_check_contract() {
    let baseline = write_eval_results("exit-base.json", 0.50, 0.60);
    let promote_candidate = write_eval_results("exit-cand-promote.json", 0.70, 0.80);
    let reject_candidate = write_eval_results("exit-cand-reject.json", 0.90, 0.95);
    let missing_baseline = unique_temp_path("exit-missing-base.json");

    let output = run_gw(promote_args(&baseline, &promote_candidate, 0, true));
    assert_exit(&output, 0);

    let output = run_gw(promote_args(&baseline, &reject_candidate, 1, true));
    assert_exit(&output, 2);

    let output = run_gw(promote_args(&missing_baseline, &promote_candidate, 0, true));
    assert_exit(&output, 1);

    let output = run_gw(promote_args(&baseline, &reject_candidate, 1, false));
    assert_exit(&output, 0);

    let _ = std::fs::remove_file(&baseline);
    let _ = std::fs::remove_file(&promote_candidate);
    let _ = std::fs::remove_file(&reject_candidate);
}

#[test]
fn incomplete_promotion_evidence_is_a_reported_gate_rejection() {
    let baseline = unique_temp_path("incomplete-base.json");
    let candidate = unique_temp_path("incomplete-cand.json");
    std::fs::write(
        &baseline,
        r#"{"aggregate":0.50,"benchmarks":{"task":0.90}}"#,
    )
    .unwrap();
    std::fs::write(&candidate, r#"{"aggregate":0.51}"#).unwrap();
    for (check, exit) in [(true, 2), (false, 0)] {
        let output = run_gw(promote_args(&baseline, &candidate, 0, check));
        assert_exit(&output, exit);
        let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(report["promote"], false);
        assert_eq!(report["evidence_valid"], false);
        assert_eq!(report["evidence_issues"][0]["kind"], "missing_metric");
        assert_eq!(report["evidence_issues"][0]["metric"], "task");
        assert_eq!(report["evidence_issues"][0]["side"], "candidate");
    }
    std::fs::remove_file(baseline).unwrap();
    std::fs::remove_file(candidate).unwrap();
}

#[test]
fn promotion_numeric_overflow_rejects_but_malformed_json_is_an_error() {
    let baseline = unique_temp_path("numeric-base.json");
    let candidate = unique_temp_path("numeric-cand.json");
    std::fs::write(
        &baseline,
        r#"{"aggregate":0.50,"benchmarks":{"task":-1.7976931348623157e308}}"#,
    )
    .unwrap();
    std::fs::write(
        &candidate,
        r#"{"aggregate":0.60,"benchmarks":{"task":1.7976931348623157e308}}"#,
    )
    .unwrap();
    for (check, exit) in [(true, 2), (false, 0)] {
        let output = run_gw(promote_args(&baseline, &candidate, 0, check));
        assert_exit(&output, exit);
        let report: gw_eval::PromotionReport = serde_json::from_slice(&output.stdout).unwrap();
        assert!(!report.promote && !report.evidence_valid);
        assert_eq!(
            report.evidence_issues,
            [gw_eval::EvidenceIssue::NonFiniteDelta {
                metric: "task".into()
            }]
        );
        assert!(!report.rederive(0.0, 0.0));
    }

    std::fs::write(&candidate, r#"{"aggregate":NaN}"#).unwrap();
    for check in [true, false] {
        let output = run_gw(promote_args(&baseline, &candidate, 0, check));
        assert_exit(&output, 1);
        assert!(
            output.stdout.is_empty(),
            "a parse error cannot produce a decision report"
        );
    }
    std::fs::remove_file(baseline).unwrap();
    std::fs::remove_file(candidate).unwrap();
}

async fn seed_low_data_store(path: &Path) {
    let records = [
        record(
            "m-pass",
            "run-1",
            Some(Verdict::Admit),
            Some(0.9),
            true,
            "m",
        ),
        record(
            "m-fail",
            "run-1",
            Some(Verdict::Reject),
            Some(0.2),
            false,
            "m",
        ),
        record(
            "ap-hi",
            "run-1",
            Some(Verdict::Admit),
            Some(0.9),
            true,
            "ap",
        ),
        record(
            "ap-lo",
            "run-1",
            Some(Verdict::Admit),
            Some(0.5),
            true,
            "ap",
        ),
    ];
    let store = seed_store(path, "run-1", &records).await;
    drop(store);
}

#[tokio::test]
async fn audit_separation_check_returns_two_on_low_data_store() {
    let db = unique_temp_path("exit-separation-low-data.sqlite");
    seed_low_data_store(&db).await;

    let output = run_gw(vec![
        OsString::from("eval"),
        OsString::from("audit-separation"),
        OsString::from("--db"),
        db.as_os_str().to_owned(),
        OsString::from("--check"),
    ]);
    assert_exit(&output, 2);

    cleanup_db(&db);
}
