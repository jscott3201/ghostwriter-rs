//! Integration tests for the PURE `eval` handlers: `audit-separation` over a real (file-backed) store
//! of synthetic records, and `promote` over fixture `eval_results.json` bytes. Both assert (a) the
//! handler runs end-to-end (open → scan/parse → analyze → serialize), and (b) the underlying gw-eval
//! report matches the expected values over the same inputs. No network.

mod common;

use std::io::Write;

use gw_cli::CommandOutcome;
use gw_cli::cli::{AuditSeparationArgs, PromoteArgs};
use gw_cli::commands::eval::{audit_separation, promote_cmd};
use gw_eval::{SeparationConfig, promote::EvalResults, promote::promote, separation};
use gw_schema::Verdict;
use gw_storage::{RecordFilter, Store};

use common::{cleanup_db, record, seed_store, unique_temp_path};

/// A corpus with one mixed group (verifier-decidable) and one all-pass group with two scored
/// siblings (selector-eligible): a deterministic separation signal.
async fn seed_separation_store(path: &std::path::Path) -> Store {
    let recs = vec![
        // mixed group "m": one pass, one fail.
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
        // all-pass group "ap": aggregates 0.9 and 0.5 (argmax 0.9 > mean 0.7 → selector wins).
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
    seed_store(path, "run-1", &recs).await
}

#[tokio::test]
async fn audit_separation_handler_runs_and_signal_matches() {
    let db = unique_temp_path("sep.sqlite");
    let store = seed_separation_store(&db).await;

    // (a) The underlying diagnostic over the same store yields the expected, deterministic signal.
    let report = separation::analyze_store(
        &store,
        &RecordFilter::new().run_id("run-1"),
        &SeparationConfig::default(),
    )
    .await
    .expect("analyze");
    assert_eq!(report.n_mixed, 1, "the m group is verifier-decidable");
    assert_eq!(report.n_allpass, 1, "the ap group is all-pass");
    assert_eq!(report.n_selector_eligible, 1, "ap has 2 scored siblings");
    assert!(
        (report.selector_mean_gap - 0.2).abs() < 1e-12,
        "argmax 0.9 - mean 0.7 == 0.2"
    );
    assert!((report.selector_winrate - 1.0).abs() < 1e-12);
    drop(store);

    // (b) The handler runs end-to-end (open → scan → analyze → serialize → print) and returns Ok.
    let args = AuditSeparationArgs {
        db: db.clone(),
        run_id: Some("run-1".into()),
        min_decidable_groups: Some(1),
        min_decidable_fraction: Some(0.0),
        check: false,
    };
    let outcome = audit_separation(args)
        .await
        .expect("audit-separation handler runs");
    assert_eq!(outcome, CommandOutcome::Success);

    cleanup_db(&db);
}

#[tokio::test]
async fn audit_separation_check_rejects_on_low_data_and_passes_on_signal() {
    let db = unique_temp_path("sep-check.sqlite");
    let store = seed_separation_store(&db).await;
    drop(store);

    let reject = AuditSeparationArgs {
        db: db.clone(),
        run_id: Some("run-1".into()),
        min_decidable_groups: None,
        min_decidable_fraction: None,
        check: true,
    };
    let outcome = audit_separation(reject)
        .await
        .expect("audit-separation check reject runs");
    assert_eq!(outcome, CommandOutcome::GateRejected);

    let pass = AuditSeparationArgs {
        db: db.clone(),
        run_id: Some("run-1".into()),
        min_decidable_groups: Some(1),
        min_decidable_fraction: Some(0.0),
        check: true,
    };
    let outcome = audit_separation(pass)
        .await
        .expect("audit-separation check pass runs");
    assert_eq!(outcome, CommandOutcome::Success);

    cleanup_db(&db);
}

/// Write a tiny `eval_results.json` to a unique temp file and return the path.
fn write_eval_results(suffix: &str, aggregate: f64, gsm8k: f64) -> std::path::PathBuf {
    let path = unique_temp_path(suffix);
    let json = format!(r#"{{"aggregate": {aggregate}, "benchmarks": {{"gsm8k": {gsm8k}}}}}"#);
    let mut f = std::fs::File::create(&path).expect("create eval_results");
    f.write_all(json.as_bytes()).expect("write eval_results");
    path
}

#[tokio::test]
async fn promote_handler_runs_and_decision_matches() {
    // Candidate clearly beats baseline on both the aggregate and gsm8k → PROMOTE (no σ priors → band
    // collapses to ab_min_delta == 0, so any positive delta is a win).
    let baseline = write_eval_results("base.json", 0.50, 0.60);
    let candidate = write_eval_results("cand.json", 0.70, 0.80);

    // (a) The underlying gate over the parsed fixtures yields PROMOTE.
    let base = EvalResults::from_json(&std::fs::read(&baseline).unwrap()).unwrap();
    let cand = EvalResults::from_json(&std::fs::read(&candidate).unwrap()).unwrap();
    let report = promote(&base, &cand, 0, &gw_eval::PromoteConfig::default());
    assert!(report.promote, "a clean win must promote");
    assert!(report.drift_pass);
    assert!(report.ab_pass);

    // (b) The handler runs end-to-end (read → parse → gate → serialize → print) and returns Ok.
    let args = PromoteArgs {
        baseline: baseline.clone(),
        candidate: candidate.clone(),
        drift_exit: 0,
        config: None,
        check: false,
    };
    let outcome = promote_cmd(args).await.expect("promote handler runs");
    assert_eq!(outcome, CommandOutcome::Success);

    let _ = std::fs::remove_file(&baseline);
    let _ = std::fs::remove_file(&candidate);
}

#[tokio::test]
async fn promote_check_promote_returns_success() {
    let baseline = write_eval_results("base-check-pass.json", 0.50, 0.60);
    let candidate = write_eval_results("cand-check-pass.json", 0.70, 0.80);

    let args = PromoteArgs {
        baseline: baseline.clone(),
        candidate: candidate.clone(),
        drift_exit: 0,
        config: None,
        check: true,
    };
    let outcome = promote_cmd(args)
        .await
        .expect("promote check succeeds on a promote decision");
    assert_eq!(outcome, CommandOutcome::Success);

    let _ = std::fs::remove_file(&baseline);
    let _ = std::fs::remove_file(&candidate);
}

#[tokio::test]
async fn promote_handler_rejects_on_nonzero_drift() {
    let baseline = write_eval_results("base2.json", 0.50, 0.60);
    let candidate = write_eval_results("cand2.json", 0.90, 0.95);

    // A non-zero drift exit hard-rejects regardless of the A/B win.
    let base = EvalResults::from_json(&std::fs::read(&baseline).unwrap()).unwrap();
    let cand = EvalResults::from_json(&std::fs::read(&candidate).unwrap()).unwrap();
    let report = promote(&base, &cand, 1, &gw_eval::PromoteConfig::default());
    assert!(!report.promote, "drift exit 1 must hard-reject");
    assert!(!report.drift_pass);

    let args = PromoteArgs {
        baseline: baseline.clone(),
        candidate: candidate.clone(),
        drift_exit: 1,
        config: None,
        check: false,
    };
    let outcome = promote_cmd(args)
        .await
        .expect("promote handler runs even on a reject decision");
    assert_eq!(
        outcome,
        CommandOutcome::Success,
        "default-off check preserves Ok-on-reject behavior"
    );

    let _ = std::fs::remove_file(&baseline);
    let _ = std::fs::remove_file(&candidate);
}

#[tokio::test]
async fn promote_check_reject_returns_gate_rejected() {
    let baseline = write_eval_results("base-check-reject.json", 0.50, 0.60);
    let candidate = write_eval_results("cand-check-reject.json", 0.90, 0.95);

    let args = PromoteArgs {
        baseline: baseline.clone(),
        candidate: candidate.clone(),
        drift_exit: 1,
        config: None,
        check: true,
    };
    let outcome = promote_cmd(args)
        .await
        .expect("promote check reject still runs");
    assert_eq!(outcome, CommandOutcome::GateRejected);

    let _ = std::fs::remove_file(&baseline);
    let _ = std::fs::remove_file(&candidate);
}

#[tokio::test]
async fn promote_handler_errors_on_missing_file() {
    let args = PromoteArgs {
        baseline: unique_temp_path("does-not-exist-base.json"),
        candidate: unique_temp_path("does-not-exist-cand.json"),
        drift_exit: 0,
        config: None,
        check: true,
    };
    let err = promote_cmd(args)
        .await
        .expect_err("a missing baseline must error");
    assert!(format!("{err:#}").contains("reading baseline"));
}
