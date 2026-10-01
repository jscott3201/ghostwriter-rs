//! Real CLI outcome-analysis and exit contracts. All labels and model scores are synthetic.

mod common;

use std::ffi::OsString;
use std::process::{Command, Output};

use common::{cleanup_db, record, seed_store, unique_temp_path};
use gw_eval::outcomes::*;
use gw_eval::{OutcomeEvidence, OutcomeStatus, SeparationReport};
use gw_schema::{Content, Verdict};

struct Fixture {
    db: std::path::PathBuf,
    file: std::path::PathBuf,
    evidence: OutcomeEvidence,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        cleanup_db(&self.db);
        let _ = std::fs::remove_file(&self.file);
    }
}

impl Fixture {
    async fn new(prompts: usize) -> Self {
        let db = unique_temp_path("outcome-process.sqlite");
        let file = unique_temp_path("outcome-process.json");
        let mut records = Vec::with_capacity(prompts * 2);
        for prompt in 0..prompts {
            for index in 0..2 {
                let mut candidate = record(
                    &format!("p{prompt}-c{index}"),
                    "run-1",
                    Some(Verdict::Admit),
                    Some(if index == 0 { 0.9 } else { 0.1 }),
                    true,
                    &format!("p{prompt}"),
                );
                candidate.generation.completion_index = Some(index);
                candidate.messages[1].content =
                    Content::Text(format!("independent candidate answer {index}"));
                records.push(candidate);
            }
        }
        let store = seed_store(&db, "run-1", &records).await;
        let corpus: Vec<_> = records
            .iter()
            .map(|record| gw_storage::capture_candidate_binding(record).unwrap())
            .collect();
        let outcomes = corpus
            .iter()
            .enumerate()
            .map(|(i, candidate)| CandidateOutcome {
                candidate: candidate.clone(),
                outcome: ReferenceOutcome::Known {
                    value: if i % 2 == 0 { 1.0 } else { 0.0 },
                },
            })
            .collect();
        let evidence = OutcomeEvidence {
            version: 1,
            run_id: "run-1".into(),
            training_area: "rust-async".into(),
            metric: OutcomeMetric {
                name: "synthetic_reference_success".into(),
                version: "v1".into(),
                direction: MetricDirection::HigherIsBetter,
            },
            provenance: OutcomeProvenance {
                source: ReferenceSource::AdjudicatedReference,
                protocol_revision: "hand-authored-fixture-v1".into(),
                reference_artifact_digest: ReferenceDigest {
                    algorithm: DigestAlgorithm::Blake3,
                    hex: "b".repeat(64),
                },
            },
            sampling_assumption: SamplingAssumption::IndependentPrompts,
            corpus,
            outcomes,
        };
        drop(store);
        let fixture = Self { db, file, evidence };
        fixture.write();
        fixture
    }

    fn write(&self) {
        std::fs::write(&self.file, serde_json::to_vec(&self.evidence).unwrap()).unwrap();
    }

    fn run(&self, check: bool, extra: &[&str]) -> Output {
        let mut args = vec![
            OsString::from("eval"),
            OsString::from("audit-separation"),
            OsString::from("--db"),
            self.db.as_os_str().to_owned(),
            OsString::from("--outcomes"),
            self.file.as_os_str().to_owned(),
        ];
        if check {
            args.push("--check".into());
        }
        args.extend(extra.iter().map(OsString::from));
        Command::new(env!("CARGO_BIN_EXE_gw"))
            .args(args)
            .output()
            .unwrap()
    }
}

fn report(output: &Output, exit: i32, expected: OutcomeStatus) -> SeparationReport {
    assert_eq!(
        output.status.code(),
        Some(exit),
        "stdout:{}\nstderr:{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let report: SeparationReport = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report.outcome_evaluation.status, expected);
    report
}

#[tokio::test]
async fn predictive_fixture_qualifies_but_inverted_outcomes_reject_only_under_check() {
    let mut fixture = Fixture::new(64).await;
    for check in [false, true] {
        let parsed = report(&fixture.run(check, &[]), 0, OutcomeStatus::Qualified);
        assert_eq!(parsed.outcome_evaluation.coverage.evaluated_prompts, 64);
        assert!(parsed.outcome_evaluation.statistics.unwrap().lower_bound > 0.0);
    }
    for (i, label) in fixture.evidence.outcomes.iter_mut().enumerate() {
        label.outcome = ReferenceOutcome::Known {
            value: if i % 2 == 0 { 0.0 } else { 1.0 },
        };
    }
    fixture.write();
    for (check, exit) in [(false, 0), (true, 2)] {
        let parsed = report(&fixture.run(check, &[]), exit, OutcomeStatus::Inconclusive);
        assert_eq!(parsed.outcome_evaluation.statistics.unwrap().mean_gap, -0.5);
    }
}

#[tokio::test]
async fn incomplete_unknown_and_small_positive_evidence_are_successful_nonqualifying_analyses() {
    let mut fixture = Fixture::new(64).await;
    let original = fixture.evidence.clone();
    for unknown in [false, true] {
        fixture.evidence = original.clone();
        if unknown {
            fixture.evidence.outcomes[1].outcome = ReferenceOutcome::Unknown {
                reason: "not measured".into(),
            };
        } else {
            fixture.evidence.outcomes.remove(1);
        }
        fixture.write();
        for (check, exit) in [(false, 0), (true, 2)] {
            let parsed = report(
                &fixture.run(check, &[]),
                exit,
                OutcomeStatus::InsufficientEvidence,
            );
            assert!(parsed.outcome_evaluation.statistics.is_none());
        }
    }
    fixture.evidence = original;
    fixture.evidence.corpus.truncate(2);
    fixture.evidence.outcomes.truncate(2);
    fixture.write();
    let parsed = report(&fixture.run(true, &[]), 2, OutcomeStatus::Inconclusive);
    assert_eq!(parsed.outcome_evaluation.statistics.unwrap().mean_gap, 0.5);
}

#[tokio::test]
async fn stale_duplicate_and_run_filtered_evidence_are_reported_errors_in_both_modes() {
    let mut fixture = Fixture::new(2).await;
    let original = fixture.evidence.clone();
    fixture.evidence.corpus[0].record_hash = "c".repeat(64);
    fixture.evidence.outcomes[0].candidate.record_hash = "c".repeat(64);
    fixture.write();
    for check in [false, true] {
        let parsed = report(&fixture.run(check, &[]), 1, OutcomeStatus::InvalidEvidence);
        assert!(
            parsed
                .outcome_evaluation
                .reasons
                .contains(&OutcomeReason::IdentityMismatch {
                    record_id: "p0-c0".into(),
                    field: "record_hash".into()
                })
        );
    }
    fixture.evidence = original.clone();
    fixture
        .evidence
        .outcomes
        .push(fixture.evidence.outcomes[0].clone());
    fixture.write();
    report(&fixture.run(true, &[]), 1, OutcomeStatus::InvalidEvidence);
    fixture.evidence = original;
    fixture.write();
    let parsed = report(
        &fixture.run(true, &["--run-id", "other-run"]),
        1,
        OutcomeStatus::InvalidEvidence,
    );
    assert_eq!(parsed.outcome_evaluation.coverage.matched_records, 0);
    let parsed = report(
        &fixture.run(true, &["--confidence-level", "NaN"]),
        1,
        OutcomeStatus::InvalidEvidence,
    );
    assert!(parsed.outcome_evaluation.confidence_level.is_none());
}

#[tokio::test]
async fn malformed_conflicting_contracts_and_missing_files_exit_one_without_a_report() {
    let fixture = Fixture::new(2).await;
    for bytes in [b"{\"version\":NaN}".as_slice(), b"not json".as_slice()] {
        std::fs::write(&fixture.file, bytes).unwrap();
        for check in [false, true] {
            let output = fixture.run(check, &[]);
            assert_eq!(output.status.code(), Some(1));
            assert!(output.stdout.is_empty());
        }
    }
    let mut json = serde_json::to_value(&fixture.evidence).unwrap();
    json["outcomes"][0]["metric"] = serde_json::json!({"name":"mixed-contract"});
    std::fs::write(&fixture.file, serde_json::to_vec(&json).unwrap()).unwrap();
    let output = fixture.run(true, &[]);
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    std::fs::remove_file(&fixture.file).unwrap();
    let output = fixture.run(true, &[]);
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
}
