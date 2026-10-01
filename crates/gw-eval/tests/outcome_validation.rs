//! Adversarial evidence joins and contracts, using only synthetic offline records.

mod common;

use common::*;
use gw_eval::outcomes::*;
use gw_eval::{OutcomeEvidence, SeparationConfig, analyze};
use gw_schema::Content;

fn invalid(records: &[gw_schema::TrainingRecord], evidence: &OutcomeEvidence) -> OutcomeReport {
    let report = analyze(records, &SeparationConfig::default(), Some(evidence)).outcome_evaluation;
    assert_eq!(report.status, OutcomeStatus::InvalidEvidence);
    assert!(report.statistics.is_none());
    assert_eq!(report.coverage.evaluated_prompts, 0);
    report
}

#[test]
fn duplicate_labels_corpus_members_and_input_records_are_invalid() {
    let (records, evidence) = corpus(2, true);
    let mut duplicate = evidence.clone();
    duplicate.outcomes.push(duplicate.outcomes[0].clone());
    assert!(
        invalid(&records, &duplicate)
            .reasons
            .contains(&OutcomeReason::DuplicateOutcome {
                record_id: "p0-c0".into()
            })
    );
    duplicate.outcomes.last_mut().unwrap().outcome = ReferenceOutcome::Known { value: 0.0 };
    invalid(&records, &duplicate);
    let mut duplicate = evidence.clone();
    duplicate.corpus.push(duplicate.corpus[0].clone());
    assert!(invalid(&records, &duplicate).reasons.contains(
        &OutcomeReason::DuplicateCorpusMember {
            record_id: "p0-c0".into()
        }
    ));
    let mut repeated = records.clone();
    repeated.push(records[0].clone());
    assert!(
        invalid(&repeated, &evidence)
            .reasons
            .contains(&OutcomeReason::DuplicateRecord {
                record_id: "p0-c0".into()
            })
    );
}

#[test]
fn missing_members_and_outcomes_outside_the_frozen_membership_are_invalid() {
    let (records, evidence) = corpus(2, true);
    assert!(
        invalid(&records[1..], &evidence)
            .reasons
            .contains(&OutcomeReason::MissingRecord {
                record_id: "p0-c0".into()
            })
    );
    let mut extra = evidence.clone();
    extra.corpus.remove(0);
    assert!(
        invalid(&records, &extra)
            .reasons
            .contains(&OutcomeReason::UnexpectedOutcome {
                record_id: "p0-c0".into()
            })
    );
}

#[test]
fn candidate_bindings_reject_every_conflicting_identity_axis() {
    let (records, evidence) = corpus(2, true);
    for field in ["run_id", "training_area", "prompt_hash", "record_hash"] {
        let mut changed = evidence.clone();
        let candidate = &mut changed.outcomes[0].candidate;
        match field {
            "run_id" => candidate.run_id = "other-run".into(),
            "training_area" => candidate.training_area = "other-area".into(),
            "prompt_hash" => candidate.prompt_hash = "b".repeat(64),
            "record_hash" => candidate.record_hash = "b".repeat(64),
            _ => unreachable!(),
        }
        assert!(
            invalid(&records, &changed)
                .reasons
                .contains(&OutcomeReason::IdentityMismatch {
                    record_id: "p0-c0".into(),
                    field: field.into()
                })
        );
    }
    let mut wrong_run = evidence.clone();
    wrong_run.run_id = "other-run".into();
    invalid(&records, &wrong_run);
    let mut wrong_area = evidence.clone();
    wrong_area.training_area = "other-area".into();
    invalid(&records, &wrong_area);
}

#[test]
fn full_content_hash_is_recomputed_even_if_the_stored_hash_was_not_updated() {
    let (mut records, evidence) = corpus(2, true);
    let before = records[0].hashes.record_hash.clone();
    records[0].messages[1].content = Content::Text("changed candidate contents".into());
    assert_eq!(records[0].hashes.record_hash, before);
    assert!(
        invalid(&records, &evidence)
            .reasons
            .contains(&OutcomeReason::IdentityMismatch {
                record_id: "p0-c0".into(),
                field: "record_hash".into()
            })
    );
}

#[test]
fn shortened_hashes_and_stale_tail_bytes_are_rejected() {
    let (records, evidence) = corpus(2, true);
    for hash in [
        evidence.corpus[0].record_hash[..16].to_string(),
        format!("{}ff", &evidence.corpus[0].record_hash[..62]),
    ] {
        let mut changed = evidence.clone();
        changed.corpus[0].record_hash = hash.clone();
        changed.outcomes[0].candidate.record_hash = hash;
        invalid(&records, &changed);
    }
}

#[test]
fn duplicate_or_missing_completion_indices_never_use_input_order_to_select() {
    for missing in [false, true] {
        let (mut records, evidence) = corpus(2, true);
        records[1]
            .origin
            .generated_mut()
            .expect("generated record")
            .generation
            .completion_index = if missing { None } else { Some(0) };
        let report = invalid(&records, &evidence);
        if missing {
            assert!(
                report
                    .reasons
                    .contains(&OutcomeReason::MissingCompletionIndex {
                        record_id: "p0-c1".into()
                    })
            );
        } else {
            assert!(report.reasons.iter().any(|r| matches!(
                r,
                OutcomeReason::DuplicateCompletionIndex {
                    completion_index: 0,
                    ..
                }
            )));
        }
        records.reverse();
        assert_eq!(invalid(&records, &evidence), report);
    }
}

#[test]
fn nonfinite_and_out_of_range_outcomes_and_selector_scores_are_invalid() {
    for value in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MAX,
        -0.1,
        1.1,
    ] {
        let (records, mut evidence) = corpus(2, true);
        evidence.outcomes[0].outcome = ReferenceOutcome::Known { value };
        let report = invalid(&records, &evidence);
        assert!(report.reasons.contains(&OutcomeReason::InvalidOutcome {
            record_id: "p0-c0".into()
        }));
        assert!(serde_json::to_value(&report).unwrap()["statistics"].is_null());
        let (mut records, evidence) = corpus(2, true);
        records[0].judging.aggregate = Some(value);
        assert!(
            invalid(&records, &evidence)
                .reasons
                .contains(&OutcomeReason::InvalidJudgeScore {
                    record_id: "p0-c0".into()
                })
        );
    }
}

#[test]
fn invalid_configuration_never_emits_a_nonfinite_numerical_result() {
    let (records, evidence) = corpus(100, true);
    for value in [
        f64::NAN,
        f64::INFINITY,
        0.0,
        1.0,
        -0.1,
        1.1,
        f64::MIN_POSITIVE,
    ] {
        let mut cfg = SeparationConfig::default();
        cfg.outcomes.confidence_level = value;
        let report = analyze(&records, &cfg, Some(&evidence)).outcome_evaluation;
        assert_eq!(report.status, OutcomeStatus::InvalidEvidence);
        assert!(report.confidence_level.is_none());
        assert!(report.statistics.is_none());
        assert!(serde_json::to_value(report).unwrap()["confidence_level"].is_null());
    }
    let mut cfg = SeparationConfig::default();
    cfg.outcomes.min_evaluated_prompts = 0;
    assert_eq!(
        analyze(&records, &cfg, Some(&evidence))
            .outcome_evaluation
            .status,
        OutcomeStatus::InvalidEvidence
    );
    cfg = SeparationConfig::default();
    cfg.min_decidable_fraction = f64::NAN;
    let report = analyze(&records, &cfg, Some(&evidence));
    assert_eq!(
        report.outcome_evaluation.status,
        OutcomeStatus::InvalidEvidence
    );
    assert!(report.outcome_evaluation.statistics.is_none());
    assert!(report.diagnostics.low_decidability.is_none());
}

#[test]
fn unsupported_versions_and_empty_or_malformed_contracts_are_invalid() {
    let (records, evidence) = corpus(2, true);
    let mut changed = evidence.clone();
    changed.version += 1;
    assert!(
        invalid(&records, &changed)
            .reasons
            .contains(&OutcomeReason::UnsupportedVersion { version: 2 })
    );
    let mut changed = evidence.clone();
    changed.metric.name.clear();
    invalid(&records, &changed);
    let mut changed = evidence.clone();
    changed.metric.version.clear();
    invalid(&records, &changed);
    let mut changed = evidence.clone();
    changed.provenance.protocol_revision.clear();
    invalid(&records, &changed);
    let mut changed = evidence.clone();
    changed.provenance.reference_artifact_digest.hex = "truncated".into();
    invalid(&records, &changed);
    let mut changed = evidence.clone();
    changed.outcomes[0].outcome = ReferenceOutcome::Unknown { reason: " ".into() };
    invalid(&records, &changed);
}

#[test]
fn strict_single_contract_decoding_rejects_judge_references_and_mixed_contracts() {
    let (_, evidence) = corpus(2, true);
    let original = serde_json::to_value(&evidence).unwrap();
    let bytes = serde_json::to_vec(&evidence).unwrap();
    assert_eq!(OutcomeEvidence::from_json(&bytes).unwrap(), evidence);
    let mut json = original.clone();
    json["provenance"]["source"] = "judge_aggregate".into();
    assert!(OutcomeEvidence::from_json(&serde_json::to_vec(&json).unwrap()).is_err());
    let mut json = original.clone();
    json["outcomes"][0]["metric"] =
        serde_json::json!({"name":"other", "version":"v2", "direction":"higher_is_better"});
    assert!(OutcomeEvidence::from_json(&serde_json::to_vec(&json).unwrap()).is_err());
    let mut json = original;
    json["outcomes"][0]["provenance"] =
        serde_json::json!({"source":"adjudicated_reference", "protocol_revision":"other"});
    assert!(OutcomeEvidence::from_json(&serde_json::to_vec(&json).unwrap()).is_err());
    let duplicate_contract = String::from_utf8(bytes).unwrap().replacen(
        "\"version\":1",
        "\"version\":1,\"version\":2",
        1,
    );
    assert!(OutcomeEvidence::from_json(duplicate_contract.as_bytes()).is_err());
}
