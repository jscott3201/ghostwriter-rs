//! Untrusted intake failures, with self-consistent forged declarations kept distinct from authority.
mod calibration_support;
use calibration_support::*;
use gw_judge::{PanelJudge, ResolvedCalibrationPanel};
use gw_schema::*;

fn status(fixture: &Fixture, expected: CalibrationStatus, reason: &str) {
    let report = fixture.report();
    assert_eq!(report.status, expected, "{:?}", report.reasons);
    assert!(report.snapshot.is_none());
    assert!(
        report
            .reasons
            .iter()
            .any(|message| message.contains(reason)),
        "{:?}",
        report.reasons
    );
}

#[test]
fn missing_unknown_uncertain_and_failed_cells_never_become_zero_score_observations() {
    let mut fixture = Fixture::new();
    fixture.evidence.fit[0].observations.pop();
    status(&fixture, CalibrationStatus::IncompleteEvidence, "coverage");
    fixture = Fixture::new();
    fixture.evidence.fit[0].label = CalibrationLabel::Unknown {
        reason: "not adjudicated".into(),
    };
    status(
        &fixture,
        CalibrationStatus::IncompleteEvidence,
        "unknown reference",
    );
    fixture = Fixture::new();
    fixture.evidence.fit[0].observations[0] = fixture
        .panel
        .observation(
            &fixture.records[0],
            0,
            CalibrationCollection::Observed,
            Some(r#"{"score":0.4,"verdict":"a-new-unknown-verdict"}"#.into()),
        )
        .unwrap();
    status(&fixture, CalibrationStatus::IncompleteEvidence, "uncertain");
    let cell = &mut fixture.evidence.fit[0].observations[0];
    cell.verdict = Some(CalibrationVerdict::Accept);
    rehash(cell);
    status(
        &fixture,
        CalibrationStatus::InvalidEvidence,
        "verdict contradicts",
    );
    fixture = Fixture::new();
    for collection in [
        CalibrationCollection::Missing,
        CalibrationCollection::Unknown,
        CalibrationCollection::Failed,
    ] {
        fixture.evidence.fit[0].observations[0] = fixture
            .panel
            .observation(&fixture.records[0], 0, collection, None)
            .unwrap();
        status(
            &fixture,
            CalibrationStatus::IncompleteEvidence,
            "collection state",
        );
    }
}

#[test]
fn valid_hashes_do_not_excuse_contradictory_claims_or_malformed_payloads() {
    let mut fixture = Fixture::new();
    let cell = &mut fixture.evidence.fit[0].observations[0];
    cell.raw.as_mut().unwrap().response = r#"{"score":0.2,"verdict":"accept"}"#.into();
    cell.score = Some(0.9);
    rehash(cell);
    status(
        &fixture,
        CalibrationStatus::InvalidEvidence,
        "score contradicts",
    );
    let original = Fixture::new().evidence.fit[0].observations[0].clone();
    for raw in [Some("malformed response"), None] {
        let cell = &mut fixture.evidence.fit[0].observations[0];
        *cell = original.clone();
        cell.raw = raw.map(|response| CalibrationRawResponse {
            response: response.into(),
        });
        rehash(cell);
        status(
            &fixture,
            CalibrationStatus::InvalidEvidence,
            if raw.is_some() {
                "malformed"
            } else {
                "lack raw.response"
            },
        );
    }
    fixture.evidence.fit[0].observations[0] = original;
    let cell = &mut fixture.evidence.fit[0].observations[0];
    cell.interpretation_version += 1;
    rehash(cell);
    status(
        &fixture,
        CalibrationStatus::InvalidEvidence,
        "interpretation version",
    );
}

#[test]
fn production_fences_prose_raw_score_nine_and_admit_alias_remain_usable() {
    let mut fixture = Fixture::new();
    for response in [
        "```json\n{\"score\":9,\"verdict\":\"ADMIT\"}\n```",
        "Explanation before {\"score\":9,\"verdict\":\"admit\"} after",
    ] {
        fixture.evidence.fit[0].observations[0] = fixture
            .panel
            .observation(
                &fixture.records[0],
                0,
                CalibrationCollection::Observed,
                Some(response.into()),
            )
            .unwrap();
        let cell = &fixture.evidence.fit[0].observations[0];
        assert_eq!(cell.score.unwrap().to_bits(), 0.9_f64.to_bits());
        assert_eq!(cell.verdict, Some(CalibrationVerdict::Accept));
        fixture.snapshot();
    }
    fixture.evidence.fit[0].observations[0].score = Some(9.0);
    rehash(&mut fixture.evidence.fit[0].observations[0]);
    status(
        &fixture,
        CalibrationStatus::InvalidEvidence,
        "score contradicts",
    );
}

#[test]
fn cross_run_overlap_cannot_be_hidden_by_different_group_labels() {
    let mut fixture = Fixture::new();
    let mut record = fixture.records[0].clone();
    record
        .origin
        .generated_mut()
        .expect("generated record")
        .provenance
        .run_id = "independent-looking-run".into();
    record.record_id = "assessment-duplicate".into();
    fixture.evidence.assessment = vec![row(
        &fixture.panel,
        &record,
        "different-group-label",
        &[0.1, 0.2],
    )];
    fixture.records.push(record);
    status(
        &fixture,
        CalibrationStatus::InvalidEvidence,
        "prompt hashes overlap",
    );
    fixture.evidence.assessment[0].prompt_group = "fit-a".into();
    status(
        &fixture,
        CalibrationStatus::InvalidEvidence,
        "prompt groups overlap",
    );
    fixture = Fixture::new();
    fixture.evidence.fit[1].prompt_group = "another-global-group".into();
    status(
        &fixture,
        CalibrationStatus::InvalidEvidence,
        "equal prompt hashes",
    );
    fixture = Fixture::new();
    fixture.evidence.fit.push(fixture.evidence.fit[0].clone());
    status(
        &fixture,
        CalibrationStatus::InvalidEvidence,
        "duplicate candidate membership",
    );
}

#[test]
fn stale_bindings_wrong_requests_and_empty_slot_substitutions_reject() {
    let mut fixture = Fixture::new();
    fixture.evidence.fit[0].candidate.record_hash = "forged-content-hash".into();
    fixture.evidence.fit[0].observations[0].candidate = fixture.evidence.fit[0].candidate.clone();
    rehash(&mut fixture.evidence.fit[0].observations[0]);
    status(
        &fixture,
        CalibrationStatus::InvalidEvidence,
        "stale or contradictory",
    );
    fixture = Fixture::new();
    let second_request = fixture.evidence.fit[1].observations[0].request.clone();
    fixture.evidence.fit[0].observations[0].request = second_request;
    rehash(&mut fixture.evidence.fit[0].observations[0]);
    status(
        &fixture,
        CalibrationStatus::InvalidEvidence,
        "rebuilt production request",
    );
    fixture.evidence.fit[0].observations[0].request = fixture.panel.declaration().judges[0].clone();
    rehash(&mut fixture.evidence.fit[0].observations[0]);
    status(
        &fixture,
        CalibrationStatus::InvalidEvidence,
        "rebuilt production request",
    );
    fixture = Fixture::new();
    let wrong = ResolvedCalibrationPanel::new(
        "synthetic-area",
        &[
            PanelJudge::new("judge-a", "a"),
            PanelJudge::new("judge-b", "b"),
        ],
        "self-consistent wrong rubric",
    )
    .unwrap();
    fixture.evidence.fit[0].observations[0].request =
        wrong.request_contract(&fixture.records[0], 0).unwrap();
    rehash(&mut fixture.evidence.fit[0].observations[0]);
    status(
        &fixture,
        CalibrationStatus::InvalidEvidence,
        "rebuilt production request",
    );
    fixture.evidence.panel = wrong.declaration().clone();
    fixture.evidence.panel_identity = wrong.identity().into();
    status(
        &fixture,
        CalibrationStatus::InvalidEvidence,
        "independently resolved",
    );
}

#[test]
fn missing_empty_and_invalid_targets_or_partitions_cannot_produce_snapshots() {
    let mut fixture = Fixture::new();
    fixture.evidence.fit.clear();
    status(&fixture, CalibrationStatus::IncompleteEvidence, "nonempty");
    fixture = Fixture::new();
    fixture.evidence.assessment.clear();
    status(&fixture, CalibrationStatus::IncompleteEvidence, "nonempty");
    fixture = Fixture::new();
    fixture.evidence.target.protocol_revision.clear();
    status(
        &fixture,
        CalibrationStatus::InvalidEvidence,
        "reference protocol",
    );
    fixture = Fixture::new();
    fixture.evidence.fit[0].label = CalibrationLabel::Known { value: f64::NAN };
    status(
        &fixture,
        CalibrationStatus::InvalidEvidence,
        "label must be finite",
    );
    fixture = Fixture::new();
    fixture.records.remove(0);
    status(
        &fixture,
        CalibrationStatus::InvalidEvidence,
        "was not supplied",
    );
}

#[test]
fn aliased_judges_cannot_become_distinct_calibration_columns() {
    let first = PanelJudge::new("same-model", "family-a");
    let second = PanelJudge::new("same-model", "family-b").with_rubric("another-audit-label");
    assert!(matches!(
        ResolvedCalibrationPanel::new("area", &[first, second], "rubric"),
        Err(gw_judge::JudgeError::DuplicateJudgeEvidence {
            first: 0,
            duplicate: 1
        })
    ));
}
