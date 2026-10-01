//! Pure offline numerical and partition contracts. Scores/labels are synthetic, never model evidence.
mod calibration_support;
use calibration_support::*;

#[test]
fn unequal_prompt_groups_have_equal_influence_and_match_independent_loss_oracle() {
    let fixture = Fixture::new();
    let snapshot = fixture.snapshot();
    // By hand: A: mean(0, 1)=1/2; B: mean(1/4, 0)=1/8.
    // Incorrect row-equal losses are 1/4 and 3/16, so this also detects row weighting.
    assert_eq!(snapshot.fit.losses, vec![0.5, 0.125]);
    assert_eq!(
        (
            snapshot.fit.candidates,
            snapshot.fit.groups,
            snapshot.fit.exclusions
        ),
        (4, 2, 0)
    );
    // Independent analytic logistic(-3) reference constants; this is an accuracy check,
    // separate from the pinned backend exact-bit regression added after implementation.
    assert!((snapshot.fit.weights[0] - 0.047_425_873_177_566_78).abs() < 2e-17);
    assert!((snapshot.fit.weights[1] - 0.952_574_126_822_433_2).abs() < 2e-16);
    assert_ne!(snapshot.fit.identity, snapshot.identity);
}

use gw_judge::{PanelJudge, ResolvedCalibrationPanel, verify_calibration_snapshot};
use gw_schema::*;

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn held_out_labels_membership_and_source_container_never_change_fit_identity_or_weights() {
    let mut fixture = Fixture::new();
    let original = fixture.snapshot();
    fixture.evidence.assessment[0].label = CalibrationLabel::Known { value: 0.8 };
    let labels = fixture.snapshot();
    let mut other = candidate("different-held-member", "another held-out prompt");
    other
        .origin
        .generated_mut()
        .expect("generated record")
        .provenance
        .run_id = "another-run".into();
    fixture
        .evidence
        .assessment
        .push(row(&fixture.panel, &other, "assessment-b", &[0.8, 0.1]));
    fixture.records.push(other);
    fixture.evidence.provenance.reference_digest =
        "changed-container-with-different-held-out-labels".into();
    fixture.evidence.provenance.source = "different-held-out-container".into();
    let changed = fixture.snapshot();
    for snapshot in [&labels, &changed] {
        assert_eq!(original.fit.identity, snapshot.fit.identity);
        assert_eq!(bits(&original.fit.weights), bits(&snapshot.fit.weights));
        assert_ne!(original.identity, snapshot.identity);
    }
    assert_ne!(
        original.assessment.weighted_score_loss,
        labels.assessment.weighted_score_loss
    );
    fixture.evidence.fit[0].label = CalibrationLabel::Known { value: 1.0 };
    let fitted = fixture.snapshot();
    assert_ne!(original.fit.identity, fitted.fit.identity);
    assert_ne!(bits(&original.fit.weights), bits(&fitted.fit.weights));
}

#[test]
fn correlation_is_sealed_in_complete_snapshot_without_entering_the_fit() {
    let mut fixture = Fixture::new();
    let original = fixture.snapshot();
    fixture.evidence.correlation = CalibrationCorrelation::AssumedConstantRho { rho: 0.5 };
    let changed = fixture.snapshot();
    assert_eq!(original.fit.identity, changed.fit.identity);
    assert_eq!(bits(&original.fit.weights), bits(&changed.fit.weights));
    assert_ne!(original.identity, changed.identity);
    for rho in [f64::NAN, f64::INFINITY, -0.1, 0.0, 1.1] {
        fixture.evidence.correlation = CalibrationCorrelation::AssumedConstantRho { rho };
        assert_eq!(fixture.report().status, CalibrationStatus::InvalidEvidence);
        assert!(fixture.report().snapshot.is_none());
    }
}

#[test]
fn canonical_collection_order_preserves_every_saved_bit_and_identity() {
    let mut fixture = Fixture::new();
    // Unequal values make order-sensitive floating-point accumulation a real concern.
    for (index, score) in [0.000_000_01, 0.731_231_237, 0.998_762_311]
        .into_iter()
        .enumerate()
    {
        fixture.evidence.fit[index].observations[0] = fixture
            .panel
            .observation(
                &fixture.records[index],
                0,
                CalibrationCollection::Observed,
                Some(format!(r#"{{"score":{score},"verdict":"accept"}}"#)),
            )
            .unwrap();
    }
    let extra = candidate("held-extra", "held-out prompt");
    fixture.evidence.assessment.push(row(
        &fixture.panel,
        &extra,
        "assessment-a",
        &[0.312_321_9, 0.813_019_7],
    ));
    fixture.records.push(extra);
    let original = fixture.snapshot();
    fixture.records.reverse();
    fixture.evidence.fit.reverse();
    fixture.evidence.assessment.reverse();
    let reordered = fixture.snapshot();
    assert_eq!(
        serde_json::to_vec(&original).unwrap(),
        serde_json::to_vec(&reordered).unwrap()
    );
    verify_calibration_snapshot(&fixture.panel, &fixture.records, &original).unwrap();
}

#[test]
fn panel_reordering_requires_aligned_columns_and_maps_losses_and_weights() {
    let mut fixture = Fixture::new();
    let original = fixture.snapshot();
    let reversed = ResolvedCalibrationPanel::new(
        "synthetic-area",
        &[
            PanelJudge::new("judge-b", "b"),
            PanelJudge::new("judge-a", "a"),
        ],
        "synthetic rubric",
    )
    .unwrap();
    fixture.panel = reversed;
    fixture.evidence.panel = fixture.panel.declaration().clone();
    fixture.evidence.panel_identity = fixture.panel.identity().into();
    assert_eq!(fixture.report().status, CalibrationStatus::InvalidEvidence);
    for row in fixture
        .evidence
        .fit
        .iter_mut()
        .chain(&mut fixture.evidence.assessment)
    {
        row.observations.reverse();
    }
    let reordered = fixture.snapshot();
    assert_eq!(
        reordered.fit.losses,
        original.fit.losses.into_iter().rev().collect::<Vec<_>>()
    );
    assert_eq!(
        bits(&reordered.fit.weights),
        bits(&original.fit.weights)
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
    );
    assert_ne!(reordered.fit.identity, original.fit.identity);
}

#[test]
fn zero_beta_equal_loss_and_underflow_have_explicit_outcomes() {
    let mut fixture = Fixture::new();
    for beta in [0.0, -0.0] {
        fixture.evidence.beta = beta;
        assert_eq!(fixture.snapshot().fit.weights, vec![0.5, 0.5]);
    }
    fixture.evidence.beta = 4_000.0;
    let underflow = fixture.report();
    assert_eq!(underflow.status, CalibrationStatus::InvalidEvidence);
    assert!(underflow.snapshot.is_none());
    assert!(
        underflow
            .reasons
            .iter()
            .any(|reason| reason.contains("underflow"))
    );
    for beta in [f64::NAN, f64::INFINITY, -1.0] {
        fixture.evidence.beta = beta;
        assert_eq!(fixture.report().status, CalibrationStatus::InvalidEvidence);
    }
    fixture.evidence.beta = f64::MAX;
    for (index, row) in fixture.evidence.fit.iter_mut().enumerate() {
        row.observations[1] = fixture
            .panel
            .observation(
                &fixture.records[index],
                1,
                CalibrationCollection::Observed,
                row.observations[0]
                    .raw
                    .as_ref()
                    .map(|raw| raw.response.clone()),
            )
            .unwrap();
    }
    let equal = fixture.snapshot();
    assert_eq!(equal.fit.losses[0], equal.fit.losses[1]);
    assert_eq!(equal.fit.weights, vec![0.5, 0.5]);
}

#[test]
fn exact_refit_rejects_one_ulp_output_changes_and_noncomputed_status() {
    let fixture = Fixture::new();
    let original = fixture.snapshot();
    verify_calibration_snapshot(&fixture.panel, &fixture.records, &original).unwrap();
    let mut changed = original.clone();
    changed.fit.weights[0] = changed.fit.weights[0].next_up();
    assert!(verify_calibration_snapshot(&fixture.panel, &fixture.records, &changed).is_err());
    changed = original.clone();
    changed.status = CalibrationStatus::IncompleteEvidence;
    assert!(verify_calibration_snapshot(&fixture.panel, &fixture.records, &changed).is_err());
    changed = original;
    changed.identity = "forged identity".into();
    assert!(verify_calibration_snapshot(&fixture.panel, &fixture.records, &changed).is_err());
}

#[test]
fn unreferenced_unique_records_are_only_lookup_pool_entries() {
    let mut fixture = Fixture::new();
    let original = fixture.snapshot();
    // Deliberately shares a measured prompt: this pool-only record is not another fitted row.
    let mut extra = candidate("unreferenced-candidate", "first prompt");
    extra.training_area = "outside-the-measured-area".into();
    fixture.records.push(extra);
    let with_extra = fixture.snapshot();
    assert_eq!(
        serde_json::to_vec(&original).unwrap(),
        serde_json::to_vec(&with_extra).unwrap()
    );
    verify_calibration_snapshot(&fixture.panel, &fixture.records, &original).unwrap();
}
