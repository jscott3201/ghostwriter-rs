//! Admission configuration and persisted decisive-subset evidence.

use gw_judge::{
    AreaThresholds, CorrelationMatrix, Decision, Grade, HybridGrader, Verdict, assess_panel,
    rederive_verdict,
};
use gw_schema::{AdmissionIntent, Judging};

fn grade(index: usize, verdict: Verdict) -> Grade {
    Grade {
        judge_model: format!("judge-{index}"),
        score: 0.95,
        verdict,
        confidence: 0.5,
        meta_prediction: None,
        dimensions: None,
        rationale: None,
        raw: serde_json::Value::Null,
        temperature: 0.0,
        top_p: None,
        seed: None,
        rubric_id: None,
    }
}

fn subset_thresholds() -> AreaThresholds {
    AreaThresholds {
        accept_threshold: 0.8,
        reject_below: 0.6,
        min_n_eff: 1.5,
        min_n_eff_ratio: 0.7,
    }
}

#[test]
fn decisive_subset_admission_survives_persistence_and_rederivation() {
    let panel = [
        grade(0, Verdict::Accept),
        grade(1, Verdict::Accept),
        grade(2, Verdict::Uncertain),
        grade(3, Verdict::Uncertain),
    ];
    let outcome = HybridGrader::new(subset_thresholds())
        .grade(
            None,
            &panel,
            &[],
            None,
            &CorrelationMatrix::uniform_offdiagonal(4, 0.2),
        )
        .unwrap();
    assert!(matches!(outcome.decision, Decision::Accept { .. }));
    let persisted = serde_json::to_string(&outcome.judging).unwrap();
    let restored = serde_json::from_str(&persisted).unwrap();
    assert!(matches!(
        rederive_verdict(&restored, subset_thresholds()).unwrap(),
        Decision::Accept { .. }
    ));
}

#[test]
fn feasibility_considers_decisive_subsets_instead_of_only_the_full_panel() {
    let assessment = assess_panel(4, subset_thresholds(), 0.2, AdmissionIntent::Automatic).unwrap();
    assert_eq!(assessment.feasible_decisive_count, Some(2));
    assert!(4.0 / (1.0 + 3.0 * 0.2) / 4.0 < subset_thresholds().min_n_eff_ratio);
}

#[test]
fn absolute_and_relative_floors_are_inclusive_and_must_both_pass() {
    let exact = AreaThresholds {
        min_n_eff: 1.6,
        min_n_eff_ratio: 0.8,
        ..AreaThresholds::default()
    };
    assert_eq!(
        assess_panel(2, exact, 0.25, AdmissionIntent::Automatic)
            .unwrap()
            .feasible_decisive_count,
        Some(2)
    );
    for thresholds in [
        AreaThresholds {
            min_n_eff: 1.6_f64.next_up(),
            ..exact
        },
        AreaThresholds {
            min_n_eff_ratio: 0.8_f64.next_up(),
            ..exact
        },
    ] {
        assert!(assess_panel(2, thresholds, 0.25, AdmissionIntent::Automatic).is_err());
    }
    let single = AreaThresholds {
        min_n_eff: 1.0,
        min_n_eff_ratio: 1.0,
        accept_threshold: 1.0,
        reject_below: 0.0,
    };
    assert!(assess_panel(1, single, 0.0, AdmissionIntent::Automatic).is_ok());
    assert!(assess_panel(2, single, 1.0, AdmissionIntent::Automatic).is_ok());
}

#[test]
fn conservative_defaults_require_explicit_review_only_at_any_panel_size() {
    for size in [1, 2, 4, 128] {
        assert!(
            assess_panel(
                size,
                AreaThresholds::default(),
                gw_judge::DEFAULT_CORRELATION_RHO,
                AdmissionIntent::Automatic
            )
            .is_err()
        );
        assert_eq!(
            assess_panel(
                size,
                AreaThresholds::default(),
                gw_judge::DEFAULT_CORRELATION_RHO,
                AdmissionIntent::ReviewOnly
            )
            .unwrap()
            .feasible_decisive_count,
            None
        );
    }
    assert!(
        assess_panel(
            0,
            AreaThresholds::default(),
            0.7,
            AdmissionIntent::ReviewOnly
        )
        .is_err()
    );
}

#[test]
fn invalid_numeric_domains_are_rejected_even_for_review_only() {
    let base = AreaThresholds {
        min_n_eff: 1.0,
        ..AreaThresholds::default()
    };
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 1.1] {
        for thresholds in [
            AreaThresholds {
                accept_threshold: bad,
                ..base
            },
            AreaThresholds {
                reject_below: bad,
                ..base
            },
            AreaThresholds {
                min_n_eff_ratio: bad,
                ..base
            },
        ] {
            assert!(assess_panel(2, thresholds, 0.7, AdmissionIntent::ReviewOnly).is_err());
        }
        for size in [1, 2] {
            assert!(assess_panel(size, base, bad, AdmissionIntent::ReviewOnly).is_err());
        }
    }
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1] {
        assert!(
            assess_panel(
                2,
                AreaThresholds {
                    min_n_eff: bad,
                    ..base
                },
                0.7,
                AdmissionIntent::ReviewOnly
            )
            .is_err()
        );
    }
    assert!(
        assess_panel(
            2,
            AreaThresholds {
                reject_below: 0.9,
                ..base
            },
            0.7,
            AdmissionIntent::ReviewOnly
        )
        .is_err()
    );
    for rho in [0.0, 1e-13] {
        assert!(assess_panel(2, base, rho, AdmissionIntent::ReviewOnly).is_err());
    }
}

#[test]
fn review_only_persists_across_more_permissive_rederivation() {
    let thresholds = AreaThresholds {
        min_n_eff: 1.0,
        ..AreaThresholds::default()
    };
    let outcome = HybridGrader::new(thresholds)
        .with_admission_intent(AdmissionIntent::ReviewOnly)
        .grade(
            None,
            &[grade(0, Verdict::Accept)],
            &[],
            None,
            &CorrelationMatrix::identity(1),
        )
        .unwrap();
    assert_eq!(
        outcome.judging.verdict,
        Some(gw_schema::Verdict::NeedsReview)
    );
    assert_eq!(
        outcome.judging.verdict_reason.as_deref(),
        Some("review_only")
    );
    let restored: Judging =
        serde_json::from_str(&serde_json::to_string(&outcome.judging).unwrap()).unwrap();
    assert_eq!(restored.admission_intent, AdmissionIntent::ReviewOnly);
    let permissive = AreaThresholds {
        accept_threshold: 0.1,
        reject_below: 0.0,
        min_n_eff: 0.0,
        min_n_eff_ratio: 0.0,
    };
    let decision = rederive_verdict(&restored, permissive).unwrap();
    assert_eq!(
        decision.to_schema_verdict(),
        Some(gw_schema::Verdict::NeedsReview)
    );
    assert_eq!(decision.reason().as_str(), "review_only");
}

#[test]
fn decisive_count_is_validated_and_missing_historical_evidence_stays_conservative() {
    let panel = [
        grade(0, Verdict::Accept),
        grade(1, Verdict::Accept),
        grade(2, Verdict::Uncertain),
        grade(3, Verdict::Uncertain),
    ];
    let outcome = HybridGrader::new(subset_thresholds())
        .grade(
            None,
            &panel,
            &[],
            None,
            &CorrelationMatrix::uniform_offdiagonal(4, 0.2),
        )
        .unwrap();
    assert_eq!(outcome.judging.decisive_count, Some(2));
    for count in [0, 5, usize::MAX] {
        let mut invalid = outcome.judging.clone();
        invalid.decisive_count = Some(count);
        assert!(rederive_verdict(&invalid, subset_thresholds()).is_err());
    }
    for n_eff in [
        None,
        Some(f64::NAN),
        Some(f64::INFINITY),
        Some(0.0),
        Some(-1.0),
        Some(3.0),
    ] {
        let mut invalid = outcome.judging.clone();
        invalid.n_eff = n_eff;
        assert!(rederive_verdict(&invalid, subset_thresholds()).is_err());
    }
    let mut historic = outcome.judging;
    historic.decisive_count = None;
    assert!(matches!(
        rederive_verdict(&historic, subset_thresholds()).unwrap(),
        Decision::Escalate { .. }
    ));
    let all_uncertain = HybridGrader::new(subset_thresholds())
        .grade(
            None,
            &[grade(0, Verdict::Uncertain)],
            &[],
            None,
            &CorrelationMatrix::identity(1),
        )
        .unwrap();
    assert_eq!(all_uncertain.judging.decisive_count, Some(0));
    assert!(matches!(
        rederive_verdict(&all_uncertain.judging, subset_thresholds()).unwrap(),
        Decision::Reject { .. }
    ));
}
