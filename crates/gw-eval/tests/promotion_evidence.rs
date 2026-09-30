//! Promotion must reject incomplete or invalid evidence without emitting invalid JSON numbers.

use gw_eval::{EvalResults, PromoteConfig, PromotionReport, promote_gate};

fn results(aggregate: f64, benchmarks: &[(&str, f64)]) -> EvalResults {
    EvalResults {
        aggregate,
        benchmarks: benchmarks
            .iter()
            .map(|(name, score)| ((*name).into(), *score))
            .collect(),
    }
}

fn assert_json_safe(report: &PromotionReport) {
    for benchmark in &report.benchmarks {
        for value in [
            benchmark.baseline,
            benchmark.candidate,
            benchmark.delta,
            benchmark.sigma,
            benchmark.noise_band,
        ] {
            assert!(
                value.is_finite(),
                "report outcomes must contain only finite numbers"
            );
        }
    }
    let json = serde_json::to_vec(report).unwrap();
    let decoded: PromotionReport = serde_json::from_slice(&json).expect("JSON report round-trips");
    assert_eq!(&decoded, report);
}

fn assert_evidence_rejection(report: &PromotionReport, kind: &str) {
    assert!(!report.promote, "invalid evidence cannot promote");
    assert!(!report.ab_pass);
    assert!(!report.evidence_valid);
    assert!(
        !report.rederive(0.0, 0.0),
        "retuning must preserve evidence rejection"
    );
    assert_json_safe(report);
    let json = serde_json::to_value(report).unwrap();
    assert!(
        json["evidence_issues"]
            .as_array()
            .unwrap()
            .iter()
            .any(|issue| issue["kind"] == kind),
        "missing diagnostic {kind}: {json}"
    );
}

#[test]
fn nonfinite_scores_reject_on_both_sides_including_unused_aggregate() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for invalid_baseline in [true, false] {
            for metric in ["aggregate", "task", "unshared"] {
                let mut baseline = results(0.50, &[("task", 0.60)]);
                let mut candidate = results(0.70, &[("task", 0.80)]);
                let invalid = if invalid_baseline {
                    &mut baseline
                } else {
                    &mut candidate
                };
                let mut config = PromoteConfig::default();
                if metric == "aggregate" {
                    invalid.aggregate = value;
                    // Aggregate still must be valid even when a different headline is configured.
                    config.ab_metric = "task".into();
                } else {
                    invalid.benchmarks.insert(metric.into(), value);
                }
                let report = promote_gate(&baseline, &candidate, 0, &config);
                assert_evidence_rejection(&report, "non_finite_score");
            }
        }
    }
}

#[test]
fn finite_scores_whose_subtraction_overflows_reject() {
    for (b, c) in [(-f64::MAX, f64::MAX), (f64::MAX, -f64::MAX)] {
        let baseline = results(0.50, &[("extreme", b)]);
        let candidate = results(0.60, &[("extreme", c)]);
        assert_evidence_rejection(
            &promote_gate(&baseline, &candidate, 0, &PromoteConfig::default()),
            "non_finite_delta",
        );
    }
}

#[test]
fn finite_noise_band_arithmetic_overflow_rejects_before_clamping() {
    let baseline = results(0.50, &[("task", 0.60)]);
    let candidate = results(0.60, &[("task", 0.80)]);
    for (floor, k, sigma) in [
        (0.0, f64::MAX, 2.0),
        (0.0, -f64::MAX, 2.0),
        (f64::MAX, f64::MAX, 1.0),
        (-f64::MAX, -f64::MAX, 1.0),
    ] {
        let config = PromoteConfig {
            ab_min_delta: floor,
            ab_sigma_k: k,
            ab_benchmark_sigma: [("task".into(), sigma)].into(),
            ..Default::default()
        };
        assert_evidence_rejection(
            &promote_gate(&baseline, &candidate, 0, &config),
            "non_finite_noise_band",
        );
    }
}

#[test]
fn nonfinite_threshold_parameters_produce_json_safe_rejections() {
    let baseline = results(0.50, &[]);
    let candidate = results(0.60, &[]);
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for field in ["ab_min_delta", "ab_sigma_k"] {
            let mut config = PromoteConfig::default();
            if field == "ab_min_delta" {
                config.ab_min_delta = value;
            } else {
                config.ab_sigma_k = value;
            }
            let report = promote_gate(&baseline, &candidate, 0, &config);
            assert_evidence_rejection(&report, "non_finite_parameter");
            let json = serde_json::to_value(report).unwrap();
            assert!(
                json[field].is_null(),
                "invalid values are explicitly absent, never fabricated"
            );
        }
    }
}

#[test]
fn aggregate_headline_and_prior_aliases_have_one_canonical_outcome() {
    let baseline = results(0.50, &[]);
    let candidate = results(0.80, &[]);
    let mut reports = Vec::new();
    for headline in ["aggregate", "eval_results.aggregate"] {
        for prior_name in ["aggregate", "eval_results.aggregate"] {
            let config = PromoteConfig {
                ab_metric: headline.into(),
                ab_benchmark_sigma: [(prior_name.into(), 0.20)].into(),
                ..Default::default()
            };
            let report = promote_gate(&baseline, &candidate, 0, &config);
            assert!(report.promote && report.evidence_valid);
            assert_eq!(report.benchmarks.len(), 1);
            assert_eq!(report.benchmarks[0].name, "eval_results.aggregate");
            assert_eq!(report.benchmarks[0].sigma, 0.20);
            assert_json_safe(&report);
            reports.push(report);
        }
    }
    assert!(reports.windows(2).all(|pair| pair[0] == pair[1]));
}

#[test]
fn reserved_aggregate_names_in_benchmark_maps_are_rejected() {
    for alias in ["aggregate", "eval_results.aggregate"] {
        for on_baseline in [true, false] {
            let mut baseline = results(0.50, &[]);
            let mut candidate = results(0.60, &[]);
            let conflicting = if on_baseline {
                &mut baseline
            } else {
                &mut candidate
            };
            conflicting.benchmarks.insert(alias.into(), 0.99);
            assert_evidence_rejection(
                &promote_gate(&baseline, &candidate, 0, &PromoteConfig::default()),
                "reserved_metric_name",
            );
        }
    }
}

#[test]
fn aggregate_prior_aliases_must_agree_after_existing_sigma_normalization() {
    let baseline = results(0.50, &[]);
    let candidate = results(0.80, &[]);
    for (bare, dotted, valid, expected_sigma) in [
        (0.1, 0.2, false, 0.0),
        (0.2, 0.2, true, 0.2),
        (-1.0, f64::NAN, true, 0.0),
        (f64::INFINITY, f64::NEG_INFINITY, true, 0.0),
    ] {
        let config = PromoteConfig {
            ab_benchmark_sigma: [
                ("aggregate".into(), bare),
                ("eval_results.aggregate".into(), dotted),
            ]
            .into(),
            ..Default::default()
        };
        let report = promote_gate(&baseline, &candidate, 0, &config);
        if valid {
            assert!(report.promote && report.evidence_valid);
            assert_eq!(report.benchmarks[0].sigma, expected_sigma);
            assert_json_safe(&report);
        } else {
            assert_evidence_rejection(&report, "conflicting_aggregate_sigma");
        }
    }
}

#[test]
fn rederive_rejects_nonfinite_or_overflowing_new_thresholds() {
    let config = PromoteConfig {
        ab_sigma_k: 0.0,
        ab_benchmark_sigma: [("task".into(), f64::MAX)].into(),
        ..Default::default()
    };
    let report = promote_gate(
        &results(0.50, &[("task", 0.60)]),
        &results(0.60, &[("task", 0.80)]),
        0,
        &config,
    );
    assert!(report.promote);
    for (floor, k) in [
        (f64::NAN, 0.0),
        (0.0, f64::NAN),
        (f64::INFINITY, 0.0),
        (f64::NEG_INFINITY, 0.0),
        (0.0, f64::INFINITY),
        (0.0, f64::NEG_INFINITY),
        (0.0, 2.0),
        (0.0, -2.0),
    ] {
        assert!(
            !report.rederive(floor, k),
            "new invalid arithmetic must fail closed"
        );
    }
}

#[test]
fn valid_percentage_and_aggregate_only_reports_keep_numeric_json() {
    for (baseline, candidate) in [
        (
            results(50.0, &[("task", 90.0)]),
            results(51.0, &[("task", 91.0)]),
        ),
        (results(0.50, &[]), results(0.60, &[])),
    ] {
        let report = promote_gate(&baseline, &candidate, 0, &PromoteConfig::default());
        assert!(report.promote && report.evidence_valid);
        assert_json_safe(&report);
        let json = serde_json::to_value(report).unwrap();
        assert!(json["ab_min_delta"].is_number());
        assert!(json["ab_sigma_k"].is_number());
    }
}
