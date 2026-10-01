//! Exact wire replay, strict numeric fields, and opaque production/payload text boundaries.
mod calibration_support;
use calibration_support::*;
use gw_judge::{JudgeSampling, PanelJudge, verify_calibration_snapshot};
use gw_schema::*;
use serde_json::json;

fn exact_fixture() -> Fixture {
    let judge = PanelJudge::new("judge-a", "a").with_sampling(JudgeSampling {
        temperature: 20.0 / 13.0,
        top_p: Some(5.0 / 13.0),
        seed: Some(i64::MAX),
    });
    let mut fixture = Fixture::with_judges(&[judge, PanelJudge::new("judge-b", "b")]);
    fixture.evidence.beta = 5.0 / 13.0;
    fixture.evidence.correlation = CalibrationCorrelation::AssumedConstantRho { rho: 7.0 / 13.0 };
    fixture.evidence.fit[0].label = CalibrationLabel::Known { value: 5.0 / 13.0 };
    fixture.evidence.assessment[0].label = CalibrationLabel::Known { value: -0.0 };
    fixture
}

#[test]
fn typed_and_value_wire_replay_preserve_every_declared_numeric_bit_and_opaque_projection() {
    let mut fixture = exact_fixture();
    for value in [
        (5.0_f64 / 13.0).next_down(),
        5.0 / 13.0,
        (5.0_f64 / 13.0).next_up(),
        0.0,
        -0.0,
    ] {
        fixture.evidence.fit[0].label = CalibrationLabel::Known { value };
        let snapshot = fixture.snapshot();
        let bytes = serde_json::to_vec(&snapshot).unwrap();
        let typed: CalibrationSnapshot = serde_json::from_slice(&bytes).unwrap();
        let tree: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let from_tree: CalibrationSnapshot = serde_json::from_value(tree.clone()).unwrap();
        assert_eq!(serde_json::to_vec(&typed).unwrap(), bytes);
        assert_eq!(serde_json::to_vec(&from_tree).unwrap(), bytes);
        assert_eq!(serde_json::to_value(&from_tree).unwrap(), tree);
        assert_eq!(
            typed.evidence.panel.judges[0].temperature.to_bits(),
            (20.0_f64 / 13.0).to_bits()
        );
        assert_eq!(
            typed.evidence.panel.judges[0].top_p.unwrap().to_bits(),
            (5.0_f64 / 13.0).to_bits()
        );
        assert_eq!(
            typed.evidence.panel.judges[0].projection_json,
            fixture.panel.declaration().judges[0].projection_json
        );
        let CalibrationLabel::Known { value: replayed } = typed.evidence.fit[0].label else {
            panic!("known label")
        };
        assert_eq!(replayed.to_bits(), value.to_bits());
        verify_calibration_snapshot(&fixture.panel, &fixture.records, &typed).unwrap();
    }
}

#[test]
fn one_ulp_and_signed_zero_evidence_changes_have_distinct_identities() {
    let mut fixture = exact_fixture();
    let original = fixture.snapshot();
    fixture.evidence.fit[0].label = CalibrationLabel::Known {
        value: (5.0_f64 / 13.0).next_up(),
    };
    let adjacent = fixture.snapshot();
    assert_ne!(original.fit.identity, adjacent.fit.identity);
    assert_ne!(original.identity, adjacent.identity);
    fixture.evidence.fit[0].label = CalibrationLabel::Known { value: 0.0 };
    let positive = fixture.snapshot();
    fixture.evidence.fit[0].label = CalibrationLabel::Known { value: -0.0 };
    let negative = fixture.snapshot();
    assert_ne!(positive.fit.identity, negative.fit.identity);
    assert_eq!(positive.fit.losses, negative.fit.losses);
    let cell = &mut fixture.evidence.fit[0].observations[0];
    cell.score = Some(cell.score.unwrap().next_up());
    rehash(cell);
    let report = fixture.report();
    assert_eq!(report.status, CalibrationStatus::InvalidEvidence);
    assert!(
        report
            .reasons
            .iter()
            .any(|message| message.contains("score contradicts"))
    );
}

#[test]
fn decimal_nonfinite_malformed_and_extra_numeric_tags_reject_at_every_numeric_shape() {
    let snapshot = exact_fixture().snapshot();
    let original = serde_json::to_value(snapshot).unwrap();
    let paths = [
        "/evidence/beta",
        "/evidence/correlation/rho",
        "/evidence/panel/judges/0/temperature",
        "/evidence/panel/judges/0/top_p",
        "/evidence/fit/0/label/value",
        "/evidence/fit/0/observations/0/score",
        "/evidence/fit/0/observations/0/request/temperature",
        "/fit/losses/0",
        "/fit/weights/0",
        "/assessment/losses/0",
        "/assessment/weighted_score_loss",
    ];
    for path in paths {
        for invalid in [
            json!(0.5),
            json!({"binary64":"7ff0000000000000"}),
            json!({"binary64":"7ff8000000000001"}),
            json!({"binary64":"3FE0000000000000"}),
            json!({"binary64":"123"}),
            json!({"binary64":"3fe0000000000000", "extra":true}),
        ] {
            let mut value = original.clone();
            *value.pointer_mut(path).unwrap() = invalid;
            assert!(
                serde_json::from_value::<CalibrationSnapshot>(value).is_err(),
                "numeric path {path}"
            );
        }
    }
}

#[test]
fn literal_reserved_names_in_raw_text_target_names_and_dimensions_remain_opaque() {
    let mut fixture = exact_fixture();
    fixture.evidence.target.name = r#"{"binary64":"literal label"}"#.into();
    let response = r#"{"score":0.25,"verdict":"accept","dimensions":{"binary64":0.7},"rationale":"literal {\"binary64\":\"7ff0000000000000\"}"}"#;
    fixture.evidence.fit[0].observations[0] = fixture
        .panel
        .observation(
            &fixture.records[0],
            0,
            CalibrationCollection::Observed,
            Some(response.into()),
        )
        .unwrap();
    let snapshot = fixture.snapshot();
    let restored: CalibrationSnapshot =
        serde_json::from_slice(&serde_json::to_vec(&snapshot).unwrap()).unwrap();
    assert_eq!(
        restored.evidence.fit[0].observations[0]
            .raw
            .as_ref()
            .unwrap()
            .response,
        response
    );
    assert_eq!(restored.evidence.target.name, fixture.evidence.target.name);
    verify_calibration_snapshot(&fixture.panel, &fixture.records, &restored).unwrap();
}

#[test]
fn unsupported_targets_provenance_flags_or_matrix_shapes_cannot_deserialize() {
    let original = serde_json::to_value(Fixture::new().evidence).unwrap();
    for (path, invalid) in [
        ("/target/semantics", json!("probability_forecast")),
        ("/provenance/runtime", json!("verified")),
        (
            "/correlation",
            json!({"kind":"empirical_matrix", "matrix":[[1, 0.2],[0.3,1]]}),
        ),
    ] {
        let mut value = original.clone();
        *value.pointer_mut(path).unwrap() = invalid;
        assert!(serde_json::from_value::<CalibrationEvidence>(value).is_err());
    }
    for path in ["/fit/0", "/fit/0/observations/0", "/provenance"] {
        let mut value = original.clone();
        value.pointer_mut(path).unwrap()["verified"] = json!(true);
        assert!(serde_json::from_value::<CalibrationEvidence>(value).is_err());
    }
    let mut mixed = original;
    mixed["fit"][0]["target"] = json!({"name":"different target"});
    assert!(serde_json::from_value::<CalibrationEvidence>(mixed).is_err());
}

#[test]
fn pinned_recipe_exact_bits_are_separate_from_analytic_accuracy_reference() {
    // Exact regression for the pinned fixed Rust exp and sequential reduction recipe.
    // This test is checked on the executing target; it does not qualify untested targets/builds.
    let snapshot = Fixture::new().snapshot();
    assert_eq!(
        snapshot
            .fit
            .weights
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        vec![0x3fa8_4834_3c90_5446, 0x3fee_7b7c_bc36_fabd]
    );
    assert!((snapshot.assessment.weighted_score_loss - 0.527_492_898_478_488_6).abs() < 4e-16);
}

/// Preserve duplicate members in raw text. Parsing into Value before injection would erase the bug.
fn malformed_raw_tags(bits: &str) -> Vec<(&'static str, String)> {
    let valid = format!(r#""{bits}""#);
    let other = if bits == "0000000000000000" {
        "4020000000000000"
    } else {
        "0000000000000000"
    };
    vec![
        (
            "identical duplicate",
            format!(r#"{{"binary64":{valid},"binary64":{valid}}}"#),
        ),
        (
            "conflicting duplicate",
            format!(r#"{{"binary64":"{other}","binary64":{valid}}}"#),
        ),
        (
            "invalid-type first duplicate",
            format!(r#"{{"binary64":0,"binary64":{valid}}}"#),
        ),
        (
            "nonfinite-first duplicate",
            format!(r#"{{"binary64":"7ff0000000000000","binary64":{valid}}}"#),
        ),
        (
            "NaN-first duplicate",
            format!(r#"{{"binary64":"7ff8000000000001","binary64":{valid}}}"#),
        ),
        (
            "malformed-first duplicate",
            format!(r#"{{"binary64":"invalid","binary64":{valid}}}"#),
        ),
        (
            "unknown key first",
            format!(r#"{{"unknown":0,"binary64":{valid}}}"#),
        ),
        (
            "unknown key last",
            format!(r#"{{"binary64":{valid},"unknown":0}}"#),
        ),
    ]
}

fn inject_raw_tag(json: &str, field_prefix: &str, bits: &str, malformed: &str) -> String {
    let original = format!(r#"{field_prefix}{{"binary64":"{bits}"}}"#);
    assert!(json.contains(&original), "missing raw field {field_prefix}");
    json.replacen(&original, &format!("{field_prefix}{malformed}"), 1)
}

#[test]
fn raw_scalar_tags_reject_duplicate_members_before_value_collapse() {
    let fixture = Fixture::new();
    let json = serde_json::to_string(&fixture.evidence).unwrap();
    let bits = format!("{:016x}", fixture.evidence.beta.to_bits());
    let mut accepted = vec![];
    for (case, malformed) in malformed_raw_tags(&bits) {
        let raw = inject_raw_tag(&json, "\"beta\":", &bits, &malformed);
        if let Ok(evidence) = serde_json::from_str::<CalibrationEvidence>(&raw) {
            let report = gw_judge::fit_calibration(&fixture.panel, &fixture.records, &evidence);
            accepted.push((case, report.status));
        }
    }
    assert!(
        accepted.is_empty(),
        "malformed raw scalar tags were accepted: {accepted:?}"
    );
}

#[test]
fn raw_optional_score_and_top_p_tags_reject_duplicate_members_before_value_collapse() {
    let fixture = exact_fixture();
    let json = serde_json::to_string(&fixture.evidence).unwrap();
    let fields = [
        (
            "\"score\":",
            fixture.evidence.fit[0].observations[0].score.unwrap(),
        ),
        (
            "\"top_p\":",
            fixture.evidence.panel.judges[0].top_p.unwrap(),
        ),
    ];
    let mut accepted = vec![];
    for (prefix, value) in fields {
        let bits = format!("{:016x}", value.to_bits());
        for (case, malformed) in malformed_raw_tags(&bits) {
            let raw = inject_raw_tag(&json, prefix, &bits, &malformed);
            if let Ok(evidence) = serde_json::from_str::<CalibrationEvidence>(&raw) {
                let report = gw_judge::fit_calibration(&fixture.panel, &fixture.records, &evidence);
                accepted.push((prefix, case, report.status));
            }
        }
    }
    assert!(
        accepted.is_empty(),
        "malformed raw optional tags were accepted: {accepted:?}"
    );
}

#[test]
fn raw_snapshot_weight_and_loss_tags_reject_duplicate_members_before_value_collapse() {
    let fixture = Fixture::new();
    let snapshot = fixture.snapshot();
    let json = serde_json::to_string(&snapshot).unwrap();
    let fields = [
        ("\"weights\":[", snapshot.fit.weights[0]),
        ("\"losses\":[", snapshot.fit.losses[0]),
    ];
    let mut accepted = vec![];
    for (prefix, value) in fields {
        let bits = format!("{:016x}", value.to_bits());
        for (case, malformed) in malformed_raw_tags(&bits) {
            let raw = inject_raw_tag(&json, prefix, &bits, &malformed);
            if let Ok(snapshot) = serde_json::from_str::<CalibrationSnapshot>(&raw) {
                let verified =
                    verify_calibration_snapshot(&fixture.panel, &fixture.records, &snapshot)
                        .is_ok();
                accepted.push((prefix, case, verified));
            }
        }
    }
    assert!(
        accepted.is_empty(),
        "malformed raw vector tags were accepted: {accepted:?}"
    );
}
