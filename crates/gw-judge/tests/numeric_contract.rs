//! Numeric extraction follows the task contract and never searches for a convenient number.
use gw_judge::{NullSandboxOracle, VerifierInput, run_verifier};
use gw_schema::{
    Content, Message, Oracle, Role, VerificationContract, VerificationKind, VerificationOutcome,
    VerificationPolicy,
};

fn observed(content: &str, marker: Option<&str>) -> VerificationOutcome {
    observed_with(
        content,
        "42",
        gw_schema::NumericComparison {
            extraction: marker.map_or(gw_schema::NumericExtraction::WholeContent, |marker| {
                gw_schema::NumericExtraction::FinalMarker {
                    marker: marker.into(),
                }
            }),
            ..Default::default()
        },
    )
    .unwrap()
}

fn observed_with(
    content: &str,
    expected: &str,
    numeric: gw_schema::NumericComparison,
) -> gw_judge::Result<VerificationOutcome> {
    let contract = VerificationContract {
        answer_policy: Some(VerificationPolicy::Authoritative),
        execution_policy: Some(VerificationPolicy::Absent),
        required_tests: vec![],
        kind: VerificationKind::NumericMatch,
        oracle: Oracle::Literal {
            expected: expected.into(),
        },
        numeric: Some(numeric),
    };
    let messages = vec![Message {
        role: Role::Assistant,
        content: Content::Text(content.into()),
        reasoning: Some("An unrelated scratch number: 42".into()),
        reasoning_details: None,
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }];
    Ok(run_verifier(
        &VerifierInput {
            messages: &messages,
            reasoning_tokens: 1,
            cot_required: true,
            contract: Some(&contract),
            execution_evidence: None,
            evidence_key: Default::default(),
        },
        &NullSandboxOracle,
    )?
    .verification
    .interpretation
    .unwrap()
    .answer
    .observation
    .unwrap()
    .outcome)
}

#[test]
fn declared_final_marker_extracts_only_the_final_numeric_answer() {
    assert_eq!(
        observed("A short explanation.\nFINAL: 42", Some("FINAL:")),
        VerificationOutcome::Pass
    );
}

#[test]
fn whole_content_rejects_currency_and_separator_cleanup() {
    for text in ["$42", "4 2", "4_2", "42%"] {
        assert_eq!(observed(text, None), VerificationOutcome::Unknown, "{text}");
    }
}

#[test]
fn strict_numeric_grammar_and_binary64_edges() {
    use VerificationOutcome::{Fail, Pass, Unknown};
    let exact = gw_schema::NumericComparison {
        tolerance: gw_schema::NumericTolerance {
            absolute: 0.0,
            relative: 0.0,
        },
        ..Default::default()
    };
    for (content, expected, outcome) in [
        ("42", "42", Pass),
        (" \n+4.2E+1\t", "42", Pass),
        (".5", "0.5", Pass),
        ("1.", "1", Pass),
        ("-2.5e-1", "-.25", Pass),
        ("-0", "+0.0e9", Pass),
        ("41", "42", Fail),
        ("-42", "42", Fail),
        ("about forty-two", "42", Unknown),
        ("The answer is 42", "42", Unknown),
        ("41 or 42", "42", Unknown),
        ("", "42", Unknown),
        ("1e309", "42", Unknown),
        ("-1e309", "42", Unknown),
        ("1e-9999", "0", Unknown),
        ("NaN", "42", Unknown),
        ("inf", "42", Unknown),
        ("0x2a", "42", Unknown),
        ("4,2", "42", Unknown),
        ("４２", "42", Unknown),
        ("42e", "42", Unknown),
        ("42e+", "42", Unknown),
        ("42.0.0", "42", Unknown),
        ("--42", "42", Unknown),
        ("5e-324", "5e-324", Pass),
        // Contract explicitly uses binary64: both tokens round to the same representable integer.
        ("9007199254740993", "9007199254740992", Pass),
    ] {
        assert_eq!(
            observed_with(content, expected, exact.clone()).unwrap(),
            outcome,
            "{content:?} vs {expected}"
        );
    }
}

#[test]
fn final_marker_must_be_unique_at_the_final_numeric_line() {
    use VerificationOutcome::{Pass, Unknown};
    for (content, outcome) in [
        ("Earlier number 999.\nFINAL: +4.2e1", Pass),
        ("\n  FINAL:42\n \n", Pass),
        ("42", Unknown),
        ("FINAL:", Unknown),
        ("FINAL: 41 or 42", Unknown),
        ("FINAL: 42\nMore prose", Unknown),
        ("prefix FINAL: 42", Unknown),
        ("FINAL: 41\nFINAL: 42", Unknown),
        ("The marker is FINAL:.\nFINAL: 42", Unknown),
        ("FINAL: 42 43", Unknown),
        ("FINAL: 42%", Unknown),
        ("FINAL:\n42", Unknown),
    ] {
        assert_eq!(observed(content, Some("FINAL:")), outcome, "{content:?}");
    }
}

#[test]
fn inclusive_tolerance_boundaries_and_extremes_never_pass_by_overflow() {
    use VerificationOutcome::{Fail, Pass};
    let settings = |absolute, relative| gw_schema::NumericComparison {
        tolerance: gw_schema::NumericTolerance { absolute, relative },
        ..Default::default()
    };
    for (content, expected, absolute, relative, outcome) in [
        ("1.25".to_owned(), "1", 0.25, 0.0, Pass),
        (
            f64::from_bits(1.25f64.to_bits() + 1).to_string(),
            "1",
            0.25,
            0.0,
            Fail,
        ),
        ("-1.25".to_owned(), "-1", 0.25, 0.0, Pass),
        ("-0.25".to_owned(), "0", 0.25, 0.0, Pass),
        ("9".to_owned(), "8", 0.0, 0.125, Pass),
        (
            f64::from_bits(9.0f64.to_bits() + 1).to_string(),
            "8",
            0.0,
            0.125,
            Fail,
        ),
        (f64::MAX.to_string(), "0", f64::MAX, 0.0, Pass),
        (
            (-f64::MAX).to_string(),
            "1.7976931348623157e308",
            f64::MAX,
            0.0,
            Fail,
        ),
        (
            f64::MAX.to_string(),
            "-1.7976931348623157e308",
            0.0,
            1.0,
            Fail,
        ),
    ] {
        assert_eq!(
            observed_with(&content, expected, settings(absolute, relative)).unwrap(),
            outcome,
            "{content}/{expected}/{absolute}/{relative}"
        );
    }
    assert!(observed_with("42", "1.7976931348623157e308", settings(0.0, 2.0)).is_err());
    assert!(observed_with("42", "NaN", settings(0.0, 0.0)).is_err());
}
