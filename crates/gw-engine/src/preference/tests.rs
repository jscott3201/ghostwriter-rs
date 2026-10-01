use super::*;
use gw_schema::*;
use gw_storage::{completion_hash, prompt_hash, record_hash};

fn message(role: Role, content: &str, reasoning: Option<&str>) -> Message {
    Message {
        role,
        content: Content::Text(content.into()),
        reasoning: reasoning.map(str::to_owned),
        reasoning_details: None,
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }
}

fn record(id: &str, answer: &str, score: f64) -> TrainingRecord {
    let mut record: TrainingRecord = serde_json::from_value(serde_json::json!({
        "record_id": id,
        "schema_version": "1.0.0",
        "training_area": "math",
        "messages": [],
        "provenance": {
            "run_id": "run", "teacher": {"provider": "test", "slug": "teacher"},
            "harness_version": "test"
        },
        "generation": {},
        "lifecycle": {"state": "admitted"}
    }))
    .unwrap();
    record.messages = vec![
        message(Role::User, "first question", None),
        message(Role::Assistant, "earlier answer", Some("earlier reasoning")),
        message(Role::User, "final question", None),
        message(Role::Assistant, answer, Some("final reasoning")),
    ];
    record.verification_contract = Some(VerificationContract {
        answer_policy: Some(VerificationPolicy::Authoritative),
        execution_policy: Some(VerificationPolicy::Absent),
        required_tests: vec![],
        kind: VerificationKind::NumericMatch,
        oracle: Oracle::Literal {
            expected: "42".into(),
        },
        numeric: Some(NumericComparison::default()),
    });
    record.verification = gw_judge::run_verifier(
        &gw_judge::VerifierInput {
            messages: &record.messages,
            reasoning_tokens: 10,
            cot_required: false,
            contract: record.verification_contract.as_ref(),
            execution_evidence: None,
            evidence_key: Default::default(),
        },
        &gw_judge::NullSandboxOracle,
    )
    .unwrap()
    .verification;
    record.judging = Judging {
        aggregate: Some(score),
        verdict: Some(Verdict::Admit),
        decisive_count: Some(1),
        threshold_at_decision: Some(0.7),
        n_eff: Some(1.0),
        agreement: Some(1.0),
        panel: vec![JudgeVote {
            judge_model: "judge".into(),
            rubric_id: Some("rubric".into()),
            temperature: Some(0.0),
            top_p: Some(1.0),
            seed: Some(1),
            score: 9.0,
            dimensions: Some(std::collections::BTreeMap::from([(
                "correctness".into(),
                9.0,
            )])),
            rationale: Some("stored grade".into()),
            raw_response: None,
        }],
        ..Default::default()
    };
    record
}

fn pair() -> (TrainingRecord, TrainingRecord) {
    let chosen = record("chosen", "42", 0.9);
    let mut rejected = record("rejected", "43", 0.3);
    // Retained nonselected candidates may have an Admit grade.
    rejected.lifecycle.state = LifecycleState::Rejected;
    (chosen, rejected)
}

fn assessment(chosen: &TrainingRecord, rejected: &TrainingRecord) -> PreferenceAssessment {
    PreferenceAssessment {
        version: PREFERENCE_VERSION,
        policy: PreferencePolicy {
            version: PREFERENCE_VERSION,
            source: PreferenceSource::JudgeScoreRanking,
            direction: PreferenceDirection::HigherAggregateIsChosen,
            protocol_revision: "declared-protocol-revision".into(),
            cot_policy: CotPolicy::Supervised,
            cot_required: false,
            minimum_margin: 0.1,
            ties: PreferenceTiePolicy::Reject,
            uncertainty: PreferenceUncertaintyPolicy::RequireDecisive,
        },
        chosen: capture_preference_source(chosen).unwrap(),
        rejected: capture_preference_source(rejected).unwrap(),
    }
}

fn prepared(chosen: &TrainingRecord, rejected: &TrainingRecord) -> Result<PreferenceRecord> {
    prepare_preference_pair(chosen, rejected, &assessment(chosen, rejected))
}

#[test]
fn preference_returns_deterministic_message_arrays_and_explicit_limits() {
    let (chosen, rejected) = pair();
    let pair = prepared(&chosen, &rejected).unwrap();
    assert_eq!(pair, prepared(&chosen, &rejected).unwrap());
    assert_eq!(pair.prompt, chosen.messages[..3]);
    assert_eq!(pair.chosen, chosen.messages[3..]);
    assert_eq!(pair.rejected, rejected.messages[3..]);
    assert_eq!(pair.evidence.assessment.chosen.record_id, "chosen");
    assert_eq!(pair.evidence.assessment.rejected.record_id, "rejected");
    assert_eq!(
        pair.evidence.chosen_termination,
        PreferenceTermination::Unknown
    );
    assert_eq!(
        pair.evidence.rejected_termination,
        PreferenceTermination::Unknown
    );
    assert_eq!(
        pair.evidence.receipt_output_binding,
        PreferenceBindingStatus::Unbound
    );
    assert_eq!(
        pair.evidence.decision_execution_binding,
        PreferenceBindingStatus::Unbound
    );
    let json = serde_json::to_value(&pair).unwrap();
    assert!(json["chosen"].is_array());
    assert_eq!(json["chosen"][0]["content"], "42");
    assert_eq!(json["chosen"][0]["reasoning"], "final reasoning");
    assert_eq!(
        serde_json::from_value::<PreferenceRecord>(json).unwrap(),
        pair
    );
    // Rubric scores on a 1..10 scale are retained, not incorrectly bounded to [0,1].
    assert_eq!(pair.evidence.assessment.chosen.judging.panel[0].score, 9.0);
}

#[test]
fn preference_rejects_equal_hashes_with_different_original_prefixes() {
    for reasoning in [false, true] {
        let (mut chosen, mut rejected) = pair();
        if reasoning {
            rejected.messages[1].reasoning = Some("different earlier reasoning".into());
        } else {
            rejected.messages[1].content = Content::Text("different earlier answer".into());
        }
        chosen.hashes.prompt_hash = prompt_hash(&chosen.messages).unwrap();
        rejected.hashes.prompt_hash = prompt_hash(&rejected.messages).unwrap();
        assert_eq!(chosen.hashes.prompt_hash, rejected.hashes.prompt_hash);
        let mut evidence = assessment(&chosen, &rejected);
        evidence.policy.cot_policy = CotPolicy::Stripped;
        assert!(
            prepare_preference_pair(&chosen, &rejected, &evidence)
                .unwrap_err()
                .to_string()
                .contains("original message prefix")
        );
    }
}

#[test]
fn preference_rejects_populated_stale_hashes_on_either_side() {
    for side in 0..2 {
        for field in 0..3 {
            let (mut chosen, mut rejected) = pair();
            let evidence = assessment(&chosen, &rejected);
            let hashes = if side == 0 {
                &mut chosen.hashes
            } else {
                &mut rejected.hashes
            };
            match field {
                0 => hashes.record_hash = "forged".into(),
                1 => hashes.prompt_hash = "forged".into(),
                _ => hashes.completion_hash = "forged".into(),
            }
            assert!(prepare_preference_pair(&chosen, &rejected, &evidence).is_err());
        }
    }
}

#[test]
fn preference_missing_hashes_recompute_and_fresh_hashes_preserve_identity() {
    let (mut chosen, mut rejected) = pair();
    let original = prepared(&chosen, &rejected).unwrap();
    for record in [&mut chosen, &mut rejected] {
        record.hashes.record_hash = record_hash(record).unwrap();
        record.hashes.prompt_hash = prompt_hash(&record.messages).unwrap();
        record.hashes.completion_hash = completion_hash(&record.messages).unwrap();
    }
    assert_eq!(prepared(&chosen, &rejected).unwrap(), original);
}

#[test]
fn preference_rejects_tied_missing_reversed_and_fabricated_scores() {
    for scores in [
        (Some(0.9), Some(0.9)),
        (Some(0.3), Some(0.9)),
        (None, Some(0.3)),
        (Some(0.9), None),
    ] {
        let (mut chosen, mut rejected) = pair();
        chosen.judging.aggregate = scores.0;
        rejected.judging.aggregate = scores.1;
        assert!(prepared(&chosen, &rejected).is_err());
    }
    let (chosen, rejected) = pair();
    let mut evidence = assessment(&chosen, &rejected);
    evidence.chosen.judging.aggregate = Some(1.0);
    assert!(
        prepare_preference_pair(&chosen, &rejected, &evidence)
            .unwrap_err()
            .to_string()
            .contains("snapshots")
    );
}

#[test]
fn preference_strict_margin_boundary() {
    let (mut chosen, mut rejected) = pair();
    chosen.judging.aggregate = Some(0.75);
    rejected.judging.aggregate = Some(0.5);
    let mut evidence = assessment(&chosen, &rejected);
    evidence.policy.minimum_margin = 0.25;
    assert!(prepare_preference_pair(&chosen, &rejected, &evidence).is_err());
    evidence.policy.minimum_margin = f64::from_bits(0.25_f64.to_bits() - 1);
    assert!(prepare_preference_pair(&chosen, &rejected, &evidence).is_ok());
    evidence.policy.minimum_margin = f64::from_bits(0.25_f64.to_bits() + 1);
    assert!(prepare_preference_pair(&chosen, &rejected, &evidence).is_err());
}

#[test]
fn preference_rejects_nonfinite_material_numbers_before_hashing() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for field in 0..11 {
            let (mut chosen, rejected) = pair();
            let evidence = assessment(&chosen, &rejected);
            match field {
                0 => chosen.judging.aggregate = Some(value),
                1 => chosen.judging.agreement = Some(value),
                2 => chosen.judging.n_eff = Some(value),
                3 => chosen.judging.threshold_at_decision = Some(value),
                4 => chosen.judging.panel[0].score = value,
                5 => chosen.judging.panel[0].temperature = Some(value),
                6 => chosen.judging.panel[0].top_p = Some(value),
                7 => {
                    chosen.judging.panel[0]
                        .dimensions
                        .as_mut()
                        .unwrap()
                        .insert("correctness".into(), value);
                }
                8 => chosen.verification.checks[0].score = Some(value),
                9 => {
                    chosen.reasoning_quality = Some(ReasoningQuality {
                        reasoning_score: Some(value),
                        ..Default::default()
                    })
                }
                _ => {
                    chosen.verification_contract.as_mut().unwrap().numeric =
                        Some(NumericComparison {
                            tolerance: NumericTolerance {
                                absolute: value,
                                relative: 0.0,
                            },
                            ..Default::default()
                        })
                }
            }
            assert!(capture_preference_source(&chosen).is_err(), "field {field}");
            assert!(prepare_preference_pair(&chosen, &rejected, &evidence).is_err());
        }
    }
}

#[test]
fn preference_rejects_invalid_policy_and_aggregate_bounds() {
    let (chosen, rejected) = pair();
    for margin in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        let mut evidence = assessment(&chosen, &rejected);
        evidence.policy.minimum_margin = margin;
        assert!(prepare_preference_pair(&chosen, &rejected, &evidence).is_err());
    }
    for score in [-0.1, 1.1] {
        let mut changed = chosen.clone();
        changed.judging.aggregate = Some(score);
        assert!(capture_preference_source(&changed).is_err());
    }
    for which in 0..3 {
        let mut evidence = assessment(&chosen, &rejected);
        match which {
            0 => evidence.version += 1,
            1 => evidence.policy.version += 1,
            _ => evidence.policy.protocol_revision = " ".into(),
        }
        assert!(prepare_preference_pair(&chosen, &rejected, &evidence).is_err());
    }
}

#[test]
fn preference_rejects_unselected_chosen_and_preserved_admit_losers() {
    for state in [
        LifecycleState::Rejected,
        LifecycleState::Judged,
        LifecycleState::NeedsReview,
        LifecycleState::Error,
    ] {
        let (mut chosen, rejected) = pair();
        chosen.lifecycle.state = state;
        assert!(!is_selected_admitted(&chosen));
        assert!(prepared(&chosen, &rejected).is_err());
    }
    for state in [
        LifecycleState::Admitted,
        LifecycleState::Formatted,
        LifecycleState::Exported,
    ] {
        let (mut chosen, rejected) = pair();
        chosen.lifecycle.state = state;
        assert!(prepared(&chosen, &rejected).is_ok());
    }
}

#[test]
fn preference_rejects_uncertain_or_review_only_sides() {
    for side in 0..2 {
        for field in 0..7 {
            let (mut chosen, mut rejected) = pair();
            let record = if side == 0 {
                &mut chosen
            } else {
                &mut rejected
            };
            match field {
                0 => record.judging.admission_intent = AdmissionIntent::ReviewOnly,
                1 => record.judging.verdict = Some(Verdict::NeedsReview),
                2 => record.judging.decisive_count = None,
                3 => record.judging.decisive_count = Some(0),
                4 => record.verification.needs_review = Some("held".into()),
                5 => {
                    record
                        .verification
                        .interpretation
                        .as_mut()
                        .unwrap()
                        .answer
                        .observation
                        .as_mut()
                        .unwrap()
                        .outcome = VerificationOutcome::Unknown
                }
                _ => record.verification.interpretation = None,
            }
            assert!(
                prepared(&chosen, &rejected).is_err(),
                "side {side}, field {field}"
            );
        }
    }
}

#[test]
fn preference_authoritative_failure_blocks_chosen_even_with_legacy_pass_and_admit() {
    let (mut chosen, rejected) = pair();
    chosen
        .verification
        .interpretation
        .as_mut()
        .unwrap()
        .answer
        .observation
        .as_mut()
        .unwrap()
        .outcome = VerificationOutcome::Fail;
    chosen.verification.all_passed = true;
    assert!(is_selected_admitted(&chosen));
    assert!(
        prepared(&chosen, &rejected)
            .unwrap_err()
            .to_string()
            .contains("authoritative verification")
    );
}

#[test]
fn preference_rejects_mismatched_authority_or_missing_contract() {
    let (mut chosen, mut rejected) = pair();
    chosen
        .verification
        .interpretation
        .as_mut()
        .unwrap()
        .answer
        .policy = VerificationPolicy::Advisory;
    chosen
        .verification
        .interpretation
        .as_mut()
        .unwrap()
        .answer
        .observation = Some(VerificationObservation {
        outcome: VerificationOutcome::Pass,
        reason: "declared".into(),
    });
    assert!(prepared(&chosen, &rejected).is_err());
    chosen.verification_contract = None;
    rejected.verification_contract = None;
    assert!(prepared(&chosen, &rejected).is_err());
}

#[test]
fn preference_identity_and_source_evidence_change_independently_of_content() {
    let (chosen, rejected) = pair();
    let original = prepared(&chosen, &rejected).unwrap();
    let old_assessment = assessment(&chosen, &rejected);
    for field in 0..8 {
        let mut changed = chosen.clone();
        match field {
            0 => changed.judging.aggregate = Some(0.95),
            1 => changed.judging.panel[0].rationale = Some("different evidence".into()),
            2 => changed.judging.threshold_at_decision = Some(0.8),
            3 => changed.verification.checks[0].detail = Some("different observation".into()),
            4 => {
                changed
                    .verification
                    .interpretation
                    .as_mut()
                    .unwrap()
                    .answer
                    .observation
                    .as_mut()
                    .unwrap()
                    .reason = "different authority evidence".into()
            }
            5 => changed.lifecycle.state = LifecycleState::Formatted,
            6 => {
                changed
                    .origin
                    .generated_mut()
                    .expect("generated fixture")
                    .provenance
                    .teacher
                    .model_card_revision = Some("new revision".into())
            }
            _ => {
                changed.reasoning_quality = Some(ReasoningQuality {
                    reasoning_score: Some(0.8),
                    ..Default::default()
                })
            }
        }
        assert_eq!(
            record_hash(&chosen).unwrap(),
            record_hash(&changed).unwrap()
        );
        assert!(prepare_preference_pair(&changed, &rejected, &old_assessment).is_err());
        let pair = prepared(&changed, &rejected).unwrap();
        assert_ne!(
            pair.evidence.decision_evidence_hash,
            original.evidence.decision_evidence_hash
        );
        assert_ne!(pair.pair_id, original.pair_id);
    }
    let mut evidence = old_assessment;
    evidence.policy.protocol_revision = "another-declared-revision".into();
    assert_ne!(
        prepare_preference_pair(&chosen, &rejected, &evidence)
            .unwrap()
            .pair_id,
        original.pair_id
    );
}

#[test]
fn preference_stripped_policy_applies_to_all_arrays_and_retains_original_prefix() {
    let (chosen, rejected) = pair();
    let mut evidence = assessment(&chosen, &rejected);
    evidence.policy.cot_policy = CotPolicy::Stripped;
    let pair = prepare_preference_pair(&chosen, &rejected, &evidence).unwrap();
    assert!(
        pair.prompt
            .iter()
            .chain(&pair.chosen)
            .chain(&pair.rejected)
            .all(|message| message.reasoning.is_none())
    );
    assert_eq!(
        pair.evidence.original_prefix[1].reasoning.as_deref(),
        Some("earlier reasoning")
    );
}

#[test]
fn preference_rejects_identical_stripped_completions_and_empty_reasoning_distinctions() {
    let (chosen, mut rejected) = pair();
    rejected.messages[3].content = chosen.messages[3].content.clone();
    rejected.messages[3].reasoning = Some("different reasoning".into());
    assert!(prepared(&chosen, &rejected).is_ok());
    let mut evidence = assessment(&chosen, &rejected);
    evidence.policy.cot_policy = CotPolicy::Stripped;
    assert!(prepare_preference_pair(&chosen, &rejected, &evidence).is_err());
    let mut chosen = chosen;
    chosen.messages[3].reasoning = None;
    rejected.messages[3].reasoning = Some(String::new());
    assert!(prepared(&chosen, &rejected).is_err());
}

#[test]
fn preference_rejects_unqualified_reasoning_mask() {
    let (chosen, rejected) = pair();
    let mut evidence = assessment(&chosen, &rejected);
    evidence.policy.cot_policy = CotPolicy::Masked;
    assert!(prepare_preference_pair(&chosen, &rejected, &evidence).is_err());
}

#[test]
fn preference_rejects_trailing_nonassistant_or_unsupported_structures() {
    for shape in 0..11 {
        let (chosen, mut rejected) = pair();
        match shape {
            0 => rejected
                .messages
                .push(message(Role::User, "trailing user", None)),
            1 => rejected
                .messages
                .push(message(Role::Tool, "trailing result", None)),
            2 => rejected.messages[3].content = Content::Null,
            3 => {
                rejected.messages[3].content =
                    Content::Parts(vec![ContentPart::Text { text: "43".into() }])
            }
            4 => rejected.messages[3].content = Content::Text("  ".into()),
            5 => {
                rejected.messages[3].reasoning_details = Some(vec![ReasoningDetail::Summary {
                    summary: "summary".into(),
                    id: None,
                    format: None,
                    index: 0,
                }])
            }
            6 => rejected.messages[3].tool_calls = Some(vec![]),
            7 => rejected.messages[3].tool_call_id = Some("call".into()),
            8 => rejected.tools = Some(vec![]),
            9 => rejected.messages[3].content = Content::Text("<think>bad</think>43".into()),
            _ => rejected.messages[0].name = Some("speaker".into()),
        }
        assert!(prepared(&chosen, &rejected).is_err(), "shape {shape}");
    }
}

#[test]
fn preference_rejects_same_identity_cross_run_and_cross_area() {
    for field in 0..5 {
        let (chosen, mut rejected) = pair();
        match field {
            0 => rejected.record_id = chosen.record_id.clone(),
            1 => {
                rejected
                    .origin
                    .generated_mut()
                    .expect("generated record")
                    .provenance
                    .run_id = "different run".into()
            }
            2 => rejected.training_area = "different area".into(),
            3 => rejected.record_id.clear(),
            _ => rejected
                .origin
                .generated_mut()
                .expect("generated record")
                .provenance
                .run_id
                .clear(),
        }
        if let Ok(snapshot) = capture_preference_source(&rejected) {
            let mut evidence = assessment(&chosen, &record("placeholder", "43", 0.3));
            evidence.rejected = snapshot;
            assert!(prepare_preference_pair(&chosen, &rejected, &evidence).is_err());
        } else {
            assert!(field >= 3);
        }
    }
}

#[test]
fn preference_new_envelopes_reject_unknown_fields() {
    let (chosen, rejected) = pair();
    let pair = prepared(&chosen, &rejected).unwrap();
    for pointer in [
        "",
        "/evidence",
        "/evidence/assessment",
        "/evidence/assessment/policy",
        "/evidence/assessment/chosen",
    ] {
        let mut json = serde_json::to_value(&pair).unwrap();
        json.pointer_mut(pointer)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("unknown".into(), serde_json::json!(true));
        assert!(
            serde_json::from_value::<PreferenceRecord>(json).is_err(),
            "{pointer}"
        );
    }
}

mod evidence;

mod reasoning;
