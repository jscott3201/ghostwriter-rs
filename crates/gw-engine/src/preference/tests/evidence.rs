use super::*;

fn task_pair() -> (TrainingRecord, TrainingRecord) {
    let tasks = NumericTaskDocument::from_json(include_str!(
        "../../../../../examples/reviewed-numeric-tasks.json"
    ))
    .unwrap();
    let task = &tasks.tasks[0];
    let (mut chosen, mut rejected) = pair();
    for (record, answer) in [(&mut chosen, "FINAL:5"), (&mut rejected, "FINAL:6")] {
        record.messages = vec![
            message(Role::User, task.prompt.text(), None),
            message(Role::Assistant, answer, None),
        ];
        record.task_provenance = Some(TaskProvenance::from_task(task).unwrap());
        record.verification_contract = Some(task.verification.contract());
        record.verification = gw_judge::run_verifier(
            &gw_judge::VerifierInput {
                messages: &record.messages,
                reasoning_tokens: 0,
                cot_required: false,
                contract: record.verification_contract.as_ref(),
                execution_evidence: None,
                evidence_key: Default::default(),
            },
            &gw_judge::NullSandboxOracle,
        )
        .unwrap()
        .verification;
    }
    (chosen, rejected)
}

#[test]
fn preference_retains_and_revalidates_task_group_and_split_lineage() {
    let (chosen, rejected) = task_pair();
    let original = prepared(&chosen, &rejected).unwrap();
    assert_eq!(
        original.evidence.assessment.chosen.task_provenance,
        chosen.task_provenance
    );
    for field in 0..4 {
        let mut changed = rejected.clone();
        let task = changed.task_provenance.as_mut().unwrap();
        match field {
            0 => task.group.id = "different group".into(),
            1 => task.split.revision = "different split revision".into(),
            2 => task.split.role = TaskSplitRole::Test,
            _ => task.identity.digest = "fabricated identity".into(),
        }
        assert_eq!(
            record_hash(&changed).unwrap(),
            record_hash(&rejected).unwrap()
        );
        assert!(prepared(&chosen, &changed).is_err());
    }
    let mut chosen = chosen;
    let mut rejected = rejected;
    for record in [&mut chosen, &mut rejected] {
        record.task_provenance.as_mut().unwrap().split.revision = "next declared split".into();
    }
    assert!(prepare_preference_pair(&chosen, &rejected, &original.evidence.assessment).is_err());
    let changed = prepared(&chosen, &rejected).unwrap();
    assert_ne!(changed.pair_id, original.pair_id);
    assert_ne!(
        changed.evidence.decision_evidence_hash,
        original.evidence.decision_evidence_hash
    );
}

#[test]
fn preference_authority_change_invalidates_evidence_with_unchanged_content() {
    let (mut chosen, mut rejected) = pair();
    let original = prepared(&chosen, &rejected).unwrap();
    let content = record_hash(&chosen).unwrap();
    for record in [&mut chosen, &mut rejected] {
        record.verification_contract.as_mut().unwrap().answer_policy =
            Some(VerificationPolicy::Advisory);
        record
            .verification
            .interpretation
            .as_mut()
            .unwrap()
            .answer
            .policy = VerificationPolicy::Advisory;
    }
    assert_eq!(record_hash(&chosen).unwrap(), content);
    assert!(prepare_preference_pair(&chosen, &rejected, &original.evidence.assessment).is_err());
    let changed = prepared(&chosen, &rejected).unwrap();
    assert_ne!(
        changed.evidence.decision_evidence_hash,
        original.evidence.decision_evidence_hash
    );
}

#[test]
fn preference_finite_checks_cover_remaining_quality_and_tolerance_values() {
    for field in 0..3 {
        let (mut chosen, _) = pair();
        match field {
            0 => {
                chosen.reasoning_quality = Some(ReasoningQuality {
                    fsf: Some(f64::NAN),
                    ..Default::default()
                })
            }
            1 => {
                chosen.reasoning_quality = Some(ReasoningQuality {
                    steps: vec![StepVerdict {
                        index: 0,
                        score: f64::INFINITY,
                        passed: true,
                        label: None,
                        rationale: None,
                    }],
                    ..Default::default()
                })
            }
            _ => {
                chosen
                    .verification_contract
                    .as_mut()
                    .unwrap()
                    .numeric
                    .as_mut()
                    .unwrap()
                    .tolerance
                    .relative = f64::NEG_INFINITY
            }
        }
        assert!(capture_preference_source(&chosen).is_err());
    }
}

#[test]
fn preference_supplied_snapshots_preserve_exact_numeric_representation() {
    let (chosen, rejected) = pair();
    let mut evidence = assessment(&chosen, &rejected);
    assert_eq!(evidence.chosen.judging.panel[0].temperature, Some(0.0));
    evidence.chosen.judging.panel[0].temperature = Some(-0.0);
    assert!(prepare_preference_pair(&chosen, &rejected, &evidence).is_err());
}

#[test]
fn preference_old_uncertainty_and_new_policy_disagreement_remain_blocked() {
    let (chosen, rejected) = pair();
    let mut evidence = assessment(&chosen, &rejected);
    evidence.policy.cot_required = true;
    assert!(prepare_preference_pair(&chosen, &rejected, &evidence).is_err());
    let mut chosen = chosen;
    chosen.verification.interpretation.as_mut().unwrap().version += 1;
    assert!(prepared(&chosen, &rejected).is_err());
}

#[test]
fn preference_precise_finite_evidence_survives_json_roundtrip() {
    let (mut chosen, mut rejected) = pair();
    chosen.judging.panel[0].score = 20.0 / 13.0;
    chosen.judging.aggregate = Some(10.0 / 13.0);
    rejected.judging.aggregate = Some(2.0 / 13.0);
    let mut evidence = assessment(&chosen, &rejected);
    evidence.policy.minimum_margin = 1.0 / 13.0;
    let original = prepare_preference_pair(&chosen, &rejected, &evidence).unwrap();
    let persisted: PreferenceAssessment =
        serde_json::from_slice(&serde_json::to_vec(&evidence).unwrap()).unwrap();
    let replay = prepare_preference_pair(&chosen, &rejected, &persisted).unwrap();
    assert_eq!(replay, original);
    let persisted: PreferenceRecord =
        serde_json::from_slice(&serde_json::to_vec(&original).unwrap()).unwrap();
    assert_eq!(persisted, original);
    assert_eq!(
        serde_json::to_vec(&persisted).unwrap(),
        serde_json::to_vec(&original).unwrap()
    );
}

#[test]
fn preference_all_material_numbers_roundtrip_and_reserved_names_remain_data() {
    let (mut chosen, mut rejected) = pair();
    let precise = 20.0 / 13.0;
    let prose = r#"{"binary64":"3ff89d89d89d89d9"}"#;
    for record in [&mut chosen, &mut rejected] {
        record.judging.agreement = Some(10.0 / 13.0);
        record.judging.n_eff = Some(10.0 / 13.0);
        record.judging.threshold_at_decision = Some(10.0 / 13.0);
        let vote = &mut record.judging.panel[0];
        vote.score = precise;
        vote.temperature = Some(precise);
        vote.top_p = Some(10.0 / 13.0);
        vote.dimensions
            .as_mut()
            .unwrap()
            .insert("binary64".into(), precise);
        vote.rationale = Some(prose.into());
        vote.raw_response = Some(prose.into());
        record.verification.checks[0].score = Some(10.0 / 13.0);
        record.verification.checks[0].detail = Some(prose.into());
        let tolerance = &mut record
            .verification_contract
            .as_mut()
            .unwrap()
            .numeric
            .as_mut()
            .unwrap()
            .tolerance;
        tolerance.absolute = precise * 1e-12;
        tolerance.relative = (10.0 / 13.0) * 1e-12;
        record.reasoning_quality = Some(ReasoningQuality {
            reasoning_score: Some(10.0 / 13.0),
            fsf: Some(2.0 / 13.0),
            steps: vec![StepVerdict {
                index: 0,
                score: 10.0 / 13.0,
                passed: true,
                label: Some(prose.into()),
                rationale: Some(prose.into()),
            }],
            aggregation: Some(StepAggregation::Min),
        });
    }
    let original = prepared(&chosen, &rejected).unwrap();
    // Exercise both the direct typed parser and the existing Value parsing path.
    let json = serde_json::to_vec(&original).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&json).unwrap();
    assert_eq!(
        value["evidence"]["assessment"]["chosen"]["judging"]["panel"][0]["dimensions"]["binary64"],
        serde_json::json!({"binary64": format!("{:016x}", precise.to_bits())})
    );
    let restored: PreferenceRecord = serde_json::from_value(value).unwrap();
    assert_eq!(serde_json::to_vec(&restored).unwrap(), json);
    assert_eq!(
        restored.evidence.assessment.chosen.judging.panel[0]
            .raw_response
            .as_deref(),
        Some(prose)
    );
    assert_eq!(
        prepare_preference_pair(&chosen, &rejected, &restored.evidence.assessment).unwrap(),
        original
    );
}

#[test]
fn preference_adjacent_finite_scores_remain_distinct_after_persistence() {
    let (mut chosen, rejected) = pair();
    let score = 20.0_f64 / 13.0;
    chosen.judging.panel[0].score = score;
    let first = prepared(&chosen, &rejected).unwrap();
    chosen.judging.panel[0].score = f64::from_bits(score.to_bits() + 1);
    let second = prepared(&chosen, &rejected).unwrap();
    assert_ne!(first.pair_id, second.pair_id);
    assert_ne!(
        first.evidence.decision_evidence_hash,
        second.evidence.decision_evidence_hash
    );
    for pair in [first, second] {
        let wire = serde_json::to_vec(&pair).unwrap();
        let parsed: PreferenceRecord = serde_json::from_slice(&wire).unwrap();
        assert_eq!(serde_json::to_vec(&parsed).unwrap(), wire);
    }
}

#[test]
fn preference_signed_zero_is_exact_evidence_without_changing_score_ordering() {
    let (mut chosen, rejected) = pair();
    chosen.judging.panel[0].temperature = Some(0.0);
    let positive = prepared(&chosen, &rejected).unwrap();
    chosen.judging.panel[0].temperature = Some(-0.0);
    let negative = prepared(&chosen, &rejected).unwrap();
    assert_ne!(positive.pair_id, negative.pair_id);
    let restored: PreferenceRecord =
        serde_json::from_slice(&serde_json::to_vec(&negative).unwrap()).unwrap();
    assert_eq!(
        restored.evidence.assessment.chosen.judging.panel[0]
            .temperature
            .unwrap()
            .to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(
        prepare_preference_pair(&chosen, &rejected, &restored.evidence.assessment).unwrap(),
        negative
    );
    assert!(prepare_preference_pair(&chosen, &rejected, &positive.evidence.assessment).is_err());
}

#[test]
fn preference_wire_rejects_malformed_or_nonfinite_numeric_encodings() {
    let (chosen, rejected) = pair();
    let evidence = assessment(&chosen, &rejected);
    let json = serde_json::to_value(&evidence).unwrap();
    for replacement in [
        serde_json::json!({"binary64":"7ff0000000000000"}),
        serde_json::json!({"binary64":"fff0000000000000"}),
        serde_json::json!({"binary64":"7ff8000000000001"}),
        serde_json::json!({"binary64":"3FF0000000000000"}),
        serde_json::json!({"binary64":"3ff0"}),
        serde_json::json!({"binary64":"xxxxxxxxxxxxxxxx"}),
        serde_json::json!({"binary64":0}),
        serde_json::json!({"binary64":"3ff0000000000000", "extra":true}),
        serde_json::json!(0.9),
        serde_json::Value::Null,
    ] {
        for pointer in ["/policy/minimum_margin", "/chosen/judging/panel/0/score"] {
            let mut changed = json.clone();
            *changed.pointer_mut(pointer).unwrap() = replacement.clone();
            assert!(
                serde_json::from_value::<PreferenceAssessment>(changed).is_err(),
                "{pointer}: {replacement}"
            );
        }
    }
    // Some(NaN) must not silently become None during serialization of optional numeric fields.
    let mut evidence = evidence;
    evidence.chosen.judging.aggregate = Some(f64::NAN);
    assert!(serde_json::to_value(evidence).is_err());
}
