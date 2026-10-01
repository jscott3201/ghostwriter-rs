use super::*;

fn detail(text: &str, index: u32, metadata: &str) -> ReasoningDetail {
    ReasoningDetail::Text {
        text: text.into(),
        index,
        id: Some(metadata.into()),
        signature: Some(metadata.into()),
        format: Some("plaintext".into()),
    }
}

fn reasoning_pair() -> (TrainingRecord, TrainingRecord, PreferenceAssessment) {
    let (mut chosen, mut rejected) = pair();
    for record in [&mut chosen, &mut rejected] {
        record.messages[1].reasoning = Some("前の推論 café".into());
        record.messages[1].reasoning_details = Some(vec![
            detail("前の推論 ", 1, "prefix-one"),
            detail("café", 3, "prefix-three"),
        ]);
        record.messages[3].reasoning = Some("計算すると 42 ✓".into());
        record.messages[3].reasoning_details = Some(vec![
            detail("計算すると ", 0, "completion-zero"),
            detail("42 ✓", 7, "completion-seven"),
        ]);
        record.verification = gw_judge::run_verifier(
            &gw_judge::VerifierInput {
                messages: &record.messages,
                reasoning_tokens: 12,
                cot_required: true,
                contract: record.verification_contract.as_ref(),
                execution_evidence: None,
                evidence_key: Default::default(),
            },
            &gw_judge::NullSandboxOracle,
        )
        .unwrap()
        .verification;
    }
    let mut evidence = assessment(&chosen, &rejected);
    evidence.policy.cot_required = true;
    (chosen, rejected, evidence)
}

#[test]
fn preference_required_cot_accepts_exact_ordered_unicode_text_redundancy() {
    let (chosen, rejected, evidence) = reasoning_pair();
    assert!(
        chosen
            .verification
            .interpretation
            .as_ref()
            .unwrap()
            .gate()
            .unwrap()
            .0
    );
    let pair = prepare_preference_pair(&chosen, &rejected, &evidence).unwrap();
    assert_eq!(pair.chosen[0].reasoning.as_deref(), Some("計算すると 42 ✓"));
    assert_eq!(pair.prompt[1].reasoning.as_deref(), Some("前の推論 café"));
    assert!(
        pair.prompt
            .iter()
            .chain(&pair.chosen)
            .chain(&pair.rejected)
            .all(|message| message.reasoning_details.is_none())
    );
    assert_eq!(pair.evidence.original_prefix, chosen.messages[..3]);
    assert_eq!(
        pair.evidence.assessment.chosen.original_messages,
        chosen.messages
    );
    assert_eq!(
        pair.evidence.assessment.rejected.original_messages,
        rejected.messages
    );
    let mut stripped = evidence;
    stripped.policy.cot_policy = CotPolicy::Stripped;
    let pair = prepare_preference_pair(&chosen, &rejected, &stripped).unwrap();
    assert!(
        pair.prompt
            .iter()
            .chain(&pair.chosen)
            .chain(&pair.rejected)
            .all(|message| message.reasoning.is_none() && message.reasoning_details.is_none())
    );
    assert_eq!(
        pair.evidence.assessment.chosen.original_messages,
        chosen.messages
    );
}

#[test]
fn preference_rejects_missing_mismatched_or_ambiguous_reasoning_details_before_stripping() {
    for which in 0..7 {
        let (chosen, mut rejected, _) = reasoning_pair();
        match which {
            0 => rejected.messages[3].reasoning = None,
            1 => rejected.messages[3].reasoning = Some("different flat text".into()),
            2 => {
                rejected.messages[3].reasoning_details = Some(vec![
                    detail("計算すると ", 7, "one"),
                    detail("42 ✓", 7, "two"),
                ])
            }
            3 => {
                rejected.messages[3].reasoning_details = Some(vec![
                    detail("計算すると ", 7, "one"),
                    detail("42 ✓", 0, "two"),
                ])
            }
            4 => {
                rejected.messages[3].reasoning_details = Some(vec![ReasoningDetail::Summary {
                    summary: "計算すると 42 ✓".into(),
                    id: None,
                    format: None,
                    index: 0,
                }])
            }
            5 => {
                rejected.messages[3].reasoning_details = Some(vec![ReasoningDetail::Encrypted {
                    data: "opaque".into(),
                    id: None,
                    format: None,
                    index: 0,
                }])
            }
            _ => rejected.messages[3]
                .reasoning_details
                .as_mut()
                .unwrap()
                .push(ReasoningDetail::Summary {
                    summary: "summary".into(),
                    id: None,
                    format: None,
                    index: 8,
                }),
        }
        let mut evidence = assessment(&chosen, &rejected);
        evidence.policy.cot_required = true;
        for cot in [CotPolicy::Supervised, CotPolicy::Stripped] {
            evidence.policy.cot_policy = cot;
            assert!(
                prepare_preference_pair(&chosen, &rejected, &evidence).is_err(),
                "case {which}, policy {cot:?}"
            );
        }
    }
}

#[test]
fn preference_detail_metadata_partition_or_presence_cannot_invent_preference() {
    for which in 0..3 {
        let (chosen, mut rejected, _) = reasoning_pair();
        rejected.messages[3].content = chosen.messages[3].content.clone();
        rejected.messages[3].reasoning_details = match which {
            0 => Some(vec![detail("計算すると 42 ✓", 3, "different metadata")]),
            1 => None,
            _ => Some(vec![]),
        };
        let mut evidence = assessment(&chosen, &rejected);
        evidence.policy.cot_required = true;
        assert!(
            prepare_preference_pair(&chosen, &rejected, &evidence)
                .unwrap_err()
                .to_string()
                .contains("identical")
        );
    }
}

#[test]
fn preference_original_prefix_details_must_match_before_stripping() {
    let (chosen, mut rejected, _) = reasoning_pair();
    rejected.messages[1].reasoning_details = Some(vec![detail(
        "前の推論 café",
        0,
        "same flat text but different original details",
    )]);
    assert_eq!(
        prompt_hash(&chosen.messages).unwrap(),
        prompt_hash(&rejected.messages).unwrap()
    );
    let mut evidence = assessment(&chosen, &rejected);
    evidence.policy.cot_required = true;
    evidence.policy.cot_policy = CotPolicy::Stripped;
    assert!(
        prepare_preference_pair(&chosen, &rejected, &evidence)
            .unwrap_err()
            .to_string()
            .contains("original message prefix")
    );
}

#[test]
fn preference_split_control_token_cannot_be_erased_by_stripping() {
    let (chosen, mut rejected, _) = reasoning_pair();
    rejected.messages[3].reasoning = Some("<think>bad".into());
    rejected.messages[3].reasoning_details =
        Some(vec![detail("<thi", 0, "one"), detail("nk>bad", 1, "two")]);
    let mut evidence = assessment(&chosen, &rejected);
    evidence.policy.cot_required = true;
    evidence.policy.cot_policy = CotPolicy::Stripped;
    assert!(matches!(
        prepare_preference_pair(&chosen, &rejected, &evidence),
        Err(EngineError::Format(
            gw_format::FormatError::ControlTokenInContent { .. }
        ))
    ));
}
