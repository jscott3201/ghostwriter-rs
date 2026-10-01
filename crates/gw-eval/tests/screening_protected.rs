mod screening_support;
use gw_eval::screening::*;
use gw_schema::*;
use screening_support::*;

fn rehash(sets: &mut [ProtectedScreeningSet]) {
    for set in sets {
        set.content_digest = protected_screening_content_digest(&set.items).unwrap();
    }
}

#[test]
fn protected_reasoning_quarantines_components_without_merging_them() {
    let text = "alpha beta gamma delta epsilon zeta eta theta iota kappa";
    let mut a = record("a", "first complete task");
    let sibling = record("sibling", "first complete task");
    let mut b = record("b", "different complete task");
    a.messages[1].reasoning = Some(text.into());
    b.messages[1].reasoning = Some(text.into());
    let rows = vec![a, sibling, b];
    let mut sets = protected();
    sets[0].items[0].responses[0].reasoning = Some(text.into());
    rehash(&mut sets);
    let plan = prepare_screening(&rows, &declaration(&rows), &sets, None).unwrap();
    assert_eq!(
        plan.lexical_status,
        LexicalScreeningStatus::MatchQuarantined
    );
    assert_eq!(plan.groups.len(), 2);
    assert!(plan.groups.iter().all(|group| group.quarantined));
    assert!(
        plan.groups
            .iter()
            .any(|group| group.members.contains(&key(&rows[1])) && group.members.len() == 2)
    );
    assert!(
        !plan
            .protected_matches
            .iter()
            .any(|found| found.record == key(&rows[1]))
    );
    assert!(plan.eligible_output.is_empty());
    let json = serde_json::to_string(&plan).unwrap();
    assert!(!json.contains(text));
    assert!(!json.contains("fixture-private-"));
}

#[test]
fn short_exact_complete_prompt_normalizes_only_pinned_ascii_and_whitespace_rules() {
    let rows = vec![record("a", " HI\tThere! ")];
    let mut sets = protected();
    sets[0].items[0].prompt = vec![message(Role::User, "hi there!")];
    rehash(&mut sets);
    let plan = prepare_screening(&rows, &declaration(&rows), &sets, None).unwrap();
    let found = &plan.protected_matches;
    assert_eq!(found.len(), 1);
    assert_eq!(
        (
            found[0].evidence.n,
            found[0].evidence.intersection,
            found[0].evidence.union
        ),
        (None, 0, 0)
    );
    sets[0].items[0].prompt[0].content = Content::Text("hi there".into());
    rehash(&mut sets);
    assert_eq!(
        prepare_screening(&rows, &declaration(&rows), &sets, None)
            .unwrap()
            .lexical_status,
        LexicalScreeningStatus::CompleteNoMatch
    );
}

#[test]
fn any_n_set_jaccard_records_independent_counts_without_repetition_weight() {
    let mut row = record("a", "ordinary independent task");
    row.messages[1].reasoning = Some("a b a b".into());
    let rows = vec![row];
    let mut declared = declaration(&rows);
    declared.policy.ngram = [2, 3];
    declared.policy.min_overlap_tokens = 2;
    declared.policy.jaccard_threshold = 1.0 / 3.0;
    let mut sets = protected();
    sets[0].items[0].responses = vec![message(Role::Assistant, "a b c")];
    rehash(&mut sets);
    let plan = prepare_screening(&rows, &declared, &sets, None).unwrap();
    assert_eq!(plan.protected_matches.len(), 1);
    let evidence = &plan.protected_matches[0].evidence;
    assert_eq!(
        (evidence.n, evidence.intersection, evidence.union),
        (Some(2), 1, 3)
    );
    declared.policy.jaccard_threshold = f64::from_bits((1.0_f64 / 3.0).to_bits() + 1);
    assert!(
        prepare_screening(&rows, &declared, &sets, None)
            .unwrap()
            .protected_matches
            .is_empty()
    );
}

#[test]
fn shingles_never_cross_fields_parts_or_turns_and_overlap_gate_is_independent() {
    let mut row = record("a", "independent question");
    row.messages[1].reasoning = Some("one two three four".into());
    row.messages[1].content = Content::Parts(vec![
        ContentPart::Text {
            text: "five six".into(),
        },
        ContentPart::Text {
            text: "seven eight".into(),
        },
    ]);
    let rows = vec![row];
    let mut sets = protected();
    sets[0].items[0].responses = vec![message(
        Role::Assistant,
        "one two three four five six seven eight",
    )];
    rehash(&mut sets);
    let plan = prepare_screening(&rows, &declaration(&rows), &sets, None).unwrap();
    assert!(plan.protected_matches.is_empty());
    assert_eq!(plan.lexical_status, LexicalScreeningStatus::CompleteNoMatch);
    let mut rows = vec![record("a", "independent question")];
    rows[0].messages[1].reasoning = Some("a b x c d".into());
    sets[0].items[0].responses = vec![message(Role::Assistant, "a b y c d")];
    rehash(&mut sets);
    let mut declared = declaration(&rows);
    declared.policy.ngram = [2, 2];
    declared.policy.min_overlap_tokens = 3;
    declared.policy.jaccard_threshold = 0.3;
    assert!(
        prepare_screening(&rows, &declared, &sets, None)
            .unwrap()
            .protected_matches
            .is_empty()
    );
    declared.policy.min_overlap_tokens = 2;
    assert!(
        !prepare_screening(&rows, &declared, &sets, None)
            .unwrap()
            .protected_matches
            .is_empty()
    );
}

#[test]
fn unsupported_payloads_and_protected_rights_union_or_coverage_are_incomplete() {
    let rows = vec![record("a", "ordinary question")];
    let declared = declaration(&rows);
    let full = protected();
    for variant in 0..6 {
        let mut sets = full.clone();
        match variant {
            0 => {
                sets.pop();
            }
            1 => sets[0].rights = None,
            2 => sets[0].rights.as_mut().unwrap().screening_permitted = false,
            3 => sets[0].coverage.fields.clear(),
            4 => sets[0].coverage.languages = vec!["fr".into()],
            _ => sets[0].coverage.complete = false,
        }
        let plan = prepare_screening(&rows, &declared, &sets, None).unwrap();
        assert_eq!(
            plan.lexical_status,
            LexicalScreeningStatus::Incomplete,
            "{variant}"
        );
        assert!(plan.eligible_output.is_empty());
    }
    for content in [
        Content::Parts(vec![ContentPart::ImageUrl {
            image_url: "opaque-local-image".into(),
        }]),
        Content::Parts(vec![ContentPart::InputAudio {
            audio_url: None,
            format: None,
        }]),
    ] {
        let mut changed = rows.clone();
        changed[0].messages[1].content = content;
        let plan = prepare_screening(&changed, &declared, &full, None).unwrap();
        assert_eq!(plan.lexical_status, LexicalScreeningStatus::Incomplete);
        assert!(
            plan.incomplete
                .iter()
                .any(|reason| reason.code == "unsupported_media")
        );
    }
    let mut changed = rows;
    changed[0].messages[1].reasoning_details = Some(vec![ReasoningDetail::Encrypted {
        data: "opaque cipher text".into(),
        id: None,
        format: None,
        index: 0,
    }]);
    assert_eq!(
        prepare_screening(&changed, &declared, &full, None)
            .unwrap()
            .lexical_status,
        LexicalScreeningStatus::Incomplete
    );
}

#[test]
fn every_resource_limit_fails_explicitly_before_a_complete_no_match_result() {
    let rows = vec![record(
        "a",
        "one two three four five six seven eight nine ten",
    )];
    let baseline = declaration(&rows);
    for (name, limits) in [
        (
            "total_text",
            ScreeningLimits {
                total_text_bytes: 1,
                ..Default::default()
            },
        ),
        (
            "segment_bytes",
            ScreeningLimits {
                segment_bytes: 1,
                ..Default::default()
            },
        ),
        (
            "tokens",
            ScreeningLimits {
                segment_tokens: 1,
                ..Default::default()
            },
        ),
        (
            "segments",
            ScreeningLimits {
                segments: 1,
                ..Default::default()
            },
        ),
        (
            "shingles",
            ScreeningLimits {
                distinct_shingles: 1,
                ..Default::default()
            },
        ),
        (
            "shingle_work",
            ScreeningLimits {
                shingle_token_work: 1,
                ..Default::default()
            },
        ),
        (
            "comparisons",
            ScreeningLimits {
                comparisons: 1,
                ..Default::default()
            },
        ),
    ] {
        let mut declared = baseline.clone();
        declared.policy.limits = limits;
        let plan = prepare_screening(&rows, &declared, &protected(), None).unwrap();
        assert_eq!(
            plan.lexical_status,
            LexicalScreeningStatus::Incomplete,
            "{name}"
        );
        assert!(
            plan.incomplete
                .iter()
                .any(|reason| reason.code.ends_with("limit")),
            "{name}: {:?}",
            plan.incomplete
        );
        assert!(plan.eligible_output.is_empty());
        assert_eq!(plan.strata[0].tokens, None);
        validate_screening_plan(&rows, &protected(), &plan).unwrap();
    }
    let mut exact = baseline;
    exact.policy.limits.segments = 22; // Two source fields plus ten two-field synthetic sets.
    assert_eq!(
        prepare_screening(&rows, &exact, &protected(), None)
            .unwrap()
            .lexical_status,
        LexicalScreeningStatus::CompleteNoMatch
    );
    exact.policy.limits.segments = 21;
    assert_eq!(
        prepare_screening(&rows, &exact, &protected(), None)
            .unwrap()
            .lexical_status,
        LexicalScreeningStatus::Incomplete
    );
}

#[test]
fn an_empty_protected_item_cannot_claim_complete_required_contents() {
    let rows = vec![record("a", "ordinary question")];
    let mut sets = protected();
    sets[0].items[0].prompt.clear();
    sets[0].items[0].responses.clear();
    rehash(&mut sets);
    let plan = prepare_screening(&rows, &declaration(&rows), &sets, None).unwrap();
    assert_eq!(plan.lexical_status, LexicalScreeningStatus::Incomplete);
    assert!(
        plan.incomplete
            .iter()
            .any(|reason| reason.code == "incomplete_protected_prompt")
    );
    assert!(
        !plan
            .protected_inputs
            .iter()
            .find(|input| input.canonical_id == sets[0].canonical_id)
            .unwrap()
            .complete
    );
}
