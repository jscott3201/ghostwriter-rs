mod screening_support;
use gw_eval::screening::*;
use gw_schema::*;
use screening_support::*;

#[test]
fn protected_input_identity_is_independent_of_source_coverage_results() {
    let rows = vec![record("a", "ordinary question")];
    let mut sets = protected();
    sets[0]
        .coverage
        .fields
        .retain(|field| *field != ScreeningField::Reasoning);
    let before = prepare_screening(&rows, &declaration(&rows), &sets, None).unwrap();
    assert_eq!(
        before.lexical_status,
        LexicalScreeningStatus::CompleteNoMatch
    );
    let mut changed = rows;
    changed[0].messages[1].reasoning = Some("additional source reasoning".into());
    let after = prepare_screening(&changed, &declaration(&changed), &sets, None).unwrap();
    assert_eq!(after.lexical_status, LexicalScreeningStatus::Incomplete);
    assert_eq!(before.protected_input_id, after.protected_input_id);
    assert_ne!(before.screening_input_id, after.screening_input_id);
    assert_ne!(before.plan_id, after.plan_id);
}

#[test]
fn missing_source_sibling_or_parent_evidence_never_shrinks_the_population() {
    let a = record("a", "ordinary question");
    let b = record("b", "another question");
    let rows = vec![a, b];
    let baseline = declaration(&rows);
    for variant in 0..4 {
        let mut rows = rows.clone();
        let mut declared = baseline.clone();
        let code = match variant {
            0 => {
                declared.expected_tasks.pop();
                "undeclared_source_item"
            }
            1 => {
                declared.siblings.pop();
                "undeclared_sibling_set"
            }
            2 => {
                rows[1].provenance.parent_ids = vec!["external-parent".into()];
                "missing_or_ambiguous_parent"
            }
            _ => {
                rows[1].task_provenance = None;
                "missing_task_evidence"
            }
        };
        rows[1].lifecycle.state = LifecycleState::Rejected;
        let plan = prepare_screening(&rows, &declared, &protected(), None).unwrap();
        assert_eq!(plan.population.len(), 2);
        assert_eq!(plan.lexical_status, LexicalScreeningStatus::Incomplete);
        assert!(plan.incomplete.iter().any(|reason| reason.code == code));
        assert!(plan.eligible_output.is_empty());
    }
}

#[test]
fn missing_expected_source_is_distinct_from_a_rejected_sibling_completion() {
    let mut a = record("a", "ordinary shared sibling prompt");
    a.generation.n_completions = Some(2);
    let mut sibling = a.clone();
    sibling.record_id = "sibling".into();
    sibling.generation.completion_index = Some(1);
    sibling.lifecycle.state = LifecycleState::Rejected;
    let expected = record("missing-source", "expected separate source item");
    let full = vec![a.clone(), sibling.clone(), expected];
    let declared = declaration(&full);
    let before = prepare_screening(&[a.clone()], &declared, &protected(), None).unwrap();
    let after = prepare_screening(&[a, sibling], &declared, &protected(), None).unwrap();
    assert!(
        before
            .incomplete
            .iter()
            .any(|r| r.code == "incomplete_or_inconsistent_siblings")
    );
    assert!(
        !after
            .incomplete
            .iter()
            .any(|r| r.code == "incomplete_or_inconsistent_siblings")
    );
    assert!(
        after
            .incomplete
            .iter()
            .any(|r| r.code == "expected_task_membership")
    );
    assert_eq!(after.population.len(), 2);
    assert_eq!(after.lexical_status, LexicalScreeningStatus::Incomplete);
}

#[test]
fn ambiguous_parent_ids_and_out_of_scope_parents_are_not_resolved_by_guessing() {
    let a = record("a", "first question");
    let mut b = record("b", "second question");
    b.record_id = a.record_id.clone();
    let mut child = record("child", "third question");
    child.provenance.parent_ids = vec![a.record_id.clone()];
    let rows = vec![a, b, child];
    let plan = prepare_screening(&rows, &declaration(&rows), &protected(), None).unwrap();
    assert!(
        plan.incomplete
            .iter()
            .any(|r| r.code == "missing_or_ambiguous_parent")
    );
    assert!(!plan.edges.iter().any(|edge| edge.kind == "declared_parent"));
    let mut child = record("child", "third question");
    child.provenance.parent_ids = vec!["external".into()];
    let declared = declaration(&[child.clone()]);
    let before = prepare_screening(&[child.clone()], &declared, &protected(), None).unwrap();
    let after = prepare_screening(
        &[child, record("external", "external parent")],
        &declared,
        &protected(),
        None,
    )
    .unwrap();
    assert_eq!(
        serde_json::to_vec(&before).unwrap(),
        serde_json::to_vec(&after).unwrap()
    );
}

#[test]
fn excluded_dependencies_eligibility_and_in_scope_additions_invalidate_old_plans() {
    let a = record("a", "first question");
    let mut b = record("b", "second question");
    b.lifecycle.state = LifecycleState::Rejected;
    let rows = vec![a, b];
    let declared = declaration(&rows);
    let sets = protected();
    let plan = prepare_screening(&rows, &declared, &sets, None).unwrap();
    for variant in 0..7 {
        let mut changed = rows.clone();
        match variant {
            0 => changed[1].provenance.parent_ids = vec![changed[0].record_id.clone()],
            1 => {
                changed[1].task_provenance.as_mut().unwrap().split.role = TaskSplitRole::Validation
            }
            2 => changed[1].lifecycle.state = LifecycleState::Admitted,
            3 => {
                let mut extra = record("extra", "unrelated added row");
                extra.provenance.run_id = rows[1].provenance.run_id.clone();
                extra.lifecycle.state = LifecycleState::Rejected;
                changed.push(extra);
            }
            4 => {
                changed.pop();
            }
            5 => changed[1].generation.n_completions = Some(2),
            _ => changed[1].judging.verdict = Some(Verdict::Reject),
        }
        if matches!(variant, 0 | 1 | 2 | 5 | 6) {
            assert_eq!(
                gw_storage::record_hash(&changed[1]).unwrap(),
                gw_storage::record_hash(&rows[1]).unwrap()
            );
        }
        assert!(
            validate_screening_plan(&changed, &sets, &plan).is_err(),
            "variant {variant}"
        );
    }
}

#[test]
fn publication_history_and_equivalent_eligible_states_do_not_self_invalidate() {
    let rows = vec![record("a", "ordinary question")];
    let declared = declaration(&rows);
    let sets = protected();
    let plan = prepare_screening(&rows, &declared, &sets, None).unwrap();
    let mut changed = rows;
    changed[0].lifecycle.state = LifecycleState::Exported;
    changed[0].lifecycle.history.push(StateTransition {
        state: LifecycleState::Exported,
        at: "2026-09-30T00:00:00Z".into(),
        attempt: 7,
    });
    changed[0].dataset_version = Some(semver::Version::new(1, 2, 3));
    changed[0].hashes.record_hash = "publication cache only".into();
    validate_screening_plan(&changed, &sets, &plan).unwrap();
}

fn detailed_record(id: &str, answer: &str) -> TrainingRecord {
    let mut row = record(id, "initial question");
    row.messages[1].reasoning_details = Some(vec![ReasoningDetail::Text {
        text: "intermediate rationale".into(),
        id: Some("detail-1".into()),
        index: 0,
        signature: Some("signature-1".into()),
        format: Some("format-1".into()),
    }]);
    row.messages.extend([
        message(Role::User, "final question"),
        message(Role::Assistant, answer),
    ]);
    rebind_task(&mut row);
    row
}

fn change_detail(row: &mut TrainingRecord, field: &str) {
    let ReasoningDetail::Text {
        id,
        index,
        signature,
        format,
        ..
    } = &mut row.messages[1].reasoning_details.as_mut().unwrap()[0]
    else {
        panic!("text fixture")
    };
    match field {
        "id" => *id = Some("detail-2".into()),
        "index" => *index = 1,
        "signature" => *signature = Some("signature-2".into()),
        "format" => *format = Some("format-2".into()),
        _ => panic!("unknown fixture field"),
    }
}

#[test]
fn every_reasoning_shape_field_invalidates_frozen_inputs_and_predecessors() {
    let rows = vec![detailed_record("a", "41")];
    let mut declared = declaration(&rows);
    declared.policy.multi_turn_loss = MultiTurnLoss::FinalTurnOnly;
    let sets = protected();
    let before = prepare_screening(&rows, &declared, &sets, None).unwrap();
    assert!(before.incomplete.is_empty());
    let mut unbound = Vec::new();
    for field in ["id", "index", "signature", "format"] {
        let mut changed = rows.clone();
        change_detail(&mut changed[0], field);
        // Legacy content identity intentionally omits these shape fields; retain that contract.
        assert_eq!(
            gw_storage::record_hash(&rows[0]).unwrap(),
            gw_storage::record_hash(&changed[0]).unwrap()
        );
        let after = prepare_screening(&changed, &declared, &sets, None).unwrap();
        assert!(after.incomplete.is_empty());
        if before.population[0].screening_input_id == after.population[0].screening_input_id
            || before.screening_input_id == after.screening_input_id
            || before.plan_id == after.plan_id
            || validate_screening_plan(&changed, &sets, &before).is_ok()
            || prepare_screening(&changed, &declared, &sets, Some(&before)).is_ok()
        {
            unbound.push(field);
        }
    }
    assert!(
        unbound.is_empty(),
        "unbound reasoning metadata: {unbound:?}"
    );
}

#[test]
fn reasoning_metadata_changes_full_prompt_edges_without_equal_example_masking() {
    let rows = vec![detailed_record("a", "41"), detailed_record("b", "42")];
    let mut declared = declaration(&rows);
    declared.policy.multi_turn_loss = MultiTurnLoss::FinalTurnOnly;
    let sets = protected();
    let before = prepare_screening(&rows, &declared, &sets, None).unwrap();
    assert_eq!(before.groups.len(), 1);
    assert!(
        before
            .edges
            .iter()
            .any(|edge| edge.kind == "exact_complete_prompt")
    );
    assert!(
        !before
            .edges
            .iter()
            .any(|edge| edge.kind == "exact_training_example")
    );
    for field in ["id", "index", "signature", "format"] {
        let mut changed = rows.clone();
        change_detail(&mut changed[1], field);
        let after = prepare_screening(&changed, &declared, &sets, None).unwrap();
        assert!(after.incomplete.is_empty());
        assert_eq!(after.groups.len(), 2, "{field}");
        assert!(after.edges.is_empty(), "{field}");
    }
}

#[test]
fn changed_contents_claimed_hashes_and_forged_reports_cannot_certify_themselves() {
    let rows = vec![record("a", "ordinary question")];
    let declared = declaration(&rows);
    let sets = protected();
    let plan = prepare_screening(&rows, &declared, &sets, None).unwrap();
    let mut forged = plan.clone();
    forged.groups.clear();
    forged.edges.clear();
    forged.plan_id = String::new();
    forged.plan_id =
        gw_storage::canonical_json_hash(&serde_json::json!(["frozen-screening-plan-v2", &forged]))
            .unwrap();
    assert!(validate_screening_plan(&rows, &sets, &forged).is_err());
    assert!(prepare_screening(&rows, &declared, &sets, Some(&forged)).is_err());
    let mut changed = sets;
    changed[0].items[0].responses[0].content = Content::Text("substituted content".into());
    assert!(
        prepare_screening(&rows, &declared, &changed, None)
            .unwrap_err()
            .to_string()
            .contains("digest")
    );
    let mut changed = rows;
    changed[0].messages[0].content = Content::Text("substituted task prompt".into());
    let report = prepare_screening(&changed, &declared, &protected(), None).unwrap();
    assert!(
        report
            .incomplete
            .iter()
            .any(|r| r.code == "invalid_task_declaration")
    );
}

#[test]
fn cyclic_lineage_is_incomplete_and_never_recurses() {
    let mut rows = vec![
        record("a", "first question"),
        record("b", "second question"),
    ];
    rows[0].provenance.parent_ids = vec!["b".into()];
    rows[1].provenance.parent_ids = vec!["a".into()];
    let plan = prepare_screening(&rows, &declaration(&rows), &protected(), None).unwrap();
    assert!(
        plan.incomplete
            .iter()
            .any(|reason| reason.code == "cyclic_parent_lineage")
    );
    assert!(plan.eligible_output.is_empty());
}

#[test]
fn an_unknown_relative_split_is_missing_evidence_not_a_claimed_split_conflict() {
    let a = record("a", "ordinary question");
    let mut b = record("b", "another question");
    b.task_provenance = None;
    b.provenance.parent_ids = vec!["a".into()];
    let rows = vec![a, b];
    let plan = prepare_screening(&rows, &declaration(&rows), &protected(), None).unwrap();
    assert_eq!(plan.lexical_status, LexicalScreeningStatus::Incomplete);
    assert_eq!(plan.groups.len(), 1);
    assert_eq!(plan.groups[0].split, None);
    assert!(
        !plan.groups[0]
            .reasons
            .iter()
            .any(|reason| reason.code == "conflicting_split_assignments")
    );
}
