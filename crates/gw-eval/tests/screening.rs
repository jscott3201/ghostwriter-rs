mod screening_support;
use gw_eval::screening::*;
use gw_schema::*;
use screening_support::*;

#[test]
fn exact_complete_prompts_group_across_runs_sources_and_tags() {
    let a = record("a", "How much is twelve plus thirty?");
    let mut b = record("b", "How much is twelve plus thirty?");
    b.tags = vec!["different".into()];
    b.training_area = "other".into();
    b.messages[1].content = Content::Text("a different answer".into());
    let rows = vec![a, b];
    let plan = prepare_screening(&rows, &declaration(&rows), &protected(), None).unwrap();
    assert_eq!(plan.groups.len(), 1);
    assert_eq!(plan.groups[0].members, vec![key(&rows[0]), key(&rows[1])]);
    assert_eq!(plan.lexical_status, LexicalScreeningStatus::CompleteNoMatch);
    assert_eq!(plan.semantic_status, SemanticScreeningStatus::NotRun);
    assert_eq!(
        plan.population_check,
        ScreeningPopulationCheck::SuppliedFilesOnly
    );
    validate_screening_plan(&rows, &protected(), &plan).unwrap();
}

#[test]
fn equal_short_answers_do_not_join_distinct_tasks() {
    let rows = vec![
        record("a", "How many pears remain?"),
        record("b", "What is the length in metres?"),
    ];
    let plan = prepare_screening(&rows, &declaration(&rows), &protected(), None).unwrap();
    assert_eq!(plan.groups.len(), 2);
    assert!(plan.edges.is_empty());
}

#[test]
fn complete_prefix_units_follow_all_assistant_and_final_only_policies() {
    let short = record("short", "initial independent question");
    let mut long = record("long", "initial independent question");
    long.messages.extend([
        message(Role::User, "a later unrelated question"),
        message(Role::Assistant, "99"),
    ]);
    rebind_task(&mut long);
    let rows = vec![short, long];
    let mut declared = declaration(&rows);
    let all = prepare_screening(&rows, &declared, &protected(), None).unwrap();
    assert_eq!(all.groups.len(), 1);
    assert!(
        all.edges
            .iter()
            .any(|edge| edge.kind == "exact_training_example")
    );
    assert!(
        all.edges
            .iter()
            .any(|edge| edge.kind == "exact_complete_prompt")
    );
    declared.policy.multi_turn_loss = MultiTurnLoss::FinalTurnOnly;
    let final_only = prepare_screening(&rows, &declared, &protected(), None).unwrap();
    assert_eq!(final_only.groups.len(), 2);
    assert!(final_only.edges.is_empty());
    assert_ne!(all.policy_id, final_only.policy_id);
}

#[test]
fn final_prefix_keeps_prior_assistant_reasoning_and_tool_history() {
    let mut a = record("a", "initial question");
    let mut b = record("b", "initial question");
    for row in [&mut a, &mut b] {
        row.messages.extend([
            message(Role::User, "the same final question"),
            message(Role::Assistant, "7"),
        ]);
        rebind_task(row);
    }
    a.messages[1].reasoning = Some("different earlier account".into());
    b.messages[1].reasoning = Some("another earlier account".into());
    let rows = vec![a, b];
    let mut declared = declaration(&rows);
    declared.policy.multi_turn_loss = MultiTurnLoss::FinalTurnOnly;
    // Supervised export retains those source-distinct reasoning prefixes. Actual model-specific
    // tokenizer/templates may erase them; the source report must explicitly leave that unknown.
    let plan = prepare_screening(&rows, &declared, &protected(), None).unwrap();
    assert_eq!(plan.groups.len(), 2);
    assert_eq!(
        plan.effective_prompt_separation,
        EffectivePromptSeparation::Unknown
    );
    assert_eq!(
        plan.lexical_scope,
        ScreeningLexicalScope::CanonicalSourceAndPinnedExportPolicy
    );
}

#[test]
fn shared_boilerplate_or_one_reasoning_segment_is_not_a_grouping_edge() {
    let text = "shared system instruction with enough common boilerplate to exceed eight tokens";
    let mut a = record("a", "calculate a sum for pears");
    let mut b = record("b", "find the capital city now");
    for row in [&mut a, &mut b] {
        row.messages.insert(1, message(Role::System, text));
        row.messages.last_mut().unwrap().reasoning = Some(text.into());
        rebind_task(row);
    }
    let rows = vec![a, b];
    let plan = prepare_screening(&rows, &declaration(&rows), &protected(), None).unwrap();
    assert_eq!(plan.groups.len(), 2);
    assert!(plan.edges.is_empty());
}

#[test]
fn order_is_canonical_and_out_of_scope_records_do_not_change_the_claim() {
    let rows = vec![
        record("a", "one question"),
        record("b", "different question"),
    ];
    let declared = declaration(&rows);
    let sets = protected();
    let plan = prepare_screening(&rows, &declared, &sets, None).unwrap();
    let mut reversed = rows.clone();
    reversed.reverse();
    reversed.push(record("outside", "one question"));
    let mut unordered = declared.clone();
    unordered.runs.run_ids.reverse();
    unordered.expected_tasks.reverse();
    unordered.siblings.reverse();
    let mut protected_reversed = sets;
    protected_reversed.reverse();
    let reordered = prepare_screening(&reversed, &unordered, &protected_reversed, None).unwrap();
    assert_eq!(
        serde_json::to_vec(&plan).unwrap(),
        serde_json::to_vec(&reordered).unwrap()
    );
}

#[test]
fn declared_sibling_completion_preserves_the_frozen_group_and_split() {
    let mut a = record("a", "the shared sibling prompt");
    a.origin
        .generated_mut()
        .expect("generated record")
        .generation
        .n_completions = Some(2);
    let mut b = a.clone();
    b.record_id = "b".into();
    b.origin
        .generated_mut()
        .expect("generated record")
        .generation
        .completion_index = Some(1);
    b.messages[1].content = Content::Text("different sibling answer".into());
    b.lifecycle.state = LifecycleState::Rejected;
    let all = vec![a.clone(), b];
    let complete_declaration = declaration(&all);
    let previous = prepare_screening(&[a], &complete_declaration, &protected(), None).unwrap();
    assert_eq!(previous.lexical_status, LexicalScreeningStatus::Incomplete);
    let complete =
        prepare_screening(&all, &complete_declaration, &protected(), Some(&previous)).unwrap();
    assert_eq!(
        complete.lexical_status,
        LexicalScreeningStatus::CompleteNoMatch
    );
    assert_eq!(complete.groups.len(), 1);
    assert_eq!(complete.groups[0].group_id, previous.groups[0].group_id);
    assert_eq!(complete.groups[0].split, previous.groups[0].split);
    assert_eq!(complete.eligible_output, vec![key(&all[0])]);
    assert!(
        complete.exclusions[0]
            .reasons
            .contains(&"not_selected_admitted".into())
    );
}

#[test]
fn an_extension_joining_prior_groups_is_an_explicit_persistent_conflict() {
    let rows = vec![record("a", "a b c"), record("b", "c d e")];
    let mut declared = declaration(&rows);
    declared.policy.ngram = [2, 2];
    declared.policy.min_overlap_tokens = 2;
    declared.policy.jaccard_threshold = 0.5;
    let old = prepare_screening(&rows, &declared, &protected(), None).unwrap();
    assert_eq!(old.groups.len(), 2);
    let mut extended = rows;
    extended.push(record("bridge", "a b c d e"));
    let mut next = declaration(&extended);
    next.policy = declared.policy.clone();
    let joined = prepare_screening(&extended, &next, &protected(), Some(&old)).unwrap();
    assert_eq!(joined.groups.len(), 1);
    assert!(joined.groups[0].quarantined);
    assert!(
        joined.groups[0]
            .reasons
            .iter()
            .any(|r| r.code == "joined_prior_groups")
    );
    assert!(joined.eligible_output.is_empty());
    extended.push(record("unrelated", "something entirely distinct"));
    let mut third = declaration(&extended);
    third.policy = declared.policy;
    let later = prepare_screening(&extended, &third, &protected(), Some(&joined)).unwrap();
    assert!(
        later
            .groups
            .iter()
            .any(|g| g.group_id == joined.groups[0].group_id
                && g.reasons.iter().any(|r| r.code == "joined_prior_groups"))
    );
}

#[test]
fn held_out_or_excluded_relatives_quarantine_the_entire_conflicting_component() {
    let a = record("a", "common selected prompt");
    let mut b = record("b", "common selected prompt");
    b.task_provenance.as_mut().unwrap().split.role = TaskSplitRole::Test;
    b.lifecycle.state = LifecycleState::Rejected;
    let rows = vec![a, b];
    let plan = prepare_screening(&rows, &declaration(&rows), &protected(), None).unwrap();
    assert_eq!(plan.groups.len(), 1);
    assert!(plan.groups[0].quarantined);
    assert_eq!(plan.groups[0].split, None);
    assert!(
        plan.groups[0]
            .reasons
            .iter()
            .any(|r| r.code == "conflicting_split_assignments")
    );
    assert_eq!(plan.population.len(), 2);
    assert!(plan.eligible_output.is_empty());
}

#[test]
fn final_prompt_retains_linked_tool_history() {
    let mut rows = vec![
        record("a", "initial tool task"),
        record("b", "initial tool task"),
    ];
    for (i, row) in rows.iter_mut().enumerate() {
        row.messages[1].content = Content::Null;
        row.messages[1].tool_calls = Some(vec![ToolCall {
            id: Some("read-1".into()),
            function: FunctionCall {
                name: "read_value".into(),
                arguments: serde_json::json!({"source":i}),
                raw_arguments: None,
            },
        }]);
        let mut result = message(Role::Tool, "observed value");
        result.tool_call_id = Some("read-1".into());
        result.name = Some("read_value".into());
        row.messages.extend([
            result,
            message(Role::User, "same final question"),
            message(Role::Assistant, "42"),
        ]);
        rebind_task(row);
    }
    let mut declared = declaration(&rows);
    declared.policy.multi_turn_loss = MultiTurnLoss::FinalTurnOnly;
    let plan = prepare_screening(&rows, &declared, &protected(), None).unwrap();
    assert_eq!(plan.lexical_status, LexicalScreeningStatus::CompleteNoMatch);
    assert_eq!(plan.groups.len(), 2);
    assert!(plan.edges.is_empty());
}

#[test]
fn exact_source_prompts_preserve_null_and_empty_but_equal_emitted_examples_are_reported() {
    let mut rows = vec![record("a", "first question"), record("b", "first question")];
    rows[0].messages[1].content = Content::Null;
    rows[1].messages[1].content = Content::Text(String::new());
    for row in &mut rows {
        row.messages.extend([
            message(Role::User, "same final question"),
            message(Role::Assistant, "42"),
        ]);
        rebind_task(row);
    }
    let mut declared = declaration(&rows);
    declared.policy.multi_turn_loss = MultiTurnLoss::FinalTurnOnly;
    let plan = prepare_screening(&rows, &declared, &protected(), None).unwrap();
    assert!(
        !plan
            .edges
            .iter()
            .any(|edge| edge.kind == "exact_complete_prompt"
                || edge.kind == "lexical_complete_prompt")
    );
    // The existing OpenAI-messages renderer explicitly maps null content to empty text. Its
    // complete emitted examples therefore match, with that distinct edge named in the report.
    assert!(
        plan.edges
            .iter()
            .any(|edge| edge.kind == "exact_training_example")
    );
    assert_eq!(plan.groups.len(), 1);
}

#[test]
fn declared_source_revisions_and_cross_run_parent_edges_are_kept() {
    let a = record("a", "first question");
    let mut revision = record("revision", "revised independent question");
    revision.task_provenance.as_mut().unwrap().source.item =
        a.task_provenance.as_ref().unwrap().source.item.clone();
    revision.task_provenance.as_mut().unwrap().source.revision = "revision-two".into();
    rebind_task(&mut revision);
    let mut child = record("child", "another independent task");
    child
        .origin
        .generated_mut()
        .expect("generated fixture")
        .provenance
        .parent_ids = vec![revision.record_id.clone()];
    let rows = vec![a, revision, child];
    let plan = prepare_screening(&rows, &declaration(&rows), &protected(), None).unwrap();
    assert_eq!(plan.groups.len(), 1);
    assert!(
        plan.edges
            .iter()
            .any(|edge| edge.kind == "source_item_revision")
    );
    assert!(plan.edges.iter().any(|edge| edge.kind == "declared_parent"));
}
