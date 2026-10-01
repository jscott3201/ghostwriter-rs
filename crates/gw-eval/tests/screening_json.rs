mod screening_support;
use gw_eval::screening::*;
use gw_schema::*;
use screening_support::*;
use serde_json::{Value, json};

const PHRASE: &str = "alpha beta gamma delta epsilon zeta eta theta";

fn tool_turns(arguments: Value) -> Vec<Message> {
    let mut call = message(Role::Assistant, "");
    call.content = Content::Null;
    call.tool_calls = Some(vec![ToolCall {
        id: Some("read-1".into()),
        function: FunctionCall {
            name: "read_value".into(),
            arguments,
            raw_arguments: None,
        },
    }]);
    let mut result = message(Role::Tool, "observed value");
    result.tool_call_id = Some("read-1".into());
    result.name = Some("read_value".into());
    vec![call, result, message(Role::Assistant, "42")]
}

fn with_arguments(arguments: Value) -> TrainingRecord {
    let mut row = record("a", "ordinary tool task");
    row.messages.truncate(1);
    row.messages.extend(tool_turns(arguments));
    rebind_task(&mut row);
    gw_format::validate_tool_links(&row.messages).unwrap();
    row
}

fn protected_phrase() -> Vec<ProtectedScreeningSet> {
    let mut sets = protected();
    sets[0].items[0].responses = vec![message(Role::Assistant, PHRASE)];
    sets[0].content_digest = protected_screening_content_digest(&sets[0].items).unwrap();
    sets
}

fn assert_key_match(row: TrainingRecord) {
    let rows = vec![row];
    let plan = prepare_screening(&rows, &declaration(&rows), &protected_phrase(), None).unwrap();
    assert!(plan.incomplete.is_empty());
    assert_eq!(
        plan.lexical_status,
        LexicalScreeningStatus::MatchQuarantined
    );
    assert!(plan.eligible_output.is_empty());
    assert!(plan.groups[0].quarantined);
    // The key is the only eight-token source segment; a single distinct 8-gram matches.
    assert_eq!(plan.protected_matches.len(), 1);
    let found = &plan.protected_matches[0].evidence;
    assert_eq!((found.n, found.intersection, found.union), (Some(8), 1, 1));
}

#[test]
fn tool_argument_keys_are_independent_screened_text() {
    assert_key_match(with_arguments(json!({PHRASE: 1})));
}

#[test]
fn tool_definition_keys_are_independent_screened_text() {
    let mut row = record("a", "ordinary definition task");
    row.tools = Some(vec![json!({"parameters": {PHRASE: 1}})]);
    assert_key_match(row);
}

#[test]
fn protected_nested_object_keys_never_enter_report_coordinates() {
    let mut row = record("a", "ordinary reasoning task");
    row.messages[1].reasoning = Some(PHRASE.into());
    let rows = vec![row];
    let declared = declaration(&rows);
    let mut previous = None;
    // Reordered objects must retain the same coordinates and complete plan identity. Both
    // object levels contain protected key text, with slash/tilde characters in the nested case.
    for arguments in [
        r#"{"private protected phrase":"alpha beta gamma delta epsilon zeta eta theta","nested private/~phrase":[{"inner private/~phrase":"alpha beta gamma delta epsilon zeta eta theta","a":1}]}"#,
        r#"{"nested private/~phrase":[{"a":1,"inner private/~phrase":"alpha beta gamma delta epsilon zeta eta theta"}],"private protected phrase":"alpha beta gamma delta epsilon zeta eta theta"}"#,
    ] {
        let mut sets = protected();
        sets[0].items[0].responses = tool_turns(serde_json::from_str(arguments).unwrap());
        sets[0].content_digest = protected_screening_content_digest(&sets[0].items).unwrap();
        let plan = prepare_screening(&rows, &declared, &sets, None).unwrap();
        assert_eq!(
            plan.lexical_status,
            LexicalScreeningStatus::MatchQuarantined
        );
        assert_eq!(plan.protected_matches.len(), 2);
        let serialized = serde_json::to_string(&plan).unwrap();
        for protected_text in [
            PHRASE,
            "private protected phrase",
            "nested private",
            "inner private",
        ] {
            assert!(
                !serialized.contains(protected_text),
                "leaked protected text: {protected_text}"
            );
        }
        if let Some(previous) = &previous {
            assert_eq!(&serialized, previous);
        }
        previous = Some(serialized);
    }
}

#[test]
fn keys_values_and_nested_members_do_not_share_shingles() {
    let rows = vec![with_arguments(json!({
        "alpha beta gamma delta": "epsilon zeta eta theta",
        "nested": {"alpha beta gamma delta": 1, "epsilon zeta eta theta": 2}
    }))];
    let plan = prepare_screening(&rows, &declaration(&rows), &protected_phrase(), None).unwrap();
    assert_eq!(plan.lexical_status, LexicalScreeningStatus::CompleteNoMatch);
    assert!(plan.protected_matches.is_empty());
    assert_eq!(plan.eligible_output, vec![key(&rows[0])]);
}

#[test]
fn object_key_text_obeys_segment_limits() {
    let rows = vec![with_arguments(json!({"x".repeat(100): 1}))];
    let mut declared = declaration(&rows);
    declared.policy.limits.segment_bytes = 99;
    let plan = prepare_screening(&rows, &declared, &protected(), None).unwrap();
    assert_eq!(plan.lexical_status, LexicalScreeningStatus::Incomplete);
    assert!(
        plan.incomplete
            .iter()
            .any(|issue| issue.code == "segment_bytes_limit")
    );
    assert!(plan.eligible_output.is_empty());
}

#[test]
fn object_keys_remain_exact_prompt_structure() {
    let a = with_arguments(json!({"source": 1}));
    let mut b = a.clone();
    b.record_id = "b".into();
    b.origin
        .generated_mut()
        .expect("generated record")
        .provenance
        .run_id = "run-b".into();
    let task = b.task_provenance.as_mut().unwrap();
    task.task_id = "b".into();
    task.source.item = "b".into();
    task.group.id = "b".into();
    b.messages.last_mut().unwrap().content = Content::Text("43".into());
    let mut rows = vec![a, b];
    let mut declared = declaration(&rows);
    declared.policy.multi_turn_loss = MultiTurnLoss::FinalTurnOnly;
    let before = prepare_screening(&rows, &declared, &protected(), None).unwrap();
    assert_eq!(before.groups.len(), 1);
    assert!(
        before
            .edges
            .iter()
            .any(|edge| edge.kind == "exact_complete_prompt")
    );
    rows[1].messages[1].tool_calls.as_mut().unwrap()[0]
        .function
        .arguments = json!({"different": 1});
    rebind_task(&mut rows[1]);
    declared = declaration(&rows);
    declared.policy.multi_turn_loss = MultiTurnLoss::FinalTurnOnly;
    let after = prepare_screening(&rows, &declared, &protected(), None).unwrap();
    assert!(after.incomplete.is_empty());
    assert_eq!(after.groups.len(), 2);
    assert!(after.edges.is_empty());
}
