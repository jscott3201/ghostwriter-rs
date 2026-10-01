mod screening_support;
use gw_eval::screening::*;
use gw_schema::*;
use screening_support::*;

#[test]
fn exact_threshold_bits_survive_typed_value_and_plan_replay() {
    let rows = vec![record("a", "ordinary independent question")];
    let mut declared = declaration(&rows);
    let mut identities = std::collections::BTreeSet::new();
    for threshold in [
        -0.0,
        0.0,
        0.3333333333333333,
        f64::from_bits(0.3333333333333333_f64.to_bits() + 1),
    ] {
        declared.policy.jaccard_threshold = threshold;
        let encoded = serde_json::to_vec(&declared).unwrap();
        let decoded: ScreeningDeclaration = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(
            decoded.policy.jaccard_threshold.to_bits(),
            threshold.to_bits()
        );
        let value: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
        let from_value: ScreeningDeclaration = serde_json::from_value(value).unwrap();
        assert_eq!(
            from_value.policy.jaccard_threshold.to_bits(),
            threshold.to_bits()
        );
        let plan = prepare_screening(&rows, &from_value, &protected(), None).unwrap();
        assert!(identities.insert(plan.policy_id.clone()));
        let value = serde_json::to_value(&plan).unwrap();
        let round: FrozenScreeningPlan = serde_json::from_value(value).unwrap();
        assert_eq!(
            serde_json::to_vec(&plan).unwrap(),
            serde_json::to_vec(&round).unwrap()
        );
        validate_screening_plan(&rows, &protected(), &round).unwrap();
    }
}

#[test]
fn ambiguous_numeric_wire_and_requested_unsupported_modes_reject() {
    let rows = vec![record("a", "ordinary independent question")];
    let declared = declaration(&rows);
    let json = serde_json::to_string(&declared).unwrap();
    let tag = r#"{"binary64":"3fe999999999999a"}"#;
    assert!(json.contains(tag));
    for replacement in [
        r#"{"binary64":"3fe999999999999a","binary64":"3fe999999999999a"}"#,
        r#"{"binary64":"7ff0000000000000","binary64":"3fe999999999999a"}"#,
        r#"{"binary64":0,"binary64":"3fe999999999999a"}"#,
        r#"{"binary64":"3fe999999999999A"}"#,
        r#"{"binary64":"7ff8000000000000"}"#,
        r#"{"binary64":"3fe999999999999a","extra":true}"#,
        "0.8",
    ] {
        assert!(
            serde_json::from_str::<ScreeningDeclaration>(&json.replace(tag, replacement)).is_err(),
            "{replacement}"
        );
    }
    for field in ["embedding_decontam", "semantic_required"] {
        let mut value = serde_json::to_value(&declared).unwrap();
        value["policy"][field] = serde_json::json!(true);
        assert!(serde_json::from_value::<ScreeningDeclaration>(value).is_err());
    }
}

#[test]
fn strict_protected_messages_do_not_drop_unknown_content_fields() {
    let original = serde_json::to_value(&protected()[0]).unwrap();
    for path in ["/items/0/prompt/0", "/items/0/responses/0"] {
        let mut value = original.clone();
        value.pointer_mut(path).unwrap()["hidden_payload"] =
            serde_json::json!("must not disappear");
        assert!(
            serde_json::from_value::<ProtectedScreeningSet>(value).is_err(),
            "unknown field at {path}"
        );
    }
}

#[test]
fn strict_protected_nested_fields_and_duplicate_message_fields_reject() {
    let mut set = protected().remove(0);
    set.items[0].responses[0].content = Content::Parts(vec![ContentPart::Text {
        text: "part".into(),
    }]);
    set.items[0].responses[0].reasoning_details = Some(vec![ReasoningDetail::Text {
        text: "detail".into(),
        signature: None,
        id: None,
        format: None,
        index: 0,
    }]);
    set.items[0].responses[0].tool_calls = Some(vec![ToolCall {
        id: Some("call".into()),
        function: FunctionCall {
            name: "tool".into(),
            arguments: serde_json::json!({"opaque":{"binary64":"not a number"}}),
            raw_arguments: None,
        },
    }]);
    let original = serde_json::to_value(&set).unwrap();
    for path in [
        "/items/0/responses/0/content/0",
        "/items/0/responses/0/reasoning_details/0",
        "/items/0/responses/0/tool_calls/0",
        "/items/0/responses/0/tool_calls/0/function",
    ] {
        let mut value = original.clone();
        value.pointer_mut(path).unwrap()["hidden_payload"] =
            serde_json::json!("must not disappear");
        assert!(
            serde_json::from_value::<ProtectedScreeningSet>(value).is_err(),
            "{path}"
        );
    }
    let round: ProtectedScreeningSet = serde_json::from_value(original).unwrap();
    assert_eq!(
        round.items[0].responses[0].tool_calls.as_ref().unwrap()[0]
            .function
            .arguments["opaque"]["binary64"],
        "not a number"
    );
    let raw = serde_json::to_string(&set).unwrap().replace(
        "\"role\":\"user\"",
        "\"role\":\"user\",\"role\":\"assistant\"",
    );
    assert!(serde_json::from_str::<ProtectedScreeningSet>(&raw).is_err());
}

#[test]
fn invalid_ranges_limits_and_duplicate_declared_ids_reject_before_work() {
    let rows = vec![record("a", "ordinary question")];
    let baseline = declaration(&rows);
    for variant in 0..8 {
        let mut declared = baseline.clone();
        match variant {
            0 => declared.policy.ngram = [1, u32::MAX],
            1 => declared.policy.ngram = [3, 2],
            2 => declared.policy.min_overlap_tokens = 0,
            3 => declared.policy.jaccard_threshold = f64::NAN,
            4 => declared.policy.limits.shingle_token_work = u64::MAX,
            5 => declared.policy.limits.comparisons = 0,
            6 => declared.runs.run_ids.push(declared.runs.run_ids[0].clone()),
            _ => declared.siblings.push(declared.siblings[0].clone()),
        }
        assert!(
            prepare_screening(&rows, &declared, &protected(), None).is_err(),
            "{variant}"
        );
    }
    let mut duplicated = rows.clone();
    duplicated.push(rows[0].clone());
    assert!(prepare_screening(&duplicated, &baseline, &protected(), None).is_err());
    let mut sets = protected();
    sets.push(sets[0].clone());
    assert!(prepare_screening(&rows, &baseline, &sets, None).is_err());
}
