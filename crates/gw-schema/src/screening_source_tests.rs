use super::*;
use crate::{FunctionCall, ToolCall};
use serde_json::json;

fn message() -> Message {
    Message {
        role: Role::Assistant,
        content: Content::Null,
        reasoning: None,
        reasoning_details: None,
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }
}

#[test]
fn source_field_presence_preserves_empty_text_null_json_keys_and_raw_arguments() {
    let mut cases = vec![(message(), vec![])];
    let mut row = message();
    row.content = Content::Text(String::new());
    cases.push((row, vec![ScreeningField::Content]));
    let mut row = message();
    row.content = Content::Parts(vec![]);
    cases.push((row, vec![]));
    let mut row = message();
    row.role = Role::Tool;
    row.content = Content::Parts(vec![ContentPart::Text {
        text: String::new(),
    }]);
    cases.push((row, vec![ScreeningField::ToolResult]));
    let mut row = message();
    row.reasoning = Some(String::new());
    cases.push((row, vec![ScreeningField::Reasoning]));
    for detail in [
        ReasoningDetail::Text {
            text: String::new(),
            signature: None,
            id: None,
            format: None,
            index: 0,
        },
        ReasoningDetail::Summary {
            summary: String::new(),
            id: None,
            format: None,
            index: 0,
        },
    ] {
        let mut row = message();
        row.reasoning_details = Some(vec![detail]);
        cases.push((row, vec![ScreeningField::ReasoningDetail]));
    }
    let mut row = message();
    row.name = Some(String::new());
    cases.push((row, vec![ScreeningField::ToolName]));
    let mut row = message();
    row.tool_calls = Some(vec![]);
    cases.push((row, vec![]));
    for (arguments, raw, expected) in [
        (json!({}), None, vec![ScreeningField::ToolName]),
        (
            json!({"": 0}),
            None,
            vec![ScreeningField::ToolArguments, ScreeningField::ToolName],
        ),
        (
            json!({}),
            Some(String::new()),
            vec![ScreeningField::ToolArguments, ScreeningField::ToolName],
        ),
    ] {
        let mut row = message();
        row.tool_calls = Some(vec![ToolCall {
            id: None,
            function: FunctionCall {
                name: String::new(),
                arguments,
                raw_arguments: raw,
            },
        }]);
        cases.push((row, expected));
    }
    for (row, expected) in cases {
        let shape = classify_screening_source(std::slice::from_ref(&row), None);
        assert_eq!(shape.required_fields, expected, "{row:?}");
        assert!(shape.unsupported_reasons.is_empty(), "{row:?}");
    }
    for (definitions, expected) in [
        (vec![], vec![]),
        (
            vec![json!(null), json!(12), json!({}), json!([false, []])],
            vec![],
        ),
        (vec![json!({"": 0})], vec![ScreeningField::ToolDefinition]),
        (vec![json!([[""]])], vec![ScreeningField::ToolDefinition]),
    ] {
        assert_eq!(
            classify_screening_source(&[], Some(&definitions)).required_fields,
            expected
        );
    }
}

#[test]
fn unsupported_classification_retains_field_presence_without_claiming_complete_coverage() {
    let mut row = message();
    row.content = Content::Parts(vec![
        ContentPart::Text {
            text: String::new(),
        },
        ContentPart::InputAudio {
            audio_url: None,
            format: None,
        },
    ]);
    row.reasoning_details = Some(vec![ReasoningDetail::Encrypted {
        data: String::new(),
        id: None,
        format: None,
        index: 0,
    }]);
    row.tool_calls = Some(vec![ToolCall {
        id: None,
        function: FunctionCall {
            name: String::new(),
            arguments: json!([""]),
            raw_arguments: None,
        },
    }]);
    let shape = classify_screening_source(&[row], None);
    assert_eq!(
        shape.required_fields,
        vec![
            ScreeningField::Content,
            ScreeningField::ToolArguments,
            ScreeningField::ToolName
        ]
    );
    assert_eq!(
        shape.unsupported_reasons,
        vec![
            "encrypted_reasoning",
            "nonobject_tool_arguments",
            "unsupported_media"
        ]
    );
}
