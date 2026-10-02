use super::*;
use serde_json::json;
fn source() -> Value {
    serde_json::from_str(include_str!("../tests/fixtures/tool-training-source.json")).unwrap()
}
fn validate(value: &Value) -> Result<()> {
    let messages: Vec<Message> = serde_json::from_value(value["messages"].clone()).unwrap();
    let tools: Option<Vec<Value>> = serde_json::from_value(value["tools"].clone()).unwrap();
    validate_tool_training_source(&messages, tools.as_deref())
}
#[test]
fn parallel_same_name_reversed_results_are_complete_sources() {
    validate(&source()).unwrap();
}
#[test]
fn incomplete_or_unsupported_sources_fail_without_coercion() {
    for mutation in 0..16 {
        let mut value = source();
        match mutation {
            0 => value["tools"] = Value::Null,
            1 => {
                let duplicate = value["tools"][0].clone();
                value["tools"].as_array_mut().unwrap().push(duplicate);
            }
            2 => value["tools"][0]["function"]["name"] = json!(""),
            3 => {
                value["messages"].as_array_mut().unwrap().remove(3);
            }
            4 => {
                let duplicate = value["messages"][2].clone();
                value["messages"].as_array_mut().unwrap().push(duplicate);
            }
            5 => {
                value["messages"][2]["tool_call_id"] = json!("absent");
            }
            6 => {
                value["messages"].as_array_mut().unwrap().swap(1, 2);
            }
            7 => {
                value["messages"][1]["role"] = json!("user");
            }
            8 => {
                value["messages"][2]["name"] = json!("wrong");
            }
            9 => {
                value["messages"][1]["tool_calls"][0]["function"]["arguments"] = json!("{}");
            }
            10 => {
                value["messages"][1]["tool_calls"][1]["id"] = json!("a");
            }
            11 => {
                value["messages"][2]["tool_call_id"] = Value::Null;
            }
            12 => {
                value["messages"][0]["content"] =
                    json!([{"type":"image_url","image_url":"https://example.invalid/image"}]);
            }
            13 => {
                value["tools"][0]["function"]["parameters"]["$ref"] = json!("remote");
            }
            14 => {
                value["tools"][0]["type"] = json!("custom");
            }
            _ => {
                value["tools"][0]["function"]["parameters"]["properties"]["query"]["type"] =
                    json!(["string", "null"]);
            }
        };
        assert!(validate(&value).is_err(), "mutation {mutation}");
    }
}
#[test]
fn permissive_capture_and_projection_ambiguity_remain_distinct() {
    let mut value = source();
    value["messages"].as_array_mut().unwrap().remove(3);
    let messages: Vec<Message> = serde_json::from_value(value["messages"].clone()).unwrap();
    crate::validate_tool_links(&messages).unwrap();
    assert!(validate(&value).is_err());
    for mutation in 0..5 {
        let mut value = source();
        match mutation {
            0 => value["tools"][0]["function"]["description"] = json!("<|tool_call>"),
            1 => {
                value["tools"][0]["function"]["parameters"]["properties"]["<|tool_call>"] =
                    json!({"type":"string"})
            }
            2 => {
                value["messages"][1]["tool_calls"][0]["function"]["arguments"]["nested"] =
                    json!({"key":["<|tool_call>"]})
            }
            3 => value["messages"][1]["reasoning"] = json!("<|tool_call>"),
            _ => value["messages"][0]["content"] = json!("<|tool_call>"),
        }
        validate(&value).unwrap();
        let messages: Vec<Message> = serde_json::from_value(value["messages"].clone()).unwrap();
        assert!(
            validate_tool_projection_delimiters(
                &messages,
                value["tools"].as_array().map(Vec::as_slice),
                &["<|tool_call>"]
            )
            .is_err()
        );
    }
    let mut value = source();
    value["messages"][1]["tool_calls"][0]["function"]["raw_arguments"] = json!("<|tool_call>");
    let messages: Vec<Message> = serde_json::from_value(value["messages"].clone()).unwrap();
    validate_tool_projection_delimiters(
        &messages,
        value["tools"].as_array().map(Vec::as_slice),
        &["<|tool_call>"],
    )
    .unwrap();
}
