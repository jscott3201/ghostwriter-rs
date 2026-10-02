use gw_providers::serving_profile::*;
use serde_json::{Value, json};
fn chunk(delta: Value) -> Value {
    json!({"choices":[{"index":0,"delta":delta,"finish_reason":null}]})
}
fn decode(value: Value) -> NormalizedProfileChunk {
    normalize_chat_chunk(&value.to_string()).unwrap()
}

#[test]
fn reasoning_aliases_are_equivalent_without_duplicate_text_or_content_inlining() {
    for delta in [
        json!({"content":"answer","reasoning":"steps"}),
        json!({"content":"answer","reasoning_content":"steps"}),
        json!({"content":"answer","reasoning":"steps","reasoning_content":"steps"}),
    ] {
        let result = decode(chunk(delta));
        assert_eq!(result.content.as_deref(), Some("answer"));
        assert_eq!(result.reasoning.as_deref(), Some("steps"));
        assert!(result.usage.is_none());
        assert_eq!(result.termination.kind, ProfileTerminationKind::Unspecified);
    }
    for (a, b) in [("steps-a", "steps-b"), ("", "steps"), ("steps", "")] {
        assert_eq!(
            normalize_chat_chunk(&chunk(json!({"reasoning":a,"reasoning_content":b})).to_string())
                .unwrap_err(),
            ProfileError::ConflictingReasoning
        );
    }
    let empty = decode(chunk(
        json!({"content":"","reasoning":"","reasoning_content":""}),
    ));
    assert_eq!(empty.content.as_deref(), Some(""));
    assert_eq!(empty.reasoning.as_deref(), Some(""));
    let absent = decode(chunk(json!({"role":"assistant"})));
    assert_eq!(absent.content, None);
    assert_eq!(absent.reasoning, None);
}

#[test]
fn structured_reasoning_objects_are_preserved_without_flattening_or_loss() {
    let details = json!([
        {"type":"reasoning.text","text":"detail text","index":0,"id":"block-a","extension":{"format":"custom"}},
        {"type":"reasoning.encrypted","data":"opaque","index":1},
        {"type":"future.reasoning","opaque":[1,2],"index":2}
    ]);
    let result = decode(chunk(
        json!({"content":"answer","reasoning_content":"flat","reasoning_details":details}),
    ));
    assert_eq!(result.content.as_deref(), Some("answer"));
    assert_eq!(result.reasoning.as_deref(), Some("flat"));
    assert_eq!(
        serde_json::to_value(result.reasoning_details).unwrap(),
        details
    );
    for bad in [json!("text"), json!(["text"]), json!([null])] {
        assert_eq!(
            normalize_chat_chunk(&chunk(json!({"reasoning_details":bad})).to_string()).unwrap_err(),
            ProfileError::InvalidResponse
        );
    }
}

#[test]
fn missing_usage_is_not_zero_and_text_never_implies_tokenizer_truth() {
    let missing = decode(chunk(
        json!({"reasoning":"nonempty steps","content":"answer"}),
    ));
    assert!(missing.usage.is_none());
    let zero = decode(
        json!({"choices":[],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0,"completion_tokens_details":{"reasoning_tokens":0},"cost":0.0}}),
    );
    let usage = zero.usage.unwrap();
    assert_eq!(usage.prompt_tokens, Some(0));
    assert_eq!(usage.completion_tokens, Some(0));
    assert_eq!(usage.reasoning_tokens, Some(0));
    assert_eq!(usage.total_tokens, Some(0));
    assert_eq!(usage.cost_usd, Some(0.0));
    let partial = decode(json!({"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":9}}))
        .usage
        .unwrap();
    assert_eq!(partial.prompt_tokens, Some(7));
    assert_eq!(partial.completion_tokens, Some(9));
    assert_eq!(partial.total_tokens, None);
    assert_eq!(partial.reasoning_tokens, None);
    assert_eq!(partial.cost_usd, None);
    let empty = decode(json!({"choices":[],"usage":{}})).usage.unwrap();
    assert_eq!(empty.cost_usd, None);
    assert_eq!(empty.total_tokens, None);
    for usage in [
        json!({"prompt_tokens":-1}),
        json!({"completion_tokens":1.5}),
        json!({"cost":-0.1}),
        json!({"prompt_tokens":"zero"}),
    ] {
        assert_eq!(
            normalize_chat_chunk(&json!({"choices":[],"usage":usage}).to_string()).unwrap_err(),
            ProfileError::InvalidResponse
        );
    }
}

#[test]
fn native_normalized_length_and_error_termination_remain_distinct_observations() {
    for (finish, native, kind) in [
        ("stop", "eos_token", ProfileTerminationKind::Stop),
        ("length", "max_tokens", ProfileTerminationKind::Length),
        (
            "content_filter",
            "policy",
            ProfileTerminationKind::ContentFilter,
        ),
        ("tool_calls", "tool_end", ProfileTerminationKind::ToolCalls),
        ("error", "backend_error", ProfileTerminationKind::Error),
        (
            "future_reason",
            "native_future",
            ProfileTerminationKind::Other,
        ),
    ] {
        let result = decode(
            json!({"id":"response-1","model":"reported-alias","provider":"reported-route",
            "choices":[{"delta":{"content":"answer"},"finish_reason":finish,"native_finish_reason":native}]}),
        );
        assert_eq!(result.termination.kind, kind);
        assert_eq!(result.termination.finish_reason.as_deref(), Some(finish));
        assert_eq!(
            result.termination.native_finish_reason.as_deref(),
            Some(native)
        );
        assert_eq!(result.model.as_deref(), Some("reported-alias"));
        assert!(result.usage.is_none());
    }
    let error = json!({"code":503,"message":"cold-start-like fixture"});
    let result = decode(json!({"error":error}));
    assert_eq!(result.error, Some(error));
    assert_eq!(result.termination.kind, ProfileTerminationKind::Error);
    assert!(result.termination.finish_reason.is_none());
    assert!(result.usage.is_none());
    let result =
        decode(json!({"choices":[{"finish_reason":"stop"}],"error":{"code":"backend_error"}}));
    assert_eq!(result.termination.kind, ProfileTerminationKind::Error);
    assert_eq!(result.termination.finish_reason.as_deref(), Some("stop"));
    let native_only = decode(json!({"choices":[{"native_finish_reason":"eos_token"}]}));
    assert_eq!(
        native_only.termination.kind,
        ProfileTerminationKind::Unspecified
    );
    assert_eq!(
        native_only.termination.native_finish_reason.as_deref(),
        Some("eos_token")
    );
}

#[test]
fn ambiguous_choices_invalid_known_fields_and_mixed_delta_message_are_rejected() {
    for value in [
        json!({}),
        json!({"choices":[{},{}]}),
        json!({"choices":[{"index":1,"delta":{}}]}),
        json!({"choices":[{"delta":{},"message":{}}]}),
        chunk(json!({"reasoning":3})),
        chunk(json!({"content":false})),
    ] {
        assert_eq!(
            normalize_chat_chunk(&value.to_string()).unwrap_err(),
            ProfileError::InvalidResponse
        );
    }
    let full = decode(
        json!({"choices":[{"message":{"content":"answer","reasoning_content":"steps"},"finish_reason":"stop"}]}),
    );
    assert_eq!(full.reasoning.as_deref(), Some("steps"));
    assert_eq!(full.content.as_deref(), Some("answer"));
    let malformed = normalize_chat_chunk("sensitive-sentinel").unwrap_err();
    assert!(!format!("{malformed:?} {malformed}").contains("sensitive-sentinel"));
}

#[test]
fn complete_message_calls_use_array_order_but_stream_fragments_require_indices() {
    let calls = json!([
        {"id":"a","type":"function","function":{"name":"read_file","arguments":"{ \"path\": \"a\" }"}},
        {"id":"b","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"b\"}"}}
    ]);
    let full = decode(
        json!({"choices":[{"message":{"role":"assistant","content":null,"reasoning_content":"two reads","tool_calls":calls},"finish_reason":"tool_calls"}]}),
    );
    let normalized = full.tool_calls.unwrap();
    assert_eq!(
        normalized.iter().map(|v| v.index).collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(normalized[0].id.as_deref(), Some("a"));
    assert_eq!(normalized[1].id.as_deref(), Some("b"));
    assert_eq!(
        normalized[0]
            .function
            .as_ref()
            .unwrap()
            .arguments
            .as_deref(),
        Some("{ \"path\": \"a\" }")
    );
    assert_eq!(full.content, None);
    assert_eq!(full.reasoning.as_deref(), Some("two reads"));
    assert_eq!(full.termination.kind, ProfileTerminationKind::ToolCalls);
    assert_eq!(
        normalize_chat_chunk(&chunk(json!({"tool_calls":calls})).to_string()).unwrap_err(),
        ProfileError::InvalidResponse
    );
    let mut indexed = calls;
    indexed[0]["index"] = json!(1);
    indexed[1]["index"] = json!(0);
    let fragments = decode(chunk(json!({"tool_calls":indexed})))
        .tool_calls
        .unwrap();
    assert_eq!(
        fragments.iter().map(|v| v.index).collect::<Vec<_>>(),
        vec![1, 0]
    );
}
