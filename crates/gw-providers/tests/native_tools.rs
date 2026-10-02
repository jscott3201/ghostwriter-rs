//! Offline requests and local HTTP capture for the native function-tool boundary.
mod common;
#[path = "support/serving_profile.rs"]
mod support;
use futures::StreamExt;
use gw_providers::serving_profile::*;
use gw_providers::{
    ChatCompletionsProvider, ChatRequest, FunctionDefinition, Provider, ToolChoice, ToolConfig,
    ToolDefinition,
};
use gw_schema::{Content, FunctionCall, Message, ModelOperation, Role, ToolCall};
use serde_json::{Value, json};

fn tools() -> ToolConfig {
    ToolConfig {
        tools: vec![ToolDefinition::Function {
            function: FunctionDefinition {
                name: "read_file".into(),
                description: Some("Read a file".into()),
                parameters: json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}),
                strict: None,
            },
        }],
        tool_choice: Some(ToolChoice::Function("read_file".into())),
        parallel_tool_calls: Some(false),
    }
}
fn history() -> Vec<Message> {
    vec![
        Message {
            role: Role::Assistant,
            content: Content::Null,
            reasoning: Some("Read first".into()),
            reasoning_details: None,
            tool_calls: Some(vec![ToolCall {
                id: Some("a".into()),
                function: FunctionCall {
                    name: "read_file".into(),
                    arguments: json!({"path":"a"}),
                    raw_arguments: Some("{ \"path\" : \"a\" }".into()),
                },
            }]),
            tool_call_id: None,
            name: None,
        },
        Message {
            role: Role::Tool,
            content: Content::Text("contents".into()),
            reasoning: None,
            reasoning_details: None,
            tool_calls: None,
            tool_call_id: Some("a".into()),
            name: Some("read_file".into()),
        },
    ]
}
#[test]
fn native_preparation_requires_declared_support_and_changes_identity() {
    for dialect in [ServingDialect::OpenRouterV1, ServingDialect::VllmV1] {
        let mut profile = support::profile(dialect);
        let mut request = support::chat();
        request.controls = vec![support::required(ControlValue::Tools(tools()))];
        assert!(matches!(
            prepare_request(
                &profile,
                &support::semantics(&profile, ModelOperation::ChatCompletion),
                &request
            ),
            Err(ProfileError::UnsupportedRequiredControl(
                ProfileControl::Tools
            ))
        ));
        request.controls[0].required = false;
        let omitted = prepare_request(
            &profile,
            &support::semantics(&profile, ModelOperation::ChatCompletion),
            &request,
        )
        .unwrap();
        assert_eq!(omitted.degradations()[0].control, ProfileControl::Tools);
        assert!(
            serde_json::from_slice::<Value>(omitted.body())
                .unwrap()
                .get("tools")
                .is_none()
        );
        profile.behavior.capabilities.push(ProfileControl::Tools);
        request.controls[0].required = true;
        let supported = prepare_request(
            &profile,
            &support::semantics(&profile, ModelOperation::ChatCompletion),
            &request,
        )
        .unwrap();
        assert_ne!(omitted.body_digest(), supported.body_digest());
        assert_ne!(omitted.target(), supported.target());
        let body: Value = serde_json::from_slice(supported.body()).unwrap();
        assert_eq!(
            body["tool_choice"],
            json!({"type":"function","function":{"name":"read_file"}})
        );
        assert_eq!(body["parallel_tool_calls"], false);
        assert_eq!(
            body["tools"][0]["function"]["parameters"],
            json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]})
        );
        request.input = ProfileInput::Chat(history());
        let prepared = prepare_request(
            &profile,
            &support::semantics(&profile, ModelOperation::ChatCompletion),
            &request,
        )
        .unwrap();
        let body: Value = serde_json::from_slice(prepared.body()).unwrap();
        assert_eq!(
            body["messages"][0]["tool_calls"][0]["function"]["arguments"],
            "{ \"path\" : \"a\" }"
        );
        assert_eq!(body["messages"][1]["tool_call_id"], "a");
        assert_eq!(body["messages"][0]["reasoning"], "Read first");
    }
}
#[test]
fn invalid_selection_and_malformed_history_are_rejected() {
    let mut config = tools();
    config.tool_choice = Some(ToolChoice::Function("missing".into()));
    assert!(config.validate().is_err());
    let mut messages = history();
    messages[0].tool_calls.as_mut().unwrap()[0]
        .function
        .arguments = json!("{}");
    assert!(serde_json::to_vec(&ChatRequest::new("m", messages)).is_err());
    let mut messages = history();
    messages[0].tool_calls.as_mut().unwrap()[0]
        .function
        .raw_arguments = Some("{\"stale\":true}".into());
    let body = serde_json::to_value(ChatRequest::new("m", messages)).unwrap();
    assert_eq!(
        body["messages"][0]["tool_calls"][0]["function"]["arguments"],
        "{\"path\":\"a\"}"
    );
    assert!(
        body["messages"][0]["tool_calls"][0]["function"]
            .get("raw_arguments")
            .is_none()
    );
}
#[tokio::test]
async fn captured_http_bytes_and_attempt_digest_include_native_tools_and_history() {
    let server = common::Server::responses(vec![(
        200,
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
            .into(),
    )])
    .await;
    let store = gw_storage::Store::open_in_memory().await.unwrap();
    let ctx = common::context(
        &store,
        gw_schema::AttemptRole::Teacher,
        gw_schema::AttemptPurpose::Initial,
    )
    .await;
    let provider = ChatCompletionsProvider::builder()
        .base_url(&server.url)
        .build_with_key("fixture")
        .unwrap();
    let request = ChatRequest::new("teacher", history()).with_tools(tools());
    let expected = serde_json::to_vec(&request).unwrap();
    let call = ctx.call();
    let deltas: Vec<_> = provider
        .stream_chat_observed(request, call.clone())
        .await
        .unwrap()
        .collect()
        .await;
    assert!(deltas.iter().all(Result::is_ok));
    assert_eq!(*server.requests.lock().unwrap(), vec![expected.clone()]);
    let receipt = store.model_attempts("run").await.unwrap().pop().unwrap();
    assert_eq!(Some(receipt.attempt_id.clone()), call.attempt_id());
    assert_eq!(
        receipt.intent.request_digest,
        blake3::hash(&expected).to_hex().to_string()
    );
    assert_eq!(receipt.metadata.prompt_tokens, None);
    assert_eq!(receipt.metadata.cost_usd, gw_schema::ReportedCost::Missing);
}

#[test]
fn outbound_histories_validate_links_across_all_messages_without_requiring_results() {
    let valid = history();
    let mut dangling = valid.clone();
    dangling[1].tool_call_id = Some("missing".into());
    let premature = vec![valid[1].clone(), valid[0].clone()];
    let duplicate_result = vec![valid[0].clone(), valid[1].clone(), valid[1].clone()];
    let duplicate_call = vec![valid[0].clone(), valid[1].clone(), valid[0].clone()];
    let mut profile = support::profile(ServingDialect::VllmV1);
    profile.behavior.capabilities.push(ProfileControl::Tools);
    let semantics = support::semantics(&profile, ModelOperation::ChatCompletion);
    for messages in [dangling, premature, duplicate_result, duplicate_call] {
        assert!(serde_json::to_vec(&ChatRequest::new("m", messages.clone())).is_err());
        let request = ProfileRequest {
            input: ProfileInput::Chat(messages),
            controls: vec![],
        };
        assert_eq!(
            prepare_request(&profile, &semantics, &request).unwrap_err(),
            ProfileError::InvalidRequest
        );
    }
    for messages in [valid.clone(), vec![valid[0].clone()]] {
        assert!(serde_json::to_vec(&ChatRequest::new("m", messages.clone())).is_ok());
        let request = ProfileRequest {
            input: ProfileInput::Chat(messages),
            controls: vec![],
        };
        assert!(prepare_request(&profile, &semantics, &request).is_ok());
    }
}
