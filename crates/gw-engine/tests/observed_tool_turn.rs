//! Collector boundary acceptance through local HTTP and canonical SQLite persistence.
#[path = "../../gw-providers/tests/common/mod.rs"]
mod common;
use common::*;
use futures::StreamExt;
use gw_generate::generate_turn_observed;
use gw_providers::{
    ChatCompletionsProvider, ChatRequest, FunctionDefinition, Provider, RetryPolicy, ToolChoice,
    ToolConfig, ToolDefinition,
};
use gw_schema::{
    AttemptPurpose, AttemptRole, OutputInterpretation, ReportedCost, TrainingRecord,
    TransportOutcome,
};
use gw_storage::Store;
use serde_json::json;
use std::{sync::atomic::Ordering, time::Duration};

fn request() -> ChatRequest {
    ChatRequest::new("teacher", vec![]).with_tools(ToolConfig {
        tools: vec![ToolDefinition::Function {
            function: FunctionDefinition {
                name: "read_file".into(),
                description: None,
                parameters: json!({"type":"object"}),
                strict: None,
            },
        }],
        tool_choice: Some(ToolChoice::Required),
        parallel_tool_calls: Some(false),
    })
}
fn body(id: &str, finish: &str) -> String {
    let call = json!({"id":id,"model":"resolved","provider":"owned","choices":[{"delta":{"content":null,"reasoning_content":"inspect first","tool_calls":[{"index":0,"id":"call-a","type":"function","function":{"name":"read_file","arguments":"{ \"path\" : \"a.rs\", \"nested\": [1,true,null] }"}}]}}]});
    let terminal = json!({"choices":[{"delta":{},"finish_reason":finish,"native_finish_reason":"backend-stop"}]});
    format!("data: {call}\n\ndata: {terminal}\n\ndata: [DONE]\n\n")
}
fn provider(server: &Server) -> ChatCompletionsProvider {
    ChatCompletionsProvider::builder()
        .base_url(&server.url)
        .rpm(60_000)
        .retry_policy(RetryPolicy {
            max_attempts: 2,
            base_delay: Duration::ZERO,
            max_delay: Duration::ZERO,
        })
        .build_with_key("fixture")
        .unwrap()
}
fn record(message: &gw_schema::Message) -> TrainingRecord {
    serde_json::from_value(json!({
        "record_id":"record","schema_version":"1.0.0","training_area":"fixture",
        "messages":[{"role":"user","content":"inspect source"},message],
        "provenance":{"run_id":"run","teacher":{"provider":"fixture","slug":"teacher"},"harness_version":"fixture"},
        "generation":{},"lifecycle":{"state":"assistant_generated","history":[
            {"state":"user_synthesized","at":"2026-01-01T00:00:00Z","attempt":0},
            {"state":"assistant_generated","at":"2026-01-01T00:00:00Z","attempt":0}
        ]}
    })).unwrap()
}
#[tokio::test]
async fn observed_tool_turn_survives_http_retry_and_canonical_storage() {
    let store = Store::open_in_memory().await.unwrap();
    let ctx = context(&store, AttemptRole::Teacher, AttemptPurpose::Initial).await;
    let server=Server::responses(vec![
        (503,json!({"id":"failed-attempt","usage":{"cost":0.25},"choices":[{"delta":{"tool_calls":[{"index":0,"id":"bad","function":{"name":"wrong","arguments":"{}"}}]}}]}).to_string()),
        (200,body("success-attempt","tool_calls")),
    ]).await;
    let call = ctx.call();
    let expected = serde_json::to_vec(&request()).unwrap();
    let turn = generate_turn_observed(&provider(&server), request(), call.clone())
        .await
        .unwrap();
    assert_eq!(turn.generation_id.as_deref(), Some("success-attempt"));
    assert_eq!(turn.resolved_model.as_deref(), Some("resolved"));
    assert_eq!(turn.native_finish_reason.as_deref(), Some("backend-stop"));
    assert_eq!(turn.prompt_tokens, None);
    assert_eq!(turn.cost, None);
    assert_eq!(turn.message.content, gw_schema::Content::Null);
    assert_eq!(turn.message.reasoning.as_deref(), Some("inspect first"));
    assert_eq!(
        turn.message.tool_calls.as_ref().unwrap()[0].function.name,
        "read_file"
    );
    let stored = store
        .insert_record(&record(&turn.message))
        .await
        .unwrap()
        .record;
    let restored = store.get("record").await.unwrap();
    assert_eq!(stored, restored);
    assert_eq!(restored.messages[1], turn.message);
    assert_eq!(
        *server.requests.lock().unwrap(),
        vec![expected.clone(), expected]
    );
    let mut attempts = store.model_attempts("run").await.unwrap();
    attempts.sort_by_key(|a| a.intent.retry_ordinal);
    assert_eq!(attempts.len(), 2);
    assert_ne!(attempts[0].attempt_id, attempts[1].attempt_id);
    assert_eq!(attempts[0].metadata.cost_usd, ReportedCost::Known(0.25));
    assert_eq!(
        attempts[0].transport.as_ref().unwrap().outcome,
        TransportOutcome::HttpError
    );
    assert_eq!(attempts[1].metadata.cost_usd, ReportedCost::Missing);
    assert_eq!(
        attempts[1].interpretation,
        Some(OutputInterpretation::Accepted)
    );
    assert_eq!(call.attempt_id(), Some(attempts[1].attempt_id.clone()));
}
#[tokio::test]
async fn invalid_tool_completion_is_interpreted_without_losing_usage() {
    let store = Store::open_in_memory().await.unwrap();
    let ctx = context(&store, AttemptRole::Teacher, AttemptPurpose::Initial).await;
    let response = body("partial", "length").replace(
        "data: [DONE]",
        "data: {\"choices\":[],\"usage\":{\"completion_tokens\":9,\"cost\":0.1}}\n\ndata: [DONE]",
    );
    let server = Server::responses(vec![(200, response)]).await;
    assert!(
        generate_turn_observed(&provider(&server), request(), ctx.call())
            .await
            .is_err()
    );
    let attempt = store.model_attempts("run").await.unwrap().pop().unwrap();
    assert_eq!(attempt.metadata.completion_tokens, Some(9));
    assert_eq!(attempt.metadata.cost_usd, ReportedCost::Known(0.1));
    assert_eq!(
        attempt.interpretation,
        Some(OutputInterpretation::Truncated)
    );
    assert_eq!(server.posts.load(Ordering::SeqCst), 1);
}
#[tokio::test]
async fn terminal_backend_error_and_later_usage_survive_without_retry_or_usable_turn() {
    let store = Store::open_in_memory().await.unwrap();
    let ctx = context(&store, AttemptRole::Teacher, AttemptPurpose::Initial).await;
    let response=body("failed","tool_calls").replace("data: [DONE]","data: {\"error\":{\"code\":500,\"message\":\"fixture backend failure\"}}\n\ndata: {\"choices\":[],\"usage\":{\"cost\":0.2}}\n\ndata: [DONE]");
    let server = Server::responses(vec![(200, response)]).await;
    let error = generate_turn_observed(&provider(&server), request(), ctx.call())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("fixture backend failure"));
    let attempt = store.model_attempts("run").await.unwrap().pop().unwrap();
    assert_eq!(attempt.metadata.cost_usd, ReportedCost::Known(0.2));
    assert_eq!(attempt.interpretation, Some(OutputInterpretation::Invalid));
    assert_eq!(server.posts.load(Ordering::SeqCst), 1);
}
#[tokio::test]
async fn dropping_partial_tool_stream_leaves_attempt_unsettled_and_next_call_independent() {
    let store = Store::open_in_memory().await.unwrap();
    let ctx = context(&store, AttemptRole::Teacher, AttemptPurpose::Initial).await;
    let server = Server::responses(vec![
        (200, body("cancelled-response", "tool_calls")),
        (200, body("new-response", "tool_calls")),
    ])
    .await;
    let client = provider(&server);
    let abandoned = ctx.call();
    let mut stream = client
        .stream_chat_observed(request(), abandoned.clone())
        .await
        .unwrap();
    assert!(stream.next().await.unwrap().unwrap().tool_calls.is_some());
    drop(stream);
    let complete = ctx.call();
    let turn = generate_turn_observed(&client, request(), complete.clone())
        .await
        .unwrap();
    assert_eq!(turn.generation_id.as_deref(), Some("new-response"));
    assert_ne!(abandoned.attempt_id(), complete.attempt_id());
    let attempts = store.model_attempts("run").await.unwrap();
    let cancelled = attempts
        .iter()
        .find(|v| Some(v.attempt_id.clone()) == abandoned.attempt_id())
        .unwrap();
    assert!(cancelled.transport.is_none());
    assert!(cancelled.interpretation.is_none());
    assert_eq!(cancelled.metadata.cost_usd, ReportedCost::Missing);
    assert_eq!(server.posts.load(Ordering::SeqCst), 2);
}
