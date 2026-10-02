//! Authored native tool-call fixtures; no tool is executed.
use gw_generate::accumulate;
use gw_providers::decode_sse;
use gw_schema::Content;
use serde_json::{Value, json};

fn stream(chunks: Vec<Value>, done: bool) -> gw_providers::DeltaStream {
    let mut body: String = chunks
        .into_iter()
        .map(|c| format!("data: {c}\n\n"))
        .collect();
    if done {
        body.push_str("data: [DONE]\n\n");
    }
    Box::pin(decode_sse(futures::stream::iter(vec![Ok(
        body.into_bytes()
    )])))
}
fn call(index: u32, id: Option<&str>, name: &str, arguments: &str) -> Value {
    json!({"index":index,"id":id,"type":"function","function":{"name":name,"arguments":arguments}})
}
fn chunk(calls: Vec<Value>) -> Value {
    json!({"id":"response-1","model":"teacher","provider":"owned","choices":[{"index":0,"delta":{"content":null,"tool_calls":calls}}]})
}
fn finish(reason: &str) -> Value {
    json!({"choices":[{"index":0,"delta":{},"finish_reason":reason,"native_finish_reason":"native-end"}]})
}
#[tokio::test]
async fn interleaved_calls_preserve_raw_arguments_reasoning_and_canonical_roundtrip() {
    let args = "{ \"items\" : [1,true,null,{\"λ\":\"quoted\\\"\"}] }";
    let mut first = chunk(vec![
        call(1, Some("b"), "read_", "{"),
        call(0, Some("a"), "read", &args[..12]),
    ]);
    first["choices"][0]["delta"]["reasoning_content"] = json!("Need two reads.");
    let acc = accumulate(stream(
        vec![
            first,
            chunk(vec![
                call(0, None, "_file", &args[12..]),
                call(1, None, "file", "\"path\":\"b\"}"),
            ]),
            finish("tool_calls"),
        ],
        true,
    ))
    .await
    .unwrap();
    let turn = acc.into_turn().unwrap();
    assert_eq!(turn.message.content, Content::Null);
    assert_eq!(turn.message.reasoning.as_deref(), Some("Need two reads."));
    let calls = turn.message.tool_calls.as_ref().unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].id.as_deref(), Some("a"));
    assert_eq!(calls[1].id.as_deref(), Some("b"));
    assert!(calls.iter().all(|c| c.function.name == "read_file"));
    assert_eq!(calls[0].function.raw_arguments.as_deref(), Some(args));
    assert_eq!(
        calls[0].function.arguments,
        serde_json::from_str::<Value>(args).unwrap()
    );
    assert_eq!(turn.prompt_tokens, None);
    assert_eq!(turn.cost, None);
    let stored = serde_json::to_vec(&turn.message).unwrap();
    assert_eq!(
        serde_json::from_slice::<gw_schema::Message>(&stored).unwrap(),
        turn.message
    );
    let whole = accumulate(stream(
        vec![
            chunk(vec![
                call(0, Some("a"), "read_file", args),
                call(1, Some("b"), "read_file", "{\"path\":\"b\"}"),
            ]),
            finish("tool_calls"),
        ],
        true,
    ))
    .await
    .unwrap()
    .into_turn()
    .unwrap();
    assert_eq!(whole.message.tool_calls, turn.message.tool_calls);
}
#[tokio::test]
async fn incomplete_malformed_and_unsupported_calls_cannot_become_turns() {
    for chunks in [
        vec![chunk(vec![call(0, Some("a"), "f", "{}")])],
        vec![chunk(vec![call(0, Some("a"), "f", "{}")]), finish("length")],
        vec![
            chunk(vec![call(0, Some("a"), "f", "{")]),
            finish("tool_calls"),
        ],
        vec![chunk(vec![call(0, None, "f", "{}")]), finish("tool_calls")],
        vec![
            chunk(vec![call(0, Some("a"), "f", "[]")]),
            finish("tool_calls"),
        ],
        vec![chunk(vec![call(0, Some("a"), "f", "{}")]), finish("stop")],
        vec![
            json!({"choices":[{"delta":{"function_call":{"name":"f","arguments":"{}"}},"finish_reason":"function_call"}]}),
        ],
    ] {
        let result = accumulate(stream(chunks, true))
            .await
            .and_then(|a| a.into_turn());
        assert!(
            result.is_err(),
            "unsupported output became a turn: {result:?}"
        );
    }
    assert!(
        accumulate(stream(
            vec![
                chunk(vec![call(0, Some("a"), "f", "{}")]),
                finish("tool_calls")
            ],
            false
        ))
        .await
        .is_err()
    );
}

struct Fixture(Vec<Value>);
impl gw_providers::Provider for Fixture {
    fn stream_chat(
        &self,
        _request: gw_providers::ChatRequest,
    ) -> gw_providers::StreamChatFuture<'_> {
        Box::pin(async move { Ok(stream(self.0.clone(), true)) })
    }
}
fn tools(choice: gw_providers::ToolChoice, parallel: bool) -> gw_providers::ToolConfig {
    gw_providers::ToolConfig {
        tools: vec![gw_providers::ToolDefinition::Function {
            function: gw_providers::FunctionDefinition {
                name: "read_file".into(),
                description: None,
                parameters: json!({"type":"object"}),
                strict: None,
            },
        }],
        tool_choice: Some(choice),
        parallel_tool_calls: Some(parallel),
    }
}
#[tokio::test]
async fn generation_enforces_requested_names_selection_and_parallel_policy() {
    use gw_providers::{ChatRequest, ToolChoice};
    let fixture = Fixture(vec![
        chunk(vec![call(0, Some("a"), "read_file", "{}")]),
        finish("tool_calls"),
    ]);
    assert!(
        gw_generate::generate_turn(&fixture, ChatRequest::new("m", vec![]))
            .await
            .is_err(),
        "text-only caller accepted an unsolicited call"
    );
    for choice in [
        ToolChoice::Auto,
        ToolChoice::Required,
        ToolChoice::Function("read_file".into()),
    ] {
        let turn = gw_generate::generate_turn(
            &fixture,
            ChatRequest::new("m", vec![]).with_tools(tools(choice, false)),
        )
        .await
        .unwrap();
        assert_eq!(turn.message.tool_calls.unwrap()[0].id.as_deref(), Some("a"));
        assert_eq!(turn.resolved_model.as_deref(), Some("teacher"));
        assert_eq!(turn.native_finish_reason.as_deref(), Some("native-end"));
    }
    assert!(
        gw_generate::generate_turn(
            &fixture,
            ChatRequest::new("m", vec![]).with_tools(tools(ToolChoice::None, true))
        )
        .await
        .is_err()
    );
    let unknown = Fixture(vec![
        chunk(vec![call(0, Some("a"), "write_file", "{}")]),
        finish("tool_calls"),
    ]);
    assert!(
        gw_generate::generate_turn(
            &unknown,
            ChatRequest::new("m", vec![]).with_tools(tools(ToolChoice::Auto, true))
        )
        .await
        .is_err()
    );
    let parallel = Fixture(vec![
        chunk(vec![
            call(0, Some("a"), "read_file", "{}"),
            call(1, Some("b"), "read_file", "{}"),
        ]),
        finish("tool_calls"),
    ]);
    assert!(
        gw_generate::generate_turn(
            &parallel,
            ChatRequest::new("m", vec![]).with_tools(tools(ToolChoice::Required, false))
        )
        .await
        .is_err()
    );
    let text = Fixture(vec![
        json!({"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}),
    ]);
    assert!(
        gw_generate::generate_turn(
            &text,
            ChatRequest::new("m", vec![]).with_tools(tools(ToolChoice::Required, true))
        )
        .await
        .is_err()
    );
    assert!(
        gw_generate::generate_turn(
            &text,
            ChatRequest::new("m", vec![]).with_tools(tools(ToolChoice::Auto, true))
        )
        .await
        .is_ok()
    );
    let incomplete = Fixture(vec![
        json!({"choices":[{"delta":{"content":"partial"},"finish_reason":"length"}]}),
    ]);
    assert!(
        gw_generate::generate_turn(&incomplete, ChatRequest::new("m", vec![]))
            .await
            .is_ok()
    );
    assert!(
        gw_generate::generate_turn(
            &incomplete,
            ChatRequest::new("m", vec![]).with_tools(tools(ToolChoice::Auto, true))
        )
        .await
        .is_err()
    );
}
#[tokio::test]
async fn byte_boundaries_and_empty_string_content_preserve_identical_turns() {
    let chunks = vec![
        chunk(vec![call(
            0,
            Some("a"),
            "read_file",
            "{\"λ\":[false,null,2]}",
        )]),
        finish("tool_calls"),
    ];
    let expected = accumulate(stream(chunks.clone(), true))
        .await
        .unwrap()
        .into_turn()
        .unwrap();
    let body = format!(
        "{}data: [DONE]\n\n",
        chunks
            .iter()
            .map(|v| format!("data: {v}\n\n"))
            .collect::<String>()
    );
    for width in [1, 2, 7, 43] {
        let bytes: Vec<_> = body
            .as_bytes()
            .chunks(width)
            .map(|v| Ok(v.to_vec()))
            .collect();
        let actual = accumulate(Box::pin(decode_sse(futures::stream::iter(bytes))))
            .await
            .unwrap()
            .into_turn()
            .unwrap();
        assert_eq!(expected, actual);
    }
    let mut chunks = chunks;
    chunks[0]["choices"][0]["delta"]["content"] = json!("");
    let actual = accumulate(stream(chunks, true))
        .await
        .unwrap()
        .into_turn()
        .unwrap();
    assert_eq!(actual.message.content, Content::Text(String::new()));
}
#[tokio::test]
async fn conflicting_ids_indices_aliases_and_post_terminal_payload_fail_closed() {
    let bad = vec![
        vec![
            chunk(vec![call(0, Some("a"), "f", "{")]),
            chunk(vec![call(0, Some("b"), "", "}")]),
            finish("tool_calls"),
        ],
        vec![
            chunk(vec![
                call(0, Some("a"), "f", "{}"),
                call(1, Some("a"), "f", "{}"),
            ]),
            finish("tool_calls"),
        ],
        vec![
            chunk(vec![call(1, Some("a"), "f", "{}")]),
            finish("tool_calls"),
        ],
        vec![
            chunk(vec![
                json!({"index":0,"id":"a","type":"custom","function":{"name":"f","arguments":"{}"}}),
            ]),
            finish("tool_calls"),
        ],
        vec![
            chunk(vec![call(0, Some("a"), "f", "{}")]),
            finish("tool_calls"),
            chunk(vec![call(0, None, "extra", "")]),
        ],
        vec![
            chunk(vec![call(0, Some("a"), "f", "{}")]),
            finish("tool_calls"),
            finish("length"),
        ],
        vec![
            json!({"choices":[{"delta":{"content":"x","reasoning":"a","reasoning_content":"b"}}]}),
            finish("stop"),
        ],
        vec![
            json!({"choices":[{"index":1,"delta":{"content":"wrong choice"}}]}),
            finish("stop"),
        ],
    ];
    for chunks in bad {
        assert!(
            accumulate(stream(chunks, true))
                .await
                .and_then(|v| v.into_turn())
                .is_err()
        );
    }
    let error = accumulate(stream(
        vec![
            chunk(vec![call(0, Some("a"), "f", "{")]),
            json!({"error":{"message":"fixture backend failed","code":500}}),
        ],
        true,
    ))
    .await
    .unwrap_err();
    assert!(error.to_string().contains("fixture backend failed"));
}
