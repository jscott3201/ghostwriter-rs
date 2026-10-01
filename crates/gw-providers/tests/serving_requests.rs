#[path = "support/serving_profile.rs"]
mod support;
use gw_providers::serving_profile::*;
use gw_schema::*;
use serde_json::{Value, json};
use support::*;

fn sampling() -> JudgeSampling {
    JudgeSampling {
        temperature: 0.6,
        top_p: Some(0.95),
        seed: Some(7),
    }
}
fn body(prepared: &PreparedProfileRequest) -> Value {
    serde_json::from_slice(prepared.body()).unwrap()
}

#[test]
fn openrouter_exact_wire_body_is_restricted_and_uses_only_declared_extensions() {
    let profile = profile(ServingDialect::OpenRouterV1);
    let mut request = chat();
    request.controls = vec![
        required(ControlValue::Sampling(sampling())),
        required(ControlValue::MaxOutputTokens(512)),
        required(ControlValue::ReasoningEffort("xhigh".into())),
        required(ControlValue::Usage),
        required(ControlValue::Logprobs(3)),
        required(ControlValue::JsonSchema {
            name: "answer".into(),
            schema: json!({"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}),
        }),
    ];
    let prepared = prepare_request(
        &profile,
        &semantics(&profile, ModelOperation::ChatCompletion),
        &request,
    )
    .unwrap();
    // Independently specified v1 wire contract; canonical JSON uses sorted object keys only.
    let expected = r#"{"logprobs":true,"max_tokens":512,"messages":[{"content":"hello","role":"user"}],"model":"private-fixture","provider":{"allow_fallbacks":false,"data_collection":"deny","only":["replica-a","replica-b"],"require_parameters":true},"reasoning":{"effort":"xhigh"},"response_format":{"json_schema":{"name":"answer","schema":{"additionalProperties":false,"properties":{"answer":{"type":"string"}},"required":["answer"],"type":"object"},"strict":true},"type":"json_schema"},"seed":7,"stream":true,"temperature":0.6,"top_logprobs":3,"top_p":0.95,"usage":{"include":true}}"#;
    assert_eq!(prepared.body(), expected.as_bytes());
    assert_eq!(
        prepared.body_digest().hex,
        blake3::hash(expected.as_bytes()).to_hex().to_string()
    );
    assert!(prepared.degradations().is_empty());
    assert_eq!(
        prepared.endpoint().as_str(),
        "https://serve.example.test/v1/chat/completions"
    );
    assert!(body(&prepared).get("reasoning_effort").is_none());
    assert!(body(&prepared).get("chat_template_kwargs").is_none());
}

#[test]
fn vllm_exact_wire_body_maps_literal_effort_and_template_kwargs_separately() {
    let profile = support::profile(ServingDialect::VllmV1);
    let mut request = chat();
    request.controls = vec![
        required(ControlValue::Sampling(sampling())),
        required(ControlValue::MaxOutputTokens(512)),
        required(ControlValue::ReasoningEffort("max".into())),
        required(ControlValue::EnableThinking(true)),
        required(ControlValue::PreserveThinking(true)),
        required(ControlValue::ClearThinking(false)),
        required(ControlValue::Usage),
    ];
    let prepared = prepare_request(
        &profile,
        &semantics(&profile, ModelOperation::ChatCompletion),
        &request,
    )
    .unwrap();
    let expected = r#"{"chat_template_kwargs":{"clear_thinking":false,"enable_thinking":true,"preserve_thinking":true},"max_tokens":512,"messages":[{"content":"hello","role":"user"}],"model":"private-fixture","reasoning_effort":"max","seed":7,"stream":true,"stream_options":{"include_usage":true},"temperature":0.6,"top_p":0.95}"#;
    assert_eq!(prepared.body(), expected.as_bytes());
    for unsupported in ["provider", "reasoning", "usage"] {
        assert!(body(&prepared).get(unsupported).is_none());
    }
    request.controls.reverse();
    let reordered = prepare_request(
        &profile,
        &semantics(&profile, ModelOperation::ChatCompletion),
        &request,
    )
    .unwrap();
    assert_eq!(prepared.body(), reordered.body());
}

#[test]
fn required_controls_fail_and_optional_omissions_are_explicit() {
    let mut profile = profile(ServingDialect::VllmV1);
    profile
        .behavior
        .capabilities
        .retain(|v| *v != ProfileControl::Seed);
    let semantics = semantics(&profile, ModelOperation::ChatCompletion);
    let mut request = chat();
    request.controls = vec![required(ControlValue::Sampling(sampling()))];
    assert_eq!(
        prepare_request(&profile, &semantics, &request).unwrap_err(),
        ProfileError::UnsupportedRequiredControl(ProfileControl::Seed)
    );
    request.controls = vec![
        optional(ControlValue::Sampling(sampling())),
        optional(ControlValue::ReasoningBudget(100)),
    ];
    let prepared = prepare_request(&profile, &semantics, &request).unwrap();
    assert_eq!(
        prepared.degradations(),
        [
            ControlDegradation {
                control: ProfileControl::Seed,
                reason: DegradationReason::Unsupported
            },
            ControlDegradation {
                control: ProfileControl::ReasoningBudget,
                reason: DegradationReason::Unsupported
            }
        ]
    );
    assert_eq!(body(&prepared)["temperature"], 0.6);
    assert_eq!(body(&prepared)["top_p"], 0.95);
    assert!(body(&prepared).get("seed").is_none());
    assert!(body(&prepared).get("reasoning").is_none());
    request.controls = vec![required(ControlValue::ReasoningEffort("xhigh".into()))];
    assert_eq!(
        prepare_request(&profile, &semantics, &request).unwrap_err(),
        ProfileError::UnsupportedRequiredControl(ProfileControl::ReasoningEffort)
    );
    request.controls[0].required = false;
    let prepared = prepare_request(&profile, &semantics, &request).unwrap();
    assert_eq!(
        prepared.degradations(),
        [ControlDegradation {
            control: ProfileControl::ReasoningEffort,
            reason: DegradationReason::UnsupportedValue
        }]
    );
    assert!(body(&prepared).get("reasoning_effort").is_none());
}

#[test]
fn every_control_obeys_required_capability_and_request_values_affect_body_binding() {
    let controls = [
        ControlValue::MaxOutputTokens(8),
        ControlValue::ReasoningEffort("max".into()),
        ControlValue::EnableThinking(false),
        ControlValue::PreserveThinking(false),
        ControlValue::ClearThinking(false),
        ControlValue::JsonSchema {
            name: "answer".into(),
            schema: json!({"type":"string"}),
        },
        ControlValue::Logprobs(0),
        ControlValue::Usage,
    ];
    for value in controls {
        let mut profile = profile(ServingDialect::VllmV1);
        let mut request = chat();
        request.controls = vec![required(value)];
        let before = prepare_request(
            &profile,
            &semantics(&profile, ModelOperation::ChatCompletion),
            &request,
        )
        .unwrap();
        assert_ne!(before.body_digest(), prepared(&profile).body_digest());
        profile.behavior.capabilities.clear();
        profile.behavior.reasoning_efforts.clear();
        assert!(matches!(
            prepare_request(
                &profile,
                &semantics(&profile, ModelOperation::ChatCompletion),
                &request
            ),
            Err(ProfileError::UnsupportedRequiredControl(_))
        ));
        request.controls[0].required = false;
        let after = prepare_request(
            &profile,
            &semantics(&profile, ModelOperation::ChatCompletion),
            &request,
        )
        .unwrap();
        assert_eq!(after.degradations().len(), 1);
    }
    let profile = profile(ServingDialect::OpenRouterV1);
    let mut request = chat();
    request.controls = vec![required(ControlValue::ReasoningBudget(100))];
    assert_eq!(
        body(
            &prepare_request(
                &profile,
                &semantics(&profile, ModelOperation::ChatCompletion),
                &request
            )
            .unwrap()
        )["reasoning"],
        json!({"max_tokens":100})
    );
}

#[test]
fn malformed_or_duplicate_controls_cannot_be_hidden_as_optional() {
    let profile = profile(ServingDialect::OpenRouterV1);
    let semantics = semantics(&profile, ModelOperation::ChatCompletion);
    for controls in [
        vec![optional(ControlValue::Sampling(JudgeSampling {
            temperature: f64::NAN,
            top_p: None,
            seed: None,
        }))],
        vec![optional(ControlValue::Sampling(JudgeSampling {
            temperature: 0.5,
            top_p: Some(2.0),
            seed: None,
        }))],
        vec![optional(ControlValue::MaxOutputTokens(0))],
        vec![optional(ControlValue::ReasoningBudget(0))],
        vec![optional(ControlValue::ReasoningEffort(" ".into()))],
        vec![optional(ControlValue::Logprobs(21))],
        vec![optional(ControlValue::JsonSchema {
            name: "schema".into(),
            schema: json!([]),
        })],
        vec![required(ControlValue::Usage), optional(ControlValue::Usage)],
        vec![
            optional(ControlValue::ReasoningEffort("xhigh".into())),
            optional(ControlValue::ReasoningBudget(100)),
        ],
    ] {
        let mut request = chat();
        request.controls = controls;
        assert_eq!(
            prepare_request(&profile, &semantics, &request).unwrap_err(),
            ProfileError::InvalidRequest
        );
    }
}

#[test]
fn completion_and_embedding_use_explicit_template_absence_and_keep_order() {
    let profile = support::profile(ServingDialect::VllmV1);
    for (operation, input, expected, route) in [
        (
            ModelOperation::Completion,
            ProfileInput::Completion("rendered prompt".into()),
            r#"{"model":"private-fixture","prompt":"rendered prompt","stream":true}"#,
            "completions",
        ),
        (
            ModelOperation::Embedding,
            ProfileInput::Embedding(vec!["first".into(), "second".into()]),
            r#"{"encoding_format":"float","input":["first","second"],"model":"private-fixture"}"#,
            "embeddings",
        ),
    ] {
        let mut semantics = semantics(&profile, operation);
        assert_eq!(semantics.chat_template, Declaration::Declared(None));
        let request = ProfileRequest {
            input,
            controls: vec![],
        };
        let prepared = prepare_request(&profile, &semantics, &request).unwrap();
        assert_eq!(prepared.body(), expected.as_bytes());
        assert_eq!(
            prepared.endpoint().as_str(),
            format!("https://serve.example.test/v1/{route}")
        );
        semantics.chat_template = Declaration::Unknown;
        assert_eq!(
            prepare_request(&profile, &semantics, &request).unwrap_err(),
            ProfileError::InvalidSemantics
        );
        semantics.chat_template = Declaration::Declared(None);
        semantics.tokenizer = Declaration::Unknown;
        assert_eq!(
            prepare_request(&profile, &semantics, &request).unwrap_err(),
            ProfileError::InvalidSemantics
        );
    }
    let mut request = ProfileRequest {
        input: ProfileInput::Embedding(vec!["first".into(), "second".into()]),
        controls: vec![],
    };
    let target = semantics(&profile, ModelOperation::Embedding);
    let first = prepare_request(&profile, &target, &request).unwrap();
    if let ProfileInput::Embedding(values) = &mut request.input {
        values.reverse();
    }
    assert_ne!(
        first.body_digest(),
        prepare_request(&profile, &target, &request)
            .unwrap()
            .body_digest()
    );
}

#[test]
fn chat_requires_present_template_and_all_semantic_fields_must_be_declared() {
    let profile = support::profile(ServingDialect::VllmV1);
    let baseline = semantics(&profile, ModelOperation::ChatCompletion);
    let mut variants = Vec::new();
    let mut value = baseline.clone();
    value.chat_template = Declaration::Declared(None);
    variants.push(value);
    let mut value = baseline.clone();
    value.chat_template = Declaration::Unknown;
    variants.push(value);
    let mut value = baseline.clone();
    value.artifact = Declaration::Unknown;
    variants.push(value);
    let mut value = baseline.clone();
    value.additional_artifacts = Declaration::Unknown;
    variants.push(value);
    let mut value = baseline.clone();
    value.tokenizer = Declaration::Unknown;
    variants.push(value);
    let mut value = baseline.clone();
    value.runtime = Declaration::Unknown;
    variants.push(value);
    let mut value = baseline.clone();
    value.parser = Declaration::Unknown;
    variants.push(value);
    let mut value = baseline.clone();
    value.configuration = Declaration::Unknown;
    variants.push(value);
    let mut value = baseline.clone();
    value.serving_profile = Declaration::Unknown;
    variants.push(value);
    let mut value = baseline.clone();
    value.adapter_behavior.declaration.revision = "2".into();
    variants.push(value);
    for value in variants {
        assert_eq!(
            prepare_request(&profile, &value, &chat()).unwrap_err(),
            ProfileError::InvalidSemantics
        );
    }
    let wrong_operation = ProfileRequest {
        input: ProfileInput::Completion("raw".into()),
        controls: vec![],
    };
    assert_eq!(
        prepare_request(&profile, &baseline, &wrong_operation).unwrap_err(),
        ProfileError::InvalidSemantics
    );
}

#[test]
fn canonical_reasoning_history_is_preserved_separately_and_unsupported_message_data_denies() {
    let mut profile = profile(ServingDialect::OpenRouterV1);
    let mut previous = message();
    previous.role = Role::Assistant;
    previous.content = Content::Text("answer".into());
    previous.reasoning = Some("separate reasoning".into());
    previous.reasoning_details = Some(
        serde_json::from_value(json!([{"type":"reasoning.text","text":"detail","index":0}]))
            .unwrap(),
    );
    let request = ProfileRequest {
        input: ProfileInput::Chat(vec![previous.clone(), message()]),
        controls: vec![],
    };
    let prepared = prepare_request(
        &profile,
        &semantics(&profile, ModelOperation::ChatCompletion),
        &request,
    )
    .unwrap();
    assert_eq!(body(&prepared)["messages"][0]["content"], "answer");
    assert_eq!(
        body(&prepared)["messages"][0]["reasoning"],
        "separate reasoning"
    );
    assert_eq!(
        body(&prepared)["messages"][0]["reasoning_details"][0]["text"],
        "detail"
    );
    profile
        .behavior
        .capabilities
        .retain(|v| *v != ProfileControl::ReasoningHistory);
    assert_eq!(
        prepare_request(
            &profile,
            &semantics(&profile, ModelOperation::ChatCompletion),
            &request
        )
        .unwrap_err(),
        ProfileError::InvalidRequest
    );
    let profile = support::profile(ServingDialect::VllmV1);
    for change in [0, 1, 2, 3] {
        let mut message = message();
        match change {
            0 => message.tool_calls = Some(vec![]),
            1 => message.role = Role::Tool,
            2 => message.content = Content::Parts(vec![]),
            _ => message.reasoning = Some("invalid role".into()),
        }
        let request = ProfileRequest {
            input: ProfileInput::Chat(vec![message]),
            controls: vec![],
        };
        assert_eq!(
            prepare_request(
                &profile,
                &semantics(&profile, ModelOperation::ChatCompletion),
                &request
            )
            .unwrap_err(),
            ProfileError::InvalidRequest
        );
    }
}
