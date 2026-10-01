//! The new execution identity separates built-in adapter behavior from request location.
use gw_providers::{EmbeddingsClient, OpenRouterProvider, RetryPolicy, builtin_adapter_behavior};
use gw_schema::{Declaration, ModelExecutionSemantics, ModelOperation, SemanticDeclaration};

fn execution(operation: ModelOperation, client: SemanticDeclaration) -> ModelExecutionSemantics {
    ModelExecutionSemantics {
        version: 1,
        alias: "example-model".into(),
        operation,
        adapter_behavior: builtin_adapter_behavior(&client).unwrap(),
        serving_profile: Declaration::Unknown,
        artifact: Declaration::Unknown,
        additional_artifacts: Declaration::Declared(vec![]),
        tokenizer: Declaration::Unknown,
        chat_template: Declaration::Unknown,
        runtime: Declaration::Unknown,
        parser: Declaration::Unknown,
        configuration: Declaration::Unknown,
    }
}

#[test]
fn built_in_chat_location_changes_do_not_change_execution_semantics() {
    let builder = OpenRouterProvider::builder()
        .base_url("https://primary.example.test/v1")
        .retry_policy(RetryPolicy {
            max_attempts: 1,
            ..Default::default()
        });
    let original = builder.semantic_declaration().unwrap();
    let replica = builder
        .clone()
        .base_url("https://replica.example.test/api/v2")
        .semantic_declaration()
        .unwrap();
    assert_ne!(
        original, replica,
        "existing run declarations still bind the base endpoint"
    );
    assert_eq!(
        execution(ModelOperation::ChatCompletion, original.clone())
            .identity()
            .unwrap(),
        execution(ModelOperation::ChatCompletion, replica)
            .identity()
            .unwrap()
    );
    let changed = builder
        .retry_policy(RetryPolicy {
            max_attempts: 2,
            ..Default::default()
        })
        .semantic_declaration()
        .unwrap();
    assert_ne!(
        execution(ModelOperation::ChatCompletion, original)
            .identity()
            .unwrap(),
        execution(ModelOperation::ChatCompletion, changed)
            .identity()
            .unwrap()
    );
}

#[test]
fn built_in_embedding_location_changes_do_not_change_execution_semantics() {
    let builder = EmbeddingsClient::builder()
        .base_url("https://primary.example.test/v1")
        .model("example-model")
        .dim(12);
    let original = builder.semantic_declaration().unwrap();
    let replica = builder
        .clone()
        .base_url("https://replica.example.test/api/v2")
        .semantic_declaration()
        .unwrap();
    assert_ne!(
        original, replica,
        "existing run declarations still bind the base endpoint"
    );
    assert_eq!(
        execution(ModelOperation::Embedding, original.clone())
            .identity()
            .unwrap(),
        execution(ModelOperation::Embedding, replica)
            .identity()
            .unwrap()
    );
    let changed = builder.dim(24).semantic_declaration().unwrap();
    assert_ne!(
        execution(ModelOperation::Embedding, original)
            .identity()
            .unwrap(),
        execution(ModelOperation::Embedding, changed)
            .identity()
            .unwrap()
    );
}

#[test]
fn original_v1_client_declarations_remain_byte_identical() {
    use gw_providers::Provider;
    let chat = OpenRouterProvider::builder()
        .base_url("https://primary.example.test/v1/")
        .retry_policy(RetryPolicy {
            max_attempts: 1,
            ..Default::default()
        });
    let embedding = EmbeddingsClient::builder()
        .base_url("https://primary.example.test/v1/")
        .model("example-model")
        .dim(12)
        .declared_revision(Some("requested-revision".into()))
        .declared_index(gw_schema::VectorIndex::Usearch);
    // Explicit pre-repair v1 wire contracts, including endpoint and unenforced declarations.
    let chat_bytes = r#"{"implementation":"gw-providers/openai-compatible-chat-sse","revision":"1","configuration":{"base_endpoint":"https://primary.example.test/v1","redirects":"reject","request_method":"POST","retry_classification":"provider-error-retryable-v1","route":"chat/completions","stream_contract":"content-and-reasoning-details-v1","transport_attempts":1}}"#;
    let embedding_bytes = r#"{"implementation":"gw-providers/openai-compatible-embeddings","revision":"1","configuration":{"base_endpoint":"https://primary.example.test/v1","batch_order":"response-index-v1","configured_declarations":{"index":"usearch","index_selects_runtime_implementation":false,"model_revision":"requested-revision","model_revision_enforced":false},"dimension":12,"normalization":"none-preserve-finite-nonzero-vectors-v1","requested_model":"example-model","retries":0,"route":"embeddings"}}"#;
    for (declaration, expected) in [
        (chat.semantic_declaration().unwrap(), chat_bytes),
        (embedding.semantic_declaration().unwrap(), embedding_bytes),
        (
            chat.build_with_key("DUMMY-TEST-KEY")
                .unwrap()
                .semantic_declaration()
                .unwrap(),
            chat_bytes,
        ),
        (
            embedding
                .build_with_key(None)
                .unwrap()
                .semantic_declaration(),
            embedding_bytes,
        ),
    ] {
        assert_eq!(serde_json::to_string(&declaration).unwrap(), expected);
        let behavior = builtin_adapter_behavior(&declaration).unwrap();
        assert_eq!(
            serde_json::to_string(&declaration).unwrap(),
            expected,
            "projection must not mutate the original descriptor"
        );
        assert!(
            behavior
                .declaration
                .configuration
                .get("base_endpoint")
                .is_none()
        );
        assert!(
            gw_schema::ModelAdapterBehavior::from_json(expected).is_err(),
            "a full client descriptor is not an adapter-behavior document"
        );
        assert_eq!(
            behavior,
            gw_schema::ModelAdapterBehavior::from_json(&serde_json::to_string(&behavior).unwrap())
                .unwrap()
        );
    }
}

#[test]
fn every_supported_behavior_field_contributes_to_execution_identity() {
    let chat = OpenRouterProvider::builder()
        .semantic_declaration()
        .unwrap();
    let embedding = EmbeddingsClient::builder()
        .dim(12)
        .semantic_declaration()
        .unwrap();
    for (operation, descriptor, fields) in [
        (
            ModelOperation::ChatCompletion,
            chat,
            vec![
                "route",
                "request_method",
                "stream_contract",
                "transport_attempts",
                "retry_classification",
                "redirects",
            ],
        ),
        (
            ModelOperation::Embedding,
            embedding,
            vec![
                "route",
                "dimension",
                "normalization",
                "batch_order",
                "retries",
            ],
        ),
    ] {
        let original = execution(operation, descriptor.clone()).identity().unwrap();
        for field in fields {
            let mut changed = descriptor.clone();
            let value = &mut changed.configuration[field];
            *value = if let Some(number) = value.as_u64() {
                serde_json::json!(number + 1)
            } else {
                serde_json::json!("changed-declared-behavior")
            };
            assert_ne!(
                original,
                execution(operation, changed).identity().unwrap(),
                "lost behavior field {field}"
            );
        }
    }
}

#[test]
fn embedding_alias_and_unenforced_labels_are_explicitly_separate_from_adapter_behavior() {
    let builder = EmbeddingsClient::builder().model("example-model").dim(12);
    let original = builder.semantic_declaration().unwrap();
    let declared = builder
        .clone()
        .declared_revision(Some("unenforced-revision".into()))
        .declared_index(gw_schema::VectorIndex::Usearch)
        .semantic_declaration()
        .unwrap();
    let other_alias = builder.model("other-model").semantic_declaration().unwrap();
    assert_ne!(original, declared);
    assert_ne!(original, other_alias);
    assert_eq!(
        builtin_adapter_behavior(&original).unwrap(),
        builtin_adapter_behavior(&declared).unwrap()
    );
    assert_eq!(
        builtin_adapter_behavior(&original).unwrap(),
        builtin_adapter_behavior(&other_alias).unwrap()
    );
    let original_execution = execution(ModelOperation::Embedding, original);
    let mut changed_execution = execution(ModelOperation::Embedding, other_alias);
    changed_execution.alias = "other-model".into();
    assert_ne!(
        original_execution.identity().unwrap(),
        changed_execution.identity().unwrap()
    );
}

#[test]
fn unsupported_descriptor_shapes_are_rejected_without_silent_field_loss_or_secret_errors() {
    use serde_json::json;
    let chat = OpenRouterProvider::builder()
        .semantic_declaration()
        .unwrap();
    let embedding = EmbeddingsClient::builder().semantic_declaration().unwrap();
    for original in [&chat, &embedding] {
        for variant in 0..6 {
            let mut changed = original.clone();
            match variant {
                0 => changed.implementation = "unknown-sensitive-sentinel".into(),
                1 => changed.revision = "unknown-sensitive-sentinel".into(),
                2 => changed.configuration["unknown_behavior"] = json!("sensitive-sentinel"),
                3 => {
                    changed
                        .configuration
                        .as_object_mut()
                        .unwrap()
                        .remove("route");
                }
                4 => {
                    changed.configuration["base_endpoint"] =
                        json!("https://user:sensitive-sentinel@host.test/v1")
                }
                _ => {
                    changed.configuration["base_endpoint"] =
                        json!("https://host.test/v1?key=sensitive-sentinel")
                }
            }
            let error = builtin_adapter_behavior(&changed).unwrap_err();
            assert!(!format!("{error:?}: {error}").contains("sensitive-sentinel"));
        }
    }
    for key in [
        "model_revision_enforced",
        "index_selects_runtime_implementation",
        "unknown_behavior",
    ] {
        let mut changed = embedding.clone();
        changed.configuration["configured_declarations"][key] = json!(true);
        assert!(builtin_adapter_behavior(&changed).is_err());
    }
    for (mut changed, field) in [(chat, "transport_attempts"), (embedding, "dimension")] {
        changed.configuration[field] = json!(0);
        assert!(builtin_adapter_behavior(&changed).is_err());
    }
}
