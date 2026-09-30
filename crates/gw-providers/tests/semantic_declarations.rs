//! Pure endpoint and implementation declarations shared with constructed clients.
use gw_providers::{
    EmbeddingsClient, OpenRouterProvider, Provider, RetryPolicy, normalize_endpoint,
};
use std::time::Duration;
#[test]
fn normalized_endpoint_preserves_origin_and_path_and_rejects_secret_bearing_forms() {
    assert_eq!(
        normalize_endpoint("HTTPS://EXAMPLE.TEST:443/v1///").unwrap(),
        "https://example.test/v1"
    );
    for different in [
        "http://example.test/v1",
        "https://example.test/v2",
        "https://other.test/v1",
        "https://example.test:444/v1",
        "https://example.test/v1/route",
    ] {
        assert_ne!(
            normalize_endpoint(different).unwrap(),
            normalize_endpoint("https://example.test/v1").unwrap()
        );
    }
    for endpoint in [
        "https://@example.test/v1",
        "https://user:SECRET@example.test/v1",
        "https://SECRET@example.test/v1",
        "https://example.test/v1?key=SECRET",
        "https://example.test/v1#SECRET",
        "SECRET",
        "file:///SECRET",
    ] {
        let error = normalize_endpoint(endpoint).unwrap_err().to_string();
        assert!(!error.contains("SECRET"));
        let builder = OpenRouterProvider::builder()
            .base_url(endpoint)
            .api_key_env("KEY_NEVER_PRESENT_MANIFEST_TEST");
        let error = match builder.build() {
            Ok(_) => panic!("unsupported URL"),
            Err(error) => error.to_string(),
        };
        assert!(!error.contains("SECRET"));
        assert!(!error.contains("KEY_NEVER_PRESENT"));
    }
}
#[test]
fn built_chat_client_and_pure_builder_share_effective_identity() {
    let builder = OpenRouterProvider::builder()
        .base_url("https://EXAMPLE.test:443/v1/")
        .rpm(4);
    let declaration = builder.semantic_declaration().unwrap();
    let client = builder.build_with_key("DUMMY-TEST-KEY").unwrap();
    assert_eq!(client.semantic_declaration().unwrap(), declaration);
    assert_eq!(
        OpenRouterProvider::builder()
            .base_url("https://example.test/v1")
            .rpm(999)
            .semantic_declaration()
            .unwrap(),
        declaration
    );
    let zero = OpenRouterProvider::builder()
        .retry_policy(RetryPolicy {
            max_attempts: 0,
            ..Default::default()
        })
        .semantic_declaration()
        .unwrap();
    let one = OpenRouterProvider::builder()
        .retry_policy(RetryPolicy {
            max_attempts: 1,
            ..Default::default()
        })
        .semantic_declaration()
        .unwrap();
    assert_eq!(zero, one);
}
#[test]
fn embedding_declares_actual_no_normalization_and_unenforced_config_separately() {
    let builder = EmbeddingsClient::builder()
        .base_url("https://example.test/v1/")
        .model("embed")
        .dim(12)
        .declared_revision(Some("requested-revision".into()))
        .declared_index(gw_schema::VectorIndex::Usearch);
    let declaration = builder.semantic_declaration().unwrap();
    let client = builder.clone().build().unwrap();
    assert_eq!(client.semantic_declaration(), declaration);
    let config = &declaration.configuration;
    assert_eq!(
        config["normalization"],
        "none-preserve-finite-nonzero-vectors-v1"
    );
    assert_eq!(
        config["configured_declarations"]["model_revision_enforced"],
        false
    );
    assert_eq!(
        config["configured_declarations"]["index_selects_runtime_implementation"],
        false
    );
    assert_eq!(
        builder
            .clone()
            .timeout(Duration::from_secs(999))
            .api_key_env(Some("UNREAD_KEY".into()))
            .semantic_declaration()
            .unwrap(),
        declaration
    );
    for changed in [
        builder.clone().model("other"),
        builder.clone().dim(24),
        builder.clone().base_url("https://example.test/v2"),
        builder
            .clone()
            .declared_revision(Some("other-revision".into())),
    ] {
        assert_ne!(changed.semantic_declaration().unwrap(), declaration);
    }
    assert!(builder.dim(0).semantic_declaration().is_err());
}
