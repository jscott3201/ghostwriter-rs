//! Template absence is an explicit declaration, never inferred from missing data.
use gw_schema::{Declaration, ModelExecutionSemantics, PinnedModelArtifact};
use serde_json::{Value, json};

fn declaration(value: Value) -> Value {
    json!({"status":"declared","value":value})
}
fn component() -> Value {
    json!({"source":{"reference":"urn:model:template","revision":"1"},
        "file":{"path":"chat.jinja","purpose":"chat_template",
        "content":{"algorithm":"sha256","hex":"a".repeat(64)}}})
}
fn artifact(template: Value) -> Value {
    json!({"version":1,"label":"fixture","source":{"reference":"urn:model:template","revision":"1"},
        "files":[component()["file"].clone()],"lineage":{"kind":"base"},
        "tokenizer":{"status":"unknown"},"chat_template":template})
}
fn semantics(template: Value) -> Value {
    json!({"version":1,"alias":"fixture","operation":"embedding",
        "adapter_behavior":{"declaration":{"implementation":"fixture","revision":"1","configuration":{}}},
        "serving_profile":{"status":"unknown"},"artifact":{"status":"unknown"},
        "additional_artifacts":{"status":"unknown"},"tokenizer":{"status":"unknown"},
        "chat_template":template,"runtime":{"status":"unknown"},"parser":{"status":"unknown"},
        "configuration":{"status":"unknown"}})
}

#[test]
fn explicitly_absent_template_is_preserved_in_both_documents() {
    let template = declaration(Value::Null);
    let artifact = PinnedModelArtifact::from_json(&artifact(template.clone()).to_string()).unwrap();
    let semantics =
        ModelExecutionSemantics::from_json(&semantics(template.clone()).to_string()).unwrap();
    assert_eq!(
        serde_json::to_value(artifact).unwrap()["chat_template"],
        template
    );
    assert_eq!(
        serde_json::to_value(semantics).unwrap()["chat_template"],
        template
    );
}

#[test]
fn absent_unknown_and_present_template_claims_have_distinct_identities() {
    let mut artifacts = std::collections::BTreeSet::new();
    let mut executions = std::collections::BTreeSet::new();
    for template in [
        json!({"status":"unknown"}),
        declaration(Value::Null),
        declaration(component()),
    ] {
        let a = PinnedModelArtifact::from_json(&artifact(template.clone()).to_string()).unwrap();
        let s =
            ModelExecutionSemantics::from_json(&semantics(template.clone()).to_string()).unwrap();
        assert_eq!(serde_json::to_value(&a).unwrap()["chat_template"], template);
        assert_eq!(serde_json::to_value(&s).unwrap()["chat_template"], template);
        artifacts.insert(a.identity().unwrap().digest);
        executions.insert(s.identity().unwrap().digest);
    }
    assert_eq!(artifacts.len(), 3);
    assert_eq!(executions.len(), 3);
}

#[test]
fn template_declaration_requires_value_and_preserves_component_validation() {
    for bad in [
        json!({"status":"declared"}),
        json!({"status":"declared","value":null,"approved":true}),
        declaration(
            json!({"source":component()["source"], "file":{"path":"chat.jinja","purpose":"weights", "content":component()["file"]["content"]}}),
        ),
    ] {
        assert!(PinnedModelArtifact::from_json(&artifact(bad.clone()).to_string()).is_err());
        assert!(ModelExecutionSemantics::from_json(&semantics(bad).to_string()).is_err());
    }
    let mut a = artifact(declaration(component()));
    a["files"][0]["content"]["hex"] = json!("b".repeat(64));
    assert!(PinnedModelArtifact::from_json(&a.to_string()).is_err());
    let mut absent = artifact(declaration(Value::Null));
    absent.as_object_mut().unwrap().remove("chat_template");
    assert!(PinnedModelArtifact::from_json(&absent.to_string()).is_err());
    let unknown: Declaration<Option<String>> =
        serde_json::from_value(json!({"status":"unknown"})).unwrap();
    assert_eq!(unknown, Declaration::Unknown);
}

#[test]
fn present_template_keeps_the_original_v1_canonical_artifact_bytes() {
    let value = artifact(declaration(component()));
    let parsed = PinnedModelArtifact::from_json(&value.to_string()).unwrap();
    // Independent canonical wire vector from the original present-component v1 shape.
    // Option::Some must not add an envelope or alter the existing identity domain.
    let canonical = r#"{"chat_template":{"status":"declared","value":{"file":{"content":{"algorithm":"sha256","hex":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},"path":"chat.jinja","purpose":"chat_template"},"source":{"reference":"urn:model:template","revision":"1"}}},"files":[{"content":{"algorithm":"sha256","hex":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},"path":"chat.jinja","purpose":"chat_template"}],"lineage":{"kind":"base"},"source":{"reference":"urn:model:template","revision":"1"},"tokenizer":{"status":"unknown"},"version":1}"#;
    let mut hasher = blake3::Hasher::new_derive_key("ghostwriter.model-artifact.v1");
    hasher.update(canonical.as_bytes());
    assert_eq!(
        parsed.identity().unwrap().digest,
        hasher.finalize().to_hex().to_string()
    );
}
