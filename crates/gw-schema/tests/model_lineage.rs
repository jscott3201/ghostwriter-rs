//! Optional lineage distinguishes an unknown link, declared absence, and a supplied identity.
use gw_schema::{ArtifactIdentity, Declaration, ModelArtifactLineage, PinnedModelArtifact};
use serde_json::{Value, json};

fn identity() -> Value {
    json!({"version":1, "digest":"a".repeat(64)})
}

fn declared(value: Value) -> Value {
    json!({"status":"declared", "value":value})
}

fn artifact(lineage: Value) -> Value {
    json!({
        "version":1, "label":"Lineage fixture",
        "source":{"reference":"urn:model:fixture", "revision":"1"},
        "files":[{"path":"weights.bin", "purpose":"weights",
                  "content":{"algorithm":"sha256", "hex":"b".repeat(64)}}],
        "lineage":lineage,
        "tokenizer":{"status":"unknown"}, "chat_template":{"status":"unknown"}
    })
}

fn adapter() -> Value {
    json!({"kind":"adapter", "base":identity(), "parent_checkpoint":declared(Value::Null),
           "configuration":{"implementation":"lora", "revision":"1", "configuration":{}}})
}

fn checkpoint() -> Value {
    json!({"kind":"checkpoint", "base":identity(),
           "parent":declared(Value::Null), "adapter":declared(Value::Null)})
}

#[test]
fn initial_adapter_can_explicitly_declare_no_prior_checkpoint() {
    let input = artifact(adapter());
    let parsed = PinnedModelArtifact::from_json(&input.to_string())
        .expect("an initial adapter can declare that no prior checkpoint exists");
    assert_eq!(serde_json::to_value(parsed).unwrap(), input);
}

#[test]
fn first_unadapted_checkpoint_can_explicitly_declare_no_parent_or_adapter() {
    let input = artifact(checkpoint());
    let parsed = PinnedModelArtifact::from_json(&input.to_string())
        .expect("a first checkpoint can declare neither a prior checkpoint nor an adapter");
    assert_eq!(serde_json::to_value(parsed).unwrap(), input);
}

fn edges() -> [(Value, &'static str); 3] {
    [
        (adapter(), "parent_checkpoint"),
        (checkpoint(), "parent"),
        (checkpoint(), "adapter"),
    ]
}

#[test]
fn a_declared_optional_link_requires_an_explicit_value_key() {
    for (mut lineage, field) in edges() {
        lineage[field] = json!({"status":"declared"});
        assert!(
            PinnedModelArtifact::from_json(&artifact(lineage).to_string()).is_err(),
            "missing value was inferred as absent for {field}"
        );
    }
}

fn edge<'a>(
    artifact: &'a mut PinnedModelArtifact,
    field: &str,
) -> &'a mut Declaration<Option<ArtifactIdentity>> {
    match (&mut artifact.lineage, field) {
        (
            ModelArtifactLineage::Adapter {
                parent_checkpoint, ..
            },
            "parent_checkpoint",
        ) => parent_checkpoint,
        (ModelArtifactLineage::Checkpoint { parent, .. }, "parent") => parent,
        (ModelArtifactLineage::Checkpoint { adapter, .. }, "adapter") => adapter,
        _ => panic!("incorrect fixture edge"),
    }
}

#[test]
fn every_optional_link_round_trips_three_distinct_claims_and_artifact_hashes() {
    for (lineage, field) in edges() {
        let mut identities = std::collections::BTreeSet::new();
        for (index, claim) in [
            json!({"status":"unknown"}),
            declared(Value::Null),
            declared(identity()),
        ]
        .into_iter()
        .enumerate()
        {
            let mut lineage = lineage.clone();
            lineage[field] = claim;
            let input = artifact(lineage);
            let mut parsed = PinnedModelArtifact::from_json(&input.to_string()).unwrap();
            match (index, edge(&mut parsed, field)) {
                (0, Declaration::Unknown) | (1, Declaration::Declared(None)) => {}
                (2, Declaration::Declared(Some(identity))) => {
                    assert_eq!(identity.version, 1);
                    assert_eq!(identity.digest, "a".repeat(64));
                }
                _ => panic!("claim state changed for {field}"),
            }
            assert_eq!(serde_json::to_value(&parsed).unwrap(), input, "{field}");
            assert_eq!(
                serde_json::from_str::<PinnedModelArtifact>(&input.to_string()).unwrap(),
                parsed
            );
            identities.insert(parsed.identity().unwrap().digest);
        }
        assert_eq!(
            identities.len(),
            3,
            "claim identities collapsed for {field}"
        );
    }
}

fn assert_invalid(input: &str) {
    let error = PinnedModelArtifact::from_json(input).unwrap_err();
    assert_eq!(error.to_string(), "invalid model identity document");
    let error = serde_json::from_str::<PinnedModelArtifact>(input).unwrap_err();
    assert!(
        error
            .to_string()
            .starts_with("invalid model identity document")
    );
    assert!(!error.to_string().contains("sensitive-sentinel"));
}

#[test]
fn omission_of_any_optional_link_is_rejected_instead_of_inferred() {
    for (mut lineage, field) in edges() {
        lineage.as_object_mut().unwrap().remove(field);
        assert_invalid(&artifact(lineage).to_string());
    }
}

#[test]
fn optional_links_reject_invalid_identities_and_unknown_nested_fields() {
    for (lineage, field) in edges() {
        for invalid in [
            Value::Null,
            declared(json!({"version":2, "digest":"a".repeat(64)})),
            declared(json!({"version":1, "digest":"sensitive-sentinel"})),
            declared(
                json!({"version":1, "digest":"a".repeat(64), "verified":"sensitive-sentinel"}),
            ),
            json!({"status":"declared", "value":null, "verified":"sensitive-sentinel"}),
            json!({"status":"unknown", "verified":"sensitive-sentinel"}),
            json!({"status":"sensitive-sentinel", "value":null}),
        ] {
            let mut lineage = lineage.clone();
            lineage[field] = invalid;
            assert_invalid(&artifact(lineage).to_string());
        }
    }
}

#[test]
fn required_base_identity_cannot_be_omitted_unknown_or_absent() {
    for lineage in [adapter(), checkpoint()] {
        let mut missing = lineage.clone();
        missing.as_object_mut().unwrap().remove("base");
        assert_invalid(&artifact(missing).to_string());
        for invalid in [
            Value::Null,
            json!({"status":"unknown"}),
            declared(Value::Null),
        ] {
            let mut lineage = lineage.clone();
            lineage["base"] = invalid;
            assert_invalid(&artifact(lineage).to_string());
        }
    }
}

#[test]
fn mutated_present_links_are_revalidated_before_artifact_hashing() {
    for (lineage, field) in edges() {
        for identity in [
            ArtifactIdentity {
                version: 2,
                digest: "a".repeat(64),
            },
            ArtifactIdentity {
                version: 1,
                digest: "sensitive-sentinel".into(),
            },
        ] {
            let mut parsed =
                PinnedModelArtifact::from_json(&artifact(lineage.clone()).to_string()).unwrap();
            *edge(&mut parsed, field) = Declaration::Declared(Some(identity));
            assert!(parsed.validate().is_err());
            let error = parsed.identity().unwrap_err();
            assert!(!error.to_string().contains("sensitive-sentinel"));
        }
    }
}

#[test]
fn raw_declaration_envelopes_preserve_duplicate_key_rejection() {
    for (mut lineage, field) in edges() {
        // Insert raw declaration JSON after serializing the surrounding fixture, so duplicate
        // fields reach the streaming decoder instead of being normalized by serde_json::Value.
        lineage[field] = json!("raw-declaration-placeholder");
        let fixture = artifact(lineage).to_string();
        for envelope in [
            r#"{"status":"declared","status":"declared","value":null}"#,
            r#"{"status":"declared","value":null,"value":null}"#,
            r#"{"value":null,"value":null,"status":"declared"}"#,
            r#"{"value":null,"status":"declared","status":"declared"}"#,
            r#"{"status":"declared","value":null,"sensitive-sentinel":true}"#,
        ] {
            let input = fixture.replace("\"raw-declaration-placeholder\"", envelope);
            assert_invalid(&input);
        }
    }
}
