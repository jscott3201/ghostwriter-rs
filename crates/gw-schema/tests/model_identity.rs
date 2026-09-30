//! Independent structural and canonical-contract checks for model declarations.
use gw_schema::*;
use serde_json::{Value, json};

fn content(byte: char) -> Value {
    json!({"algorithm":"sha256", "hex":byte.to_string().repeat(64)})
}
fn source() -> Value {
    json!({"reference":"https://models.example.test/team/model", "revision":"commit-a1"})
}
fn file(path: &str, purpose: &str, byte: char) -> Value {
    json!({"path":path, "purpose":purpose, "content":content(byte)})
}
fn component(path: &str, purpose: &str, byte: char) -> Value {
    json!({"source":source(), "file":file(path, purpose, byte)})
}
fn declared(value: Value) -> Value {
    json!({"status":"declared", "value":value})
}
fn unknown() -> Value {
    json!({"status":"unknown"})
}
fn identity(byte: char) -> Value {
    json!({"version":1, "digest":byte.to_string().repeat(64)})
}
fn semantic(name: &str) -> Value {
    json!({"implementation":name, "revision":"1", "configuration":{"z":2,"a":1}})
}
fn artifact_json() -> Value {
    json!({
        "version":1, "label":"Example model", "source":source(),
        "files":[file("weights/model.safetensors", "weights", 'a'),
                 file("tokenizer.json", "tokenizer", 'b'),
                 file("chat-template.jinja", "chat_template", 'c')],
        "lineage":{"kind":"base"},
        "tokenizer":declared(component("tokenizer.json", "tokenizer", 'b')),
        "chat_template":declared(component("chat-template.jinja", "chat_template", 'c'))
    })
}
fn semantics_json() -> Value {
    json!({
        "version":1, "alias":"example/model", "operation":"chat_completion",
        "client":semantic("chat-adapter"), "serving_profile":declared(semantic("profile")),
        "artifact":declared(identity('a')), "additional_artifacts":declared(json!([identity('c'),identity('b')])),
        "tokenizer":declared(component("tokenizer.json","tokenizer",'b')),
        "chat_template":declared(component("chat-template.jinja","chat_template",'c')),
        "runtime":declared(semantic("runtime")), "parser":declared(semantic("reasoning-parser")),
        "configuration":declared(content('d'))
    })
}
fn policy_json() -> Value {
    json!({"version":1, "kind":"output_terms", "subject":identity('d'), "document":{"source":source(),"content":content('a')},
           "role":"embedding", "intended_use":"dataset_generation"})
}
fn request_json() -> Value {
    json!({"version":1, "endpoint":"https://serve.example.test/v1/chat/completions",
           "semantics":semantics_json(), "policy_evidence":declared(json!([identity('a'),identity('b')]))})
}
fn deployment_json() -> Value {
    json!({"version":1, "method":semantic("measurement"), "issuer":"urn:issuer:example",
           "verifier":declared(semantic("evidence-verifier")),
           "raw_evidence":{"source":{"reference":"urn:evidence:report-1","revision":"1"},"content":content('e')},
           "endpoint":"https://serve.example.test/v1/chat/completions", "instance":"server-a", "incarnation":"boot-a",
           "claimed_loaded_artifacts":declared(json!([identity('a'),identity('c')])), "effective":semantics_json(),
           "validity":{"not_before_unix_ms":100,"expires_at_unix_ms":Some(200),
                       "revocation":declared(json!({"document":{"source":source(),"content":content('f')},"observed_at_unix_ms":120}))}})
}
fn observed_json() -> Value {
    json!({"version":1,"attempt":{"run_id":"run-a","launch_id":"launch-a","attempt_id":"attempt-a","observation_sequence":0},
           "requested":declared(identity('a')), "deployment_evidence":declared(identity('b')),
           "termination":{"native":"eos_token","normalized":"stop"}})
}
fn artifact(value: &Value) -> PinnedModelArtifact {
    PinnedModelArtifact::from_json(&value.to_string()).unwrap()
}
fn semantics(value: &Value) -> ModelExecutionSemantics {
    ModelExecutionSemantics::from_json(&value.to_string()).unwrap()
}
fn deployment(value: &Value) -> ModelDeploymentEvidence {
    ModelDeploymentEvidence::from_json(&value.to_string()).unwrap()
}
fn observed(value: &Value) -> ObservedModelExecution {
    ObservedModelExecution::from_json(&value.to_string()).unwrap()
}
fn replace(value: &Value, pointer: &str, replacement: Value) -> Value {
    let mut value = value.clone();
    *value.pointer_mut(pointer).unwrap() = replacement;
    value
}

#[test]
fn strict_documents_round_trip_without_inferred_authority() {
    macro_rules! round_trip {
        ($ty:ty, $value:expr) => {{
            let value = $value;
            let parsed = <$ty>::from_json(&value.to_string()).unwrap();
            assert_eq!(serde_json::to_value(&parsed).unwrap(), value);
            assert_eq!(
                serde_json::from_str::<$ty>(&value.to_string()).unwrap(),
                parsed
            );
        }};
    }
    round_trip!(PinnedModelArtifact, artifact_json());
    round_trip!(ModelExecutionSemantics, semantics_json());
    round_trip!(ModelPolicyEvidence, policy_json());
    round_trip!(RequestedModelExecution, request_json());
    round_trip!(ModelDeploymentEvidence, deployment_json());
    round_trip!(ObservedModelExecution, observed_json());
    for lineage in [
        json!({"kind":"derived","base":identity('d'),"additional_parents":declared(json!([identity('e'),identity('f')])),"transformation":semantic("merge")}),
        json!({"kind":"quantized","base":identity('d'),"quantization":semantic("quantizer")}),
        json!({"kind":"adapter","base":identity('d'),"parent_checkpoint":unknown(),"configuration":semantic("lora")}),
        json!({"kind":"checkpoint","base":identity('d'),"parent":declared(identity('e')),"adapter":unknown()}),
    ] {
        round_trip!(
            PinnedModelArtifact,
            replace(&artifact_json(), "/lineage", lineage)
        );
    }
}

#[test]
fn unknown_fields_and_forged_authority_are_rejected_at_nested_boundaries() {
    for (value, pointers, decoder) in [
        (
            artifact_json(),
            vec![
                "",
                "/source",
                "/files/0",
                "/files/0/content",
                "/lineage",
                "/tokenizer",
                "/tokenizer/value",
                "/tokenizer/value/source",
            ],
            0,
        ),
        (
            request_json(),
            vec![
                "",
                "/semantics",
                "/semantics/client",
                "/semantics/artifact",
                "/semantics/artifact/value",
                "/policy_evidence/value/0",
            ],
            1,
        ),
        (
            policy_json(),
            vec![
                "",
                "/subject",
                "/document",
                "/document/source",
                "/document/content",
            ],
            2,
        ),
        (
            deployment_json(),
            vec![
                "",
                "/method",
                "/verifier",
                "/verifier/value",
                "/raw_evidence",
                "/effective",
                "/validity",
                "/validity/revocation/value",
            ],
            3,
        ),
        (
            observed_json(),
            vec![
                "",
                "/attempt",
                "/requested/value",
                "/deployment_evidence/value",
                "/termination",
            ],
            4,
        ),
    ] {
        for pointer in pointers {
            let mut value = value.clone();
            value
                .pointer_mut(pointer)
                .unwrap()
                .as_object_mut()
                .unwrap()
                .insert("verified".into(), json!(true));
            let input = value.to_string();
            let rejected = match decoder {
                0 => PinnedModelArtifact::from_json(&input).is_err(),
                1 => RequestedModelExecution::from_json(&input).is_err(),
                2 => ModelPolicyEvidence::from_json(&input).is_err(),
                3 => ModelDeploymentEvidence::from_json(&input).is_err(),
                _ => ObservedModelExecution::from_json(&input).is_err(),
            };
            assert!(
                rejected,
                "accepted forged flag at decoder {decoder}, {pointer}"
            );
        }
    }
    for value in [
        json!({"status":"verified","value":identity('a')}),
        json!({"status":"unknown","verified":true}),
    ] {
        assert!(
            ModelExecutionSemantics::from_json(
                &replace(&semantics_json(), "/artifact", value).to_string()
            )
            .is_err()
        );
    }
}

#[test]
fn unsupported_versions_and_invalid_nested_declarations_are_rejected_by_deserialization() {
    for pointer in [
        "/version",
        "/semantics/version",
        "/semantics/artifact/value/version",
        "/policy_evidence/value/0/version",
    ] {
        let input = replace(&request_json(), pointer, json!(2)).to_string();
        assert!(
            serde_json::from_str::<RequestedModelExecution>(&input).is_err(),
            "accepted {pointer}"
        );
    }
    for pointer in [
        "/version",
        "/effective/version",
        "/claimed_loaded_artifacts/value/0/version",
    ] {
        assert!(
            ModelDeploymentEvidence::from_json(
                &replace(&deployment_json(), pointer, json!(2)).to_string()
            )
            .is_err()
        );
    }
    for pointer in [
        "/version",
        "/requested/value/version",
        "/deployment_evidence/value/version",
    ] {
        assert!(
            ObservedModelExecution::from_json(
                &replace(&observed_json(), pointer, json!(2)).to_string()
            )
            .is_err()
        );
    }
    assert!(
        PinnedModelArtifact::from_json(
            &replace(&artifact_json(), "/version", json!(2)).to_string()
        )
        .is_err()
    );
    assert!(
        ModelPolicyEvidence::from_json(&replace(&policy_json(), "/version", json!(2)).to_string())
            .is_err()
    );
    for (pointer, value) in [
        ("/files", json!([])),
        ("/source/revision", json!("")),
        ("/files/0/content/hex", json!("A".repeat(64))),
        ("/files/0/content/hex", json!("a".repeat(63))),
        ("/files/0/content/algorithm", json!("md5")),
        ("/files/0/purpose", json!("other")),
        ("/tokenizer/value/file/purpose", json!("weights")),
        ("/tokenizer/value/file/content/hex", json!("d".repeat(64))),
    ] {
        assert!(
            PinnedModelArtifact::from_json(&replace(&artifact_json(), pointer, value).to_string())
                .is_err(),
            "accepted {pointer}"
        );
    }
    for (pointer, value) in [
        ("/validity/expires_at_unix_ms", json!(100)),
        ("/effective/client/configuration", json!([])),
        ("/instance", json!("")),
    ] {
        assert!(
            ModelDeploymentEvidence::from_json(
                &replace(&deployment_json(), pointer, value).to_string()
            )
            .is_err()
        );
    }
}

#[test]
fn unsafe_and_duplicate_file_identifiers_are_rejected() {
    for path in [
        "",
        "/model.bin",
        "../model.bin",
        "a/../b",
        "a/./b",
        "a//b",
        "a\\b",
        "C:weights.bin",
        "a%2fb",
        "a\nb",
        "CON",
        "aux.bin",
        "COM1.txt",
        "a.",
        "a ",
        "mödél.bin",
    ] {
        assert!(
            PinnedModelArtifact::from_json(
                &replace(&artifact_json(), "/files/0/path", json!(path)).to_string()
            )
            .is_err(),
            "accepted {path:?}"
        );
    }
    for path in ["weights/model.safetensors", "WEIGHTS/MODEL.SAFETENSORS"] {
        let mut value = artifact_json();
        value["files"]
            .as_array_mut()
            .unwrap()
            .push(file(path, "configuration", 'f'));
        assert!(PinnedModelArtifact::from_json(&value.to_string()).is_err());
    }
}

#[test]
fn credential_forms_fail_without_echoing_secrets() {
    for reference in [
        "https://user:sensitive-sentinel@host.test/v1",
        "https://host.test/v1?key=sensitive-sentinel",
        "https://host.test/#sensitive-sentinel",
        "https://sensitive-sentinel%40host.test/v1",
        "https://host.test/%2e%2e/sensitive-sentinel",
        "https://host.test\\sensitive-sentinel",
        "https://host.test:99999/sensitive-sentinel",
        "https://host.test/../sensitive-sentinel",
        "https://[::1]sensitive-sentinel",
        "file:///sensitive-sentinel",
        "urn:evidence:sensitive-sentinel?key=x",
    ] {
        let error = ModelReference::new(reference).unwrap_err();
        assert!(!format!("{error:?}: {error}").contains("sensitive-sentinel"));
        let input = replace(&request_json(), "/endpoint", json!(reference)).to_string();
        for error in [
            RequestedModelExecution::from_json(&input)
                .unwrap_err()
                .to_string(),
            serde_json::from_str::<RequestedModelExecution>(&input)
                .unwrap_err()
                .to_string(),
        ] {
            assert!(!error.contains("sensitive-sentinel"));
        }
        let input = replace(
            &policy_json(),
            "/document/source/reference",
            json!(reference),
        )
        .to_string();
        assert!(
            !ModelPolicyEvidence::from_json(&input)
                .unwrap_err()
                .to_string()
                .contains("sensitive-sentinel")
        );
    }
    for value in [
        "http://localhost:8080/v1",
        "https://127.0.0.1/v1",
        "https://[::1]:443/v1",
        "urn:evidence:report-1",
    ] {
        let reference = ModelReference::new(value).unwrap();
        assert_eq!(reference.as_str(), value);
    }
    let input = replace(&request_json(), "/endpoint", json!("urn:service:chat")).to_string();
    assert!(RequestedModelExecution::from_json(&input).is_err());
    let input = replace(
        &artifact_json(),
        "/lineage/kind",
        json!("sensitive-sentinel"),
    )
    .to_string();
    assert!(
        !serde_json::from_str::<PinnedModelArtifact>(&input)
            .unwrap_err()
            .to_string()
            .contains("sensitive-sentinel")
    );
}

#[test]
fn canonical_artifact_identity_ignores_order_and_label_but_pins_lineage_and_bytes() {
    let original = artifact(&artifact_json());
    let id = original.identity().unwrap();
    let mut reordered = original.clone();
    reordered.files.reverse();
    reordered.label = "Renamed display label".into();
    assert_eq!(id, reordered.identity().unwrap());
    for (pointer, value) in [
        (
            "/source/reference",
            json!("https://mirror.example.test/model"),
        ),
        ("/source/revision", json!("commit-a2")),
        ("/files/0/content/hex", json!("f".repeat(64))),
        ("/files/0/content/algorithm", json!("blake3")),
        ("/files/0/path", json!("weights/shard.safetensors")),
        ("/files/0/purpose", json!("checkpoint")),
        ("/tokenizer", unknown()),
        ("/chat_template", unknown()),
        (
            "/lineage",
            json!({"kind":"quantized","base":identity('d'),"quantization":semantic("q4")}),
        ),
    ] {
        let mut value = replace(&artifact_json(), pointer, value);
        // A changed artifact source retains explicit component pins to the original source.
        if pointer == "/source/revision" || pointer == "/source/reference" {
            assert_ne!(value["source"], value["tokenizer"]["value"]["source"]);
        }
        assert_ne!(
            id,
            artifact(&value).identity().unwrap(),
            "unchanged at {pointer}"
        );
        // Prevent accidental reliance on the fixture's mutable object identity.
        value["label"] = json!("another display label");
        assert_ne!(id, artifact(&value).identity().unwrap());
    }
    let lineage = json!({"kind":"derived", "base":identity('d'),
        "additional_parents":declared(json!([identity('f'),identity('e')])), "transformation":semantic("merge")});
    let mut value = replace(&artifact_json(), "/lineage", lineage);
    let id = artifact(&value).identity().unwrap();
    value["lineage"]["additional_parents"]["value"]
        .as_array_mut()
        .unwrap()
        .reverse();
    assert_eq!(id, artifact(&value).identity().unwrap());
    value["lineage"]["additional_parents"]["value"] = json!([identity('e'), identity('e')]);
    assert!(PinnedModelArtifact::from_json(&value.to_string()).is_err());
    value["lineage"]["additional_parents"]["value"] = json!([identity('d')]);
    assert!(PinnedModelArtifact::from_json(&value.to_string()).is_err());
}

#[test]
fn semantic_changes_and_evidence_bindings_are_distinct() {
    let requested = RequestedModelExecution::from_json(&request_json().to_string()).unwrap();
    let semantic_id = requested.semantic_identity().unwrap();
    for (pointer, value) in [
        ("/alias", json!("different/model")),
        ("/operation", json!("embedding")),
        ("/artifact/value/digest", json!("f".repeat(64))),
        ("/tokenizer/value/file/content/hex", json!("f".repeat(64))),
        (
            "/chat_template/value/file/content/hex",
            json!("f".repeat(64)),
        ),
        ("/runtime/value/revision", json!("2")),
        ("/parser/value/revision", json!("2")),
        ("/serving_profile/value/revision", json!("2")),
        ("/client/revision", json!("2")),
        ("/configuration/value/hex", json!("e".repeat(64))),
        ("/artifact", unknown()),
    ] {
        assert_ne!(
            semantic_id,
            semantics(&replace(&semantics_json(), pointer, value))
                .identity()
                .unwrap(),
            "unchanged at {pointer}"
        );
    }
    let declared_empty = semantics(&replace(
        &semantics_json(),
        "/additional_artifacts",
        declared(json!([])),
    ));
    let undeclared = semantics(&replace(
        &semantics_json(),
        "/additional_artifacts",
        unknown(),
    ));
    assert_ne!(
        declared_empty.identity().unwrap(),
        undeclared.identity().unwrap()
    );
    let mut changed = requested.clone();
    changed.endpoint =
        ModelReference::new("https://replica.example.test/v1/chat/completions").unwrap();
    if let Declaration::Declared(policies) = &mut changed.policy_evidence {
        policies.reverse();
    }
    assert_eq!(semantic_id, changed.semantic_identity().unwrap());
    if let Declaration::Declared(policies) = &mut changed.policy_evidence {
        policies.push(policies[0].clone());
    }
    assert!(changed.semantic_identity().is_err());
    let original = deployment(&deployment_json());
    assert_eq!(semantic_id, original.effective.identity().unwrap());
    let binding = original.identity().unwrap();
    for (pointer, value) in [
        ("/incarnation", json!("boot-b")),
        ("/instance", json!("server-b")),
        (
            "/endpoint",
            json!("https://replica.example.test/v1/chat/completions"),
        ),
        ("/raw_evidence/content/hex", json!("d".repeat(64))),
        ("/validity/expires_at_unix_ms", json!(201)),
        ("/validity/revocation/value/observed_at_unix_ms", json!(121)),
    ] {
        let value = deployment(&replace(&deployment_json(), pointer, value));
        assert_ne!(binding, value.identity().unwrap());
        assert_eq!(semantic_id, value.effective.identity().unwrap());
    }
    let mismatched = deployment(&replace(
        &deployment_json(),
        "/effective/artifact/value/digest",
        json!("f".repeat(64)),
    ));
    assert_ne!(semantic_id, mismatched.effective.identity().unwrap());
    let mut reordered = original.clone();
    if let Declaration::Declared(values) = &mut reordered.claimed_loaded_artifacts {
        values.reverse();
    }
    if let Declaration::Declared(values) = &mut reordered.effective.additional_artifacts {
        values.reverse();
    }
    assert_eq!(binding, reordered.identity().unwrap());
    if let Declaration::Declared(values) = &mut reordered.claimed_loaded_artifacts {
        values.push(values[0].clone());
    }
    assert!(reordered.identity().is_err());
}

#[test]
fn artifact_lineage_changes_flow_into_execution_semantics() {
    let mut execution = semantics(&semantics_json());
    let lineage = json!({"kind":"quantized","base":identity('d'),"quantization":semantic("q4")});
    let value = replace(&artifact_json(), "/lineage", lineage);
    execution.artifact = Declaration::Declared(artifact(&value).identity().unwrap());
    let original = execution.identity().unwrap();
    for (pointer, replacement) in [
        ("/lineage/base/digest", json!("e".repeat(64))),
        ("/lineage/quantization/revision", json!("2")),
    ] {
        let artifact = artifact(&replace(&value, pointer, replacement));
        execution.artifact = Declaration::Declared(artifact.identity().unwrap());
        assert_ne!(original, execution.identity().unwrap());
    }
}

#[test]
fn observed_references_preserve_missing_and_bind_the_exact_attempt() {
    let original = observed(&observed_json());
    let binding = original.identity().unwrap();
    for (pointer, value) in [
        ("/attempt/run_id", json!("run-b")),
        ("/attempt/launch_id", json!("launch-b")),
        ("/attempt/attempt_id", json!("attempt-b")),
        ("/attempt/observation_sequence", json!(1)),
        ("/attempt/observation_sequence", Value::Null),
        ("/deployment_evidence", unknown()),
        ("/requested", unknown()),
        ("/termination", Value::Null),
        ("/termination/native", Value::Null),
        ("/termination/normalized", json!("length")),
    ] {
        assert_ne!(
            binding,
            observed(&replace(&observed_json(), pointer, value))
                .identity()
                .unwrap()
        );
    }
    let mut missing = observed_json();
    missing["attempt"]["observation_sequence"] = Value::Null;
    missing["requested"] = unknown();
    missing["deployment_evidence"] = unknown();
    missing["termination"] = Value::Null;
    assert_eq!(serde_json::to_value(observed(&missing)).unwrap(), missing);
    for key in ["cost_usd", "tokens", "model", "provider", "response_id"] {
        let mut value = observed_json();
        value.as_object_mut().unwrap().insert(key.into(), json!(0));
        assert!(ObservedModelExecution::from_json(&value.to_string()).is_err());
    }
}

#[test]
fn policy_identity_pins_role_use_terms_and_document() {
    let original = ModelPolicyEvidence::from_json(&policy_json().to_string())
        .unwrap()
        .identity()
        .unwrap();
    for (pointer, value) in [
        ("/kind", json!("serving_terms")),
        ("/subject/digest", json!("e".repeat(64))),
        ("/document/source/revision", json!("2")),
        ("/document/source/reference", json!("urn:policy:terms")),
        ("/document/content/hex", json!("b".repeat(64))),
        ("/intended_use", json!("training")),
    ] {
        let changed =
            ModelPolicyEvidence::from_json(&replace(&policy_json(), pointer, value).to_string())
                .unwrap();
        assert_ne!(original, changed.identity().unwrap());
    }
    for role in [
        "teacher",
        "judge",
        "user_synthesis",
        "student",
        "derivative",
    ] {
        let changed = ModelPolicyEvidence::from_json(
            &replace(&policy_json(), "/role", json!(role)).to_string(),
        )
        .unwrap();
        assert_ne!(original, changed.identity().unwrap());
    }
}

#[test]
fn semantic_digest_matches_independently_written_canonical_wire_vector() {
    let value = json!({"version":1,"alias":"m","operation":"embedding",
        "client":{"implementation":"client","revision":"1","configuration":{"z":2,"a":1}},
        "serving_profile":unknown(),"artifact":unknown(),"additional_artifacts":unknown(),
        "tokenizer":unknown(),"chat_template":unknown(),"runtime":unknown(),"parser":unknown(),"configuration":unknown()});
    // Explicit sorted field order, including nested keys and unknown statuses; no production
    // serializer or canonicalization helper constructs this reference preimage.
    let canonical = r#"{"additional_artifacts":{"status":"unknown"},"alias":"m","artifact":{"status":"unknown"},"chat_template":{"status":"unknown"},"client":{"configuration":{"a":1,"z":2},"implementation":"client","revision":"1"},"configuration":{"status":"unknown"},"operation":"embedding","parser":{"status":"unknown"},"runtime":{"status":"unknown"},"serving_profile":{"status":"unknown"},"tokenizer":{"status":"unknown"},"version":1}"#;
    let mut hasher = blake3::Hasher::new_derive_key("ghostwriter.model-execution-semantics.v1");
    hasher.update(canonical.as_bytes());
    assert_eq!(
        semantics(&value).identity().unwrap().digest,
        hasher.finalize().to_hex().to_string()
    );
}
