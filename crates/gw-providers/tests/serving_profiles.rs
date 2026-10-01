#[path = "support/serving_profile.rs"]
mod support;
use gw_providers::serving_profile::*;
use gw_schema::*;
use serde_json::json;
use support::*;

#[test]
fn profiles_strictly_decode_without_environment_or_secret_resolution() {
    let mut profile = profile(ServingDialect::VllmV1);
    profile.authentication = ProfileAuthentication::ModalProxy {
        token_id: SecretReference::new("INTENTIONALLY_UNRESOLVED_FIXTURE_ID").unwrap(),
        token_secret: SecretReference::new("INTENTIONALLY_UNRESOLVED_FIXTURE_SECRET").unwrap(),
    };
    let encoded = serde_json::to_string(&profile).unwrap();
    let parsed = ServingProfile::from_json(&encoded).unwrap();
    assert_eq!(parsed, profile);
    assert_eq!(
        prepared(&parsed).endpoint().as_str(),
        "https://serve.example.test/v1/chat/completions"
    );
    let error = resolve_authentication(&parsed.authentication, |_| Ok(None)).unwrap_err();
    assert_eq!(error, AuthenticationError::Missing);
    for pointer in ["", "/behavior", "/operations", "/authentication"] {
        let mut value = serde_json::to_value(&profile).unwrap();
        value
            .pointer_mut(pointer)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("approved".into(), json!("sensitive-sentinel"));
        let error = ServingProfile::from_json(&value.to_string()).unwrap_err();
        assert_eq!(error, ProfileError::InvalidProfile);
        assert!(!format!("{error:?} {error}").contains("sensitive-sentinel"));
    }
    for pointer in ["/version", "/behavior/version"] {
        let mut value = serde_json::to_value(&profile).unwrap();
        *value.pointer_mut(pointer).unwrap() = json!(2);
        assert!(ServingProfile::from_json(&value.to_string()).is_err());
    }
}

#[test]
fn route_secret_replica_and_operational_defaults_do_not_enter_behavior_identity() {
    let original = profile(ServingDialect::OpenRouterV1);
    let expected = prepared(&original).target().clone();
    let mut changed = original.clone();
    changed.endpoint = "https://replacement.example.test/another-api".into();
    changed.authentication = ProfileAuthentication::Bearer {
        secret: SecretReference::new("ALTERNATE_REFERENCE").unwrap(),
    };
    changed.routing.as_mut().unwrap().only = vec!["alternate-provider".into()];
    changed.operations = ProfileOperations {
        accounting: AccountingPolicy::FiniteUsd { limit_usd: 9.0 },
        max_in_flight: 32,
        requests_per_minute: Some(400),
        backend_batch_capacity: Some(64),
        replicas: Some(ReplicaPolicy {
            min_containers: 2,
            max_containers: 8,
            buffer_containers: 1,
            target_concurrency: 40,
        }),
        request_timeout_ms: 80_000,
        idle_stream_timeout_ms: 10_000,
        max_attempts: 4,
    };
    assert_ne!(
        serde_json::to_value(&original).unwrap(),
        serde_json::to_value(&changed).unwrap()
    );
    assert_eq!(
        original.behavior.declaration().unwrap(),
        changed.behavior.declaration().unwrap()
    );
    assert_eq!(expected, *prepared(&changed).target());
    assert_ne!(
        prepared(&original).body_digest(),
        prepared(&changed).body_digest()
    );
    assert_ne!(
        prepared(&original).endpoint(),
        prepared(&changed).endpoint()
    );
    let declaration = serde_json::to_string(&changed.behavior.declaration().unwrap()).unwrap();
    for absent in [
        "replacement.example",
        "ALTERNATE_REFERENCE",
        "max_in_flight",
        "accounting",
        "replicas",
        "alternate-provider",
    ] {
        assert!(!declaration.contains(absent));
    }
}

#[test]
fn behavior_changes_affect_identity_but_declared_set_order_does_not() {
    let original = profile(ServingDialect::VllmV1);
    let baseline = prepared(&original).target().clone();
    let mut reordered = original.clone();
    reordered.behavior.operations.reverse();
    reordered.behavior.capabilities.reverse();
    reordered.behavior.reasoning_efforts.reverse();
    assert_eq!(baseline, *prepared(&reordered).target());
    let mut changed = original.clone();
    changed
        .behavior
        .capabilities
        .retain(|value| *value != ProfileControl::Seed);
    assert_ne!(baseline, *prepared(&changed).target());
    changed = original.clone();
    changed.behavior.reasoning_efforts.push("high".into());
    assert_ne!(baseline, *prepared(&changed).target());
    let other = profile(ServingDialect::OpenRouterV1);
    assert_ne!(baseline, *prepared(&other).target());
    assert_eq!(
        prepare_request(
            &changed,
            &semantics(&original, ModelOperation::ChatCompletion),
            &chat()
        )
        .unwrap_err(),
        ProfileError::InvalidSemantics
    );
}

#[test]
fn malformed_profiles_cannot_skip_validation_through_public_fields() {
    let baseline = profile(ServingDialect::VllmV1);
    let mut variants = Vec::new();
    let mut value = baseline.clone();
    value.endpoint = "https://user:sensitive-sentinel@serve.example/v1".into();
    variants.push(value);
    let mut value = baseline.clone();
    value.endpoint = "https://serve.example/v1?token=sensitive-sentinel".into();
    variants.push(value);
    let mut value = baseline.clone();
    value
        .behavior
        .capabilities
        .push(ProfileControl::Temperature);
    variants.push(value);
    let mut value = baseline.clone();
    value.behavior.reasoning_efforts.push("max".into());
    variants.push(value);
    let mut value = baseline.clone();
    value
        .behavior
        .capabilities
        .push(ProfileControl::ReasoningBudget);
    variants.push(value);
    let mut value = baseline.clone();
    value.operations.max_in_flight = 0;
    variants.push(value);
    let mut value = baseline.clone();
    value.operations.accounting = AccountingPolicy::FiniteUsd {
        limit_usd: f64::NAN,
    };
    variants.push(value);
    let mut value = baseline.clone();
    value.operations.requests_per_minute = Some(0);
    variants.push(value);
    let mut value = baseline.clone();
    value.operations.replicas = Some(ReplicaPolicy {
        min_containers: 9,
        max_containers: 2,
        buffer_containers: 1,
        target_concurrency: 1,
    });
    variants.push(value);
    let mut value = baseline.clone();
    value.routing = profile(ServingDialect::OpenRouterV1).routing;
    variants.push(value);
    let mut value = profile(ServingDialect::OpenRouterV1);
    value.routing = None;
    variants.push(value);
    for value in variants {
        assert_eq!(
            prepare_request(
                &value,
                &semantics(&baseline, ModelOperation::ChatCompletion),
                &chat()
            )
            .unwrap_err(),
            ProfileError::InvalidProfile
        );
    }
}

#[test]
fn modal_defaults_are_observation_only_and_pair_is_required_and_sensitive() {
    let id = SecretReference::new("PROXY_ID").unwrap();
    let secret = SecretReference::new("PROXY_SECRET").unwrap();
    let profile = ServingProfile::modal(
        "https://fixture.modal.run/v1",
        id.clone(),
        secret.clone(),
        profile(ServingDialect::VllmV1).behavior,
    )
    .unwrap();
    assert_eq!(
        profile.operations.accounting,
        AccountingPolicy::ObservationOnly
    );
    assert!(profile.operations.max_in_flight > 1);
    let mut resolved = Vec::new();
    let headers = resolve_authentication(&profile.authentication, |reference| {
        resolved.push(reference.clone());
        Ok(Some(SecretValue::new(if reference == &id {
            "fixture-id-value"
        } else {
            "fixture-secret-value"
        })))
    })
    .unwrap();
    assert_eq!(resolved, [id.clone(), secret]);
    assert_eq!(headers.headers().len(), 2);
    assert_eq!(headers.headers()["Modal-Key"], "fixture-id-value");
    assert_eq!(headers.headers()["Modal-Secret"], "fixture-secret-value");
    assert!(headers.headers()["Modal-Key"].is_sensitive());
    assert!(headers.headers()["Modal-Secret"].is_sensitive());
    assert!(headers.headers().get("Authorization").is_none());
    let debug = format!(
        "{headers:?} {:?} {:?}",
        headers.headers(),
        SecretValue::new("fixture-secret-value")
    );
    assert!(!debug.contains("fixture-id-value") && !debug.contains("fixture-secret-value"));
    assert!(
        !serde_json::to_string(&profile)
            .unwrap()
            .contains("fixture-secret-value")
    );
    let invalid = ProfileAuthentication::ModalProxy {
        token_id: id.clone(),
        token_secret: id,
    };
    assert_eq!(
        resolve_authentication(&invalid, |_| panic!("must not resolve invalid pair")).unwrap_err(),
        AuthenticationError::InvalidConfiguration
    );
}

#[test]
fn missing_empty_invalid_or_failed_secret_resolution_never_falls_back() {
    let policy = ProfileAuthentication::ModalProxy {
        token_id: SecretReference::new("ID").unwrap(),
        token_secret: SecretReference::new("SECRET").unwrap(),
    };
    for missing in ["ID", "SECRET"] {
        assert_eq!(
            resolve_authentication(&policy, |r| Ok(
                (r.as_str() != missing).then(|| SecretValue::new("fixture"))
            ))
            .unwrap_err(),
            AuthenticationError::Missing
        );
    }
    for bad in [
        "",
        " ",
        "with spaces",
        "line\nbreak",
        "carriage\rreturn",
        "tab\there",
        "nonascii-☃",
    ] {
        for bad_reference in ["ID", "SECRET"] {
            let error = resolve_authentication(&policy, |r| {
                Ok(Some(SecretValue::new(if r.as_str() == bad_reference {
                    bad
                } else {
                    "fixture"
                })))
            })
            .unwrap_err();
            assert_eq!(error, AuthenticationError::InvalidValue);
        }
    }
    assert_eq!(
        resolve_authentication(&policy, |_| Err(AuthenticationError::Resolver)).unwrap_err(),
        AuthenticationError::Resolver
    );
    for value in [
        json!({"kind":"modal_proxy","token_id":"ID"}),
        json!({"kind":"modal_proxy","token_secret":"SECRET"}),
        json!({"kind":"none","secret":"sensitive-sentinel"}),
    ] {
        assert!(serde_json::from_value::<ProfileAuthentication>(value).is_err());
    }
}

#[test]
fn no_auth_never_resolves_and_bearer_is_explicitly_separate() {
    let resolved = resolve_authentication(&ProfileAuthentication::None {}, |_| {
        panic!("no resolver for no-auth")
    })
    .unwrap();
    assert!(resolved.headers().is_empty());
    let policy = ProfileAuthentication::Bearer {
        secret: SecretReference::new("BEARER_REFERENCE").unwrap(),
    };
    let resolved = resolve_authentication(&policy, |r| {
        assert_eq!(r.as_str(), "BEARER_REFERENCE");
        Ok(Some(SecretValue::new("fixture-bearer-value")))
    })
    .unwrap();
    assert_eq!(resolved.headers().len(), 1);
    assert_eq!(
        resolved.headers()["authorization"],
        "Bearer fixture-bearer-value"
    );
    assert!(resolved.headers()["authorization"].is_sensitive());
}
