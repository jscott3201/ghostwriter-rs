#[path = "support/serving_profile.rs"]
mod support;
use gw_providers::serving_profile::*;
use gw_schema::*;
use serde_json::json;
use support::*;

#[test]
fn complete_supplied_snapshots_can_agree_without_any_execution_capability() {
    let fixture = GatewayFixture::new();
    let report = fixture.report();
    assert_eq!(report.aggregate(), ConsistencyState::Consistent);
    for axis in [report.semantics(), report.binding(), report.validity()] {
        assert_eq!(axis.state(), ConsistencyState::Consistent);
        assert!(axis.reasons().is_empty());
    }
    assert_eq!(report.target(), Some(fixture.prepared.target()));
    assert_eq!(
        report.deployment_evidence(),
        Some(&fixture.deployment.identity().unwrap())
    );
    let data = serde_json::to_value(report).unwrap();
    for absent in ["authenticated", "qualified", "authorized", "verified"] {
        assert!(data.get(absent).is_none());
    }
    let request =
        GatewayRequest::from_json(&serde_json::to_string(&fixture.request).unwrap()).unwrap();
    let response =
        GatewayResponseEvidence::from_json(&serde_json::to_string(&fixture.response).unwrap())
            .unwrap();
    let validity =
        SuppliedValiditySnapshot::from_json(&serde_json::to_string(&fixture.validity).unwrap())
            .unwrap();
    assert_eq!(request, fixture.request);
    assert_eq!(response, fixture.response);
    assert_eq!(validity, fixture.validity);
}

#[test]
fn equivalent_replicas_share_target_but_have_distinct_actual_evidence() {
    let first = GatewayFixture::new();
    let mut second = first.clone();
    second.request = gateway_request(&second.prepared, "attempt-2");
    second.deployment = deployment(&second.prepared, "replica-b", "generation-6");
    second.response = gateway_response(&second.request, &second.deployment);
    second.validity = supplied_validity(&second.deployment);
    assert!(
        first.request.replica_constraint.is_none() && second.request.replica_constraint.is_none()
    );
    assert_eq!(first.request.target, second.request.target);
    assert_ne!(
        first.response.deployment_evidence,
        second.response.deployment_evidence
    );
    assert_eq!(first.report().aggregate(), ConsistencyState::Consistent);
    assert_eq!(second.report().aggregate(), ConsistencyState::Consistent);
    assert_ne!(
        first.report().deployment_evidence(),
        second.report().deployment_evidence()
    );
}

#[test]
fn optional_exact_pin_rejects_other_replica_or_reloaded_incarnation() {
    for change_instance in [true, false] {
        let mut fixture = GatewayFixture::new();
        fixture.request.replica_constraint = Some(ReplicaConstraint {
            instance: "replica-a".into(),
            incarnation: "generation-1".into(),
        });
        assert_eq!(fixture.report().aggregate(), ConsistencyState::Consistent);
        if change_instance {
            fixture.deployment.instance = "replica-b".into();
        } else {
            fixture.deployment.incarnation = "generation-2".into();
        }
        fixture.refresh_replica_binding();
        let report = fixture.report();
        mismatch(report.binding(), ConsistencyReason::ReplicaMismatch);
        assert_eq!(report.semantics().state(), ConsistencyState::Consistent);
    }
}

#[test]
fn retry_requires_new_attempt_correlation_and_actual_destination_evidence() {
    let original = GatewayFixture::new();
    let mut retry = original.clone();
    retry.request = gateway_request(&retry.prepared, "attempt-2");
    retry.deployment = deployment(&retry.prepared, "replica-b", "generation-new");
    retry.validity = supplied_validity(&retry.deployment);
    let stale = retry.report();
    for reason in [
        ConsistencyReason::AttemptMismatch,
        ConsistencyReason::CorrelationMismatch,
        ConsistencyReason::DeploymentIdentityMismatch,
    ] {
        mismatch(stale.binding(), reason);
    }
    retry.response = gateway_response(&retry.request, &retry.deployment);
    assert_eq!(retry.report().aggregate(), ConsistencyState::Consistent);
    assert_ne!(retry.request.attempt, original.request.attempt);
    assert_ne!(
        retry.report().deployment_evidence(),
        original.report().deployment_evidence()
    );
}

#[test]
fn mismatched_binding_fields_and_self_reported_body_pins_cannot_agree() {
    let original = GatewayFixture::new();
    for (variant, reason) in [
        (0, ConsistencyReason::RequestBodyMismatch),
        (1, ConsistencyReason::EndpointMismatch),
        (2, ConsistencyReason::CorrelationMismatch),
        (3, ConsistencyReason::AttemptMismatch),
        (4, ConsistencyReason::TargetIdentityMismatch),
        (5, ConsistencyReason::DeploymentIdentityMismatch),
    ] {
        let mut fixture = original.clone();
        match variant {
            0 => fixture.response.request_body_digest.hex = "a".repeat(64),
            1 => {
                fixture.response.endpoint =
                    ModelReference::new("https://other.example.test/v1/chat/completions").unwrap()
            }
            2 => fixture.response.correlation_id = "unrelated-correlation".into(),
            3 => fixture.response.attempt.observation_sequence = Some(1),
            4 => {
                fixture.response.target = Declaration::Declared(SemanticExecutionIdentity {
                    version: 1,
                    digest: "b".repeat(64),
                })
            }
            _ => {
                fixture.response.deployment_evidence =
                    Declaration::Declared(DeploymentEvidenceIdentity {
                        version: 1,
                        digest: "c".repeat(64),
                    })
            }
        }
        mismatch(fixture.report().binding(), reason);
    }
    // Matching request/response claims do not replace the independently prepared client body.
    let mut fixture = original.clone();
    fixture.request.request_body_digest.hex = "a".repeat(64);
    fixture.response.request_body_digest = fixture.request.request_body_digest.clone();
    mismatch(
        fixture.report().binding(),
        ConsistencyReason::RequestBodyMismatch,
    );
    let mut fixture = original.clone();
    fixture.request.endpoint =
        ModelReference::new("https://other.example.test/v1/chat/completions").unwrap();
    fixture.response.endpoint = fixture.request.endpoint.clone();
    fixture.deployment.endpoint = fixture.request.endpoint.clone();
    fixture.refresh_replica_binding();
    mismatch(
        fixture.report().binding(),
        ConsistencyReason::EndpointMismatch,
    );
    let mut fixture = original.clone();
    fixture.request.semantics.alias = "other-model".into();
    fixture.request.target = fixture.request.semantics.identity().unwrap();
    fixture.response.target = Declaration::Declared(fixture.request.target.clone());
    fixture.deployment.effective = fixture.request.semantics.clone();
    fixture.refresh_replica_binding();
    mismatch(
        fixture.report().binding(),
        ConsistencyReason::TargetIdentityMismatch,
    );
}

#[test]
fn a_changed_prepared_body_cannot_reuse_the_old_attempt_binding() {
    let mut fixture = GatewayFixture::new();
    let profile = profile(ServingDialect::VllmV1);
    let mut request = chat();
    request.controls = vec![required(ControlValue::MaxOutputTokens(256))];
    fixture.prepared = prepare_request(
        &profile,
        &semantics(&profile, ModelOperation::ChatCompletion),
        &request,
    )
    .unwrap();
    assert_eq!(fixture.request.target, *fixture.prepared.target());
    mismatch(
        fixture.report().binding(),
        ConsistencyReason::RequestBodyMismatch,
    );
}

#[test]
fn uncertainty_is_separate_for_semantics_binding_and_validity() {
    let mut fixture = GatewayFixture::new();
    fixture.deployment.effective.runtime = Declaration::Unknown;
    fixture.refresh_replica_binding();
    let report = fixture.report();
    unknown(report.semantics(), ConsistencyReason::UnknownSemantics);
    assert_eq!(report.binding().state(), ConsistencyState::Consistent);
    assert_eq!(report.validity().state(), ConsistencyState::Consistent);
    assert_eq!(report.aggregate(), ConsistencyState::Unknown);
    let mut fixture = GatewayFixture::new();
    fixture.response.deployment_evidence = Declaration::Unknown;
    let report = fixture.report();
    unknown(
        report.binding(),
        ConsistencyReason::UnknownDeploymentIdentity,
    );
    assert_eq!(report.semantics().state(), ConsistencyState::Consistent);
    assert_eq!(report.validity().state(), ConsistencyState::Consistent);
    let fixture = GatewayFixture::new();
    let report = check_gateway_consistency(
        &fixture.prepared,
        &fixture.request,
        &fixture.response,
        Some(&fixture.deployment),
        &fixture.artifacts,
        None,
        500,
    );
    unknown(report.validity(), ConsistencyReason::MissingValidity);
    assert_eq!(report.semantics().state(), ConsistencyState::Consistent);
    assert_eq!(report.binding().state(), ConsistencyState::Consistent);
}

#[test]
fn known_semantic_mismatch_outranks_unrelated_unknown_fields() {
    let mut fixture = GatewayFixture::new();
    fixture.deployment.effective.alias = "wrong-loaded-model".into();
    fixture.deployment.effective.runtime = Declaration::Unknown;
    fixture.refresh_replica_binding();
    let report = fixture.report();
    mismatch(report.semantics(), ConsistencyReason::SemanticMismatch);
    assert!(
        report
            .semantics()
            .reasons()
            .contains(&ConsistencyReason::UnknownSemantics)
    );
    assert_eq!(report.aggregate(), ConsistencyState::Mismatch);
    let mut fixture = GatewayFixture::new();
    if let Declaration::Declared(tokenizer) = &mut fixture.deployment.effective.tokenizer {
        tokenizer.file.content.hex = "e".repeat(64);
    }
    fixture.deployment.effective.additional_artifacts = Declaration::Unknown;
    fixture.refresh_replica_binding();
    mismatch(
        fixture.report().semantics(),
        ConsistencyReason::ComponentMismatch,
    );
}

#[test]
fn every_unknown_semantic_field_remains_incomplete_even_when_a_digest_matches() {
    for field in [
        "serving_profile",
        "artifact",
        "additional_artifacts",
        "tokenizer",
        "chat_template",
        "runtime",
        "parser",
        "configuration",
    ] {
        let mut fixture = GatewayFixture::new();
        let mut value = serde_json::to_value(&fixture.deployment.effective).unwrap();
        value[field] = json!({"status":"unknown"});
        fixture.deployment.effective =
            ModelExecutionSemantics::from_json(&value.to_string()).unwrap();
        fixture.refresh_replica_binding();
        let report = fixture.report();
        assert_ne!(report.aggregate(), ConsistencyState::Consistent, "{field}");
        unknown(report.semantics(), ConsistencyReason::UnknownSemantics);
    }
    let mut fixture = GatewayFixture::new();
    fixture.request.semantics.runtime = Declaration::Unknown;
    fixture.request.target = fixture.request.semantics.identity().unwrap();
    fixture.deployment.effective = fixture.request.semantics.clone();
    fixture.response.target = Declaration::Declared(fixture.request.target.clone());
    fixture.refresh_replica_binding();
    unknown(
        fixture.report().semantics(),
        ConsistencyReason::UnknownSemantics,
    );
    assert_ne!(fixture.report().aggregate(), ConsistencyState::Consistent);
}

#[test]
fn artifacts_must_resolve_and_loaded_inventory_must_match_the_target() {
    let mut fixture = GatewayFixture::new();
    fixture.artifacts.clear();
    unknown(
        fixture.report().semantics(),
        ConsistencyReason::MissingArtifact,
    );
    let mut fixture = GatewayFixture::new();
    fixture.artifacts.push(fixture.artifacts[0].clone());
    mismatch(
        fixture.report().semantics(),
        ConsistencyReason::DuplicateArtifact,
    );
    let mut fixture = GatewayFixture::new();
    fixture.artifacts[0].files[0].content.hex = "sensitive-sentinel".into();
    let report = fixture.report();
    mismatch(report.semantics(), ConsistencyReason::InvalidArtifact);
    assert!(
        !serde_json::to_string(&report)
            .unwrap()
            .contains("sensitive-sentinel")
    );
    let mut fixture = GatewayFixture::new();
    fixture.deployment.claimed_loaded_artifacts = Declaration::Unknown;
    fixture.refresh_replica_binding();
    unknown(
        fixture.report().semantics(),
        ConsistencyReason::UnknownLoadedArtifacts,
    );
    let mut fixture = GatewayFixture::new();
    fixture.deployment.claimed_loaded_artifacts = Declaration::Declared(vec![ArtifactIdentity {
        version: 1,
        digest: "a".repeat(64),
    }]);
    fixture.refresh_replica_binding();
    mismatch(
        fixture.report().semantics(),
        ConsistencyReason::LoadedArtifactsMismatch,
    );
}

#[test]
fn artifact_lineage_resolution_does_not_treat_unknown_parents_as_absent() {
    let base = artifact();
    let mut child = base.clone();
    child.source.revision = "derived-1".into();
    child.lineage = ModelArtifactLineage::Derived {
        base: base.identity().unwrap(),
        additional_parents: Declaration::Unknown,
        transformation: declaration("fixture-transform"),
    };
    let mut fixture = GatewayFixture::with_artifact(child.clone());
    fixture.artifacts.push(base.clone());
    unknown(
        fixture.report().semantics(),
        ConsistencyReason::UnknownArtifactLineage,
    );
    child.lineage = ModelArtifactLineage::Derived {
        base: base.identity().unwrap(),
        additional_parents: Declaration::Declared(vec![]),
        transformation: declaration("fixture-transform"),
    };
    let mut fixture = GatewayFixture::with_artifact(child);
    unknown(
        fixture.report().semantics(),
        ConsistencyReason::MissingArtifact,
    );
    fixture.artifacts.push(base);
    assert_eq!(fixture.report().aggregate(), ConsistencyState::Consistent);
    fixture.artifacts.reverse();
    assert_eq!(fixture.report().aggregate(), ConsistencyState::Consistent);
}

#[test]
fn unsupported_installed_only_measurements_and_unknown_verifier_provenance_do_not_qualify() {
    for method in [
        SemanticDeclaration::new(
            "ghostwriter/loaded-generation-report",
            "2",
            json!({"scope":"loaded_generation"}),
        ),
        SemanticDeclaration::new(
            "ghostwriter/loaded-generation-report",
            "1",
            json!({"scope":"installed_files"}),
        ),
        SemanticDeclaration::new("unsupported-health-endpoint", "1", json!({})),
        SemanticDeclaration::new(
            "ghostwriter/loaded-generation-report",
            "1",
            json!({"scope":"loaded_generation","authenticated":true}),
        ),
    ] {
        let mut fixture = GatewayFixture::new();
        fixture.deployment.method = method;
        fixture.refresh_replica_binding();
        mismatch(
            fixture.report().binding(),
            ConsistencyReason::UnsupportedMeasurement,
        );
    }
    let mut fixture = GatewayFixture::new();
    fixture.deployment.verifier = Declaration::Unknown;
    fixture.refresh_replica_binding();
    unknown(
        fixture.report().binding(),
        ConsistencyReason::UnknownVerifier,
    );
    let fixture = GatewayFixture::new();
    let report = check_gateway_consistency(
        &fixture.prepared,
        &fixture.request,
        &fixture.response,
        None,
        &fixture.artifacts,
        Some(&fixture.validity),
        500,
    );
    unknown(
        report.binding(),
        ConsistencyReason::MissingDeploymentEvidence,
    );
    assert_ne!(report.aggregate(), ConsistencyState::Consistent);
}

#[test]
fn unsupported_or_malformed_public_documents_cannot_yield_consistent() {
    let fixture = GatewayFixture::new();
    for version in [0, 2] {
        let mut changed = fixture.clone();
        changed.request.version = version;
        mismatch(
            changed.report().binding(),
            ConsistencyReason::InvalidRequest,
        );
        let mut changed = fixture.clone();
        changed.response.version = version;
        mismatch(
            changed.report().binding(),
            ConsistencyReason::InvalidResponse,
        );
        let mut changed = fixture.clone();
        changed.deployment.version = version;
        mismatch(
            changed.report().binding(),
            ConsistencyReason::InvalidDeployment,
        );
    }
    for (document, decode) in [
        (serde_json::to_value(&fixture.request).unwrap(), 0),
        (serde_json::to_value(&fixture.response).unwrap(), 1),
        (serde_json::to_value(&fixture.validity).unwrap(), 2),
    ] {
        for altered_version in [true, false] {
            let mut value = document.clone();
            if altered_version {
                value["version"] = json!(2);
            } else {
                value["qualified"] = json!("sensitive-sentinel");
            }
            let rejected = match decode {
                0 => GatewayRequest::from_json(&value.to_string()).is_err(),
                1 => GatewayResponseEvidence::from_json(&value.to_string()).is_err(),
                _ => SuppliedValiditySnapshot::from_json(&value.to_string()).is_err(),
            };
            assert!(rejected);
        }
    }
    let mut changed = fixture.clone();
    changed.request.target.digest = "a".repeat(64);
    mismatch(
        changed.report().semantics(),
        ConsistencyReason::TargetIdentityMismatch,
    );
    let mut changed = fixture;
    changed.deployment.effective.chat_template = Declaration::Declared(None);
    changed.refresh_replica_binding();
    mismatch(
        changed.report().semantics(),
        ConsistencyReason::ComponentMismatch,
    );
}
