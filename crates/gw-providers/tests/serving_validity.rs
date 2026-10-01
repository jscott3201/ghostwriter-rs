#[path = "support/serving_profile.rs"]
mod support;
use gw_providers::serving_profile::*;
use gw_schema::*;
use support::*;

#[test]
fn validity_uses_explicit_time_and_exclusive_expiry_without_a_clock() {
    let fixture = GatewayFixture::new();
    for (now, expected, reason) in [
        (
            99,
            ConsistencyState::Mismatch,
            Some(ConsistencyReason::NotYetValid),
        ),
        (
            199,
            ConsistencyState::Mismatch,
            Some(ConsistencyReason::NotYetValid),
        ),
        (200, ConsistencyState::Consistent, None),
        (899, ConsistencyState::Consistent, None),
        (
            900,
            ConsistencyState::Mismatch,
            Some(ConsistencyReason::Expired),
        ),
        (
            1000,
            ConsistencyState::Mismatch,
            Some(ConsistencyReason::Expired),
        ),
    ] {
        let report = check_gateway_consistency(
            &fixture.prepared,
            &fixture.request,
            &fixture.response,
            Some(&fixture.deployment),
            &fixture.artifacts,
            Some(&fixture.validity),
            now,
        );
        assert_eq!(report.validity().state(), expected, "time {now}");
        assert_eq!(report.evaluated_at_unix_ms(), now);
        if let Some(reason) = reason {
            assert!(report.validity().reasons().contains(&reason));
        }
        assert_eq!(report.semantics().state(), ConsistencyState::Consistent);
        assert_eq!(report.binding().state(), ConsistencyState::Consistent);
    }
}

#[test]
fn missing_expiry_or_revocation_remains_unknown_and_cannot_mean_perpetual_validity() {
    let mut fixture = GatewayFixture::new();
    fixture.deployment.validity.expires_at_unix_ms = None;
    fixture.refresh_replica_binding();
    unknown(
        fixture.report().validity(),
        ConsistencyReason::UnknownExpiration,
    );
    let mut fixture = GatewayFixture::new();
    fixture.deployment.validity.revocation = Declaration::Unknown;
    fixture.refresh_replica_binding();
    unknown(
        fixture.report().validity(),
        ConsistencyReason::UnknownRevocation,
    );
    let mut fixture = GatewayFixture::new();
    fixture.validity.revoked = Declaration::Unknown;
    unknown(
        fixture.report().validity(),
        ConsistencyReason::UnknownRevocation,
    );
}

#[test]
fn explicit_revocation_and_expiration_outrank_unrelated_unknowns() {
    let mut fixture = GatewayFixture::new();
    fixture.deployment.validity.expires_at_unix_ms = None;
    fixture.refresh_replica_binding();
    fixture.validity.revoked = Declaration::Declared(true);
    let report = fixture.report();
    mismatch(report.validity(), ConsistencyReason::Revoked);
    assert!(
        report
            .validity()
            .reasons()
            .contains(&ConsistencyReason::UnknownExpiration)
    );
    assert_eq!(report.aggregate(), ConsistencyState::Mismatch);
    let fixture = GatewayFixture::new();
    let expired = check_gateway_consistency(
        &fixture.prepared,
        &fixture.request,
        &fixture.response,
        Some(&fixture.deployment),
        &fixture.artifacts,
        None,
        1000,
    );
    mismatch(expired.validity(), ConsistencyReason::Expired);
    assert!(
        expired
            .validity()
            .reasons()
            .contains(&ConsistencyReason::MissingValidity)
    );
}

#[test]
fn revocation_interpretation_is_bound_to_exact_document_observation_and_replica() {
    for changed in [0, 1, 2, 3] {
        let mut fixture = GatewayFixture::new();
        let expected = match changed {
            0 => {
                fixture.validity.deployment_evidence.digest = "a".repeat(64);
                ConsistencyReason::DeploymentIdentityMismatch
            }
            1 => {
                fixture.validity.revocation.document.content.hex = "a".repeat(64);
                ConsistencyReason::RevocationBindingMismatch
            }
            2 => {
                fixture.validity.revocation.document.source.revision = "other".into();
                ConsistencyReason::RevocationBindingMismatch
            }
            _ => {
                fixture.validity.revocation.observed_at_unix_ms = 201;
                ConsistencyReason::RevocationBindingMismatch
            }
        };
        mismatch(fixture.report().validity(), expected);
    }
    let mut fixture = GatewayFixture::new();
    fixture.deployment.incarnation = "generation-2".into();
    fixture.response.deployment_evidence =
        Declaration::Declared(fixture.deployment.identity().unwrap());
    mismatch(
        fixture.report().validity(),
        ConsistencyReason::DeploymentIdentityMismatch,
    );
}

#[test]
fn malformed_versions_intervals_and_public_fields_never_report_current() {
    for changed in [0, 1, 2, 3] {
        let mut fixture = GatewayFixture::new();
        match changed {
            0 => fixture.validity.version = 2,
            1 => {
                fixture.validity.valid_until_unix_ms =
                    fixture.validity.revocation.observed_at_unix_ms
            }
            2 => fixture.validity.revocation.document.content.hex = "sensitive-sentinel".into(),
            _ => {
                fixture.deployment.validity.expires_at_unix_ms =
                    Some(fixture.deployment.validity.not_before_unix_ms)
            }
        }
        let report = fixture.report();
        mismatch(report.validity(), ConsistencyReason::InvalidValidity);
        assert_eq!(report.aggregate(), ConsistencyState::Mismatch);
        assert!(
            !serde_json::to_string(&report)
                .unwrap()
                .contains("sensitive-sentinel")
        );
    }
}

#[test]
fn consistency_reports_retain_attempt_body_endpoint_and_correlation_binding() {
    let fixture = GatewayFixture::new();
    let report = fixture.report();
    let binding = report.request_binding().unwrap();
    assert_eq!(binding.attempt, fixture.request.attempt);
    assert_eq!(binding.endpoint, *fixture.prepared.endpoint());
    assert_eq!(binding.request_body_digest, *fixture.prepared.body_digest());
    assert_eq!(binding.correlation_id, fixture.request.correlation_id);
    let mut other = fixture.clone();
    other.request = gateway_request(&other.prepared, "attempt-2");
    other.response = gateway_response(&other.request, &other.deployment);
    assert_eq!(other.report().aggregate(), ConsistencyState::Consistent);
    assert_ne!(report, other.report());
    let mut malformed = fixture;
    malformed.request.correlation_id = "invalid\nreference".into();
    assert!(malformed.report().request_binding().is_none());
}
