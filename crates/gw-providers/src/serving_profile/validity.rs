use super::*;
use gw_schema::{Declaration, DeploymentEvidenceIdentity, ModelDeploymentEvidence};

pub(super) fn check(
    deployment: Option<&ModelDeploymentEvidence>,
    identity: Option<&DeploymentEvidenceIdentity>,
    supplied: Option<&SuppliedValiditySnapshot>,
    now: u64,
    axis: &mut ConsistencyAxis,
) {
    use ConsistencyReason as R;
    use ConsistencyState::{Mismatch, Unknown};
    let Some(deployment) = deployment else {
        axis.note(Unknown, R::MissingValidity);
        return;
    };
    if deployment.validity.validate().is_err() {
        axis.note(Mismatch, R::InvalidValidity);
    }
    if now < deployment.validity.not_before_unix_ms {
        axis.note(Mismatch, R::NotYetValid);
    }
    match deployment.validity.expires_at_unix_ms {
        None => axis.note(Unknown, R::UnknownExpiration),
        Some(expires) if now >= expires => axis.note(Mismatch, R::Expired),
        Some(_) => {}
    }
    let Some(supplied) = supplied else {
        axis.note(Unknown, R::MissingValidity);
        return;
    };
    if supplied.validate().is_err() {
        axis.note(Mismatch, R::InvalidValidity);
    }
    if identity != Some(&supplied.deployment_evidence) {
        axis.note(Mismatch, R::DeploymentIdentityMismatch);
    }
    if now < supplied.revocation.observed_at_unix_ms {
        axis.note(Mismatch, R::NotYetValid);
    }
    if now >= supplied.valid_until_unix_ms {
        axis.note(Mismatch, R::Expired);
    }
    match &deployment.validity.revocation {
        Declaration::Unknown => axis.note(Unknown, R::UnknownRevocation),
        Declaration::Declared(expected) if expected != &supplied.revocation => {
            axis.note(Mismatch, R::RevocationBindingMismatch)
        }
        _ => {}
    }
    match supplied.revoked {
        Declaration::Unknown => axis.note(Unknown, R::UnknownRevocation),
        Declaration::Declared(true) => axis.note(Mismatch, R::Revoked),
        Declaration::Declared(false) => {}
    }
}
