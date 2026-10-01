use super::{binding::*, *};
use std::collections::BTreeMap;

pub(super) struct EvidenceIndex<'a> {
    pub artifacts: BTreeMap<String, &'a PinnedModelArtifact>,
    pub policies: BTreeMap<String, (&'a SuppliedModelPolicy, String)>,
    pub artifact_bindings: Vec<String>,
    pub policy_bindings: Vec<(String, String)>,
}

pub(super) fn deny(
    denials: &mut Vec<ArtifactDenial>,
    artifact: Option<&ArtifactIdentity>,
    reason: ArtifactDenialReason,
) {
    denials.push(ArtifactDenial {
        artifact: artifact.cloned(),
        reason,
    });
}

pub(super) fn index<'a>(
    bundle: ArtifactEvidenceBundle<'a>,
    denials: &mut Vec<ArtifactDenial>,
) -> Result<EvidenceIndex<'a>, ArtifactAssessmentError> {
    let mut result = EvidenceIndex {
        artifacts: BTreeMap::new(),
        policies: BTreeMap::new(),
        artifact_bindings: Vec::new(),
        policy_bindings: Vec::new(),
    };
    for artifact in bundle.artifacts {
        match artifact.identity() {
            Ok(identity) => {
                result.artifact_bindings.push(identity.digest.clone());
                if result
                    .artifacts
                    .insert(identity.digest.clone(), artifact)
                    .is_some()
                {
                    deny(
                        denials,
                        Some(&identity),
                        ArtifactDenialReason::DuplicateArtifact,
                    );
                }
            }
            Err(_) => {
                deny(denials, None, ArtifactDenialReason::InvalidArtifact);
                result.artifact_bindings.push(hash(
                    "ghostwriter.artifact-invalid-submission.v1",
                    artifact,
                )?);
            }
        }
    }
    let mut policy_scopes = BTreeMap::new();
    for policy in bundle.policies {
        let bytes = bytes_digest(&policy.bytes);
        match policy.declaration.identity() {
            Ok(identity) => {
                let declaration = &policy.declaration;
                let scope = hash(
                    "ghostwriter.policy-submission-scope.v1",
                    &(
                        &declaration.subject,
                        declaration.kind,
                        declaration.role,
                        declaration.intended_use,
                    ),
                )?;
                if policy_scopes
                    .insert(scope, identity.digest.clone())
                    .is_some_and(|previous| previous != identity.digest)
                {
                    deny(
                        denials,
                        Some(&declaration.subject),
                        ArtifactDenialReason::MultiplePoliciesForScope,
                    );
                }
                result
                    .policy_bindings
                    .push((identity.digest.clone(), bytes.clone()));
                if !policy_bytes_match(policy) {
                    deny(
                        denials,
                        Some(&policy.declaration.subject),
                        ArtifactDenialReason::PolicyContentMismatch,
                    );
                }
                if result
                    .policies
                    .insert(identity.digest, (policy, bytes))
                    .is_some()
                {
                    deny(
                        denials,
                        Some(&policy.declaration.subject),
                        ArtifactDenialReason::DuplicatePolicy,
                    );
                }
            }
            Err(_) => {
                deny(denials, None, ArtifactDenialReason::InvalidPolicy);
                result.policy_bindings.push((
                    hash(
                        "ghostwriter.policy-invalid-submission.v1",
                        &policy.declaration,
                    )?,
                    bytes,
                ));
            }
        }
    }
    result.artifact_bindings.sort();
    result.policy_bindings.sort();
    Ok(result)
}
