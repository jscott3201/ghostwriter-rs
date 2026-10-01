use super::{binding::hash, catalog::StoredPolicyReview, evidence::deny, *};
use std::collections::BTreeSet;

/// Assess only the offline artifact and reviewed intended-use scope. No positive result authorizes
/// deployment, credentials, cache reuse, or model calls. Every resolved ancestor requires its own
/// exact review for the same requested role/use; child rights are never inherited by parents.
///
/// An empty trusted catalog cannot be supplemented with approvals from the submitted bundle:
///
/// ```
/// use gw_providers::artifact_assessment::{
///     assess_artifact, ArtifactEvidenceBundle, TrustedArtifactCatalog,
/// };
/// use gw_schema::{ModelIntendedUse, ModelPolicyRole, PinnedModelArtifact};
/// # fn inspect(artifact: PinnedModelArtifact) -> Result<(), Box<dyn std::error::Error>> {
/// let catalog = TrustedArtifactCatalog::new("owner-empty-v1", vec![])?;
/// let report = assess_artifact(
///     &catalog,
///     &artifact.identity()?,
///     ModelPolicyRole::Teacher,
///     ModelIntendedUse::DatasetGeneration,
///     ArtifactEvidenceBundle { artifacts: std::slice::from_ref(&artifact), policies: &[] },
/// )?;
/// assert!(!report.is_eligible());
/// # Ok(())
/// # }
/// ```
///
/// # Errors
/// Invalid submitted declarations produce structured denials. An error is returned only if the
/// canonical report binding cannot be encoded.
pub fn assess_artifact(
    catalog: &TrustedArtifactCatalog,
    requested: &ArtifactIdentity,
    role: ModelPolicyRole,
    intended_use: ModelIntendedUse,
    bundle: ArtifactEvidenceBundle<'_>,
) -> Result<ArtifactAssessment, ArtifactAssessmentError> {
    let mut report = ArtifactAssessment {
        version: 1,
        artifact: requested.validate().ok().map(|()| requested.clone()),
        role,
        intended_use,
        catalog: catalog.identity().clone(),
        matched_reviews: vec![],
        denials: vec![],
        evidence_digest: String::new(),
    };
    let evidence = evidence::index(bundle, &mut report.denials)?;
    let mut pending = Vec::new();
    if report.artifact.is_some() {
        pending.push(requested.clone());
    } else {
        deny(
            &mut report.denials,
            None,
            ArtifactDenialReason::InvalidRequestedIdentity,
        );
    }
    let mut visited = BTreeSet::new();
    // Iterative traversal handles shared ancestors without recursive stack growth. Every node
    // comes from a recomputed content identity; supplied aliases or claimed IDs are not indexes.
    while let Some(identity) = pending.pop() {
        if !visited.insert(identity.digest.clone()) {
            continue;
        }
        let Some(artifact) = evidence.artifacts.get(&identity.digest) else {
            deny(
                &mut report.denials,
                Some(&identity),
                ArtifactDenialReason::MissingArtifact,
            );
            continue;
        };
        pending.extend(lineage::parents(
            &identity,
            artifact,
            &evidence.artifacts,
            &mut report.denials,
        ));
        let Some(review) = catalog.review(&identity, role, intended_use) else {
            deny(
                &mut report.denials,
                Some(&identity),
                ArtifactDenialReason::MissingTrustedReview,
            );
            continue;
        };
        match review.binding.openness {
            ReviewedModelOpenness::Closed => deny(
                &mut report.denials,
                Some(&identity),
                ArtifactDenialReason::ClosedArtifact,
            ),
            ReviewedModelOpenness::Unknown => deny(
                &mut report.denials,
                Some(&identity),
                ArtifactDenialReason::UnknownOpenness,
            ),
            ReviewedModelOpenness::OpenWeights | ReviewedModelOpenness::OpenTrainingArtifacts => {}
        }
        let mut policy_documents = Vec::new();
        for axis in [
            &review.binding.rights,
            &review.binding.serving_terms,
            &review.binding.output_terms,
        ] {
            match axis {
                StoredPolicyReview::Unreviewed => deny(
                    &mut report.denials,
                    Some(&identity),
                    ArtifactDenialReason::UnreviewedPolicy,
                ),
                StoredPolicyReview::NotApplicable { .. } => {}
                StoredPolicyReview::Reviewed {
                    identity: policy,
                    bytes_digest,
                    ..
                } => {
                    policy_documents.push(policy.clone());
                    match evidence.policies.get(&policy.digest) {
                        None => deny(
                            &mut report.denials,
                            Some(&identity),
                            ArtifactDenialReason::MissingPolicy,
                        ),
                        Some((_, actual)) if actual != bytes_digest => deny(
                            &mut report.denials,
                            Some(&identity),
                            ArtifactDenialReason::PolicyContentMismatch,
                        ),
                        Some(_) => {}
                    }
                }
            }
        }
        policy_documents.sort_by(|a, b| a.digest.cmp(&b.digest));
        report.matched_reviews.push(MatchedArtifactReview {
            artifact: identity,
            reference: review.binding.reference.clone(),
            reviewed_at_unix_ms: review.binding.reviewed_at_unix_ms,
            review_digest: review.digest.clone(),
            policy_documents,
        });
    }
    report
        .matched_reviews
        .sort_by(|a, b| a.artifact.digest.cmp(&b.artifact.digest));
    report.denials.sort_by(|a, b| {
        a.artifact
            .as_ref()
            .map(|identity| &identity.digest)
            .cmp(&b.artifact.as_ref().map(|identity| &identity.digest))
            .then(a.reason.cmp(&b.reason))
    });
    report.denials.dedup();
    report.evidence_digest = hash(
        "ghostwriter.artifact-assessment.v1",
        &(
            report.version,
            requested,
            role,
            intended_use,
            &report.catalog,
            evidence.artifact_bindings,
            evidence.policy_bindings,
            &report.matched_reviews,
            &report.denials,
        ),
    )?;
    Ok(report)
}
