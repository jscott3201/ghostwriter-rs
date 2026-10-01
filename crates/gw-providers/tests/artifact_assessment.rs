//! Policy decisions against private independently supplied trusted reviews, with no network I/O.
use gw_providers::artifact_assessment::*;
use gw_schema::*;

#[path = "support/artifact_assessment.rs"]
mod support;
use support::*;

#[test]
fn supplied_exact_artifact_and_policies_do_not_approve_an_empty_catalog() {
    let artifact = artifact("base");
    let (_, policies) = review(&artifact);
    let catalog = TrustedArtifactCatalog::new("empty", vec![]).unwrap();
    let report = assess(
        &catalog,
        &artifact.identity().unwrap(),
        &[artifact],
        &policies,
    );
    denied(&report, ArtifactDenialReason::MissingTrustedReview);
}

#[test]
fn independent_exact_open_weight_review_qualifies_only_the_offline_artifact() {
    let artifact = artifact("base");
    let (review, policies) = review(&artifact);
    let catalog = TrustedArtifactCatalog::new("owner-v1", vec![review]).unwrap();
    let report = assess(
        &catalog,
        &artifact.identity().unwrap(),
        &[artifact],
        &policies,
    );
    assert!(report.is_eligible(), "{:?}", report.denials());
    assert_eq!(report.matched_reviews().len(), 1);
    assert_eq!(report.matched_reviews()[0].policy_documents.len(), 3);
    assert_eq!(report.evidence_digest().len(), 64);
}

#[test]
fn closed_unknown_and_unreviewed_custom_terms_remain_denied() {
    let artifact = artifact("base");
    for (openness, reason) in [
        (
            ReviewedModelOpenness::Closed,
            ArtifactDenialReason::ClosedArtifact,
        ),
        (
            ReviewedModelOpenness::Unknown,
            ArtifactDenialReason::UnknownOpenness,
        ),
    ] {
        let (mut review, policies) = review(&artifact);
        review.openness = openness;
        let catalog = TrustedArtifactCatalog::new("owner-v1", vec![review]).unwrap();
        denied(
            &assess(
                &catalog,
                &artifact.identity().unwrap(),
                std::slice::from_ref(&artifact),
                &policies,
            ),
            reason,
        );
    }
    for axis in 0..3 {
        let (mut review, policies) = review(&artifact);
        match axis {
            0 => review.rights = TrustedPolicyReview::Unreviewed,
            1 => review.serving_terms = TrustedPolicyReview::Unreviewed,
            _ => review.output_terms = TrustedPolicyReview::Unreviewed,
        }
        let catalog = TrustedArtifactCatalog::new("owner-v1", vec![review]).unwrap();
        denied(
            &assess(
                &catalog,
                &artifact.identity().unwrap(),
                std::slice::from_ref(&artifact),
                &policies,
            ),
            ArtifactDenialReason::UnreviewedPolicy,
        );
    }
}

#[test]
fn serving_and_output_inapplicability_must_be_explicitly_owner_reviewed() {
    let artifact = artifact("base");
    let (mut review, mut policies) = review(&artifact);
    review.serving_terms = TrustedPolicyReview::NotApplicable {
        scope: "Fixture has no serving service in this reviewed use".into(),
    };
    review.output_terms = TrustedPolicyReview::NotApplicable {
        scope: "Fixture has no separate output terms in this reviewed use".into(),
    };
    policies.truncate(1);
    let catalog = TrustedArtifactCatalog::new("owner-v1", vec![review]).unwrap();
    let report = assess(
        &catalog,
        &artifact.identity().unwrap(),
        &[artifact],
        &policies,
    );
    assert!(report.is_eligible());
    assert_eq!(report.matched_reviews()[0].policy_documents.len(), 1);
}

#[test]
fn exact_review_does_not_grant_another_role_or_intended_use() {
    let artifact = artifact("base");
    let (review, policies) = review(&artifact);
    let catalog = TrustedArtifactCatalog::new("owner-v1", vec![review]).unwrap();
    for (role, intended_use) in [
        (ModelPolicyRole::Judge, ModelIntendedUse::DatasetGeneration),
        (ModelPolicyRole::Teacher, ModelIntendedUse::Training),
        (ModelPolicyRole::Embedding, ModelIntendedUse::Inference),
    ] {
        let report = assess_artifact(
            &catalog,
            &artifact.identity().unwrap(),
            role,
            intended_use,
            ArtifactEvidenceBundle {
                artifacts: std::slice::from_ref(&artifact),
                policies: &policies,
            },
        )
        .unwrap();
        denied(&report, ArtifactDenialReason::MissingTrustedReview);
        assert_eq!(report.role(), role);
        assert_eq!(report.intended_use(), intended_use);
    }
}

#[test]
fn supplied_review_catalog_or_approval_claims_cannot_create_trust() {
    let artifact = artifact("claimed-approved-model-alias");
    let mut policies = vec![];
    for kind in [
        ModelPolicyDocumentKind::Review,
        ModelPolicyDocumentKind::Catalog,
    ] {
        let policy = policy(&artifact, kind);
        for field in ["approved", "accepted_digest", "reviewer"] {
            let mut wire = serde_json::to_value(&policy.declaration).unwrap();
            wire[field] = serde_json::json!("sensitive-sentinel");
            let error = ModelPolicyEvidence::from_json(&wire.to_string()).unwrap_err();
            assert!(!error.to_string().contains("sensitive-sentinel"));
        }
        policies.push(policy);
    }
    let catalog = TrustedArtifactCatalog::new("empty", vec![]).unwrap();
    denied(
        &assess(
            &catalog,
            &artifact.identity().unwrap(),
            &[artifact],
            &policies,
        ),
        ArtifactDenialReason::MissingTrustedReview,
    );
}

#[test]
fn every_applicable_policy_and_the_exact_artifact_must_be_supplied() {
    let artifact = artifact("base");
    let (review, policies) = review(&artifact);
    let catalog = TrustedArtifactCatalog::new("owner-v1", vec![review]).unwrap();
    for missing in 0..policies.len() {
        let mut incomplete = policies.clone();
        incomplete.remove(missing);
        denied(
            &assess(
                &catalog,
                &artifact.identity().unwrap(),
                std::slice::from_ref(&artifact),
                &incomplete,
            ),
            ArtifactDenialReason::MissingPolicy,
        );
    }
    denied(
        &assess(&catalog, &artifact.identity().unwrap(), &[], &policies),
        ArtifactDenialReason::MissingArtifact,
    );
}

#[test]
fn changes_to_inventory_revision_or_lineage_cannot_reuse_artifact_review() {
    let artifact = artifact("base");
    let identity = artifact.identity().unwrap();
    let (review, policies) = review(&artifact);
    let catalog = TrustedArtifactCatalog::new("owner-v1", vec![review]).unwrap();
    let original = assess(
        &catalog,
        &identity,
        std::slice::from_ref(&artifact),
        &policies,
    );
    for change in 0..3 {
        let mut changed = artifact.clone();
        match change {
            0 => changed.files[0].content.hex = "b".repeat(64),
            1 => changed.source.revision = "different-revision".into(),
            _ => {
                changed.lineage = ModelArtifactLineage::Quantized {
                    base: identity.clone(),
                    quantization: SemanticDeclaration::new(
                        "fixture-quantizer",
                        "1",
                        serde_json::json!({}),
                    ),
                }
            }
        }
        let changed_id = changed.identity().unwrap();
        let report = assess(
            &catalog,
            &identity,
            std::slice::from_ref(&changed),
            &policies,
        );
        denied(&report, ArtifactDenialReason::MissingArtifact);
        assert_ne!(report.evidence_digest(), original.evidence_digest());
        let report = assess(&catalog, &changed_id, &[changed], &policies);
        denied(&report, ArtifactDenialReason::MissingTrustedReview);
    }
}

#[test]
fn changed_policy_bytes_or_metadata_cannot_reuse_exact_reviewed_evidence() {
    let artifact = artifact("base");
    let (review, policies) = review(&artifact);
    let catalog = TrustedArtifactCatalog::new("owner-v1", vec![review]).unwrap();
    let original = assess(
        &catalog,
        &artifact.identity().unwrap(),
        std::slice::from_ref(&artifact),
        &policies,
    );
    for change in 0..5 {
        let mut changed = policies.clone();
        match change {
            0 => changed[0].bytes.push(b'!'),
            1 => {
                changed[0].bytes.push(b'!');
                changed[0].declaration.document.content.hex =
                    blake3::hash(&changed[0].bytes).to_hex().to_string();
            }
            2 => changed[0].declaration.document.source.revision = "changed".into(),
            3 => changed[0].declaration.subject.digest = "c".repeat(64),
            _ => changed[0].declaration.intended_use = ModelIntendedUse::Redistribution,
        }
        let report = assess(
            &catalog,
            &artifact.identity().unwrap(),
            std::slice::from_ref(&artifact),
            &changed,
        );
        denied(
            &report,
            if change == 0 {
                ArtifactDenialReason::PolicyContentMismatch
            } else {
                ArtifactDenialReason::MissingPolicy
            },
        );
        assert_ne!(report.evidence_digest(), original.evidence_digest());
    }
}

#[test]
fn invalid_public_declarations_deny_without_echoing_rejected_values() {
    let artifact = artifact("base");
    let (review, policies) = review(&artifact);
    let catalog = TrustedArtifactCatalog::new("owner-v1", vec![review]).unwrap();
    let mut bad_artifact = artifact.clone();
    bad_artifact.files[0].content.hex = "sensitive-sentinel".into();
    let report = assess(
        &catalog,
        &artifact.identity().unwrap(),
        &[bad_artifact],
        &policies,
    );
    denied(&report, ArtifactDenialReason::InvalidArtifact);
    assert!(
        !serde_json::to_string(&report)
            .unwrap()
            .contains("sensitive-sentinel")
    );
    let mut bad_policy = policies.clone();
    bad_policy[0].declaration.subject.digest = "sensitive-sentinel".into();
    let report = assess(
        &catalog,
        &artifact.identity().unwrap(),
        std::slice::from_ref(&artifact),
        &bad_policy,
    );
    denied(&report, ArtifactDenialReason::InvalidPolicy);
    assert!(
        !serde_json::to_string(&report)
            .unwrap()
            .contains("sensitive-sentinel")
    );
    let requested = ArtifactIdentity {
        version: 2,
        digest: "sensitive-sentinel".into(),
    };
    let report = assess(&catalog, &requested, &[artifact], &policies);
    denied(&report, ArtifactDenialReason::InvalidRequestedIdentity);
    assert!(report.artifact().is_none());
    assert!(
        !serde_json::to_string(&report)
            .unwrap()
            .contains("sensitive-sentinel")
    );
}

#[test]
fn duplicate_or_conflicting_supplied_evidence_is_denied_independently_of_order() {
    let artifact = artifact("base");
    let (review, policies) = review(&artifact);
    let catalog = TrustedArtifactCatalog::new("owner-v1", vec![review]).unwrap();
    let mut duplicate = artifact.clone();
    duplicate.label = "different display label, same identity".into();
    let report = assess(
        &catalog,
        &artifact.identity().unwrap(),
        &[artifact.clone(), duplicate],
        &policies,
    );
    denied(&report, ArtifactDenialReason::DuplicateArtifact);
    let mut duplicate_policies = policies.clone();
    let mut conflicting = policies[0].clone();
    conflicting.bytes.push(b'!');
    duplicate_policies.push(conflicting);
    let report = assess(
        &catalog,
        &artifact.identity().unwrap(),
        std::slice::from_ref(&artifact),
        &duplicate_policies,
    );
    denied(&report, ArtifactDenialReason::DuplicatePolicy);
    denied(&report, ArtifactDenialReason::PolicyContentMismatch);
    duplicate_policies.reverse();
    let reversed = assess(
        &catalog,
        &artifact.identity().unwrap(),
        &[artifact],
        &duplicate_policies,
    );
    assert_eq!(report, reversed);
}

#[test]
fn extra_policy_documents_cannot_expand_the_single_document_reviewed_scope() {
    let artifact = artifact("base");
    let (review, mut policies) = review(&artifact);
    let catalog = TrustedArtifactCatalog::new("owner-v1", vec![review]).unwrap();
    let mut extra = policies[0].clone();
    extra
        .bytes
        .extend_from_slice(b"; additional unreviewed source");
    extra.declaration.document.content.hex = blake3::hash(&extra.bytes).to_hex().to_string();
    policies.push(extra);
    let report = assess(
        &catalog,
        &artifact.identity().unwrap(),
        std::slice::from_ref(&artifact),
        &policies,
    );
    denied(&report, ArtifactDenialReason::MultiplePoliciesForScope);
    policies.reverse();
    assert_eq!(
        report,
        assess(
            &catalog,
            &artifact.identity().unwrap(),
            &[artifact],
            &policies
        )
    );
}
