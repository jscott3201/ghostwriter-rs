//! Owner-catalog integrity, exact review scope, and independent raw document digest checks.
use gw_providers::artifact_assessment::*;
use gw_schema::*;

#[path = "support/artifact_assessment.rs"]
mod support;
use support::*;

#[test]
fn duplicate_and_conflicting_reviews_for_one_artifact_role_use_are_rejected() {
    let artifact = artifact("base");
    let (review, _) = review(&artifact);
    for conflict in [false, true] {
        let mut other = review.clone();
        if conflict {
            other.openness = ReviewedModelOpenness::Closed;
        }
        assert_eq!(
            TrustedArtifactCatalog::new("owner-v1", vec![review.clone(), other]).unwrap_err(),
            ArtifactCatalogError::DuplicateReview
        );
    }
}

#[test]
fn catalog_rejects_policy_subject_kind_role_or_use_mismatches() {
    let artifact = artifact("base");
    for change in 0..4 {
        let (mut review, _) = review(&artifact);
        let TrustedPolicyReview::Reviewed { policy, .. } = &mut review.rights else {
            unreachable!()
        };
        match change {
            0 => policy.declaration.subject.digest = "b".repeat(64),
            1 => policy.declaration.kind = ModelPolicyDocumentKind::OutputTerms,
            2 => policy.declaration.role = ModelPolicyRole::Judge,
            _ => policy.declaration.intended_use = ModelIntendedUse::Training,
        }
        assert_eq!(
            TrustedArtifactCatalog::new("owner-v1", vec![review]).unwrap_err(),
            ArtifactCatalogError::PolicyBindingMismatch
        );
    }
}

#[test]
fn catalog_requires_explicit_review_scope_and_disallows_inapplicable_rights() {
    let artifact = artifact("base");
    let (review, _) = review(&artifact);
    for revision in ["", " ", "sensitive-sentinel\n"] {
        let error = TrustedArtifactCatalog::new(revision, vec![review.clone()]).unwrap_err();
        assert_eq!(error, ArtifactCatalogError::InvalidScope);
        assert!(!error.to_string().contains("sensitive-sentinel"));
    }
    for axis in 0..3 {
        let mut changed = review.clone();
        match axis {
            0 => changed.scope.clear(),
            1 => {
                changed.serving_terms = TrustedPolicyReview::NotApplicable {
                    scope: String::new(),
                }
            }
            _ => {
                let TrustedPolicyReview::Reviewed { scope, .. } = &mut changed.output_terms else {
                    unreachable!()
                };
                scope.clear();
            }
        }
        assert_eq!(
            TrustedArtifactCatalog::new("owner-v1", vec![changed]).unwrap_err(),
            ArtifactCatalogError::InvalidScope
        );
    }
    let mut changed = review;
    changed.rights = TrustedPolicyReview::NotApplicable {
        scope: "not allowed for rights".into(),
    };
    assert_eq!(
        TrustedArtifactCatalog::new("owner-v1", vec![changed]).unwrap_err(),
        ArtifactCatalogError::RightsNotApplicable
    );
}

#[test]
fn declared_sha256_is_checked_against_an_independent_known_vector_at_both_boundaries() {
    // Published SHA-256 "abc" test vector, not a digest computed by the production helper.
    const SHA256_ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    let artifact = artifact("base");
    let (mut review, mut policies) = review(&artifact);
    policies[0].bytes = b"abc".to_vec();
    policies[0].declaration.document.content = ContentDigest {
        algorithm: DigestAlgorithm::Sha256,
        hex: SHA256_ABC.into(),
    };
    review.rights = TrustedPolicyReview::Reviewed {
        policy: policies[0].clone(),
        scope: "Independent synthetic SHA-256 fixture review".into(),
    };
    let catalog = TrustedArtifactCatalog::new("owner-v1", vec![review.clone()]).unwrap();
    assert!(
        assess(
            &catalog,
            &artifact.identity().unwrap(),
            std::slice::from_ref(&artifact),
            &policies
        )
        .is_eligible()
    );
    let mut contradictory = review;
    let TrustedPolicyReview::Reviewed { policy, .. } = &mut contradictory.rights else {
        unreachable!()
    };
    policy.declaration.document.content.hex = "0".repeat(64);
    assert_eq!(
        TrustedArtifactCatalog::new("owner-v1", vec![contradictory]).unwrap_err(),
        ArtifactCatalogError::PolicyContentMismatch
    );
    policies[0].bytes = b"abd".to_vec();
    denied(
        &assess(
            &catalog,
            &artifact.identity().unwrap(),
            &[artifact],
            &policies,
        ),
        ArtifactDenialReason::PolicyContentMismatch,
    );
}

#[test]
fn contradictory_blake3_and_invalid_public_catalog_declarations_are_rejected() {
    let artifact = artifact("base");
    let (review, _) = review(&artifact);
    for change in 0..3 {
        let mut changed = review.clone();
        match change {
            0 => changed.artifact.version = 2,
            1 => {
                let TrustedPolicyReview::Reviewed { policy, .. } = &mut changed.rights else {
                    unreachable!()
                };
                policy.declaration.version = 2;
            }
            _ => {
                let TrustedPolicyReview::Reviewed { policy, .. } = &mut changed.rights else {
                    unreachable!()
                };
                policy.bytes.push(b'!');
            }
        }
        let expected = [
            ArtifactCatalogError::InvalidArtifact,
            ArtifactCatalogError::InvalidPolicy,
            ArtifactCatalogError::PolicyContentMismatch,
        ][change];
        assert_eq!(
            TrustedArtifactCatalog::new("owner-v1", vec![changed]).unwrap_err(),
            expected
        );
    }
}

#[test]
fn every_supported_role_and_use_requires_its_own_exact_reviewed_scope() {
    let artifact = artifact("base");
    for role in [
        ModelPolicyRole::Teacher,
        ModelPolicyRole::Judge,
        ModelPolicyRole::Embedding,
        ModelPolicyRole::UserSynthesis,
        ModelPolicyRole::Student,
        ModelPolicyRole::Derivative,
    ] {
        for intended_use in [
            ModelIntendedUse::Inference,
            ModelIntendedUse::DatasetGeneration,
            ModelIntendedUse::Training,
            ModelIntendedUse::Evaluation,
            ModelIntendedUse::Redistribution,
        ] {
            let (mut review, mut policies) = review(&artifact);
            review.role = role;
            review.intended_use = intended_use;
            for (axis, policy) in [
                &mut review.rights,
                &mut review.serving_terms,
                &mut review.output_terms,
            ]
            .into_iter()
            .zip(&mut policies)
            {
                policy.declaration.role = role;
                policy.declaration.intended_use = intended_use;
                *axis = TrustedPolicyReview::Reviewed {
                    policy: policy.clone(),
                    scope: "Independently reviewed synthetic role/use pair".into(),
                };
            }
            let catalog = TrustedArtifactCatalog::new("owner-v1", vec![review]).unwrap();
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
            assert!(report.is_eligible(), "{role:?} {intended_use:?}");
        }
    }
}

#[test]
fn catalog_revision_and_each_review_scope_field_change_the_assessment_binding() {
    let artifact = artifact("base");
    let (review, policies) = review(&artifact);
    let catalog = TrustedArtifactCatalog::new("owner-v1", vec![review.clone()]).unwrap();
    let original = assess(
        &catalog,
        &artifact.identity().unwrap(),
        std::slice::from_ref(&artifact),
        &policies,
    );
    for change in 0..7 {
        let mut changed = review.clone();
        let mut revision = "owner-v1";
        match change {
            0 => revision = "owner-v2",
            1 => changed.reviewed_at_unix_ms += 1,
            2 => changed.scope.push_str("; additional reviewed limit"),
            3 => changed.reference = ModelReference::new("urn:fixture:another-review").unwrap(),
            4 => changed.openness = ReviewedModelOpenness::OpenTrainingArtifacts,
            5 => {
                let TrustedPolicyReview::Reviewed { scope, .. } = &mut changed.rights else {
                    unreachable!()
                };
                scope.push_str("; revised obligation");
            }
            _ => {
                changed.output_terms = TrustedPolicyReview::NotApplicable {
                    scope: "Explicit revised output scope".into(),
                }
            }
        }
        let changed_catalog = TrustedArtifactCatalog::new(revision, vec![changed]).unwrap();
        assert_ne!(catalog.identity(), changed_catalog.identity());
        let report = assess(
            &changed_catalog,
            &artifact.identity().unwrap(),
            std::slice::from_ref(&artifact),
            &policies,
        );
        assert_ne!(report.evidence_digest(), original.evidence_digest());
    }
}
