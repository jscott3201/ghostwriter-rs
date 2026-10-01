//! Independent parent obligations, explicit absence, exact base consistency, and set ordering.
use gw_providers::artifact_assessment::*;
use gw_schema::*;
use serde_json::json;

#[path = "support/artifact_assessment.rs"]
mod support;
use support::*;

fn derived(base: &PinnedModelArtifact, additional: &[PinnedModelArtifact]) -> PinnedModelArtifact {
    let mut value = artifact("derived");
    value.lineage = ModelArtifactLineage::Derived {
        base: base.identity().unwrap(),
        additional_parents: Declaration::Declared(
            additional
                .iter()
                .map(|parent| parent.identity().unwrap())
                .collect(),
        ),
        transformation: SemanticDeclaration::new(
            "fixture-merge",
            "1",
            json!({"steps":["first","second"]}),
        ),
    };
    value
}

fn adapter(
    base: &PinnedModelArtifact,
    parent: Option<&PinnedModelArtifact>,
) -> PinnedModelArtifact {
    let mut value = artifact("adapter");
    value.lineage = ModelArtifactLineage::Adapter {
        base: base.identity().unwrap(),
        parent_checkpoint: Declaration::Declared(parent.map(|value| value.identity().unwrap())),
        configuration: SemanticDeclaration::new("fixture-adapter", "1", json!({"rank":8})),
    };
    value
}

fn checkpoint(
    revision: &str,
    base: &PinnedModelArtifact,
    parent: Option<&PinnedModelArtifact>,
    adapter: Option<&PinnedModelArtifact>,
) -> PinnedModelArtifact {
    let mut value = artifact(revision);
    value.lineage = ModelArtifactLineage::Checkpoint {
        base: base.identity().unwrap(),
        parent: Declaration::Declared(parent.map(|value| value.identity().unwrap())),
        adapter: Declaration::Declared(adapter.map(|value| value.identity().unwrap())),
    };
    value
}

fn reviewed(
    artifacts: &[PinnedModelArtifact],
) -> (TrustedArtifactCatalog, Vec<SuppliedModelPolicy>) {
    let mut reviews = Vec::new();
    let mut policies = Vec::new();
    for artifact in artifacts {
        let (review, documents) = review(artifact);
        reviews.push(review);
        policies.extend(documents);
    }
    (
        TrustedArtifactCatalog::new("owner-v1", reviews).unwrap(),
        policies,
    )
}

#[test]
fn derivative_requires_every_parent_artifact_review_and_policy_document() {
    let base = artifact("base");
    let additional = artifact("additional");
    let child = derived(&base, std::slice::from_ref(&additional));
    let requested = child.identity().unwrap();
    let artifacts = vec![base, additional, child.clone()];
    let (catalog, policies) = reviewed(&artifacts);
    let report = assess(&catalog, &requested, &artifacts, &policies);
    assert!(report.is_eligible());
    assert_eq!(report.matched_reviews().len(), 3);
    for missing in 0..2 {
        let mut incomplete = artifacts.clone();
        incomplete.remove(missing);
        denied(
            &assess(&catalog, &requested, &incomplete, &policies),
            ArtifactDenialReason::MissingArtifact,
        );
        let reviews: Vec<_> = artifacts
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != missing)
            .map(|(_, artifact)| review(artifact).0)
            .collect();
        let incomplete_catalog = TrustedArtifactCatalog::new("owner-v1", reviews).unwrap();
        denied(
            &assess(&incomplete_catalog, &requested, &artifacts, &policies),
            ArtifactDenialReason::MissingTrustedReview,
        );
        let missing_id = artifacts[missing].identity().unwrap();
        let incomplete_policies: Vec<_> = policies
            .iter()
            .filter(|policy| policy.declaration.subject != missing_id)
            .cloned()
            .collect();
        denied(
            &assess(&catalog, &requested, &artifacts, &incomplete_policies),
            ArtifactDenialReason::MissingPolicy,
        );
    }
}

#[test]
fn a_childs_permitted_terms_do_not_override_closed_or_unreviewed_parent_obligations() {
    let base = artifact("base");
    let child = derived(&base, &[]);
    let artifacts = vec![base, child.clone()];
    for closed in [true, false] {
        let mut reviews = Vec::new();
        let mut policies = Vec::new();
        for (index, artifact) in artifacts.iter().enumerate() {
            let (mut review, documents) = review(artifact);
            if index == 0 {
                if closed {
                    review.openness = ReviewedModelOpenness::Closed;
                } else {
                    review.rights = TrustedPolicyReview::Unreviewed;
                }
            }
            reviews.push(review);
            policies.extend(documents);
        }
        let catalog = TrustedArtifactCatalog::new("owner-v1", reviews).unwrap();
        denied(
            &assess(&catalog, &child.identity().unwrap(), &artifacts, &policies),
            if closed {
                ArtifactDenialReason::ClosedArtifact
            } else {
                ArtifactDenialReason::UnreviewedPolicy
            },
        );
    }
}

#[test]
fn unknown_additional_parents_are_denied_even_when_the_exact_claim_has_a_review() {
    let base = artifact("base");
    let mut child = derived(&base, &[]);
    let ModelArtifactLineage::Derived {
        additional_parents, ..
    } = &mut child.lineage
    else {
        unreachable!()
    };
    *additional_parents = Declaration::Unknown;
    let requested = child.identity().unwrap();
    let artifacts = [base, child];
    let (catalog, policies) = reviewed(&artifacts);
    denied(
        &assess(&catalog, &requested, &artifacts, &policies),
        ArtifactDenialReason::UnknownLineage,
    );
}

#[test]
fn explicit_absence_only_qualifies_with_the_exact_review_and_never_substitutes_for_unknown() {
    let base = artifact("base");
    for field in 0..3 {
        let child = if field == 0 {
            adapter(&base, None)
        } else {
            checkpoint("first", &base, None, None)
        };
        let requested = child.identity().unwrap();
        let artifacts = [base.clone(), child.clone()];
        let (catalog, policies) = reviewed(&artifacts);
        let original = assess(&catalog, &requested, &artifacts, &policies);
        assert!(original.is_eligible());
        let mut changed = child;
        match (&mut changed.lineage, field) {
            (
                ModelArtifactLineage::Adapter {
                    parent_checkpoint, ..
                },
                0,
            ) => *parent_checkpoint = Declaration::Unknown,
            (ModelArtifactLineage::Checkpoint { parent, .. }, 1) => *parent = Declaration::Unknown,
            (ModelArtifactLineage::Checkpoint { adapter, .. }, 2) => {
                *adapter = Declaration::Unknown
            }
            _ => unreachable!(),
        }
        let changed_id = changed.identity().unwrap();
        let changed_artifacts = [base.clone(), changed];
        let report = assess(&catalog, &changed_id, &changed_artifacts, &policies);
        denied(&report, ArtifactDenialReason::MissingTrustedReview);
        denied(&report, ArtifactDenialReason::UnknownLineage);
        assert_ne!(report.evidence_digest(), original.evidence_digest());
        let (catalog, policies) = reviewed(&changed_artifacts);
        denied(
            &assess(&catalog, &changed_id, &changed_artifacts, &policies),
            ArtifactDenialReason::UnknownLineage,
        );
    }
}

#[test]
fn complete_checkpoint_and_adapter_links_resolve_shared_ancestors_once() {
    let base = artifact("base");
    let previous = checkpoint("previous", &base, None, None);
    let adapter = adapter(&base, Some(&previous));
    let child = checkpoint("current", &base, Some(&previous), Some(&adapter));
    let requested = child.identity().unwrap();
    let artifacts = [base, previous, adapter, child];
    let (catalog, policies) = reviewed(&artifacts);
    let report = assess(&catalog, &requested, &artifacts, &policies);
    assert!(report.is_eligible(), "{:?}", report.denials());
    assert_eq!(report.matched_reviews().len(), 4);
}

#[test]
fn wrong_parent_kinds_and_inconsistent_checkpoint_or_adapter_bases_are_denied() {
    let base = artifact("base");
    let other_base = artifact("other-base");
    let foreign_checkpoint = checkpoint("foreign", &other_base, None, None);
    let foreign_adapter = adapter(&other_base, None);
    for child in [
        checkpoint("wrong-parent-base", &base, Some(&foreign_checkpoint), None),
        checkpoint("wrong-parent-kind", &base, Some(&other_base), None),
        checkpoint("wrong-adapter-base", &base, None, Some(&foreign_adapter)),
        checkpoint("wrong-adapter-kind", &base, None, Some(&foreign_checkpoint)),
        adapter(&base, Some(&foreign_checkpoint)),
    ] {
        let requested = child.identity().unwrap();
        let artifacts = [
            base.clone(),
            other_base.clone(),
            foreign_checkpoint.clone(),
            foreign_adapter.clone(),
            child,
        ];
        let (catalog, policies) = reviewed(&artifacts);
        denied(
            &assess(&catalog, &requested, &artifacts, &policies),
            ArtifactDenialReason::ConflictingLineage,
        );
    }
}

#[test]
fn catalog_bundle_inventory_and_parent_set_order_do_not_change_assessment() {
    let base = artifact("base");
    let first = artifact("first");
    let second = artifact("second");
    let mut child = derived(&base, &[first.clone(), second.clone()]);
    child.files.push(ModelArtifactFile {
        path: "config.json".into(),
        purpose: ModelFilePurpose::Configuration,
        content: ContentDigest {
            algorithm: DigestAlgorithm::Blake3,
            hex: "b".repeat(64),
        },
    });
    let requested = child.identity().unwrap();
    let mut artifacts = vec![base, first, second, child];
    let (catalog, mut policies) = reviewed(&artifacts);
    let original = assess(&catalog, &requested, &artifacts, &policies);
    assert!(original.is_eligible());
    let root = artifacts.last_mut().unwrap();
    root.files.reverse();
    let ModelArtifactLineage::Derived {
        additional_parents: Declaration::Declared(parents),
        ..
    } = &mut root.lineage
    else {
        unreachable!()
    };
    parents.reverse();
    artifacts.reverse();
    policies.reverse();
    let (reordered_catalog, _) = reviewed(&artifacts);
    assert_eq!(catalog.identity(), reordered_catalog.identity());
    assert_eq!(
        original,
        assess(&reordered_catalog, &requested, &artifacts, &policies)
    );
    let ModelArtifactLineage::Derived { transformation, .. } = &mut artifacts[0].lineage else {
        unreachable!()
    };
    transformation.configuration["steps"] = json!(["second", "first"]);
    let changed = artifacts[0].identity().unwrap();
    assert_ne!(
        requested, changed,
        "semantic configuration order must be retained"
    );
    denied(
        &assess(&catalog, &changed, &artifacts, &policies),
        ArtifactDenialReason::MissingTrustedReview,
    );
}

#[test]
fn reviewing_a_parent_for_a_different_role_does_not_cover_the_requested_use() {
    let base = artifact("base");
    let child = derived(&base, &[]);
    let requested = child.identity().unwrap();
    let (mut base_review, mut policies) = review(&base);
    base_review.role = ModelPolicyRole::Derivative;
    for (axis, policy) in [
        &mut base_review.rights,
        &mut base_review.serving_terms,
        &mut base_review.output_terms,
    ]
    .into_iter()
    .zip(&mut policies)
    {
        policy.declaration.role = ModelPolicyRole::Derivative;
        *axis = TrustedPolicyReview::Reviewed {
            policy: policy.clone(),
            scope: "Different independently reviewed role".into(),
        };
    }
    let (child_review, child_policies) = review(&child);
    policies.extend(child_policies);
    let catalog = TrustedArtifactCatalog::new("owner-v1", vec![base_review, child_review]).unwrap();
    denied(
        &assess(&catalog, &requested, &[base, child], &policies),
        ArtifactDenialReason::MissingTrustedReview,
    );
}
