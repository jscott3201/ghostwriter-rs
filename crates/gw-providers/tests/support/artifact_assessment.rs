//! Private synthetic fixtures. These documents approve no real model or execution path.
use gw_providers::artifact_assessment::*;
use gw_schema::*;
use serde_json::json;

pub fn artifact(revision: &str) -> PinnedModelArtifact {
    PinnedModelArtifact::from_json(
        &json!({
            "version":1,"label":"Private synthetic fixture",
            "source":{"reference":"urn:fixture:model","revision":revision},
            "files":[{"path":"weights.bin","purpose":"weights",
                      "content":{"algorithm":"blake3","hex":"a".repeat(64)}}],
            "lineage":{"kind":"base"},"tokenizer":{"status":"unknown"},
            "chat_template":{"status":"unknown"}
        })
        .to_string(),
    )
    .unwrap()
}

pub fn policy(
    artifact: &PinnedModelArtifact,
    kind: ModelPolicyDocumentKind,
) -> SuppliedModelPolicy {
    let bytes = format!(
        "Private synthetic {kind:?} terms for {}",
        artifact.source.revision
    )
    .into_bytes();
    SuppliedModelPolicy {
        declaration: ModelPolicyEvidence {
            version: 1,
            kind,
            subject: artifact.identity().unwrap(),
            document: PinnedModelDocument {
                source: PinnedModelSource {
                    reference: ModelReference::new("urn:fixture:policy").unwrap(),
                    revision: "1".into(),
                },
                content: ContentDigest {
                    algorithm: DigestAlgorithm::Blake3,
                    hex: blake3::hash(&bytes).to_hex().to_string(),
                },
            },
            role: ModelPolicyRole::Teacher,
            intended_use: ModelIntendedUse::DatasetGeneration,
        },
        bytes,
    }
}

pub fn review(artifact: &PinnedModelArtifact) -> (TrustedArtifactReview, Vec<SuppliedModelPolicy>) {
    let policies = vec![
        policy(artifact, ModelPolicyDocumentKind::Rights),
        policy(artifact, ModelPolicyDocumentKind::ServingTerms),
        policy(artifact, ModelPolicyDocumentKind::OutputTerms),
    ];
    let reviewed = |index: usize| TrustedPolicyReview::Reviewed {
        policy: policies[index].clone(),
        scope: "Owner-reviewed private synthetic fixture only".into(),
    };
    (
        TrustedArtifactReview {
            artifact: artifact.clone(),
            role: ModelPolicyRole::Teacher,
            intended_use: ModelIntendedUse::DatasetGeneration,
            openness: ReviewedModelOpenness::OpenWeights,
            reference: ModelReference::new("urn:fixture:independent-review").unwrap(),
            reviewed_at_unix_ms: 1_700_000_000_000,
            scope: "Synthetic offline policy test; no deployment or execution approval".into(),
            rights: reviewed(0),
            serving_terms: reviewed(1),
            output_terms: reviewed(2),
        },
        policies,
    )
}

pub fn assess(
    catalog: &TrustedArtifactCatalog,
    requested: &ArtifactIdentity,
    artifacts: &[PinnedModelArtifact],
    policies: &[SuppliedModelPolicy],
) -> ArtifactAssessment {
    assess_artifact(
        catalog,
        requested,
        ModelPolicyRole::Teacher,
        ModelIntendedUse::DatasetGeneration,
        ArtifactEvidenceBundle {
            artifacts,
            policies,
        },
    )
    .unwrap()
}

pub fn denied(report: &ArtifactAssessment, reason: ArtifactDenialReason) {
    assert!(!report.is_eligible());
    assert!(
        report
            .denials()
            .iter()
            .any(|denial| denial.reason == reason),
        "{:?}",
        report.denials()
    );
}
