use super::{binding::*, *};
use gw_schema::ModelPolicyDocumentKind;
use std::collections::BTreeMap;

/// Immutable owner-trusted review catalog. It contains no built-in production approvals.
///
/// Construct only from an independently trusted application/owner review channel. This type and
/// its review inputs deliberately cannot be deserialized from submitted evidence. The constructor
/// validates shape, exact subject/use bindings and document bytes; it does not perform the review.
///
/// ```compile_fail
/// use gw_providers::artifact_assessment::TrustedArtifactCatalog;
/// let catalog: TrustedArtifactCatalog = serde_json::from_str("{}").unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct TrustedArtifactCatalog {
    identity: ArtifactCatalogIdentity,
    reviews: BTreeMap<(String, u8, u8), StoredReview>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) enum StoredPolicyReview {
    Unreviewed,
    Reviewed {
        identity: PolicyDocumentIdentity,
        bytes_digest: String,
        scope: String,
    },
    NotApplicable {
        scope: String,
    },
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct ReviewBinding {
    pub artifact: ArtifactIdentity,
    pub role: ModelPolicyRole,
    pub intended_use: ModelIntendedUse,
    pub openness: ReviewedModelOpenness,
    pub reference: ModelReference,
    pub reviewed_at_unix_ms: u64,
    pub scope: String,
    pub rights: StoredPolicyReview,
    pub serving_terms: StoredPolicyReview,
    pub output_terms: StoredPolicyReview,
}

#[derive(Debug, Clone)]
pub(super) struct StoredReview {
    pub binding: ReviewBinding,
    pub digest: String,
}

impl TrustedArtifactCatalog {
    /// Bind independent reviews to a catalog revision. An empty catalog approves no artifacts.
    /// Input order is ignored; duplicate or conflicting exact artifact/role/use reviews fail.
    ///
    /// # Errors
    /// Returns a static error for invalid declarations, mismatched review scope, inconsistent raw
    /// policy digests, or duplicate reviews. No rejected policy text or user value enters errors.
    pub fn new(
        revision: impl Into<String>,
        reviews: Vec<TrustedArtifactReview>,
    ) -> Result<Self, ArtifactCatalogError> {
        let revision = revision.into();
        if !text_valid(&revision) {
            return Err(ArtifactCatalogError::InvalidScope);
        }
        let mut indexed = BTreeMap::new();
        for review in reviews {
            let review = normalize(review)?;
            let key = review_key(
                &review.binding.artifact,
                review.binding.role,
                review.binding.intended_use,
            );
            if indexed.insert(key, review).is_some() {
                return Err(ArtifactCatalogError::DuplicateReview);
            }
        }
        let bindings: Vec<_> = indexed.values().map(|review| &review.binding).collect();
        let digest = hash(
            "ghostwriter.artifact-review-catalog.v1",
            &(1_u32, &revision, bindings),
        )
        .map_err(|_| ArtifactCatalogError::IdentityEncoding)?;
        Ok(Self {
            identity: ArtifactCatalogIdentity {
                version: 1,
                revision,
                digest,
            },
            reviews: indexed,
        })
    }

    /// Catalog content identity, including revision and every normalized trusted review.
    #[must_use]
    pub fn identity(&self) -> &ArtifactCatalogIdentity {
        &self.identity
    }

    pub(super) fn review(
        &self,
        artifact: &ArtifactIdentity,
        role: ModelPolicyRole,
        intended_use: ModelIntendedUse,
    ) -> Option<&StoredReview> {
        self.reviews.get(&review_key(artifact, role, intended_use))
    }
}

fn review_key(
    artifact: &ArtifactIdentity,
    role: ModelPolicyRole,
    intended_use: ModelIntendedUse,
) -> (String, u8, u8) {
    let role = match role {
        ModelPolicyRole::Teacher => 0,
        ModelPolicyRole::Judge => 1,
        ModelPolicyRole::Embedding => 2,
        ModelPolicyRole::UserSynthesis => 3,
        ModelPolicyRole::Student => 4,
        ModelPolicyRole::Derivative => 5,
    };
    let intended_use = match intended_use {
        ModelIntendedUse::Inference => 0,
        ModelIntendedUse::DatasetGeneration => 1,
        ModelIntendedUse::Training => 2,
        ModelIntendedUse::Evaluation => 3,
        ModelIntendedUse::Redistribution => 4,
    };
    (artifact.digest.clone(), role, intended_use)
}

fn normalize(review: TrustedArtifactReview) -> Result<StoredReview, ArtifactCatalogError> {
    if !text_valid(&review.scope) {
        return Err(ArtifactCatalogError::InvalidScope);
    }
    let artifact = review
        .artifact
        .identity()
        .map_err(|_| ArtifactCatalogError::InvalidArtifact)?;
    let policy =
        |value, kind| normalize_policy(value, &artifact, kind, review.role, review.intended_use);
    let rights = policy(review.rights, ModelPolicyDocumentKind::Rights)?;
    let serving_terms = policy(review.serving_terms, ModelPolicyDocumentKind::ServingTerms)?;
    let output_terms = policy(review.output_terms, ModelPolicyDocumentKind::OutputTerms)?;
    let binding = ReviewBinding {
        artifact,
        role: review.role,
        intended_use: review.intended_use,
        openness: review.openness,
        reference: review.reference,
        reviewed_at_unix_ms: review.reviewed_at_unix_ms,
        scope: review.scope,
        rights,
        serving_terms,
        output_terms,
    };
    let digest = hash("ghostwriter.artifact-trusted-review.v1", &binding)
        .map_err(|_| ArtifactCatalogError::IdentityEncoding)?;
    Ok(StoredReview { binding, digest })
}

fn normalize_policy(
    review: TrustedPolicyReview,
    artifact: &ArtifactIdentity,
    kind: ModelPolicyDocumentKind,
    role: ModelPolicyRole,
    intended_use: ModelIntendedUse,
) -> Result<StoredPolicyReview, ArtifactCatalogError> {
    match review {
        TrustedPolicyReview::Unreviewed => Ok(StoredPolicyReview::Unreviewed),
        TrustedPolicyReview::NotApplicable { scope } => {
            if kind == ModelPolicyDocumentKind::Rights {
                return Err(ArtifactCatalogError::RightsNotApplicable);
            }
            if !text_valid(&scope) {
                return Err(ArtifactCatalogError::InvalidScope);
            }
            Ok(StoredPolicyReview::NotApplicable { scope })
        }
        TrustedPolicyReview::Reviewed { policy, scope } => {
            if !text_valid(&scope) {
                return Err(ArtifactCatalogError::InvalidScope);
            }
            let declaration = &policy.declaration;
            let identity = declaration
                .identity()
                .map_err(|_| ArtifactCatalogError::InvalidPolicy)?;
            if declaration.subject != *artifact
                || declaration.kind != kind
                || declaration.role != role
                || declaration.intended_use != intended_use
            {
                return Err(ArtifactCatalogError::PolicyBindingMismatch);
            }
            if !policy_bytes_match(&policy) {
                return Err(ArtifactCatalogError::PolicyContentMismatch);
            }
            Ok(StoredPolicyReview::Reviewed {
                identity,
                bytes_digest: bytes_digest(&policy.bytes),
                scope,
            })
        }
    }
}
