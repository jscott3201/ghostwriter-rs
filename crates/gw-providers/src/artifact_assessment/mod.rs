//! Pure offline artifact and intended-use assessment against an owner-supplied trusted catalog.
//!
//! Catalog construction belongs to the application's trusted review channel. Supplied artifacts,
//! policy documents, review records, and catalog records cannot approve themselves. This module
//! reads no files or environment, performs no network calls, and never qualifies a client, served
//! weights, cache entry, or execution. Even a positive assessment supplies no execution authority.

use gw_schema::{
    ArtifactIdentity, ModelIntendedUse, ModelPolicyEvidence, ModelPolicyRole, ModelReference,
    PinnedModelArtifact, PolicyDocumentIdentity,
};
use serde::Serialize;

mod assess;
mod binding;
mod catalog;
mod evidence;
mod lineage;
pub use assess::assess_artifact;
pub use catalog::TrustedArtifactCatalog;

/// A supplied policy declaration and the exact offline bytes it references. Neither is trusted.
#[derive(Debug, Clone)]
pub struct SuppliedModelPolicy {
    /// Structurally validated again by catalog construction or assessment.
    pub declaration: ModelPolicyEvidence,
    /// Raw policy document bytes; no license-text interpretation is performed.
    pub bytes: Vec<u8>,
}

/// Borrowed offline inputs. All artifacts and policies are revalidated, including unused entries.
#[derive(Debug, Clone, Copy)]
pub struct ArtifactEvidenceBundle<'a> {
    /// Artifact declarations, including every referenced base, parent, and adapter.
    pub artifacts: &'a [PinnedModelArtifact],
    /// Policy declarations and raw bytes. This first contract accepts one pinned document per
    /// artifact/kind/role/use; duplicates and extra documents for that scope are rejected.
    pub policies: &'a [SuppliedModelPolicy],
}

/// The independently reviewed openness category; names of licenses never select this value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewedModelOpenness {
    /// Reviewed open weights; complete training artifacts are not additionally required.
    OpenWeights,
    /// Reviewed weights and open training artifacts.
    OpenTrainingArtifacts,
    /// Closed artifacts remain denied even when their identity is known.
    Closed,
    /// Unresolved openness remains denied.
    Unknown,
}

/// Owner-reviewed disposition of one policy axis, supplied through a trusted application channel.
/// It deliberately has no `Deserialize` implementation.
#[derive(Debug, Clone)]
pub enum TrustedPolicyReview {
    /// Missing review, including custom terms whose intended-use review is incomplete.
    Unreviewed,
    /// Independently reviewed exact policy declaration and raw bytes for this role/use.
    Reviewed {
        /// Exact evidence and bytes reviewed independently of an assessment submission.
        policy: SuppliedModelPolicy,
        /// Nonempty explanation of the reviewed applicability and permission scope.
        scope: String,
    },
    /// Explicit owner-reviewed inapplicability for serving or output terms; invalid for rights.
    NotApplicable {
        /// Nonempty reviewed rationale for this exact artifact, role, and use.
        scope: String,
    },
}

/// One owner-supplied review for an exact artifact and role/use pair. No deserialization grants trust.
#[derive(Debug, Clone)]
pub struct TrustedArtifactReview {
    /// Independently reviewed declaration; the catalog recomputes its artifact identity.
    pub artifact: PinnedModelArtifact,
    /// Role permitted by the reviewed scope.
    pub role: ModelPolicyRole,
    /// Use permitted by the reviewed scope.
    pub intended_use: ModelIntendedUse,
    /// Owner-reviewed openness classification.
    pub openness: ReviewedModelOpenness,
    /// Credential-free reference identifying the independent review.
    pub reference: ModelReference,
    /// Recorded review date as Unix milliseconds. This is provenance, not an expiry policy.
    pub reviewed_at_unix_ms: u64,
    /// Nonempty overall scope of this exact artifact/role/use review.
    pub scope: String,
    /// Separately reviewed rights. Missing review denies; inapplicability is not allowed.
    pub rights: TrustedPolicyReview,
    /// Separately reviewed serving terms or an explicit reviewed inapplicability claim.
    pub serving_terms: TrustedPolicyReview,
    /// Separately reviewed generated-output terms or an explicit reviewed inapplicability claim.
    pub output_terms: TrustedPolicyReview,
}

/// Static error in owner-supplied catalog configuration; rejected values are never interpolated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ArtifactCatalogError {
    /// Missing or invalid catalog revision or review scope.
    #[error("invalid catalog revision or review scope")]
    InvalidScope,
    /// An artifact declaration does not validate.
    #[error("invalid reviewed artifact declaration")]
    InvalidArtifact,
    /// A policy declaration does not validate.
    #[error("invalid reviewed policy declaration")]
    InvalidPolicy,
    /// Policy subject, kind, role, or use does not match the independent review.
    #[error("reviewed policy does not match its artifact, kind, role, or use")]
    PolicyBindingMismatch,
    /// The independently supplied policy bytes disagree with their declared digest.
    #[error("reviewed policy bytes do not match their declared digest")]
    PolicyContentMismatch,
    /// Rights cannot be declared inapplicable.
    #[error("artifact rights require a review")]
    RightsNotApplicable,
    /// Multiple reviews target one exact artifact/role/use, including conflicting reviews.
    #[error("duplicate or conflicting artifact review")]
    DuplicateReview,
    /// The catalog's canonical identity could not be encoded.
    #[error("cannot encode artifact catalog identity")]
    IdentityEncoding,
}

/// Failure to encode a report identity. Invalid submitted documents instead yield denials.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("cannot encode artifact assessment identity")]
pub struct ArtifactAssessmentError;

/// Exact immutable catalog binding; the revision label alone does not establish trust.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ArtifactCatalogIdentity {
    /// Catalog identity encoding version.
    pub version: u32,
    /// Owner-supplied revision, included in the digest.
    pub revision: String,
    /// BLAKE3 derive-key digest of every normalized trusted review and its scope.
    pub digest: String,
}

/// Static reason for denying this artifact assessment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactDenialReason {
    /// The requested artifact identity is malformed or unsupported.
    InvalidRequestedIdentity,
    /// A supplied artifact declaration fails structural validation.
    InvalidArtifact,
    /// Two supplied artifact declarations have the same content identity.
    DuplicateArtifact,
    /// A requested or referenced artifact is absent from the offline bundle.
    MissingArtifact,
    /// An optional parent, adapter, or additional-parent claim is explicitly unknown.
    UnknownLineage,
    /// A checkpoint/adapter link has an incompatible kind or a different declared base.
    ConflictingLineage,
    /// No trusted review exists for this exact artifact and requested role/use pair.
    MissingTrustedReview,
    /// The trusted review classifies the artifact as closed.
    ClosedArtifact,
    /// The trusted review leaves openness unknown.
    UnknownOpenness,
    /// At least one applicable rights, serving, or output axis remains unreviewed.
    UnreviewedPolicy,
    /// A supplied policy declaration fails structural validation.
    InvalidPolicy,
    /// Two supplied policy declarations have the same exact policy identity.
    DuplicatePolicy,
    /// More than one pinned document targets a subject/kind/role/use. This first contract
    /// supports one document per policy axis, without inferring contradictions in the text.
    MultiplePoliciesForScope,
    /// An exact policy document required by a trusted review is absent.
    MissingPolicy,
    /// Supplied raw bytes contradict either their declared digest or the independent review pin.
    PolicyContentMismatch,
}

/// A denial identifies only validated artifact identities, never rejected input or policy text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ArtifactDenial {
    /// Artifact affected by the failure, when its identity is structurally valid.
    pub artifact: Option<ArtifactIdentity>,
    /// Static failure category.
    pub reason: ArtifactDenialReason,
}

/// Provenance of an exact owner review matched during assessment; no execution permission.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MatchedArtifactReview {
    /// Exact artifact assessed by this independent review.
    pub artifact: ArtifactIdentity,
    /// Owner-supplied review reference.
    pub reference: ModelReference,
    /// Recorded review date in Unix milliseconds.
    pub reviewed_at_unix_ms: u64,
    /// Digest binding the complete review, including scope, openness and all policy axes.
    pub review_digest: String,
    /// Exact required policy identities; explicit inapplicability remains bound in the review.
    pub policy_documents: Vec<PolicyDocumentIdentity>,
}

/// Structured offline result. It cannot be deserialized into trust or passed to a provider gate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ArtifactAssessment {
    pub(crate) version: u32,
    pub(crate) artifact: Option<ArtifactIdentity>,
    pub(crate) role: ModelPolicyRole,
    pub(crate) intended_use: ModelIntendedUse,
    pub(crate) catalog: ArtifactCatalogIdentity,
    pub(crate) matched_reviews: Vec<MatchedArtifactReview>,
    pub(crate) denials: Vec<ArtifactDenial>,
    pub(crate) evidence_digest: String,
}
impl ArtifactAssessment {
    /// Whether this artifact and its complete resolved lineage met the offline reviewed scope.
    /// This never qualifies deployment, actual clients, caches, or physical execution.
    #[must_use]
    pub fn is_eligible(&self) -> bool {
        self.denials.is_empty()
    }
    /// Requested artifact when structurally valid.
    #[must_use]
    pub fn artifact(&self) -> Option<&ArtifactIdentity> {
        self.artifact.as_ref()
    }
    /// Selected artifact role.
    #[must_use]
    pub fn role(&self) -> ModelPolicyRole {
        self.role
    }
    /// Selected intended use.
    #[must_use]
    pub fn intended_use(&self) -> ModelIntendedUse {
        self.intended_use
    }
    /// Exact owner catalog used for this assessment.
    #[must_use]
    pub fn catalog(&self) -> &ArtifactCatalogIdentity {
        &self.catalog
    }
    /// Matched review provenance, sorted by artifact identity.
    #[must_use]
    pub fn matched_reviews(&self) -> &[MatchedArtifactReview] {
        &self.matched_reviews
    }
    /// Stable, deduplicated denial reasons.
    #[must_use]
    pub fn denials(&self) -> &[ArtifactDenial] {
        &self.denials
    }
    /// BLAKE3 binding of request, catalog, supplied evidence, matched reviews and denials.
    #[must_use]
    pub fn evidence_digest(&self) -> &str {
        &self.evidence_digest
    }
}
