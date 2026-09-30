use super::*;

/// Model role for which the supplied policy document is relevant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelPolicyRole {
    /// Teacher producing candidate reasoning and answers.
    Teacher,
    /// Judge grading model output.
    Judge,
    /// Embedding model.
    Embedding,
    /// Model synthesizing user prompts.
    UserSynthesis,
    /// Student trained from generated data.
    Student,
    /// Derivative, adapter, quantization, or checkpoint.
    Derivative,
}
/// Intended use whose policy needs separate qualification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelIntendedUse {
    /// Serving inference requests.
    Inference,
    /// Generating a dataset, including reasoning traces.
    DatasetGeneration,
    /// Training or tuning weights.
    Training,
    /// Evaluating models or outputs.
    Evaluation,
    /// Distributing artifacts or generated data.
    Redistribution,
}
/// Kind of pinned document; serving and generated-output terms are deliberately separate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelPolicyDocumentKind {
    /// A review record, without an approval implication.
    Review,
    /// Catalog record.
    Catalog,
    /// Rights/license document.
    Rights,
    /// Lineage record.
    Lineage,
    /// Terms governing the serving service.
    ServingTerms,
    /// Terms governing use of generated output.
    OutputTerms,
}
document! {
    /// Pinned source and bytes of supplied evidence; no content is fetched or trusted.
    pub struct PinnedModelDocument {
        /// Credential-free document source and declared immutable revision.
        pub source: PinnedModelSource,
        /// Digest of the raw document bytes.
        pub content: ContentDigest,
    }
}
impl PinnedModelDocument {
    /// Validate the source and digest encoding.
    pub fn validate(&self) -> Result<()> {
        self.source.validate()?;
        self.content.validate()
    }
}
document! {
    /// Version 1 reference to policy evidence for a particular role and intended use.
    /// It is independent of task-source rights and never grants model eligibility.
    pub struct ModelPolicyEvidence {
        /// Independent policy-document contract version; only 1 is supported.
        pub version: u32,
        /// Distinct document kind, including separate serving and output terms.
        pub kind: ModelPolicyDocumentKind,
        /// Artifact to which this evidence is asserted to apply; applicability is not verified.
        pub subject: ArtifactIdentity,
        /// Exact referenced document declaration.
        pub document: PinnedModelDocument,
        /// Role to which this evidence is asserted to apply.
        pub role: ModelPolicyRole,
        /// Intended use to be independently assessed.
        pub intended_use: ModelIntendedUse,
    }
}
impl ModelPolicyEvidence {
    /// Validate the version and pinned document, without evaluating its policy meaning.
    pub fn validate(&self) -> Result<()> {
        version(self.version)?;
        self.subject.validate()?;
        self.document.validate()
    }
    /// Hash every field, including subject, locator, revision, role, use, and kind, in the policy domain.
    pub fn identity(&self) -> Result<PolicyDocumentIdentity> {
        self.validate()?;
        PolicyDocumentIdentity::of(self)
    }
}
