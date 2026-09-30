use super::*;
use crate::SemanticDeclaration;

/// Requested model operation, independent of its serving endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelOperation {
    /// Structured chat completion.
    ChatCompletion,
    /// Text completion.
    Completion,
    /// Vector embedding.
    Embedding,
}
document! {
    /// Declared execution semantics shared by requests and supplied deployment claims.
    /// Unknown fields remain explicit and do not establish cache equivalence or eligibility.
    pub struct ModelExecutionSemantics {
        /// Independent semantic execution contract version; only 1 is supported.
        pub version: u32,
        /// Requested serving alias; contributes to identity even when artifacts are unknown.
        pub alias: String,
        /// Requested operation.
        pub operation: ModelOperation,
        /// Existing client/adapter semantic declaration, with its behavior revision.
        pub client: SemanticDeclaration,
        /// Serving-profile implementation, revision, and non-secret semantic configuration.
        pub serving_profile: Declaration<SemanticDeclaration>,
        /// Primary artifact and its pinned lineage, if declared.
        pub artifact: Declaration<ArtifactIdentity>,
        /// Additional artifact set; a declared empty set explicitly means no extra artifacts.
        pub additional_artifacts: Declaration<Vec<ArtifactIdentity>>,
        /// Effective tokenizer identity or an explicit unknown.
        pub tokenizer: Declaration<ModelComponentReference>,
        /// Effective template identity or an explicit unknown.
        pub chat_template: Declaration<ModelComponentReference>,
        /// Runtime implementation and behavior revision.
        pub runtime: Declaration<SemanticDeclaration>,
        /// Output/reasoning parser implementation and behavior revision.
        pub parser: Declaration<SemanticDeclaration>,
        /// Digest of other behavior-affecting configuration bytes.
        pub configuration: Declaration<ContentDigest>,
    }
}
impl ModelExecutionSemantics {
    /// Check supplied declarations without resolving artifacts or runtime behavior.
    pub fn validate(&self) -> Result<()> {
        version(self.version)?;
        nonempty(&self.alias)?;
        semantic(&self.client)?;
        self.serving_profile.check(semantic)?;
        self.artifact.check(ArtifactIdentity::validate)?;
        self.additional_artifacts.check(|values| {
            artifact_set(values)?;
            if let Declaration::Declared(base) = &self.artifact
                && values.contains(base)
            {
                return Err(ModelIdentityError("duplicate execution artifact"));
            }
            Ok(())
        })?;
        component(&self.tokenizer, ModelFilePurpose::Tokenizer)?;
        component(&self.chat_template, ModelFilePurpose::ChatTemplate)?;
        self.runtime.check(semantic)?;
        self.parser.check(semantic)?;
        self.configuration.check(ContentDigest::validate)
    }
    fn sorted(&self) -> Self {
        let mut value = self.clone();
        sort_artifacts(&mut value.additional_artifacts);
        value
    }
    /// Hash all semantic fields in a distinct domain, sorting only declared artifact sets
    /// and JSON object keys. Endpoint, policy, incarnation, validity, and attempt references
    /// are deliberately absent; equal declarations alone never authorize reuse.
    pub fn identity(&self) -> Result<SemanticExecutionIdentity> {
        self.validate()?;
        SemanticExecutionIdentity::of(&self.sorted())
    }
}
fn component(
    value: &Declaration<ModelComponentReference>,
    purpose: ModelFilePurpose,
) -> Result<()> {
    value.check(|value| {
        value.validate()?;
        if value.file.purpose == purpose {
            Ok(())
        } else {
            Err(ModelIdentityError("incorrect execution component purpose"))
        }
    })
}
document! {
    /// Requested model execution declaration, without runtime or policy authority.
    pub struct RequestedModelExecution {
        /// Independent request declaration version; only 1 is supported.
        pub version: u32,
        /// Credential-free HTTP(S) origin/path; excluded from semantic execution identity.
        pub endpoint: ModelReference,
        /// Declared behavior, artifacts, and adapter semantics.
        pub semantics: ModelExecutionSemantics,
        /// Pinned policy documents; unknown or a nonempty, duplicate-free set of identities.
        pub policy_evidence: Declaration<Vec<PolicyDocumentIdentity>>,
    }
}
impl RequestedModelExecution {
    /// Check the endpoint, semantics, and policy references without authorizing execution.
    pub fn validate(&self) -> Result<()> {
        version(self.version)?;
        self.endpoint.endpoint()?;
        self.semantics.validate()?;
        self.policy_evidence.check(|values| {
            if values.is_empty() {
                return Err(ModelIdentityError("empty policy evidence set"));
            }
            let mut seen = BTreeSet::new();
            for value in values {
                value.validate()?;
                if !seen.insert(&value.digest) {
                    return Err(ModelIdentityError("duplicate policy evidence"));
                }
            }
            Ok(())
        })
    }
    /// Return the declaration's semantic identity, independently of endpoint and policy evidence.
    pub fn semantic_identity(&self) -> Result<SemanticExecutionIdentity> {
        self.validate()?;
        self.semantics.identity()
    }
}
document! {
    /// Supplied revocation evidence; the caller must independently qualify its meaning and age.
    pub struct ModelRevocationEvidence {
        /// Pinned revocation document, with no inferred good-standing status.
        pub document: PinnedModelDocument,
        /// Producer-reported observation time, in milliseconds since the Unix epoch.
        pub observed_at_unix_ms: u64,
    }
}
impl ModelRevocationEvidence {
    /// Validate the pinned document; no clock or revocation service is consulted.
    pub fn validate(&self) -> Result<()> {
        self.document.validate()
    }
}
document! {
    /// Claimed validity interval and revocation evidence. Validation performs no clock check.
    pub struct ModelEvidenceValidity {
        /// Claimed earliest valid instant, in milliseconds since the Unix epoch.
        pub not_before_unix_ms: u64,
        /// Claimed expiration instant; null means unknown, never perpetual validity.
        pub expires_at_unix_ms: Option<u64>,
        /// Supplied revocation record, or an explicit unknown.
        pub revocation: Declaration<ModelRevocationEvidence>,
    }
}
impl ModelEvidenceValidity {
    /// Reject inverted or empty intervals while preserving unknown expiration/revocation.
    pub fn validate(&self) -> Result<()> {
        if self
            .expires_at_unix_ms
            .is_some_and(|expires| expires <= self.not_before_unix_ms)
        {
            return Err(ModelIdentityError("invalid deployment evidence interval"));
        }
        self.revocation.check(ModelRevocationEvidence::validate)
    }
}
document! {
    /// Supplied deployment evidence awaiting independent qualification.
    /// Deserializing this document never establishes measurement trust, loaded-byte equality,
    /// endpoint binding, policy eligibility, or continued deployment validity.
    pub struct ModelDeploymentEvidence {
        /// Independent deployment evidence version; only 1 is supported.
        pub version: u32,
        /// Measurement/collection method, version, and non-secret semantic parameters.
        pub method: SemanticDeclaration,
        /// Declared issuer identity, without inferred trust.
        pub issuer: ModelReference,
        /// Claimed verifier implementation and version, or unknown; not a verification result.
        pub verifier: Declaration<SemanticDeclaration>,
        /// Pinned raw evidence bytes and source.
        pub raw_evidence: PinnedModelDocument,
        /// Endpoint to which the producer binds this evidence.
        pub endpoint: ModelReference,
        /// Producer-declared serving instance identity.
        pub instance: String,
        /// Producer-declared incarnation; changes the evidence binding, not semantic identity.
        pub incarnation: String,
        /// Claimed loaded artifacts, unknown or a nonempty set; no bytes are inspected here.
        pub claimed_loaded_artifacts: Declaration<Vec<ArtifactIdentity>>,
        /// Claimed effective behavior. Differences from requested semantics remain representable.
        pub effective: ModelExecutionSemantics,
        /// Claimed time interval and revocation information, requiring later qualification.
        pub validity: ModelEvidenceValidity,
    }
}
impl ModelDeploymentEvidence {
    /// Structurally validate supplied evidence without granting any execution authority.
    pub fn validate(&self) -> Result<()> {
        version(self.version)?;
        semantic(&self.method)?;
        self.verifier.check(semantic)?;
        self.raw_evidence.validate()?;
        self.endpoint.endpoint()?;
        nonempty(&self.instance)?;
        nonempty(&self.incarnation)?;
        self.claimed_loaded_artifacts.check(|values| {
            if values.is_empty() {
                return Err(ModelIdentityError("empty loaded artifact set"));
            }
            artifact_set(values)
        })?;
        self.effective.validate()?;
        self.validity.validate()
    }
    /// Hash all evidence fields, including locators, incarnation, method, and validity.
    /// Artifact sets and semantic object keys are canonicalized; raw evidence is only referenced.
    pub fn identity(&self) -> Result<DeploymentEvidenceIdentity> {
        self.validate()?;
        let mut value = self.clone();
        sort_artifacts(&mut value.claimed_loaded_artifacts);
        value.effective = value.effective.sorted();
        DeploymentEvidenceIdentity::of(&value)
    }
}
document! {
    /// Reference to an existing durable attempt/observation, without copying receipt facts.
    pub struct ModelAttemptReference {
        /// Run identity of the durable attempt.
        pub run_id: String,
        /// Launch identity of the durable attempt.
        pub launch_id: String,
        /// Physical attempt identity, including failed/retried calls.
        pub attempt_id: String,
        /// Exact observation sequence; null references the attempt only, never an inferred latest row.
        pub observation_sequence: Option<u64>,
    }
}
impl ModelAttemptReference {
    /// Require explicit run, launch, and physical attempt identities.
    pub fn validate(&self) -> Result<()> {
        nonempty(&self.run_id)?;
        nonempty(&self.launch_id)?;
        nonempty(&self.attempt_id)
    }
}
document! {
    /// Supplementary termination facts absent from the existing attempt receipt contract.
    pub struct ModelTermination {
        /// Provider-reported native termination reason, null when missing.
        pub native: Option<String>,
        /// Adapter-normalized termination reason, null when missing.
        pub normalized: Option<String>,
    }
}
impl ModelTermination {
    /// Validate supplied reasons, preserving both as independent observations.
    pub fn validate(&self) -> Result<()> {
        for value in [&self.native, &self.normalized].into_iter().flatten() {
            nonempty(value)?;
        }
        Ok(())
    }
}
document! {
    /// Supplementary observed identity evidence bound to durable receipt facts.
    /// Resolved model, route, response ID, usage, and cost remain owned by the referenced receipt.
    pub struct ObservedModelExecution {
        /// Independent observation identity contract version; only 1 is supported.
        pub version: u32,
        /// Durable physical attempt and optional exact observation reference.
        pub attempt: ModelAttemptReference,
        /// Requested semantic declaration identity; unknown when it was not captured.
        pub requested: Declaration<SemanticExecutionIdentity>,
        /// Supplied deployment evidence identity, unknown when absent; never automatically trusted.
        pub deployment_evidence: Declaration<DeploymentEvidenceIdentity>,
        /// Supplementary termination observations, null when none were captured.
        pub termination: Option<ModelTermination>,
    }
}
impl ObservedModelExecution {
    /// Check reference structure; resolving the durable attempt and evidence is a later boundary.
    pub fn validate(&self) -> Result<()> {
        version(self.version)?;
        self.attempt.validate()?;
        self.requested.check(SemanticExecutionIdentity::validate)?;
        self.deployment_evidence
            .check(DeploymentEvidenceIdentity::validate)?;
        if let Some(value) = &self.termination {
            value.validate()?;
        }
        Ok(())
    }
    /// Hash the full attempt/observation binding and supplied supplementary facts.
    pub fn identity(&self) -> Result<ObservedExecutionIdentity> {
        self.validate()?;
        ObservedExecutionIdentity::of(self)
    }
}
