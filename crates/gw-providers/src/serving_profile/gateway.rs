use super::*;
use gw_schema::{
    ContentDigest, Declaration, DeploymentEvidenceIdentity, ModelAttemptReference,
    ModelDeploymentEvidence, ModelExecutionSemantics, ModelReference, ModelRevocationEvidence,
    PinnedModelArtifact, SemanticExecutionIdentity,
};

document! {
    /// Optional intentional pin to one serving instance and loaded-generation incarnation.
    /// An unconstrained fleet request has no such pin and never preselects deployment evidence.
    pub struct ReplicaConstraint {
        /// Exact serving instance identifier.
        pub instance: String,
        /// Exact incarnation, which a qualified loader must change on restart or model reload.
        pub incarnation: String,
    }
}
impl ReplicaConstraint {
    /// Validate explicit instance/incarnation text without resolving a deployment.
    pub fn validate(&self) -> Result<()> {
        if !text_valid(&self.instance) || !text_valid(&self.incarnation) {
            return Err(ProfileError::InvalidProfile);
        }
        Ok(())
    }
}

document! {
    /// Supplied request binding for one physical attempt. The mandatory immutable target is
    /// independent of replica evidence. Deserialization grants no execution authority.
    pub struct GatewayRequest {
        /// Request-binding contract version; only 1 is supported.
        pub version: u32,
        /// Mandatory route-independent semantic identity, recomputed during consistency checks.
        pub target: SemanticExecutionIdentity,
        /// Supplied declaration resolving the target. Artifact documents are supplied separately.
        pub semantics: ModelExecutionSemantics,
        /// Normalized operation endpoint, excluding credentials.
        pub endpoint: ModelReference,
        /// Digest of the exact outgoing JSON bytes, checked against the immutable prepared body.
        pub request_body_digest: ContentDigest,
        /// Explicit correlation identifier for this request, not inferred from a model alias.
        pub correlation_id: String,
        /// Durable physical-attempt reference. Every retry uses a distinct attempt.
        pub attempt: ModelAttemptReference,
        /// Optional deliberate instance/incarnation pin; None permits equivalent valid replicas.
        pub replica_constraint: Option<ReplicaConstraint>,
    }
}
impl GatewayRequest {
    /// Bind pure prepared bytes to an explicit attempt and optional replica constraint.
    /// This does not reserve the attempt, dispatch a request, or qualify the target.
    pub fn from_prepared(
        prepared: &PreparedProfileRequest,
        correlation_id: impl Into<String>,
        attempt: ModelAttemptReference,
        replica_constraint: Option<ReplicaConstraint>,
    ) -> Result<Self> {
        let value = Self {
            version: 1,
            target: prepared.target().clone(),
            semantics: prepared.semantics().clone(),
            endpoint: prepared.endpoint().clone(),
            request_body_digest: prepared.body_digest().clone(),
            correlation_id: correlation_id.into(),
            attempt,
            replica_constraint,
        };
        value.validate()?;
        Ok(value)
    }
    /// Check shape only. Identity/body are compared to the pure preparer's immutable result by
    /// check_gateway_consistency; a deserialized request cannot invent its own accepted body.
    pub fn validate(&self) -> Result<()> {
        if self.version != 1 || !text_valid(&self.correlation_id) {
            return Err(ProfileError::InvalidProfile);
        }
        self.target
            .validate()
            .map_err(|_| ProfileError::InvalidProfile)?;
        self.semantics
            .validate()
            .map_err(|_| ProfileError::InvalidProfile)?;
        crate::normalize_endpoint(self.endpoint.as_str())
            .map_err(|_| ProfileError::InvalidProfile)?;
        self.request_body_digest
            .validate()
            .map_err(|_| ProfileError::InvalidProfile)?;
        self.attempt
            .validate()
            .map_err(|_| ProfileError::InvalidProfile)?;
        if let Some(value) = &self.replica_constraint {
            value.validate()?;
        }
        Ok(())
    }
}

document! {
    /// Unauthenticated response-side binding to actual supplied replica evidence and one attempt.
    /// A real same-replica gateway must authenticate this evidence and retain the same loaded
    /// incarnation atomically through execution. Matching these fields proves neither property.
    pub struct GatewayResponseEvidence {
        /// Response-binding contract version; only 1 is supported.
        pub version: u32,
        /// Claimed requested semantic target, unknown when no binding was supplied.
        pub target: Declaration<SemanticExecutionIdentity>,
        /// Actual replica evidence identity, unknown when unavailable; never a trusted flag.
        pub deployment_evidence: Declaration<DeploymentEvidenceIdentity>,
        /// Endpoint whose response carries the evidence.
        pub endpoint: ModelReference,
        /// Claimed exact request-body digest.
        pub request_body_digest: ContentDigest,
        /// Claimed request correlation.
        pub correlation_id: String,
        /// Claimed physical attempt, including exact observation sequence when present.
        pub attempt: ModelAttemptReference,
    }
}
impl GatewayResponseEvidence {
    /// Check shape without authenticating any response or replica claim.
    pub fn validate(&self) -> Result<()> {
        if self.version != 1 || !text_valid(&self.correlation_id) {
            return Err(ProfileError::InvalidProfile);
        }
        if let Declaration::Declared(value) = &self.target {
            value.validate().map_err(|_| ProfileError::InvalidProfile)?;
        }
        if let Declaration::Declared(value) = &self.deployment_evidence {
            value.validate().map_err(|_| ProfileError::InvalidProfile)?;
        }
        crate::normalize_endpoint(self.endpoint.as_str())
            .map_err(|_| ProfileError::InvalidProfile)?;
        self.request_body_digest
            .validate()
            .map_err(|_| ProfileError::InvalidProfile)?;
        self.attempt
            .validate()
            .map_err(|_| ProfileError::InvalidProfile)
    }
}

document! {
    /// Separately supplied interpretation of one exact revocation document and replica evidence.
    /// A declared false value is an observation claim, not proof of an authenticated authority.
    pub struct SuppliedValiditySnapshot {
        /// Validity interpretation contract version; only 1 is supported.
        pub version: u32,
        /// Exact deployment evidence to which this interpretation belongs.
        pub deployment_evidence: DeploymentEvidenceIdentity,
        /// Exact revocation document and observation time from the deployment's validity claim.
        pub revocation: ModelRevocationEvidence,
        /// Supplied revocation interpretation; unknown never establishes current validity.
        pub revoked: Declaration<bool>,
        /// Exclusive expiry for this supplied interpretation, in Unix milliseconds.
        pub valid_until_unix_ms: u64,
    }
}
impl SuppliedValiditySnapshot {
    /// Reject unsupported versions and impossible interpretation intervals, without reading a clock.
    pub fn validate(&self) -> Result<()> {
        if self.version != 1 || self.valid_until_unix_ms <= self.revocation.observed_at_unix_ms {
            return Err(ProfileError::InvalidProfile);
        }
        self.deployment_evidence
            .validate()
            .map_err(|_| ProfileError::InvalidProfile)?;
        self.revocation
            .validate()
            .map_err(|_| ProfileError::InvalidProfile)
    }
}

/// Three-valued supplied-document result. Mismatch always outranks unrelated missing information.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsistencyState {
    /// Complete supplied fields agree; this is not authentication, qualification or permission.
    Consistent,
    /// Required supplied information is missing or explicitly unknown.
    Unknown,
    /// A malformed, unsupported, expired, revoked or contradictory input was observed.
    Mismatch,
}

/// Static diagnostics contain no rejected document values, secrets, or source error text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsistencyReason {
    /// Request fields failed structural validation.
    InvalidRequest,
    /// Response evidence fields failed structural validation.
    InvalidResponse,
    /// Deployment fields failed structural validation.
    InvalidDeployment,
    /// Declared target digest did not match the recomputed semantic declaration.
    TargetIdentityMismatch,
    /// Required semantic field was unknown.
    UnknownSemantics,
    /// Known behavior fields differed.
    SemanticMismatch,
    /// The profile/adapter declaration could not be resolved to this supported software contract.
    UnsupportedProfileBehavior,
    /// A referenced artifact declaration was not supplied.
    MissingArtifact,
    /// An artifact declaration failed structural validation.
    InvalidArtifact,
    /// Artifact evidence repeated a recomputed identity.
    DuplicateArtifact,
    /// A requested component contradicted the artifact's pinned component.
    ComponentMismatch,
    /// An artifact's lineage contains an unknown parent claim.
    UnknownArtifactLineage,
    /// Supplied parent kind or base contradicts an artifact's lineage.
    ConflictingLineage,
    /// No complete loaded-artifact set was supplied.
    UnknownLoadedArtifacts,
    /// Claimed loaded artifacts differed from the primary and additional execution artifacts.
    LoadedArtifactsMismatch,
    /// The actual deployment document was unavailable.
    MissingDeploymentEvidence,
    /// The response did not identify its actual replica evidence.
    UnknownDeploymentIdentity,
    /// Response or validity data bound a different replica evidence document.
    DeploymentIdentityMismatch,
    /// Unsupported measurement method/revision/configuration, including installed-only claims.
    UnsupportedMeasurement,
    /// The supplied verifier provenance was unknown; no authentication is inferred when known.
    UnknownVerifier,
    /// Supplied bytes or response digest did not match the requested exact body.
    RequestBodyMismatch,
    /// Request, response, and deployment endpoints differed.
    EndpointMismatch,
    /// Response bound another correlation identifier.
    CorrelationMismatch,
    /// Response bound another physical attempt or observation sequence.
    AttemptMismatch,
    /// Actual replica differed from the optional intentional pin.
    ReplicaMismatch,
    /// No separately supplied validity interpretation was available.
    MissingValidity,
    /// Supplied validity interpretation failed structural validation.
    InvalidValidity,
    /// The deployment evidence did not declare an expiry.
    UnknownExpiration,
    /// Explicit evaluation time preceded an evidence start or observation time.
    NotYetValid,
    /// Explicit evaluation time reached an exclusive expiry.
    Expired,
    /// No revocation claim or interpretation was supplied.
    UnknownRevocation,
    /// The supplied interpretation explicitly reported revocation.
    Revoked,
    /// Revocation document or its observation time did not match the deployment claim.
    RevocationBindingMismatch,
}

/// Result for one independently evaluated axis, without any trusted/qualified boolean.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ConsistencyAxis {
    state: ConsistencyState,
    reasons: Vec<ConsistencyReason>,
}
impl ConsistencyAxis {
    /// Supplied-document outcome for this axis.
    #[must_use]
    pub fn state(&self) -> ConsistencyState {
        self.state
    }
    /// Deterministic, duplicate-free diagnostics.
    #[must_use]
    pub fn reasons(&self) -> &[ConsistencyReason] {
        &self.reasons
    }
    pub(super) fn new() -> Self {
        Self {
            state: ConsistencyState::Consistent,
            reasons: vec![],
        }
    }
    pub(super) fn note(&mut self, state: ConsistencyState, reason: ConsistencyReason) {
        self.state = self.state.max(state);
        self.reasons.push(reason);
    }
    fn finish(&mut self) {
        self.reasons.sort();
        self.reasons.dedup();
    }
}

/// Immutable supplied-document report. A Consistent aggregate grants no model, cache, dispatch,
/// policy, authenticated-evidence, or loaded-generation lifecycle authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GatewayConsistency {
    semantics: ConsistencyAxis,
    binding: ConsistencyAxis,
    validity: ConsistencyAxis,
    target: Option<SemanticExecutionIdentity>,
    deployment_evidence: Option<DeploymentEvidenceIdentity>,
    request_binding: Option<GatewayAttemptBinding>,
    evaluated_at_unix_ms: u64,
}

/// Auditable request coordinates retained in a consistency report, independently of semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GatewayAttemptBinding {
    /// Exact physical attempt/observation that was compared.
    pub attempt: ModelAttemptReference,
    /// Credential-free destination endpoint that was compared.
    pub endpoint: ModelReference,
    /// Exact body digest that was compared to the immutable prepared request.
    pub request_body_digest: ContentDigest,
    /// Correlation identifier that was compared.
    pub correlation_id: String,
}
impl GatewayConsistency {
    /// Semantic declaration and artifact-resolution outcome.
    #[must_use]
    pub fn semantics(&self) -> &ConsistencyAxis {
        &self.semantics
    }
    /// Request/body/attempt/actual-replica binding outcome.
    #[must_use]
    pub fn binding(&self) -> &ConsistencyAxis {
        &self.binding
    }
    /// Explicit-time and supplied revocation consistency outcome.
    #[must_use]
    pub fn validity(&self) -> &ConsistencyAxis {
        &self.validity
    }
    /// Mismatch outranks Unknown; only complete agreement across all axes is Consistent.
    #[must_use]
    pub fn aggregate(&self) -> ConsistencyState {
        self.semantics
            .state
            .max(self.binding.state)
            .max(self.validity.state)
    }
    /// Recomputed semantic target identity, when structurally valid.
    #[must_use]
    pub fn target(&self) -> Option<&SemanticExecutionIdentity> {
        self.target.as_ref()
    }
    /// Recomputed actual replica evidence identity, when structurally valid.
    #[must_use]
    pub fn deployment_evidence(&self) -> Option<&DeploymentEvidenceIdentity> {
        self.deployment_evidence.as_ref()
    }
    /// Exact request coordinates when the supplied request was structurally valid.
    #[must_use]
    pub fn request_binding(&self) -> Option<&GatewayAttemptBinding> {
        self.request_binding.as_ref()
    }
    /// Explicit caller-supplied evaluation time; no system clock was consulted.
    #[must_use]
    pub fn evaluated_at_unix_ms(&self) -> u64 {
        self.evaluated_at_unix_ms
    }
}

/// Check supplied snapshots without I/O, authentication, a verifier callback, or execution.
///
/// The supported loader report method is implementation `ghostwriter/loaded-generation-report`,
/// revision `1`, configuration `{"scope":"loaded_generation"}`. This recognizes a wire contract,
/// not a qualified measurement implementation. The supplied verifier declaration is provenance
/// only. Installed-file inventories, model aliases, and endpoint metadata do not satisfy it.
///
/// The deployment incarnation must name a loaded generation and change on restart/reload. Real
/// qualification must establish same-replica measurement, authentication, freshness/revocation,
/// and atomic retention of that generation through inference; metadata equality cannot do so.
/// The client-side prepared object supplies the independently computed body/target binding;
/// matching self-reported request/response digests alone is insufficient. Each retry supplies
/// its own request/attempt and actual destination evidence. Policy eligibility
/// remains a separate artifact assessment; even a Consistent report never authorizes execution.
#[must_use]
pub fn check_gateway_consistency(
    prepared: &PreparedProfileRequest,
    request: &GatewayRequest,
    response: &GatewayResponseEvidence,
    deployment: Option<&ModelDeploymentEvidence>,
    artifacts: &[PinnedModelArtifact],
    validity: Option<&SuppliedValiditySnapshot>,
    now_unix_ms: u64,
) -> GatewayConsistency {
    use ConsistencyReason as R;
    use ConsistencyState::{Mismatch, Unknown};
    let mut report = GatewayConsistency {
        semantics: ConsistencyAxis::new(),
        binding: ConsistencyAxis::new(),
        validity: ConsistencyAxis::new(),
        target: request.semantics.identity().ok(),
        deployment_evidence: deployment.and_then(|v| v.identity().ok()),
        request_binding: request.validate().ok().map(|()| GatewayAttemptBinding {
            attempt: request.attempt.clone(),
            endpoint: request.endpoint.clone(),
            request_body_digest: request.request_body_digest.clone(),
            correlation_id: request.correlation_id.clone(),
        }),
        evaluated_at_unix_ms: now_unix_ms,
    };
    if request.validate().is_err() {
        report.binding.note(Mismatch, R::InvalidRequest);
    }
    if response.validate().is_err() {
        report.binding.note(Mismatch, R::InvalidResponse);
    }
    if report.target.as_ref() != Some(&request.target) {
        report.semantics.note(Mismatch, R::TargetIdentityMismatch);
    }
    if &request.target != prepared.target() {
        report.binding.note(Mismatch, R::TargetIdentityMismatch);
    }
    match &response.target {
        Declaration::Declared(value) if value != &request.target => {
            report.binding.note(Mismatch, R::TargetIdentityMismatch)
        }
        Declaration::Unknown => report.binding.note(Unknown, R::UnknownSemantics),
        _ => {}
    }
    if request.request_body_digest != *prepared.body_digest()
        || request.request_body_digest != response.request_body_digest
    {
        report.binding.note(Mismatch, R::RequestBodyMismatch);
    }
    if request.endpoint != response.endpoint || &request.endpoint != prepared.endpoint() {
        report.binding.note(Mismatch, R::EndpointMismatch);
    }
    if request.correlation_id != response.correlation_id {
        report.binding.note(Mismatch, R::CorrelationMismatch);
    }
    if request.attempt != response.attempt {
        report.binding.note(Mismatch, R::AttemptMismatch);
    }
    super::consistency::semantics(
        &request.semantics,
        deployment.map(|v| &v.effective),
        artifacts,
        &mut report.semantics,
    );
    if let Some(deployment) = deployment {
        if deployment.validate().is_err() {
            report.binding.note(Mismatch, R::InvalidDeployment);
        }
        let method = gw_schema::SemanticDeclaration::new(
            "ghostwriter/loaded-generation-report",
            "1",
            serde_json::json!({"scope":"loaded_generation"}),
        );
        if deployment.method != method {
            report.binding.note(Mismatch, R::UnsupportedMeasurement);
        }
        if matches!(deployment.verifier, Declaration::Unknown) {
            report.binding.note(Unknown, R::UnknownVerifier);
        }
        if deployment.endpoint != request.endpoint {
            report.binding.note(Mismatch, R::EndpointMismatch);
        }
        if request.replica_constraint.as_ref().is_some_and(|v| {
            v.instance != deployment.instance || v.incarnation != deployment.incarnation
        }) {
            report.binding.note(Mismatch, R::ReplicaMismatch);
        }
        match &response.deployment_evidence {
            Declaration::Unknown => report.binding.note(Unknown, R::UnknownDeploymentIdentity),
            Declaration::Declared(value) if report.deployment_evidence.as_ref() != Some(value) => {
                report.binding.note(Mismatch, R::DeploymentIdentityMismatch)
            }
            _ => {}
        }
        super::consistency::loaded_artifacts(
            &request.semantics,
            &deployment.claimed_loaded_artifacts,
            &mut report.semantics,
        );
    } else {
        report.binding.note(Unknown, R::MissingDeploymentEvidence);
    }
    super::validity::check(
        deployment,
        report.deployment_evidence.as_ref(),
        validity,
        now_unix_ms,
        &mut report.validity,
    );
    report.semantics.finish();
    report.binding.finish();
    report.validity.finish();
    report
}
