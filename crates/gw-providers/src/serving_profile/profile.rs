use super::*;
use gw_schema::{
    AccountingPolicy, DataCollection, ModelAdapterBehavior, ModelOperation, SemanticDeclaration,
};
use std::collections::BTreeSet;

/// Supported wire mappings, without any claim that an endpoint implements them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServingDialect {
    /// OpenRouter chat fields, including strict provider routing and nested reasoning controls.
    OpenRouterV1,
    /// vLLM OpenAI-compatible text/chat/embedding shapes with explicit template kwargs.
    VllmV1,
}

/// Controls whose support must be explicitly declared for the selected model/server recipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileControl {
    /// Sampling temperature.
    Temperature,
    /// Nucleus sampling.
    TopP,
    /// Random seed.
    Seed,
    /// Combined output token cap; not a reasoning-token measurement.
    MaxOutputTokens,
    /// Literal, model-specific effort string; names are never mapped to another level.
    ReasoningEffort,
    /// Explicit reasoning-token budget, supported only by the OpenRouter mapping.
    ReasoningBudget,
    /// Template-specific enable_thinking kwarg.
    EnableThinking,
    /// Template-specific preserve_thinking kwarg.
    PreserveThinking,
    /// Template-specific clear_thinking kwarg.
    ClearThinking,
    /// JSON-schema response formatting.
    JsonSchema,
    /// Chat token log-probability request.
    Logprobs,
    /// Request provider usage metadata; absence remains unknown.
    Usage,
    /// Preserve prior assistant reasoning text as a separate wire field.
    ReasoningHistory,
    /// Preserve structured reasoning blocks as a separate wire field.
    StructuredReasoningHistory,
}

document! {
    /// Route-independent configured wire behavior. Capability entries are supplied claims,
    /// not approvals or evidence of actual endpoint/model support. Sets are canonicalized.
    pub struct ProfileBehavior {
        /// Behavior contract version; only 1 is supported.
        pub version: u32,
        /// Explicit wire mapping.
        pub dialect: ServingDialect,
        /// Nonempty set of operations claimed by this profile.
        pub operations: Vec<ModelOperation>,
        /// Duplicate-free set of controls the configured recipe claims to support.
        pub capabilities: Vec<ProfileControl>,
        /// Exact allowed effort values; no automatic xhigh/max or cross-model equivalence.
        pub reasoning_efforts: Vec<String>,
    }
}
impl ProfileBehavior {
    /// Validate the supported mapping and non-contradictory capability declarations.
    pub fn validate(&self) -> Result<()> {
        if self.version != 1
            || self.operations.is_empty()
            || self
                .operations
                .iter()
                .enumerate()
                .any(|(i, v)| self.operations[..i].contains(v))
            || self.capabilities.iter().collect::<BTreeSet<_>>().len() != self.capabilities.len()
            || self.reasoning_efforts.iter().collect::<BTreeSet<_>>().len()
                != self.reasoning_efforts.len()
            || self
                .reasoning_efforts
                .iter()
                .any(|v| v.is_empty() || !v.bytes().all(|c| c.is_ascii_lowercase() || c == b'_'))
            || self.supports(ProfileControl::ReasoningEffort) != !self.reasoning_efforts.is_empty()
        {
            return Err(ProfileError::InvalidProfile);
        }
        match self.dialect {
            ServingDialect::OpenRouterV1 => {
                if self.operations != [ModelOperation::ChatCompletion]
                    || [
                        ProfileControl::EnableThinking,
                        ProfileControl::PreserveThinking,
                        ProfileControl::ClearThinking,
                    ]
                    .iter()
                    .any(|v| self.supports(*v))
                    || self.reasoning_efforts.iter().any(|v| {
                        !["none", "minimal", "low", "medium", "high", "xhigh"].contains(&v.as_str())
                    })
                {
                    return Err(ProfileError::InvalidProfile);
                }
            }
            ServingDialect::VllmV1
                if self.supports(ProfileControl::ReasoningBudget)
                    || self.supports(ProfileControl::StructuredReasoningHistory) =>
            {
                return Err(ProfileError::InvalidProfile);
            }
            ServingDialect::VllmV1 => {}
        }
        Ok(())
    }
    /// Whether this configuration declares one control; not runtime qualification.
    #[must_use]
    pub fn supports(&self, control: ProfileControl) -> bool {
        self.capabilities.contains(&control)
    }
    /// Canonical semantic declaration containing behavior only, excluding route/auth/operations policy.
    pub fn declaration(&self) -> Result<SemanticDeclaration> {
        self.validate()?;
        let mut value = self.clone();
        value.operations.sort_by_key(|v| match v {
            ModelOperation::ChatCompletion => 0,
            ModelOperation::Completion => 1,
            ModelOperation::Embedding => 2,
        });
        value.capabilities.sort();
        value.reasoning_efforts.sort();
        Ok(SemanticDeclaration::new(
            "gw-providers/offline-serving-profile",
            "1",
            serde_json::to_value(value).map_err(|_| ProfileError::Encoding)?,
        ))
    }
    /// Identity of the pure preparer's supported behavior, independently of endpoint or retries.
    pub fn adapter_behavior(&self, operation: ModelOperation) -> Result<ModelAdapterBehavior> {
        self.validate()?;
        if !self.operations.contains(&operation) {
            return Err(ProfileError::InvalidSemantics);
        }
        Ok(ModelAdapterBehavior {
            declaration: SemanticDeclaration::new(
                "gw-providers/offline-request-preparation",
                "1",
                serde_json::json!({"operation":operation}),
            ),
        })
    }
}

document! {
    /// Explicit OpenRouter allowlist; the prepared body always disables fallbacks and requires
    /// parameter support. A provider slug is routing policy, not verified artifact identity.
    pub struct StrictProviderRouting {
        /// Nonempty provider allowlist. Input order is not a priority order.
        pub only: Vec<String>,
        /// Explicit upstream data-collection policy.
        pub data_collection: DataCollection,
    }
}
impl StrictProviderRouting {
    /// Reject empty, blank, or duplicate restrictions.
    pub fn validate(&self) -> Result<()> {
        if self.only.is_empty()
            || self.only.iter().any(|v| !text_valid(v))
            || self.only.iter().collect::<BTreeSet<_>>().len() != self.only.len()
        {
            return Err(ProfileError::InvalidProfile);
        }
        Ok(())
    }
}

document! {
    /// Declared server replica defaults, independent of client in-flight and backend batch limits.
    /// These fields neither deploy a server nor establish measured capacity.
    pub struct ReplicaPolicy {
        /// Minimum containers kept warm.
        pub min_containers: u32,
        /// Maximum containers allowed by the declared server policy.
        pub max_containers: u32,
        /// Extra warm containers requested beyond current demand.
        pub buffer_containers: u32,
        /// Soft per-container autoscaling target, not a hard active-request limit.
        pub target_concurrency: u32,
    }
}
impl ReplicaPolicy {
    /// Check self-consistency of the supplied operational values.
    pub fn validate(&self) -> Result<()> {
        if self.max_containers == 0
            || self.min_containers > self.max_containers
            || self.buffer_containers > self.max_containers
            || self.target_concurrency == 0
        {
            return Err(ProfileError::InvalidProfile);
        }
        Ok(())
    }
}

document! {
    /// Client and server operational declarations, excluded from semantic model identity.
    pub struct ProfileOperations {
        /// Monetary policy; the profile default is explicitly ObservationOnly.
        pub accounting: AccountingPolicy,
        /// Client physical-request concurrency bound; not backend batch capacity.
        pub max_in_flight: u32,
        /// Optional start-rate bound; null means no declared start-rate bound, not zero.
        pub requests_per_minute: Option<u32>,
        /// Independently declared backend batching capacity, if known.
        pub backend_batch_capacity: Option<u32>,
        /// Independently declared autoscaling/warm capacity, if configured.
        pub replicas: Option<ReplicaPolicy>,
        /// Whole-request timeout declaration; not enforced by offline preparation.
        pub request_timeout_ms: u64,
        /// Idle-stream timeout declaration; not enforced by offline preparation.
        pub idle_stream_timeout_ms: u64,
        /// Maximum physical attempts, including the first. Each needs distinct evidence.
        pub max_attempts: u32,
    }
}
impl Default for ProfileOperations {
    fn default() -> Self {
        Self {
            accounting: AccountingPolicy::ObservationOnly,
            max_in_flight: 8,
            requests_per_minute: None,
            backend_batch_capacity: None,
            replicas: None,
            request_timeout_ms: 120_000,
            idle_stream_timeout_ms: 30_000,
            max_attempts: 3,
        }
    }
}
impl ProfileOperations {
    /// Check ranges without imposing monetary serialization or creating runtime limits.
    pub fn validate(&self) -> Result<()> {
        if !self.accounting.is_valid()
            || self.max_in_flight == 0
            || self.requests_per_minute == Some(0)
            || self.backend_batch_capacity == Some(0)
            || self.request_timeout_ms == 0
            || self.idle_stream_timeout_ms == 0
            || self.max_attempts == 0
        {
            return Err(ProfileError::InvalidProfile);
        }
        if let Some(value) = &self.replicas {
            value.validate()?;
        }
        Ok(())
    }
}

document! {
    /// Configured endpoint, authentication references and operational defaults, separate from
    /// semantic behavior. Parsing performs no environment access, credential discovery, or I/O.
    pub struct ServingProfile {
        /// Independent configured-profile contract version; only 1 is supported.
        pub version: u32,
        /// Credential-free API base route, normalized during preparation.
        pub endpoint: String,
        /// Explicit authentication configuration; stores references only.
        pub authentication: ProfileAuthentication,
        /// Route-independent wire behavior and supplied capability claims.
        pub behavior: ProfileBehavior,
        /// Strict routing is mandatory for OpenRouter, absent for direct vLLM serving.
        pub routing: Option<StrictProviderRouting>,
        /// Operational policy, excluded from behavior identity.
        pub operations: ProfileOperations,
    }
}
impl ServingProfile {
    /// Validate all declarations with the existing credential-free endpoint normalizer.
    pub fn validate(&self) -> Result<()> {
        if self.version != 1 {
            return Err(ProfileError::InvalidProfile);
        }
        crate::normalize_endpoint(&self.endpoint).map_err(|_| ProfileError::InvalidProfile)?;
        self.authentication.validate()?;
        self.behavior.validate()?;
        self.operations.validate()?;
        match (&self.routing, self.behavior.dialect) {
            (Some(routing), ServingDialect::OpenRouterV1) => routing.validate(),
            (None, ServingDialect::VllmV1) => Ok(()),
            _ => Err(ProfileError::InvalidProfile),
        }
    }
    /// Build an offline owned-Modal configuration using two separate proxy secret references
    /// and ObservationOnly defaults. No credentials, deployment, or inference is acquired.
    pub fn modal(
        endpoint: impl Into<String>,
        token_id: SecretReference,
        token_secret: SecretReference,
        behavior: ProfileBehavior,
    ) -> Result<Self> {
        let profile = Self {
            version: 1,
            endpoint: endpoint.into(),
            authentication: ProfileAuthentication::ModalProxy {
                token_id,
                token_secret,
            },
            behavior,
            routing: None,
            operations: ProfileOperations::default(),
        };
        profile.validate()?;
        Ok(profile)
    }
}
