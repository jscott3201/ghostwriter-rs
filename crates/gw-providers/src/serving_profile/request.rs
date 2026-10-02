use super::*;
use gw_schema::{
    Content, ContentDigest, Declaration, JudgeSampling, Message, ModelExecutionSemantics,
    ModelOperation, ModelReference, Role, SemanticExecutionIdentity,
};
use serde_json::json;

/// Canonical inputs supported by the offline text-serving mappings.
#[derive(Debug, Clone, PartialEq)]
pub enum ProfileInput {
    /// Canonical chat messages; reasoning remains a sibling of content.
    Chat(Vec<Message>),
    /// An already rendered text prompt. Preparation does not guess or run a template.
    Completion(String),
    /// Ordered input texts. Order is behaviorally meaningful and preserved.
    Embedding(Vec<String>),
}
impl ProfileInput {
    pub(super) fn operation(&self) -> ModelOperation {
        match self {
            Self::Chat(_) => ModelOperation::ChatCompletion,
            Self::Completion(_) => ModelOperation::Completion,
            Self::Embedding(_) => ModelOperation::Embedding,
        }
    }
}

/// A requested control value, separate from configured capability claims.
#[derive(Debug, Clone, PartialEq)]
pub enum ControlValue {
    /// Native function definitions and selection controls, applied as one capability.
    Tools(crate::ToolConfig),
    /// Shared canonical sampling vocabulary, without provider-extension serialization.
    Sampling(JudgeSampling),
    /// Positive combined output-token cap.
    MaxOutputTokens(u32),
    /// Exact literal effort name; no cross-model level translation.
    ReasoningEffort(String),
    /// Positive reasoning-token budget, distinct from the output cap.
    ReasoningBudget(u32),
    /// Explicit template kwarg value.
    EnableThinking(bool),
    /// Explicit template kwarg value.
    PreserveThinking(bool),
    /// Explicit template kwarg value.
    ClearThinking(bool),
    /// Strict JSON-schema response format. The supplied schema is sent unchanged.
    JsonSchema {
        /// Nonempty response-schema name.
        name: String,
        /// JSON object representing the requested schema.
        schema: Value,
    },
    /// Chat token logprobs with zero through twenty alternatives.
    Logprobs(u32),
    /// Request token/cost metadata without assuming the provider supplies it.
    Usage,
}

/// Whether an unsupported control must stop preparation or may be explicitly omitted.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestedControl {
    /// Requested control value.
    pub value: ControlValue,
    /// True denies unsupported preparation; false records each omitted control.
    pub required: bool,
}

/// Input and request-specific controls. No route, credential, or approval fields are accepted.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileRequest {
    /// Canonical ordered request data.
    pub input: ProfileInput,
    /// Controls must be unique; incompatible reasoning effort/budget requests are rejected.
    pub controls: Vec<RequestedControl>,
}

/// Why an explicitly optional control was omitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DegradationReason {
    /// The declared mapping, operation, or capability set does not support this control.
    Unsupported,
    /// This exact effort value is not included in the declared allowed set.
    UnsupportedValue,
}

/// An omission that the caller can display or persist; never a silent fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ControlDegradation {
    /// Omitted optional control.
    pub control: ProfileControl,
    /// Explicit reason for omitting it.
    pub reason: DegradationReason,
}

/// Immutable offline request bytes and identity, without authentication or execution authority.
/// No client, environment source, or deployment verifier is reachable from this type.
#[derive(Debug, Clone)]
pub struct PreparedProfileRequest {
    endpoint: ModelReference,
    semantics: ModelExecutionSemantics,
    target: SemanticExecutionIdentity,
    body: Vec<u8>,
    digest: ContentDigest,
    degradations: Vec<ControlDegradation>,
}
impl PreparedProfileRequest {
    /// Normalized endpoint including the operation's API path.
    #[must_use]
    pub fn endpoint(&self) -> &ModelReference {
        &self.endpoint
    }
    /// The supplied immutable semantic target; still not artifact or deployment approval.
    #[must_use]
    pub fn target(&self) -> &SemanticExecutionIdentity {
        &self.target
    }
    /// Full declaration resolving the target identity.
    #[must_use]
    pub fn semantics(&self) -> &ModelExecutionSemantics {
        &self.semantics
    }
    /// Exact canonical JSON request-body bytes, excluding credentials.
    #[must_use]
    pub fn body(&self) -> &[u8] {
        &self.body
    }
    /// Digest of the exact body bytes to bind each physical attempt.
    #[must_use]
    pub fn body_digest(&self) -> &ContentDigest {
        &self.digest
    }
    /// Every optional control omitted by the explicit mapping.
    #[must_use]
    pub fn degradations(&self) -> &[ControlDegradation] {
        &self.degradations
    }
}

/// Prepare exact bytes using only supplied data and explicit capability declarations.
///
/// The full OpenRouter ChatRequest is deliberately not the generic wire input. Credentials are
/// resolved separately with resolve_authentication. This function does not authenticate supplied
/// model declarations, qualify a deployment, perform I/O, or authorize any dispatch.
///
/// # Errors
/// Rejects incomplete/mismatched semantics, malformed inputs and unsupported required controls.
pub fn prepare_request(
    profile: &ServingProfile,
    semantics: &ModelExecutionSemantics,
    request: &ProfileRequest,
) -> Result<PreparedProfileRequest> {
    profile.validate()?;
    semantics
        .validate()
        .map_err(|_| ProfileError::InvalidSemantics)?;
    if !semantics_complete(semantics)
        || request.input.operation() != semantics.operation
        || semantics.adapter_behavior != profile.behavior.adapter_behavior(semantics.operation)?
        || semantics.serving_profile != Declaration::Declared(profile.behavior.declaration()?)
    {
        return Err(ProfileError::InvalidSemantics);
    }
    let (mut body, route) = match &request.input {
        ProfileInput::Chat(messages) => {
            if messages.is_empty() {
                return Err(ProfileError::InvalidRequest);
            }
            for message in messages {
                validate_message(message, &profile.behavior)?;
            }
            (
                json!({"model":semantics.alias,"messages":crate::wire_messages(messages).map_err(|_| ProfileError::InvalidRequest)?,"stream":true}),
                "chat/completions",
            )
        }
        ProfileInput::Completion(prompt) => {
            if prompt.is_empty() {
                return Err(ProfileError::InvalidRequest);
            }
            (
                json!({"model":semantics.alias,"prompt":prompt,"stream":true}),
                "completions",
            )
        }
        ProfileInput::Embedding(input) => {
            if input.is_empty() {
                return Err(ProfileError::InvalidRequest);
            }
            (
                json!({"model":semantics.alias,"input":input,"encoding_format":"float"}),
                "embeddings",
            )
        }
    };
    if let Some(routing) = &profile.routing {
        let mut only = routing.only.clone();
        only.sort();
        body["provider"] = json!({"only":only,"allow_fallbacks":false,"require_parameters":true,"data_collection":routing.data_collection});
    }
    let degradations = super::controls::apply(
        &profile.behavior,
        semantics.operation,
        &request.controls,
        &mut body,
    )?;
    let body = canonical_bytes(&body)?;
    let endpoint = format!(
        "{}/{route}",
        crate::normalize_endpoint(&profile.endpoint).map_err(|_| ProfileError::InvalidProfile)?
    );
    Ok(PreparedProfileRequest {
        endpoint: ModelReference::new(endpoint).map_err(|_| ProfileError::InvalidProfile)?,
        target: semantics
            .identity()
            .map_err(|_| ProfileError::InvalidSemantics)?,
        semantics: semantics.clone(),
        digest: body_digest(&body),
        body,
        degradations,
    })
}

pub(super) fn semantics_complete(value: &ModelExecutionSemantics) -> bool {
    matches!(value.serving_profile, Declaration::Declared(_))
        && matches!(value.artifact, Declaration::Declared(_))
        && matches!(value.additional_artifacts, Declaration::Declared(_))
        && matches!(value.tokenizer, Declaration::Declared(_))
        && match &value.chat_template {
            Declaration::Declared(Some(_)) => true,
            Declaration::Declared(None) => value.operation != ModelOperation::ChatCompletion,
            Declaration::Unknown => false,
        }
        && matches!(value.runtime, Declaration::Declared(_))
        && matches!(value.parser, Declaration::Declared(_))
        && matches!(value.configuration, Declaration::Declared(_))
}

fn validate_message(message: &Message, behavior: &ProfileBehavior) -> Result<()> {
    if (message.role == Role::Tool
        || message.tool_calls.is_some()
        || message.tool_call_id.is_some()
        || message.name.is_some())
        && !behavior.supports(ProfileControl::Tools)
        || matches!(message.content, Content::Parts(_))
        || (message.reasoning.is_some() || message.reasoning_details.is_some())
            && message.role != Role::Assistant
        || message.reasoning.is_some() && !behavior.supports(ProfileControl::ReasoningHistory)
        || message.reasoning_details.is_some()
            && !behavior.supports(ProfileControl::StructuredReasoningHistory)
    {
        return Err(ProfileError::InvalidRequest);
    }
    Ok(())
}
