use super::*;

/// Supplied usage observations. Missing is unknown, including when reasoning text exists.
/// Counts are never inferred from text, summed into other fields, or attributed to a tokenizer.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProfileUsage {
    /// Provider-reported input count, absent when unknown.
    pub prompt_tokens: Option<u64>,
    /// Provider-reported output count, absent when unknown.
    pub completion_tokens: Option<u64>,
    /// Provider-reported total; never calculated by this normalizer.
    pub total_tokens: Option<u64>,
    /// Provider-reported reasoning count, absent even if reasoning text exists.
    pub reasoning_tokens: Option<u64>,
    /// Provider-reported dollar cost; None and Some(0.0) remain distinct.
    pub cost_usd: Option<f64>,
}

/// Observed termination class; unspecified never implies successful completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileTerminationKind {
    /// No recognized terminal observation was supplied.
    Unspecified,
    /// The normalized wire reason was stop.
    Stop,
    /// The normalized wire reason was length.
    Length,
    /// The normalized wire reason was content_filter.
    ContentFilter,
    /// The normalized wire reason was tool_calls or function_call.
    ToolCalls,
    /// An error object or normalized error reason was supplied; execution/usage stay unknown.
    Error,
    /// An unrecognized normalized reason is retained verbatim.
    Other,
}

/// Both native and normalized termination observations, without collapsing one into the other.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProfileTermination {
    /// Classification derived only from explicit normalized/error observations.
    pub kind: ProfileTerminationKind,
    /// Original normalized finish reason, when supplied.
    pub finish_reason: Option<String>,
    /// Original backend/native reason, when supplied.
    pub native_finish_reason: Option<String>,
}

/// One offline chat response/chunk, retaining sibling reasoning and lossless structured details.
/// Model/provider strings are observations only and cannot establish deployment identity.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NormalizedProfileChunk {
    /// Final-answer text, distinct from reasoning. Missing/null differs from an empty string.
    pub content: Option<String>,
    /// Indexed native tool-call fragments, kept separate from content and reasoning.
    pub tool_calls: Option<Vec<crate::ToolCallDelta>>,
    /// One reasoning value after consistent aliases are reconciled.
    pub reasoning: Option<String>,
    /// Original structured objects, including provider extensions/unknown detail types.
    pub reasoning_details: Option<Vec<Value>>,
    /// Explicit refusal text, if present.
    pub refusal: Option<String>,
    /// Supplied usage, with each missing measurement kept unknown.
    pub usage: Option<ProfileUsage>,
    /// Explicit terminal facts; neither cost nor execution absence is inferred from errors.
    pub termination: ProfileTermination,
    /// Original provider error object/value for the caller's diagnostic boundary.
    pub error: Option<Value>,
    /// Response identifier, if supplied.
    pub response_id: Option<String>,
    /// Reported model alias, never a substitute for loaded-byte evidence.
    pub model: Option<String>,
    /// Reported provider route, never artifact eligibility or fleet verification.
    pub provider: Option<String>,
}

/// Normalize one supplied OpenAI-shaped chat JSON payload, independently of the live decoder.
/// Accepts a single message/delta choice, or an empty choices array for terminal usage. This
/// function performs no streaming, endpoint qualification, tokenizer measurement, or I/O.
///
/// # Errors
/// Rejects ambiguous choices, malformed known fields, non-object structured details, invalid
/// usage domains, and conflicting reasoning/reasoning_content aliases. Unknown wire extensions
/// are ignored except structured reasoning objects, which are retained losslessly.
pub fn normalize_chat_chunk(input: &str) -> Result<NormalizedProfileChunk> {
    let raw: RawChunk = serde_json::from_str(input).map_err(|_| ProfileError::InvalidResponse)?;
    if raw.choices.as_ref().is_none_or(|v| v.len() > 1) && raw.error.is_none() {
        return Err(ProfileError::InvalidResponse);
    }
    if raw.choices.as_ref().is_some_and(|v| v.len() > 1) {
        return Err(ProfileError::InvalidResponse);
    }
    let choice = raw
        .choices
        .and_then(|v| v.into_iter().next())
        .unwrap_or_default();
    if choice.index.is_some_and(|v| v != 0) || choice.delta.is_some() && choice.message.is_some() {
        return Err(ProfileError::InvalidResponse);
    }
    let delta = choice.delta.or(choice.message).unwrap_or_default();
    if delta
        .reasoning_details
        .as_ref()
        .is_some_and(|v| v.iter().any(|v| !v.is_object()))
    {
        return Err(ProfileError::InvalidResponse);
    }
    if delta.function_call.is_some() {
        return Err(ProfileError::InvalidResponse);
    }
    let reasoning = match (delta.reasoning, delta.reasoning_content) {
        (Some(a), Some(b)) if a != b => return Err(ProfileError::ConflictingReasoning),
        (Some(value), _) | (_, Some(value)) => Some(value),
        (None, None) => None,
    };
    let usage = raw
        .usage
        .map(|v| {
            if v.cost.is_some_and(|v| !v.is_finite() || v < 0.0) {
                return Err(ProfileError::InvalidResponse);
            }
            Ok(ProfileUsage {
                prompt_tokens: v.prompt_tokens,
                completion_tokens: v.completion_tokens,
                total_tokens: v.total_tokens,
                reasoning_tokens: v.completion_tokens_details.and_then(|v| v.reasoning_tokens),
                cost_usd: v.cost,
            })
        })
        .transpose()?;
    let kind = if raw.error.is_some() {
        ProfileTerminationKind::Error
    } else {
        match choice.finish_reason.as_deref() {
            None => ProfileTerminationKind::Unspecified,
            Some("stop") => ProfileTerminationKind::Stop,
            Some("length") => ProfileTerminationKind::Length,
            Some("content_filter") => ProfileTerminationKind::ContentFilter,
            Some("tool_calls" | "function_call") => ProfileTerminationKind::ToolCalls,
            Some("error") => ProfileTerminationKind::Error,
            Some(_) => ProfileTerminationKind::Other,
        }
    };
    Ok(NormalizedProfileChunk {
        content: delta.content,
        tool_calls: delta.tool_calls,
        reasoning,
        reasoning_details: delta.reasoning_details,
        refusal: delta.refusal,
        usage,
        termination: ProfileTermination {
            kind,
            finish_reason: choice.finish_reason,
            native_finish_reason: choice.native_finish_reason,
        },
        error: raw.error,
        response_id: raw.id,
        model: raw.model,
        provider: raw.provider,
    })
}

#[derive(Deserialize)]
struct RawChunk {
    choices: Option<Vec<RawChoice>>,
    usage: Option<RawUsage>,
    error: Option<Value>,
    id: Option<String>,
    model: Option<String>,
    provider: Option<String>,
}
#[derive(Default, Deserialize)]
struct RawChoice {
    index: Option<u64>,
    delta: Option<RawDelta>,
    message: Option<RawDelta>,
    finish_reason: Option<String>,
    native_finish_reason: Option<String>,
}
#[derive(Default, Deserialize)]
struct RawDelta {
    tool_calls: Option<Vec<crate::ToolCallDelta>>,
    function_call: Option<Value>,
    content: Option<String>,
    reasoning: Option<String>,
    reasoning_content: Option<String>,
    reasoning_details: Option<Vec<Value>>,
    refusal: Option<String>,
}
#[derive(Deserialize)]
struct RawUsage {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
    completion_tokens_details: Option<RawCompletionTokens>,
    cost: Option<f64>,
}
#[derive(Deserialize)]
struct RawCompletionTokens {
    reasoning_tokens: Option<u64>,
}
