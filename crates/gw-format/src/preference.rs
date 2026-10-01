//! Structural projection only. This layer does not certify score evidence or chosen eligibility.
use gw_schema::{Content, CotPolicy, Message, ReasoningDetail, Role, TrainingRecord};

use crate::{FormatError, Result};

/// Text-only conversational arrays after structural checks and reasoning projection.
/// These arrays are not evidence of a valid preference; use the engine's preparation boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct PreferenceMessages {
    /// Complete equal prefix before reasoning projection.
    pub original_prefix: Vec<Message>,
    /// Shared prefix after reasoning policy.
    pub prompt: Vec<Message>,
    /// Terminal chosen assistant message.
    pub chosen: Vec<Message>,
    /// Terminal rejected assistant message.
    pub rejected: Vec<Message>,
}

fn invalid(reason: &str) -> FormatError {
    FormatError::Projection(reason.into())
}

fn supported(record: &TrainingRecord) -> Result<(&[Message], &Message)> {
    crate::validate::validate_clean(&record.messages)?;
    if record.tools.is_some() || crate::render::first_tool_violation(&record.messages).is_some() {
        return Err(invalid(
            "preference projection supports no tool declarations or signals",
        ));
    }
    for message in &record.messages {
        if !matches!(&message.content, Content::Text(_)) || message.name.is_some() {
            return Err(invalid(
                "preference projection requires unnamed text-only messages",
            ));
        }
        validate_reasoning(message)?;
        if message.role != Role::Assistant
            && (message.reasoning.is_some() || message.reasoning_details.is_some())
        {
            return Err(invalid(
                "preference reasoning belongs only to assistant messages",
            ));
        }
    }
    let (last, prefix) = record
        .messages
        .split_last()
        .ok_or_else(|| invalid("empty preference conversation"))?;
    if last.role != Role::Assistant || prefix.is_empty() {
        return Err(invalid(
            "preference completion must be the terminal assistant turn after a prompt",
        ));
    }
    if !matches!(&last.content, Content::Text(text) if !text.trim().is_empty()) {
        return Err(invalid("preference completion must contain nonempty text"));
    }
    Ok((prefix, last))
}

/// Only redundant ordered plaintext can be represented by the single flat reasoning field.
fn validate_reasoning(message: &Message) -> Result<()> {
    let Some(details) = message
        .reasoning_details
        .as_ref()
        .filter(|details| !details.is_empty())
    else {
        return Ok(());
    };
    let flat = message
        .reasoning
        .as_deref()
        .ok_or_else(|| invalid("reasoning details require matching flat reasoning"))?;
    let mut joined = String::new();
    let mut previous = None;
    for detail in details {
        let ReasoningDetail::Text { text, index, .. } = detail else {
            return Err(invalid(
                "preference reasoning details must all be plaintext",
            ));
        };
        if previous.is_some_and(|previous| *index <= previous) {
            return Err(invalid(
                "reasoning detail indices must be strictly increasing in stored order",
            ));
        }
        previous = Some(*index);
        joined.push_str(text);
    }
    if joined != flat {
        return Err(invalid(
            "reasoning detail text must exactly equal flat reasoning",
        ));
    }
    Ok(())
}

fn project(message: &Message, cot: CotPolicy) -> Message {
    let mut message = message.clone();
    message.reasoning_details = None;
    if cot == CotPolicy::Stripped || message.reasoning.as_ref().is_some_and(String::is_empty) {
        message.reasoning = None;
    }
    message
}

/// Validate supported terminal shape and full original prefix, then project all message arrays.
/// Does not inspect hashes, grades or eligibility; the result is structural, not a prepared pair.
///
/// # Errors
/// Rejects unequal original prefixes, tools, multimodal/null content, nonredundant reasoning,
/// nonterminal/empty completions, unqualified masking, control tokens, or equal projected answers.
/// Ordered plaintext details are supported only when they exactly duplicate flat reasoning; their
/// metadata stays in source evidence and is omitted from projected arrays.
pub fn project_preference_messages(
    chosen: &TrainingRecord,
    rejected: &TrainingRecord,
    cot: CotPolicy,
) -> Result<PreferenceMessages> {
    if cot == CotPolicy::Masked {
        return Err(invalid(
            "masked reasoning is not qualified for both DPO likelihood paths",
        ));
    }
    let (chosen_prefix, chosen_last) = supported(chosen)?;
    let (rejected_prefix, rejected_last) = supported(rejected)?;
    if chosen_prefix != rejected_prefix {
        return Err(invalid(
            "preference records must share the complete original message prefix",
        ));
    }
    let chosen_last = project(chosen_last, cot);
    let rejected_last = project(rejected_last, cot);
    if chosen_last == rejected_last {
        return Err(invalid(
            "preference completions are identical under the reasoning policy",
        ));
    }
    Ok(PreferenceMessages {
        original_prefix: chosen_prefix.to_vec(),
        prompt: chosen_prefix
            .iter()
            .map(|message| project(message, cot))
            .collect(),
        chosen: vec![chosen_last],
        rejected: vec![rejected_last],
    })
}
