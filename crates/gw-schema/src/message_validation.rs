//! Pure source prerequisites shared by renderers and screened artifact verification. No rendering.
use crate::{Content, ContentPart, Message, MultiTurnLoss, Role, TrlFormat};

/// Chat-control tokens forbidden in clean content and reasoning, in diagnostic precedence order.
pub const CLEAN_FIELD_CONTROL_TOKENS: &[&str] = &[
    "<think>",
    "</think>",
    "<|im_start|>",
    "<|im_end|>",
    "<|channel>",
    "<channel|>",
    "<|channel|>",
    "<|turn>",
    "<turn|>",
    "<|think|>",
    "<bos>",
    "<|start|>",
    "<|end|>",
    "<|message|>",
    "<|return|>",
];

/// First forbidden chat-control token in one clean field, using diagnostic precedence order.
#[must_use]
pub fn first_control_token(text: &str) -> Option<&'static str> {
    CLEAN_FIELD_CONTROL_TOKENS
        .iter()
        .copied()
        .find(|token| text.contains(token))
}

/// Concatenate text parts for clean-field validation; null has no text and stays absent.
#[must_use]
pub fn content_text_parts(content: &Content) -> Option<String> {
    match content {
        Content::Text(text) => Some(text.clone()),
        Content::Parts(parts) => Some(
            parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        ),
        Content::Null => None,
    }
}

/// First control-token leak in ordered message content or flat reasoning. Detail metadata and
/// tool arguments retain their existing semantics and are not treated as clean renderer fields.
#[must_use]
pub fn first_clean_field_violation(messages: &[Message]) -> Option<(Role, &'static str)> {
    for message in messages {
        if let Some(token) = content_text_parts(&message.content)
            .as_deref()
            .and_then(first_control_token)
        {
            return Some((message.role, token));
        }
        if let Some(token) = message.reasoning.as_deref().and_then(first_control_token) {
            return Some((message.role, token));
        }
    }
    None
}

/// A conversation signal that a target without tool support would discard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolSignal {
    /// Present tool calls, including an explicitly present empty list.
    ToolCalls,
    /// Present explicit result link.
    ToolCallId,
    /// A tool-role result message.
    ToolRole,
}

/// First tool-bearing message and every observed tool signal, in stable encounter order.
#[must_use]
pub fn first_tool_signal(messages: &[Message]) -> Option<(usize, Vec<ToolSignal>)> {
    let mut first = None;
    let mut signals = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        let present = [
            (message.tool_calls.is_some(), ToolSignal::ToolCalls),
            (message.tool_call_id.is_some(), ToolSignal::ToolCallId),
            (message.role == Role::Tool, ToolSignal::ToolRole),
        ];
        for (present, signal) in present {
            if present {
                first.get_or_insert(index);
                if !signals.contains(&signal) {
                    signals.push(signal);
                }
            }
        }
    }
    first.map(|index| (index, signals))
}

/// Whether this target preserves the canonical conversation's tool signals.
#[must_use]
pub const fn preserves_tool_signals(target: TrlFormat) -> bool {
    matches!(
        target,
        TrlFormat::OpenAiMessages | TrlFormat::TrlPromptCompletion
    )
}

/// Source-level failure that the canonical renderer can diagnose before template dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderSourceIssue {
    /// A clean field contains framing reserved for the renderer.
    ControlToken {
        /// Offending message role.
        role: Role,
        /// The reserved token.
        token: &'static str,
    },
    /// The selected target cannot preserve these tool signals.
    UnsupportedTools {
        /// Requested target.
        target: TrlFormat,
        /// Every observed tool signal, in encounter order.
        signals: Vec<ToolSignal>,
        /// Index of the first message carrying a signal.
        index: usize,
    },
}

/// Check the canonical renderer's clean-field and target/tool prerequisites in their original
/// order. This performs no rendering and does not validate tool-result links.
#[must_use]
pub fn render_source_issue(messages: &[Message], target: TrlFormat) -> Option<RenderSourceIssue> {
    if let Some((role, token)) = first_clean_field_violation(messages) {
        return Some(RenderSourceIssue::ControlToken { role, token });
    }
    if !preserves_tool_signals(target)
        && let Some((index, signals)) = first_tool_signal(messages)
    {
        return Some(RenderSourceIssue::UnsupportedTools {
            target,
            signals,
            index,
        });
    }
    None
}

/// Source-level failure while selecting complete assistant prefixes for SFT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SftSourceIssue {
    /// An empty conversation or non-assistant final turn cannot form these training units.
    MissingTerminalAssistant,
    /// A selected prefix fails the canonical renderer's source prerequisites.
    Render(RenderSourceIssue),
}

/// Select supported SFT prefix targets using the same terminal and renderer prerequisites as
/// formatting. Final-turn mode checks the full conversation; all-assistant mode checks each prefix.
///
/// # Errors
/// Rejects a missing terminal assistant or a selected prefix with unsupported renderer inputs.
pub fn sft_source_targets(
    messages: &[Message],
    target: TrlFormat,
    turns: MultiTurnLoss,
) -> Result<Vec<usize>, SftSourceIssue> {
    if messages.last().map(|message| message.role) != Some(Role::Assistant) {
        return Err(SftSourceIssue::MissingTerminalAssistant);
    }
    let mut indices: Vec<_> = messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| (message.role == Role::Assistant).then_some(index))
        .collect();
    if turns == MultiTurnLoss::FinalTurnOnly {
        indices = indices.into_iter().rev().take(1).collect();
    }
    for &index in &indices {
        if let Some(issue) = render_source_issue(&messages[..=index], target) {
            return Err(SftSourceIssue::Render(issue));
        }
    }
    Ok(indices)
}
