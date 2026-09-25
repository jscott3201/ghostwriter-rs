//! Harmony renderer (gpt-oss / OpenAI open models).
//!
//! Assistant reasoning maps to the `analysis` channel and content to the `final` channel:
//!
//! ```text
//! <|start|>assistant<|channel|>analysis<|message|>{reasoning}<|end|>
//! <|start|>assistant<|channel|>final<|message|>{answer}<|return|>
//! ```
//!
//! Non-assistant turns render as `<|start|>{role}<|message|>{content}<|end|>`. When the assistant
//! has no reasoning under [`CotPolicy`] (e.g. [`CotPolicy::Stripped`]), the `analysis` channel is
//! omitted and only the `final` channel is emitted. Harmony channel tokens are SYMMETRIC
//! (`<|channel|>`), unlike Gemma-4.
//!
//! v1 omissions: a tool trajectory never reaches this renderer — [`crate::render()`] fails closed
//! with [`FormatError::UnsupportedToolCalls`](crate::FormatError::UnsupportedToolCalls) because the
//! Harmony `commentary` tool channel is out of scope and there is nowhere to carry `tool_calls`,
//! the `tool_call_id` result link or a tool-turn pairing
//! ([`OpenAiMessages`](gw_schema::TrlFormat::OpenAiMessages) and
//! [`TrlPromptCompletion`](gw_schema::TrlFormat::TrlPromptCompletion) preserve them). And
//! [`Content::Parts`](gw_schema::Content::Parts) is flattened to text (an image/audio-only turn
//! renders empty content). Noted for a future multimodal corpus.

use std::fmt::Write as _;

use gw_schema::{CotPolicy, Message, Role};

use crate::error::Result;
use crate::render::{content_text, effective_reasoning};

/// The Harmony wire role for a non-assistant turn. `developer` is preserved.
fn role_token(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::Developer => "developer",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

/// Render `messages` into Harmony under `cot`.
pub(crate) fn render(messages: &[Message], cot: CotPolicy) -> Result<String> {
    let mut out = String::new();
    for msg in messages {
        let content = content_text(&msg.content);
        match msg.role {
            Role::Assistant => {
                if let Some(reasoning) = effective_reasoning(msg, cot) {
                    let _ = write!(
                        out,
                        "<|start|>assistant<|channel|>analysis<|message|>{reasoning}<|end|>"
                    );
                }
                let _ = write!(
                    out,
                    "<|start|>assistant<|channel|>final<|message|>{content}<|return|>"
                );
            }
            role => {
                let _ = write!(
                    out,
                    "<|start|>{}<|message|>{content}<|end|>",
                    role_token(role)
                );
            }
        }
    }
    Ok(out)
}
