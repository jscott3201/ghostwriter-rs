//! ChatML renderer: `<|im_start|>{role}\n{content}<|im_end|>` per turn.
//!
//! Supervised reasoning is rendered as a `<think>{reasoning}</think>` block at the START of the
//! assistant content region (the Qwen / DeepSeek-R1 convention), generated from the separate
//! `reasoning` field — NEVER read out of `content`. [`CotPolicy::Stripped`] drops it.
//!
//! v1 omissions: a tool trajectory never reaches this renderer — [`crate::render()`] fails closed
//! with [`FormatError::UnsupportedToolCalls`](crate::FormatError::UnsupportedToolCalls) because
//! ChatML has no slot for `tool_calls`, a `tool_call_id` link, or a tool-turn pairing
//! ([`OpenAiMessages`](gw_schema::TrlFormat::OpenAiMessages) and
//! [`TrlPromptCompletion`](gw_schema::TrlFormat::TrlPromptCompletion) preserve them). And
//! [`Content::Parts`](gw_schema::Content::Parts) is flattened to text (an image/audio-only turn
//! renders empty content). Noted for a future multimodal corpus.

use std::fmt::Write as _;

use gw_schema::{CotPolicy, Message, Role};

use crate::error::Result;
use crate::render::{content_text, effective_reasoning, is_assistant};

/// The wire role token for a ChatML turn. `developer` collapses to `system`; `tool` stays `tool`.
fn role_token(role: Role) -> &'static str {
    match role {
        Role::System | Role::Developer => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

/// Render `messages` into ChatML under `cot`.
pub(crate) fn render(messages: &[Message], cot: CotPolicy) -> Result<String> {
    let mut out = String::new();
    for msg in messages {
        let role = role_token(msg.role);
        let _ = writeln!(out, "<|im_start|>{role}");
        if is_assistant(msg.role)
            && let Some(reasoning) = effective_reasoning(msg, cot)
        {
            let _ = write!(out, "<think>{reasoning}</think>");
        }
        out.push_str(&content_text(&msg.content));
        out.push_str("<|im_end|>\n");
    }
    Ok(out)
}
