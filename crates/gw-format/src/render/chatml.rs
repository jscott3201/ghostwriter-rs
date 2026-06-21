//! ChatML renderer: `<|im_start|>{role}\n{content}<|im_end|>` per turn.
//!
//! Supervised reasoning is rendered as a `<think>{reasoning}</think>` block at the START of the
//! assistant content region (the Qwen / DeepSeek-R1 convention), generated from the separate
//! `reasoning` field — NEVER read out of `content`. [`CotPolicy::Stripped`] drops it.
//!
//! v1 omissions: assistant `tool_calls` are **dropped** (only
//! [`OpenAiMessages`](gw_schema::TrlFormat::OpenAiMessages) preserves them), and
//! [`Content::Parts`](gw_schema::Content::Parts) is flattened to text (an image/audio-only turn
//! renders empty content). Noted for a future tool / multimodal corpus.

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
