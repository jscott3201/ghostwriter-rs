//! `render` — project clean `{messages, reasoning}` turns into model-specific chat templates.
//!
//! [`render`] is the single entry point: it dispatches on [`TrlFormat`] and applies the
//! [`CotPolicy`] (whether `reasoning` enters the loss/thought region) uniformly across targets.
//!
//! ## The round-trip contract (INVARIANT-a)
//!
//! The stored `content` is ALWAYS clean final-answer text. Channel tokens (`<think>`,
//! `<|channel>`, `<turn|>`, …) are PRODUCED here by the renderer from the separate `reasoning`
//! field — never read out of `content`. So format A↔B conversion is a pure function of
//! `(clean messages + reasoning + target template)`; [`crate::ingest`] is its inverse.
//!
//! ## CotPolicy semantics (uniform across targets)
//!
//! - [`Supervised`](CotPolicy::Supervised): render `reasoning` into the loss/thought region.
//! - [`Masked`](CotPolicy::Masked): render `reasoning` IDENTICALLY to `Supervised`. Label
//!   masking is a trainer-side concern; the [`crate::projection`] manifest records that the
//!   region is masked. v1 does not change the rendered bytes for `Masked`.
//! - [`Stripped`](CotPolicy::Stripped): drop `reasoning`; render the answer only (an empty
//!   thought wrapper for Gemma-4).

mod chatml;
mod gemma4;
mod harmony;
mod openai;
mod prompt_completion;
mod sharegpt;

use gw_schema::{Content, CotPolicy, Message, Role, TrlFormat};

use crate::error::Result;

/// Render `messages` into `target`, applying `cot` to the reasoning region.
///
/// The output is a single `String`: a raw template-rendered prompt for the token-stream targets
/// ([`Gemma4`](TrlFormat::Gemma4), [`ChatML`](TrlFormat::ChatML), [`Harmony`](TrlFormat::Harmony))
/// and a pretty-printed JSON document for the structured targets
/// ([`ShareGpt`](TrlFormat::ShareGpt), [`OpenAiMessages`](TrlFormat::OpenAiMessages),
/// [`TrlPromptCompletion`](TrlFormat::TrlPromptCompletion)).
///
/// # Errors
///
/// Returns [`crate::FormatError::Template`] if the Gemma-4 template fails to render,
/// [`crate::FormatError::Serde`] if a structured target fails to serialize, and
/// [`crate::FormatError::UnsupportedRole`] if a message carries a role the target cannot place.
pub fn render(messages: &[Message], target: TrlFormat, cot: CotPolicy) -> Result<String> {
    match target {
        TrlFormat::Gemma4 => gemma4::render(messages, cot),
        TrlFormat::ChatML => chatml::render(messages, cot),
        TrlFormat::ShareGpt => sharegpt::render(messages, cot),
        TrlFormat::OpenAiMessages => openai::render(messages, cot),
        TrlFormat::Harmony => harmony::render(messages, cot),
        TrlFormat::TrlPromptCompletion => prompt_completion::render(messages, cot),
    }
}

/// The clean text of a [`Content`]. `Parts` are flattened to their concatenated text segments
/// (image/audio refs carry no training text), so callers always see leak-free final-answer text.
pub(crate) fn content_text(content: &Content) -> String {
    match content {
        Content::Text(s) => s.clone(),
        Content::Parts(parts) => parts
            .iter()
            .filter_map(|p| match p {
                gw_schema::ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(""),
    }
}

/// The reasoning that should be rendered for `msg` under `cot`.
///
/// - [`Stripped`](CotPolicy::Stripped) always yields `None` (answer-only).
/// - [`Supervised`](CotPolicy::Supervised) / [`Masked`](CotPolicy::Masked) yield the message's
///   `reasoning` (v1 renders `Masked` identically to `Supervised`; the masking is recorded in the
///   projection manifest, not the bytes).
///
/// An empty-string `reasoning` is treated as present (so a deliberately-empty thought channel
/// round-trips); only `None` and `Stripped` collapse to "no reasoning".
pub(crate) fn effective_reasoning(msg: &Message, cot: CotPolicy) -> Option<&str> {
    match cot {
        CotPolicy::Stripped => None,
        CotPolicy::Supervised | CotPolicy::Masked => msg.reasoning.as_deref(),
    }
}

/// `true` if `role` is an assistant turn (`assistant`). Tool/user/system are not.
pub(crate) fn is_assistant(role: Role) -> bool {
    role == Role::Assistant
}

#[cfg(test)]
mod tests {
    use super::*;
    use gw_schema::ContentPart;

    #[test]
    fn content_text_flattens_parts() {
        let c = Content::Parts(vec![
            ContentPart::Text { text: "a".into() },
            ContentPart::ImageUrl {
                image_url: "x".into(),
            },
            ContentPart::Text { text: "b".into() },
        ]);
        assert_eq!(content_text(&c), "ab");
    }

    #[test]
    fn effective_reasoning_respects_policy() {
        let msg = Message {
            role: Role::Assistant,
            content: Content::Text("ans".into()),
            reasoning: Some("cot".into()),
            reasoning_details: None,
            tool_calls: None,
            name: None,
        };
        assert_eq!(
            effective_reasoning(&msg, CotPolicy::Supervised),
            Some("cot")
        );
        assert_eq!(effective_reasoning(&msg, CotPolicy::Masked), Some("cot"));
        assert_eq!(effective_reasoning(&msg, CotPolicy::Stripped), None);
    }
}
