//! `ingest` — parse provider responses into the clean canonical [`Message`], and the
//! channel-token stripper that enforces the round-trip rule.
//!
//! ## The round-trip rule (INVARIANT-a)
//!
//! The stored `content` is ALWAYS clean final-answer text. On ingest, any channel tokens that a
//! provider inlined into content (`<think>…</think>`, Gemma-4 `<|channel>thought…<channel|>`,
//! Harmony `analysis` channel) are STRIPPED out of `content` into `reasoning`. Combined with
//! [`render`](crate::render()) (its inverse for the TOKEN-STREAM targets), this makes
//! `render → ingest` recover the original clean messages + reasoning.
//!
//! ## Fail loud, never store a leak
//!
//! [`strip_channel_tokens`] removes the LEADING channel block. If the resulting clean content
//! STILL contains a control token (an interleaved / second channel the stripper does not model),
//! [`ingest_openrouter`] errors rather than store a leaking `content` — the same fail-loud stance
//! as the render-time guard.

use serde_json::Value;

use gw_schema::{Content, Message, ReasoningDetail, Role};

use crate::error::{FormatError, Result};
use crate::validate::first_control_token;

/// Parse ONE non-streaming OpenAI / OpenRouter assistant message into a clean [`Message`].
///
/// Accepts either a full response object (`{"choices":[{"message":{…}}]}`, first choice taken) or
/// a bare message object (`{"role":"assistant","content":…,"reasoning":…}`). The mapping mirrors
/// `gw-providers`:
///
/// - `message.content` → [`Message::content`] (after [`strip_channel_tokens`], so any inlined
///   channel tokens move to `reasoning`).
/// - `message.reasoning` (or its `reasoning_content` alias) → [`Message::reasoning`]; if absent,
///   any reasoning extracted from `content` is used instead.
/// - `message.reasoning_details` → [`Message::reasoning_details`] (parsed leniently; unparseable
///   or unknown-type fragments are dropped, never fatal).
///
/// # Errors
///
/// Returns [`FormatError::Ingest`] if no assistant message can be located (missing `choices` /
/// `message`, or a non-object payload), or if the clean `content` STILL contains a control token
/// after stripping the leading channel block (a leak that must not be stored).
pub fn ingest_openrouter(value: &Value) -> Result<Message> {
    let message = locate_message(value)?;

    let raw_content = message
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let (clean_content, extracted) = strip_channel_tokens(raw_content);

    // Fail loud: a control token surviving in the clean content means an interleaved / second
    // channel the stripper does not model — storing it would leak channel markup into `content`.
    if let Some(token) = first_control_token(&clean_content) {
        return Err(FormatError::Ingest(format!(
            "clean content still contains control token `{token}` after stripping"
        )));
    }

    // Provider `reasoning` (or `reasoning_content` alias) wins, with an empty provider string
    // collapsing to None. Otherwise use any reasoning peeled out of `content` — kept EVEN when
    // empty, so an emitted-empty channel (`Some("")`) round-trips and stays distinct from "no
    // channel at all" (`None`).
    let provider_reasoning = message
        .get("reasoning")
        .or_else(|| message.get("reasoning_content"))
        .and_then(Value::as_str)
        .map(str::to_owned)
        .filter(|s| !s.is_empty());
    let reasoning = provider_reasoning.or(extracted);

    let reasoning_details = parse_reasoning_details(message.get("reasoning_details"));

    let name = message
        .get("name")
        .and_then(Value::as_str)
        .map(str::to_owned);

    Ok(Message {
        role: Role::Assistant,
        content: Content::Text(clean_content),
        reasoning,
        reasoning_details,
        tool_calls: None,
        name,
    })
}

/// Locate the assistant message object inside `value`, accepting a full response or a bare
/// message.
fn locate_message(value: &Value) -> Result<&Value> {
    if value.get("content").is_some() || value.get("reasoning").is_some() {
        // Already a bare message object.
        return Ok(value);
    }
    value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
        .and_then(|c| c.get("message"))
        .ok_or_else(|| FormatError::Ingest("no choices[0].message in response".into()))
}

/// Parse `reasoning_details` leniently into strict [`ReasoningDetail`]s. A non-array shape yields
/// `None`; within an array, fragments that fail to parse or carry an unknown `type` are dropped.
fn parse_reasoning_details(raw: Option<&Value>) -> Option<Vec<ReasoningDetail>> {
    let array = raw?.as_array()?;
    let mut out = Vec::with_capacity(array.len());
    for frag in array {
        if let Ok(detail) = serde_json::from_value::<ReasoningDetail>(frag.clone()) {
            out.push(detail);
        }
    }
    if out.is_empty() { None } else { Some(out) }
}

/// Strip channel tokens out of `content` into extracted reasoning, enforcing the round-trip rule.
///
/// Recognizes the channel framings the renderers produce / providers inline:
///
/// - ChatML / Qwen / DeepSeek: `<think>…</think>` at the start of content.
/// - Gemma-4: `<|channel>thought\n…<channel|>` (asymmetric).
/// - Harmony: a leading `analysis` channel before the `final` channel.
///
/// Returns `(clean_content, extracted_reasoning)`. `clean_content` is the final-answer text with
/// the channel removed; `extracted_reasoning` is `Some` only when a channel was found. The trailing
/// `\n` trim is framing-specific (it matches each renderer EXACTLY): Gemma-4 frames
/// `{reasoning}\n<channel|>`, so its trailing `\n` is trimmed; ChatML (`<think>{reasoning}</think>`)
/// and Harmony (`<|message|>{reasoning}<|end|>`) add NO trailing newline, so a reasoning ending in
/// `\n` is preserved. Content with no channel tokens is returned unchanged with `None`.
#[must_use]
pub fn strip_channel_tokens(content: &str) -> (String, Option<String>) {
    // (open, close, trim_one_trailing_newline) — the bool mirrors the matching renderer's framing.
    if let Some(r) = strip_pair(content, "<think>", "</think>", false) {
        return r;
    }
    if let Some(r) = strip_pair(content, "<|channel>thought\n", "<channel|>", true) {
        return r;
    }
    if let Some(r) = strip_harmony(content) {
        return r;
    }
    (content.to_owned(), None)
}

/// Strip a single `open … close` channel anchored at the START of `content`. The reasoning is the
/// text between the markers; one trailing `\n` is trimmed ONLY when `trim_trailing_nl` (the
/// matching renderer added it). The clean content is everything after `close`.
fn strip_pair(
    content: &str,
    open: &str,
    close: &str,
    trim_trailing_nl: bool,
) -> Option<(String, Option<String>)> {
    let rest = content.strip_prefix(open)?;
    let end = rest.find(close)?;
    let raw = &rest[..end];
    let reasoning = if trim_trailing_nl {
        raw.strip_suffix('\n').unwrap_or(raw)
    } else {
        raw
    };
    let clean = &rest[end + close.len()..];
    Some((clean.to_owned(), Some(reasoning.to_owned())))
}

/// Strip a Harmony `analysis` channel that precedes the `final` channel.
fn strip_harmony(content: &str) -> Option<(String, Option<String>)> {
    const ANALYSIS: &str = "<|start|>assistant<|channel|>analysis<|message|>";
    const FINAL: &str = "<|start|>assistant<|channel|>final<|message|>";
    const ANALYSIS_END: &str = "<|end|>";
    const FINAL_END: &str = "<|return|>";

    let rest = content.strip_prefix(ANALYSIS)?;
    let a_end = rest.find(ANALYSIS_END)?;
    let reasoning = &rest[..a_end];
    let after = &rest[a_end + ANALYSIS_END.len()..];
    let after = after.strip_prefix(FINAL)?;
    let f_end = after.find(FINAL_END).unwrap_or(after.len());
    let clean = &after[..f_end];
    Some((clean.to_owned(), Some(reasoning.to_owned())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn strips_think_block() {
        let (clean, r) = strip_channel_tokens("<think>step</think>42");
        assert_eq!(clean, "42");
        assert_eq!(r.as_deref(), Some("step"));
    }

    #[test]
    fn strips_gemma_thought_channel() {
        let (clean, r) = strip_channel_tokens("<|channel>thought\nstep\n<channel|>42");
        assert_eq!(clean, "42");
        assert_eq!(r.as_deref(), Some("step"));
    }

    #[test]
    fn strips_harmony_analysis() {
        let s = "<|start|>assistant<|channel|>analysis<|message|>step<|end|>\
                 <|start|>assistant<|channel|>final<|message|>42<|return|>";
        let (clean, r) = strip_channel_tokens(s);
        assert_eq!(clean, "42");
        assert_eq!(r.as_deref(), Some("step"));
    }

    #[test]
    fn clean_content_unchanged() {
        let (clean, r) = strip_channel_tokens("42");
        assert_eq!(clean, "42");
        assert!(r.is_none());
    }

    #[test]
    fn ingest_full_response() {
        let v = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "The answer is 42.",
                    "reasoning": "Let me work through this.",
                    "reasoning_details": [
                        {"type": "reasoning.text", "text": "Let me work through this.", "index": 0}
                    ]
                }
            }]
        });
        let m = ingest_openrouter(&v).unwrap();
        assert_eq!(m.role, Role::Assistant);
        assert_eq!(m.content, Content::Text("The answer is 42.".into()));
        assert_eq!(m.reasoning.as_deref(), Some("Let me work through this."));
        assert!(m.reasoning_details.is_some());
    }

    #[test]
    fn ingest_bare_message_and_strips_inline_cot() {
        let v = json!({
            "role": "assistant",
            "content": "<think>3*4</think>12"
        });
        let m = ingest_openrouter(&v).unwrap();
        assert_eq!(m.content, Content::Text("12".into()));
        assert_eq!(m.reasoning.as_deref(), Some("3*4"));
    }

    #[test]
    fn ingest_reasoning_content_alias() {
        let v = json!({"role": "assistant", "content": "ok", "reasoning_content": "think"});
        let m = ingest_openrouter(&v).unwrap();
        assert_eq!(m.reasoning.as_deref(), Some("think"));
    }

    #[test]
    fn ingest_missing_message_errors() {
        let v = json!({"choices": []});
        assert!(matches!(ingest_openrouter(&v), Err(FormatError::Ingest(_))));
    }

    #[test]
    fn ingest_errors_when_clean_content_retains_control_token() {
        // A second interleaved channel the leading strip does not consume must NOT be stored.
        let v = json!({
            "role": "assistant",
            "content": "<think>a</think>answer <|im_end|> trailing"
        });
        let err = ingest_openrouter(&v).unwrap_err();
        assert!(matches!(err, FormatError::Ingest(_)));
        assert!(err.to_string().contains("<|im_end|>"));
    }

    #[test]
    fn chatml_reasoning_trailing_newline_is_preserved() {
        // ChatML frames `<think>{r}</think>` with no added newline, so a reasoning ending in
        // `\n` must survive ingest (framing-specific trim).
        let (clean, r) = strip_channel_tokens("<think>step\n</think>42");
        assert_eq!(clean, "42");
        assert_eq!(r.as_deref(), Some("step\n"));
    }

    #[test]
    fn gemma_reasoning_trailing_newline_is_trimmed_once() {
        // Gemma frames `{r}\n<channel|>`, so exactly one trailing newline is the framing.
        let (clean, r) = strip_channel_tokens("<|channel>thought\nstep\n<channel|>42");
        assert_eq!(clean, "42");
        assert_eq!(r.as_deref(), Some("step"));
    }

    #[test]
    fn emitted_empty_channel_round_trips_to_some_empty() {
        // An emitted-empty Gemma thought channel (`<|channel>thought\n<channel|>`) is a real,
        // distinct signal — it must ingest to Some("") (a present-but-empty channel), NOT None.
        let (clean, r) = strip_channel_tokens("<|channel>thought\n<channel|>96");
        assert_eq!(clean, "96");
        assert_eq!(r.as_deref(), Some(""));

        let v = json!({"role": "assistant", "content": "<|channel>thought\n<channel|>96"});
        let m = ingest_openrouter(&v).unwrap();
        assert_eq!(m.reasoning.as_deref(), Some(""));
    }

    #[test]
    fn empty_provider_reasoning_collapses_to_none() {
        // A provider-supplied empty `reasoning` string (no channel) is noise → None.
        let v = json!({"role": "assistant", "content": "ok", "reasoning": ""});
        let m = ingest_openrouter(&v).unwrap();
        assert!(m.reasoning.is_none());
    }
}
