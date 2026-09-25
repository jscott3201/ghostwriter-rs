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
//!
//! ## The tool-trajectory mapping (INVARIANT g / i)
//!
//! Ingest is where a provider's wire turn becomes the canonical [`Message`], so it is where the
//! tool fields must actually be POPULATED (a field merely added to the struct fixes nothing):
//!
//! - `message.tool_calls[]` → [`Message::tool_calls`], preserving each call's `id`, function name
//!   and arguments.
//! - `message.tool_call_id` → [`Message::tool_call_id`]: the explicit result link a `tool` turn
//!   points back at. A function *name* is never used to invent one.
//! - `message.role` → [`Message::role`], defaulting to [`Role::Assistant`] when absent or
//!   unrecognized.
//!
//! Two normalizations happen HERE, at the provider boundary, and nowhere else:
//!
//! 1. **Nullable content.** A wire `null` (or absent) `content` becomes [`Content::Null`], NOT
//!    `""`. Null and empty are different observations and both survive; only an explicit string
//!    goes through [`strip_channel_tokens`].
//! 2. **String-encoded arguments.** OpenAI-family providers send `function.arguments` as a JSON
//!    *string*, while the contract requires an object (INVARIANT g). A string is decoded exactly
//!    once into an object and the ORIGINAL wire text is retained in
//!    [`FunctionCall::raw_arguments`] so the byte-level edit is recoverable. An object passes
//!    through untouched (no raw duplicate).
//!
//! A malformed or non-object `arguments` — including a string that decodes to a non-object — is a
//! CONTRACT VIOLATION and is REJECTED, never coerced. Silently repairing it would fabricate an
//! argument list the provider never sent.

use serde_json::{Map, Value};

use gw_schema::{Content, ContentPart, FunctionCall, Message, ReasoningDetail, Role, ToolCall};

use crate::error::{FormatError, Result};
use crate::validate::{content_str, first_control_token};

/// Parse ONE non-streaming OpenAI / OpenRouter chat message into a clean [`Message`].
///
/// Accepts either a full response object (`{"choices":[{"message":{…}}]}`, first choice taken) or
/// a bare message object (`{"role":"assistant","content":…,"reasoning":…}`). The mapping mirrors
/// `gw-providers`:
///
/// - `message.content` → [`Message::content`]. A JSON string goes through
///   [`strip_channel_tokens`], so any inlined channel tokens move to `reasoning`; a JSON array
///   becomes [`Content::Parts`]; `null`/absent becomes [`Content::Null`] (NOT empty text).
/// - `message.reasoning` (or its `reasoning_content` alias) → [`Message::reasoning`]; if absent,
///   any reasoning extracted from `content` is used instead.
/// - `message.reasoning_details` → [`Message::reasoning_details`] (parsed leniently; unparseable
///   or unknown-type fragments are dropped, never fatal).
/// - `message.tool_calls` → [`Message::tool_calls`], with string-encoded `arguments` normalized
///   once to an object and the original text kept in [`FunctionCall::raw_arguments`].
/// - `message.tool_call_id` → [`Message::tool_call_id`] (the explicit result link).
/// - `message.role` → [`Message::role`], defaulting to [`Role::Assistant`] when absent or
///   unrecognized, so ingesting a `tool` result turn does not mislabel it as an assistant turn.
///
/// # Errors
///
/// Returns [`FormatError::Ingest`] if no message can be located (missing `choices` / `message`, or
/// a non-object payload), if the clean `content` STILL contains a control token after stripping
/// the leading channel block (a leak that must not be stored), if `content` is neither a string,
/// `null` nor a well-formed content-parts array, or if any `tool_calls` entry is malformed /
/// carries non-object `arguments` (INVARIANT g).
pub fn ingest_openrouter(value: &Value) -> Result<Message> {
    let message = locate_message(value)?;

    let (content, extracted) = parse_content(message.get("content"))?;

    // Fail loud: a control token surviving in the clean content means an interleaved / second
    // channel the stripper does not model — storing it would leak channel markup into `content`.
    if let Some(clean) = content_str(&content)
        && let Some(token) = first_control_token(&clean)
    {
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
    let tool_calls = parse_tool_calls(message.get("tool_calls"))?;

    Ok(Message {
        role: wire_role(message.get("role")),
        content,
        reasoning,
        reasoning_details,
        tool_calls,
        tool_call_id: string_field(message, "tool_call_id")?,
        name: string_field(message, "name")?,
    })
}

/// Locate the message object inside `value`, accepting a full response or a bare message.
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

/// The wire `role`, defaulting to [`Role::Assistant`] when absent or unrecognized.
///
/// Lenient on purpose: `role` is not the subject of this ingest, and the function's documented
/// contract is the assistant-turn boundary. Defaulting keeps an existing caller (which always
/// sends `"assistant"`) working while still labelling a `tool` result turn correctly.
fn wire_role(raw: Option<&Value>) -> Role {
    raw.and_then(|v| serde_json::from_value::<Role>(v.clone()).ok())
        .unwrap_or(Role::Assistant)
}

/// Read an optional identity-bearing string field (`tool_call_id`, `name`). Absent or JSON `null`
/// means "absent"; any other non-string type is a malformed wire payload and is REJECTED rather
/// than silently dropped — a dropped result link is exactly the fidelity loss this field exists
/// to prevent.
fn string_field(message: &Value, key: &str) -> Result<Option<String>> {
    match message.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(other) => Err(FormatError::Ingest(format!(
            "`{key}` must be a string; got {}",
            json_kind(other)
        ))),
    }
}

/// Parse the wire `content` into a [`Content`], preserving the null-vs-empty distinction.
///
/// - string → [`strip_channel_tokens`] then [`Content::Text`] (so `""` stays empty text, NOT null)
/// - array → [`Content::Parts`], rejecting a malformed parts array instead of flattening it away
/// - `null` / absent → [`Content::Null`]
/// - anything else → rejected (no silent coercion to a string)
///
/// Returns `(content, extracted_reasoning)`; the second element is `Some` only when a leading
/// channel block was peeled out of the text.
fn parse_content(raw: Option<&Value>) -> Result<(Content, Option<String>)> {
    match raw {
        None | Some(Value::Null) => Ok((Content::Null, None)),
        Some(Value::String(text)) => {
            let (clean, extracted) = strip_channel_tokens(text);
            Ok((Content::Text(clean), extracted))
        }
        Some(value @ Value::Array(_)) => {
            let parts: Vec<ContentPart> = serde_json::from_value(value.clone()).map_err(|e| {
                FormatError::Ingest(format!("`content` is not a valid content-parts array: {e}"))
            })?;
            Ok((Content::Parts(parts), None))
        }
        Some(other) => Err(FormatError::Ingest(format!(
            "`content` must be a string, null, or a content-parts array; got {}",
            json_kind(other)
        ))),
    }
}

/// Parse `tool_calls` into canonical [`ToolCall`]s. An absent/`null` value yields `None`; a
/// non-array shape is rejected. An empty array collapses to `None`, matching the
/// `reasoning_details` convention (an empty list declares nothing).
fn parse_tool_calls(raw: Option<&Value>) -> Result<Option<Vec<ToolCall>>> {
    let Some(value) = raw.filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let array = value.as_array().ok_or_else(|| {
        FormatError::Ingest(format!(
            "`tool_calls` must be an array; got {}",
            json_kind(value)
        ))
    })?;
    let mut out = Vec::with_capacity(array.len());
    for (i, frag) in array.iter().enumerate() {
        out.push(
            parse_tool_call(frag)
                .map_err(|e| FormatError::Ingest(format!("tool_calls[{i}]: {e}")))?,
        );
    }
    Ok(if out.is_empty() { None } else { Some(out) })
}

/// Parse ONE `tool_calls[]` entry, normalizing its `arguments` (INVARIANT g).
fn parse_tool_call(frag: &Value) -> Result<ToolCall> {
    let obj = frag.as_object().ok_or_else(|| {
        FormatError::Ingest(format!("must be an object; got {}", json_kind(frag)))
    })?;
    let id = match obj.get("id") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(other) => {
            return Err(FormatError::Ingest(format!(
                "`id` must be a string; got {}",
                json_kind(other)
            )));
        }
    };
    let function = obj
        .get("function")
        .and_then(Value::as_object)
        .ok_or_else(|| FormatError::Ingest("`function` must be an object".to_owned()))?;
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| FormatError::Ingest("`function.name` must be a string".to_owned()))?;
    let (arguments, raw_arguments) = normalize_arguments(function.get("arguments"))?;
    Ok(ToolCall {
        id,
        function: FunctionCall {
            name: name.to_owned(),
            arguments,
            raw_arguments,
        },
    })
}

/// Normalize `function.arguments` to the JSON object the contract requires (INVARIANT g).
///
/// - object → passed through UNCHANGED (nothing is normalized, so no raw duplicate is stored)
/// - string → decoded EXACTLY ONCE into an object; the original wire text is returned alongside
///   as the retained source evidence for the byte-level edit
/// - absent/`null` → the empty argument object `{}` (a zero-argument call is legitimate, and `{}`
///   is its object form)
/// - a string that fails to decode, decodes to a non-object, or any other JSON type → REJECTED
fn normalize_arguments(raw: Option<&Value>) -> Result<(Value, Option<String>)> {
    match raw {
        None | Some(Value::Null) => Ok((Value::Object(Map::new()), None)),
        Some(Value::Object(map)) => Ok((Value::Object(map.clone()), None)),
        Some(Value::String(text)) => {
            let decoded: Value = serde_json::from_str(text).map_err(|e| {
                FormatError::Ingest(format!("`function.arguments` is not valid JSON: {e}"))
            })?;
            if !decoded.is_object() {
                return Err(FormatError::Ingest(format!(
                    "`function.arguments` decoded to {}, not an object (INVARIANT g)",
                    json_kind(&decoded)
                )));
            }
            Ok((decoded, Some(text.clone())))
        }
        Some(other) => Err(FormatError::Ingest(format!(
            "`function.arguments` must be a JSON object or a JSON-encoded object string; got {}",
            json_kind(other)
        ))),
    }
}

/// A short human-readable name for a `serde_json::Value`'s type, for error messages.
fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
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

    // --- tool trajectory (INVARIANT g / i) --------------------------------------------------

    fn assistant_with_calls(calls: serde_json::Value) -> serde_json::Value {
        json!({"role": "assistant", "content": null, "tool_calls": calls})
    }

    #[test]
    fn tool_calls_are_populated_with_ids_names_and_object_arguments() {
        let v = assistant_with_calls(json!([
            {"id": "read-a", "type": "function", "function": {
                "name": "read_file", "arguments": {"filepath": "toy.py", "start_line": 1}}},
            {"id": "read-b", "type": "function", "function": {
                "name": "read_file", "arguments": {}}},
        ]));
        let m = ingest_openrouter(&v).unwrap();
        let calls = m.tool_calls.expect("tool_calls populated at ingest");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id.as_deref(), Some("read-a"));
        assert_eq!(calls[0].function.name, "read_file");
        assert_eq!(calls[0].function.arguments["start_line"], 1);
        // An object payload needs NO normalization, so no raw duplicate is stored.
        assert_eq!(calls[0].function.raw_arguments, None);
        // An empty object is a valid zero-argument call, not a missing payload.
        assert_eq!(calls[1].function.arguments, json!({}));
    }

    #[test]
    fn string_encoded_arguments_normalize_once_and_keep_the_raw_wire_text() {
        let raw = "{\"filepath\": \"toy.py\", \"start_line\": 20}";
        let v = assistant_with_calls(json!([{
            "id": "read-b", "type": "function",
            "function": {"name": "read_file", "arguments": raw},
        }]));
        let m = ingest_openrouter(&v).unwrap();
        let call = &m.tool_calls.expect("calls")[0];
        // Normalized exactly once into the contract's object form…
        assert_eq!(
            call.function.arguments,
            json!({"filepath": "toy.py", "start_line": 20})
        );
        // …with the original wire text retained as source evidence for the byte-level edit.
        assert_eq!(call.function.raw_arguments.as_deref(), Some(raw));
    }

    #[test]
    fn null_content_is_preserved_and_not_coerced_to_empty_text() {
        let m = ingest_openrouter(&assistant_with_calls(json!([]))).unwrap();
        assert_eq!(m.content, Content::Null);
        assert_ne!(m.content, Content::Text(String::new()));
        // An empty array declares nothing, so it collapses to None (and content stays null).
        assert_eq!(m.tool_calls, None);

        // An explicit empty STRING is the other observation and stays empty text.
        let empty = ingest_openrouter(&json!({"role": "assistant", "content": ""})).unwrap();
        assert_eq!(empty.content, Content::Text(String::new()));
    }

    #[test]
    fn absent_content_ingests_as_null() {
        let m = ingest_openrouter(&json!({"role": "assistant", "reasoning": "think"})).unwrap();
        assert_eq!(m.content, Content::Null);
        assert_eq!(m.reasoning.as_deref(), Some("think"));
    }

    #[test]
    fn tool_result_keeps_its_explicit_link_and_role() {
        let v = json!({
            "role": "tool", "tool_call_id": "read-b", "name": "read_file",
            "content": "{\"status\": \"error\"}",
        });
        let m = ingest_openrouter(&v).unwrap();
        assert_eq!(
            m.role,
            Role::Tool,
            "a result turn must not be labelled assistant"
        );
        assert_eq!(m.tool_call_id.as_deref(), Some("read-b"));
        assert_eq!(m.name.as_deref(), Some("read_file"));
    }

    #[test]
    fn role_defaults_to_assistant_when_absent_or_unknown() {
        let m = ingest_openrouter(&json!({"content": "hi"})).unwrap();
        assert_eq!(m.role, Role::Assistant);
        let m = ingest_openrouter(&json!({"content": "hi", "role": "wizard"})).unwrap();
        assert_eq!(m.role, Role::Assistant);
    }

    #[test]
    fn malformed_and_non_object_arguments_are_rejected() {
        for bad in [
            serde_json::json!({"id": "x", "function": {"name": "f", "arguments": "{not json"}}),
            serde_json::json!({"id": "x", "function": {"name": "f", "arguments": "[1, 2]"}}),
            serde_json::json!({"id": "x", "function": {"name": "f", "arguments": 7}}),
            serde_json::json!({"id": "x", "function": {"name": "f", "arguments": true}}),
        ] {
            let v = assistant_with_calls(json!([bad]));
            let err = ingest_openrouter(&v).unwrap_err();
            assert!(
                matches!(err, FormatError::Ingest(_)),
                "malformed arguments must be rejected, got {err:?}"
            );
        }
    }

    #[test]
    fn malformed_tool_call_shapes_are_rejected() {
        for bad in [
            json!({"id": "x", "function": "read_file"}),
            json!({"id": 7, "function": {"name": "f", "arguments": {}}}),
            json!({"id": "x", "function": {"arguments": {}}}),
            json!("not-an-object"),
        ] {
            let v = assistant_with_calls(json!([bad]));
            assert!(
                ingest_openrouter(&v).is_err(),
                "malformed call must be rejected"
            );
        }
        // A non-array `tool_calls` is rejected rather than ignored.
        assert!(ingest_openrouter(&assistant_with_calls(json!({"id": "x"}))).is_err());
    }

    #[test]
    fn non_string_non_null_content_is_rejected_not_coerced() {
        for bad in [json!(7), json!(true), json!({"text": "hi"})] {
            let v = json!({"role": "assistant", "content": bad});
            let err = ingest_openrouter(&v).unwrap_err();
            assert!(matches!(err, FormatError::Ingest(_)), "got {err:?}");
        }
    }

    #[test]
    fn content_parts_array_ingests_as_parts() {
        let v = json!({
            "role": "user",
            "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": "u"}],
        });
        let m = ingest_openrouter(&v).unwrap();
        let Content::Parts(parts) = m.content else {
            panic!("expected parts, got {:?}", m.content);
        };
        assert_eq!(parts.len(), 2);
        // A malformed parts array is rejected instead of being flattened away silently.
        let bad = json!({"role": "user", "content": [{"type": "nope"}]});
        assert!(ingest_openrouter(&bad).is_err());
    }

    #[test]
    fn absent_arguments_become_the_empty_object() {
        let v = assistant_with_calls(json!([{"id": "x", "function": {"name": "now"}}]));
        let m = ingest_openrouter(&v).unwrap();
        assert_eq!(
            m.tool_calls.expect("calls")[0].function.arguments,
            json!({})
        );
    }
}
