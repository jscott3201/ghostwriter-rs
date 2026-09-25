//! Conversation types: the leak-free `Message` and its parts.
//!
//! The single load-bearing invariant (DATA-SCHEMA §1.3, INVARIANT a): **reasoning is a
//! first-class sibling of `content` on every assistant turn, never inlined.** `content` is
//! always the clean final answer; chain-of-thought lives in [`Message::reasoning`] (+ the
//! structured [`Message::reasoning_details`]).
//!
//! ## Tool-result identity (INVARIANT i)
//!
//! A tool-using turn is a RELATION, not a pair of independent messages: an assistant turn
//! declares N [`tool_calls`](Message::tool_calls) and each one is answered by a
//! [`Role::Tool`] turn carrying the EXPLICIT
//! [`tool_call_id`](Message::tool_call_id) it answers. A function *name* is not a link — two
//! calls to the same tool with different arguments must stay distinguishable, so a result
//! without an id is only unambiguous when exactly one call in the trajectory declares that
//! name. `gw-format::validate_tool_links` is the checker; it never guesses.
//!
//! ## Nullable content policy
//!
//! [`Content::Null`] and `Content::Text("")` are DISTINCT, deliberately. A wire assistant turn
//! that carries `"content": null` (the shape for a turn whose only output is a tool call) is not
//! the same observation as a turn that emitted an empty string, so ingest preserves the
//! difference rather than coercing one into the other.

use serde::{Deserialize, Serialize};

/// One conversation turn. `content` is ALWAYS the clean final answer; CoT lives in
/// [`reasoning`](Message::reasoning) / [`reasoning_details`](Message::reasoning_details),
/// NEVER inlined into `content` (DATA-SCHEMA §1.3, INVARIANT a).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// `System | Developer | User | Assistant | Tool`.
    pub role: Role,
    /// Clean final answer (text, or multimodal parts). NEVER carries channel tokens / CoT.
    pub content: Content,

    /// FIRST-CLASS chain-of-thought for an ASSISTANT turn. NEVER serialized into `content`.
    /// Ingest strips channel tokens into here; export re-renders from here. (INVARIANT a)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,

    /// Structured, provider-native reasoning blocks (text | summary | encrypted), stored
    /// verbatim for faithful multi-turn passback AND for the Verify gate. (INVARIANT b)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_details: Option<Vec<ReasoningDetail>>,

    /// `tool_calls[].function.arguments` is ALWAYS a JSON object in history. (INVARIANT g)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    /// THE RESULT LINK: the `tool_calls[].id` this turn answers. Set on [`Role::Tool`] turns.
    ///
    /// This is the only faithful way to bind a result to its call when a trajectory issues more
    /// than one call to the same function (INVARIANT i). `None` means "this turn declares no
    /// result link" — either a non-tool turn, or a legacy/hand-built record. A `Tool` turn with
    /// `None` is only unambiguous when exactly one call in the trajectory declares the same
    /// [`name`](Message::name); `gw-format::validate_tool_links` rejects the rest rather than
    /// inferring a link from the function name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,

    /// Name for `tool` messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// The OpenAI role vocabulary the envelope keeps. The Gemma-4 renderer remaps at export
/// (`assistant → model`, `developer → system`); DATA-SCHEMA §1.3 role-mapping note.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

/// Message content: clean text, multimodal parts, or an explicitly-absent value. Serialized
/// untagged so plain string content round-trips as a JSON string.
///
/// [`Null`](Content::Null) exists so the wire's `"content": null` survives as its own observation
/// instead of being coerced to empty text. The distinction is load-bearing for tool turns: an
/// assistant turn whose only output is a `tool_calls` array carries `null`, while a turn that
/// genuinely emitted `""` is a different (and rarer) observation. Collapsing them would make a
/// tool-calling turn indistinguishable from a truncated one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    /// Multimodal parts (text / image / audio refs).
    Parts(Vec<ContentPart>),
    /// The provider sent `"content": null` (or omitted `content` entirely). NOT the same as
    /// `Text("")` — see the type docs.
    Null,
}

/// Empty text is the default: an absent value is a real, distinct observation and must be asked for
/// explicitly ([`Content::Null`]), never inferred from a defaulted struct.
impl Default for Content {
    fn default() -> Self {
        Content::Text(String::new())
    }
}

/// A single multimodal content part. Tagged on `type` to mirror the OpenAI parts shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    ImageUrl {
        image_url: String,
    },
    InputAudio {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        audio_url: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        format: Option<String>,
    },
}

/// Mirrors OpenRouter `reasoning_details[]`. Tagged on `type`, stored VERBATIM (DATA-SCHEMA §2.1).
///
/// `text` is the only plaintext-bearing variant; the Verify gate (INVARIANT b) hard-fails a
/// record whose reasoning is `summary`-only or `encrypted`-only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ReasoningDetail {
    #[serde(rename = "reasoning.text")]
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        format: Option<String>,
        index: u32,
    },
    #[serde(rename = "reasoning.summary")]
    Summary {
        summary: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        format: Option<String>,
        index: u32,
    },
    #[serde(rename = "reasoning.encrypted")]
    Encrypted {
        data: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        format: Option<String>,
        index: u32,
    },
}

/// A tool/function call emitted on an assistant turn.
///
/// `id` is the ANCHOR a later [`Role::Tool`] turn points back at via
/// [`Message::tool_call_id`] (INVARIANT i). It stays optional so a provider that omits ids is
/// still representable, but such a call cannot be the target of an explicit result link.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub function: FunctionCall,
}

/// The function name + arguments of a [`ToolCall`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    /// ALWAYS a JSON object (`serde_json::Value::Object`). A string here is a CONTRACT
    /// VIOLATION, validated at ingest + before render. (INVARIANT g)
    pub arguments: serde_json::Value,

    /// The ORIGINAL wire text of `arguments`, retained only when ingest had to NORMALIZE a
    /// string-encoded payload into the object form the contract requires (INVARIANT g). `None`
    /// when the provider already sent an object, so a record that needed no normalization carries
    /// no duplicate payload.
    ///
    /// Normalization changes the bytes even when it changes no meaning: a provider may emit
    /// `"{ \"a\" : 1 }"` and the canonical object re-renders as `{"a":1}`. Dropping the source
    /// text would make that edit unrecoverable, so the raw form is kept beside the canonical one
    /// and the ingest path normalizes exactly once (a re-ingest of an already-normalized record
    /// is a no-op, not a second decode).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_arguments: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base(role: Role) -> Message {
        Message {
            role,
            content: Content::Text(String::new()),
            reasoning: None,
            reasoning_details: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    fn call(id: &str, name: &str, args: serde_json::Value) -> ToolCall {
        ToolCall {
            id: Some(id.into()),
            function: FunctionCall {
                name: name.into(),
                arguments: args,
                raw_arguments: None,
            },
        }
    }

    /// A record written before `tool_call_id` / `raw_arguments` existed must still deserialize,
    /// with both fields defaulting to `None` (they skip-serialize when `None`, so re-encoding
    /// reproduces the original bytes).
    #[test]
    fn pre_link_records_still_deserialize() {
        let legacy = r#"{"role":"tool","content":"ok","name":"read_file"}"#;
        let m: Message = serde_json::from_str(legacy).unwrap();
        assert_eq!(m.role, Role::Tool);
        assert_eq!(m.content, Content::Text("ok".into()));
        assert_eq!(m.tool_call_id, None);
        assert_eq!(serde_json::to_string(&m).unwrap(), legacy);
    }

    /// The link is representable AND survives a serde round trip byte-for-byte.
    #[test]
    fn tool_call_id_round_trips() {
        let mut m = base(Role::Tool);
        m.content = Content::Text("{\"status\":\"ok\"}".into());
        m.name = Some("read_file".into());
        m.tool_call_id = Some("read-a".into());
        let s = serde_json::to_string(&m).unwrap();
        assert!(s.contains("\"tool_call_id\":\"read-a\""), "{s}");
        assert_eq!(m, serde_json::from_str::<Message>(&s).unwrap());
    }

    /// Two results of the SAME function stay distinguishable because the link is part of the type
    /// — the F1 identity case at its most basic.
    #[test]
    fn two_same_name_calls_stay_distinguishable_by_link() {
        let mut assistant = base(Role::Assistant);
        assistant.content = Content::Null;
        assistant.tool_calls = Some(vec![
            call("read-a", "read_file", serde_json::json!({"start_line": 1})),
            call("read-b", "read_file", serde_json::json!({"start_line": 20})),
        ]);
        let mut first = base(Role::Tool);
        first.name = Some("read_file".into());
        first.tool_call_id = Some("read-a".into());
        let mut second = base(Role::Tool);
        second.name = Some("read_file".into());
        second.tool_call_id = Some("read-b".into());

        assert_ne!(first, second);
        let round: Vec<Message> =
            serde_json::from_str(&serde_json::to_string(&[assistant, first, second]).unwrap())
                .unwrap();
        assert_eq!(round[1].tool_call_id.as_deref(), Some("read-a"));
        assert_eq!(round[2].tool_call_id.as_deref(), Some("read-b"));
    }

    /// Null and empty text are DIFFERENT observations and must stay different through serde.
    #[test]
    fn null_content_is_distinct_from_empty_text() {
        let null = Content::Null;
        let empty = Content::Text(String::new());
        assert_ne!(null, empty);
        assert_eq!(serde_json::to_string(&null).unwrap(), "null");
        assert_eq!(serde_json::to_string(&empty).unwrap(), "\"\"");
        assert_eq!(
            serde_json::from_str::<Content>("null").unwrap(),
            Content::Null
        );
        assert_eq!(
            serde_json::from_str::<Content>("\"\"").unwrap(),
            Content::Text(String::new())
        );

        let mut m = base(Role::Assistant);
        m.content = Content::Null;
        let s = serde_json::to_string(&m).unwrap();
        assert!(s.contains("\"content\":null"), "{s}");
        assert_eq!(
            serde_json::from_str::<Message>(&s).unwrap().content,
            Content::Null
        );
    }

    /// The canonical record ALWAYS carries a `content` key — an absent one is a malformed stored
    /// record, not a default. The absent-wire-field case is the ingest boundary's job to map onto
    /// `Content::Null` (see `gw_format::ingest_openrouter`), so the normalization happens exactly
    /// once and the stored form is always explicit.
    #[test]
    fn stored_record_requires_an_explicit_content_key() {
        let err = serde_json::from_str::<Message>(r#"{"role":"assistant"}"#).unwrap_err();
        assert!(err.to_string().contains("missing field `content`"), "{err}");
    }

    /// Unicode, embedded newlines and escaped quotes inside arguments and content survive a
    /// round trip unchanged.
    #[test]
    fn unicode_newline_and_quote_escapes_survive() {
        let args = serde_json::json!({"note": "λ \"quoted\"\nsecond\tline\\end"});
        let mut m = base(Role::Tool);
        m.content = Content::Text("λ\n\"quoted\"".into());
        m.tool_call_id = Some("λ-id".into());
        m.tool_calls = Some(vec![call("c-1", "read_file", args.clone())]);
        let s = serde_json::to_string(&m).unwrap();
        let back: Message = serde_json::from_str(&s).unwrap();
        assert_eq!(back, m);
        assert_eq!(
            back.tool_calls.as_ref().unwrap()[0].function.arguments,
            args
        );
    }

    /// An empty argument object is a VALID call (a zero-argument function), not a missing payload.
    #[test]
    fn empty_argument_object_is_valid() {
        let c = call("c-1", "now", serde_json::json!({}));
        assert_eq!(c.function.arguments, serde_json::json!({}));
        assert!(c.function.arguments.is_object());
    }

    /// The retained raw wire text is emitted only when a normalization actually happened.
    #[test]
    fn raw_arguments_is_absent_unless_normalized() {
        let mut c = call("c-1", "read_file", serde_json::json!({"a": 1}));
        assert_eq!(
            serde_json::to_string(&c.function).unwrap(),
            r#"{"name":"read_file","arguments":{"a":1}}"#
        );
        c.function.raw_arguments = Some("{ \"a\" : 1 }".into());
        let s = serde_json::to_string(&c.function).unwrap();
        assert!(s.contains(r#""raw_arguments":"{ \"a\" : 1 }""#), "{s}");
    }

    /// Reasoning stays a first-class sibling of content (INVARIANT a) even on a null-content
    /// tool-calling turn, where the CoT is the only prose in the message.
    #[test]
    fn reasoning_stays_a_sibling_of_null_content() {
        let mut m = base(Role::Assistant);
        m.content = Content::Null;
        m.reasoning = Some("two reads needed".into());
        m.reasoning_details = Some(vec![ReasoningDetail::Text {
            text: "two reads needed".into(),
            signature: None,
            id: None,
            format: None,
            index: 0,
        }]);
        let s = serde_json::to_string(&m).unwrap();
        assert!(s.contains(r#""content":null"#), "{s}");
        assert!(s.contains(r#""reasoning":"two reads needed""#), "{s}");
        assert_eq!(serde_json::from_str::<Message>(&s).unwrap(), m);
    }
}
