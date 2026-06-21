//! Conversation types: the leak-free `Message` and its parts.
//!
//! The single load-bearing invariant (DATA-SCHEMA §1.3, INVARIANT a): **reasoning is a
//! first-class sibling of `content` on every assistant turn, never inlined.** `content` is
//! always the clean final answer; chain-of-thought lives in [`Message::reasoning`] (+ the
//! structured [`Message::reasoning_details`]).

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

/// Message content: clean text, or multimodal parts. Serialized untagged so plain string
/// content round-trips as a JSON string.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    /// Multimodal parts (text / image / audio refs).
    Parts(Vec<ContentPart>),
}

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
}
