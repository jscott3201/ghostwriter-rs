//! [`ChatRequest`] — the serializable `/chat/completions` request body.
//!
//! An OpenAI-compatible body plus OpenRouter's unified `reasoning` param. `stream` is always
//! `true` (this crate only streams). Sampling fields are `Option`s that omit when `None`.
//!
//! The reasoning param ([`ReasoningParam`]) is the load-bearing knob: it serializes to
//! `{"effort": "xhigh"}` or `{"max_tokens": N}`. `effort` and `max_tokens` are **mutually
//! exclusive** on the wire, which the enum encodes structurally. Per the harness invariant the
//! effort is `"xhigh"`, never `"max"` (which OpenRouter rejects with HTTP 400).

use gw_schema::{Message, ReasoningEffort};
use serde::Serialize;

/// The full chat-completions request body sent to an OpenAI-compatible `/chat/completions`.
///
/// Construct via [`ChatRequest::new`] then layer sampling / reasoning with the builder-style
/// setters. `stream` is fixed `true`; `None` sampling fields are omitted from the JSON.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ChatRequest {
    /// The model slug, e.g. `"z-ai/glm-5.2"`.
    pub model: String,
    /// The conversation, reusing the canonical [`gw_schema::Message`].
    pub messages: Vec<Message>,
    /// Always `true` — this crate is streaming-only.
    pub stream: bool,

    /// Sampling temperature (Official = 1.0, Precise = 0.6).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Nucleus sampling top-p.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Deterministic seed for reproducibility.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Max completion tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// OpenRouter unified reasoning param. `None` ⇒ omit (provider default).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningParam>,

    /// Request token accounting on the final SSE chunk (`{"include": true}`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageRequest>,
}

impl ChatRequest {
    /// A minimal streaming request: `model` + `messages`, `stream: true`, everything else off.
    #[must_use]
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            stream: true,
            temperature: None,
            top_p: None,
            seed: None,
            max_tokens: None,
            reasoning: None,
            usage: None,
        }
    }

    /// Set the reasoning param (effort or max-tokens). Chainable.
    #[must_use]
    pub fn with_reasoning(mut self, reasoning: ReasoningParam) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    /// Set sampling temperature. Chainable.
    #[must_use]
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set nucleus top-p. Chainable.
    #[must_use]
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set the deterministic seed. Chainable.
    #[must_use]
    pub fn with_seed(mut self, seed: i64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the max completion tokens. Chainable.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Request `usage` accounting on the final chunk. Chainable.
    #[must_use]
    pub fn with_usage_accounting(mut self) -> Self {
        self.usage = Some(UsageRequest { include: true });
        self
    }
}

/// OpenRouter's unified `reasoning` object. `effort` and `max_tokens` are mutually exclusive
/// on the wire; the enum makes the illegal "both at once" state unrepresentable.
///
/// Serializes to `{"effort": "xhigh"}` or `{"max_tokens": 2000}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum ReasoningParam {
    /// `{"effort": "<level>"}`. Use [`ReasoningEffort::Xhigh`] for full CoT — never `"max"`.
    Effort {
        /// The effort level (reuses the shared enum; it has no `Max` variant by design).
        effort: ReasoningEffort,
    },
    /// `{"max_tokens": N}` — an explicit reasoning-token budget (mutually exclusive with
    /// `effort`).
    MaxTokens {
        /// The reasoning-token budget.
        max_tokens: u32,
    },
}

impl ReasoningParam {
    /// `{"effort": "xhigh"}` — the harness default for full chain-of-thought capture.
    #[must_use]
    pub fn xhigh() -> Self {
        ReasoningParam::Effort {
            effort: ReasoningEffort::Xhigh,
        }
    }

    /// `{"effort": <level>}` for an arbitrary level.
    #[must_use]
    pub fn effort(effort: ReasoningEffort) -> Self {
        ReasoningParam::Effort { effort }
    }

    /// `{"max_tokens": <budget>}`.
    #[must_use]
    pub fn max_tokens(budget: u32) -> Self {
        ReasoningParam::MaxTokens { max_tokens: budget }
    }
}

/// The `usage: {include: true}` request sub-object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct UsageRequest {
    /// Whether to include token accounting in the response.
    pub include: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use gw_schema::{Content, Role};
    use serde_json::Value;

    fn user_msg(text: &str) -> Message {
        Message {
            role: Role::User,
            content: Content::Text(text.into()),
            reasoning: None,
            reasoning_details: None,
            tool_calls: None,
            name: None,
        }
    }

    #[test]
    fn minimal_request_streams_and_omits_none() {
        let req = ChatRequest::new("z-ai/glm-5.2", vec![user_msg("hi")]);
        let v: Value = serde_json::to_value(&req).unwrap();
        assert_eq!(v["stream"], Value::Bool(true));
        assert_eq!(v["model"], "z-ai/glm-5.2");
        // None fields are absent, not null.
        assert!(v.get("temperature").is_none());
        assert!(v.get("top_p").is_none());
        assert!(v.get("seed").is_none());
        assert!(v.get("reasoning").is_none());
        assert!(v.get("usage").is_none());
    }

    #[test]
    fn reasoning_xhigh_serializes_to_effort_object() {
        let req =
            ChatRequest::new("m", vec![user_msg("q")]).with_reasoning(ReasoningParam::xhigh());
        let s = serde_json::to_string(&req).unwrap();
        assert!(s.contains("\"stream\":true"));
        assert!(s.contains("\"reasoning\":{\"effort\":\"xhigh\"}"));
        // never the forbidden "max".
        assert!(!s.contains("\"max\""));
    }

    #[test]
    fn reasoning_max_tokens_serializes_to_max_tokens_object() {
        let req = ChatRequest::new("m", vec![user_msg("q")])
            .with_reasoning(ReasoningParam::max_tokens(2000));
        let v: Value = serde_json::to_value(&req).unwrap();
        assert_eq!(v["reasoning"]["max_tokens"], 2000);
        // mutually exclusive: no effort key present.
        assert!(v["reasoning"].get("effort").is_none());
    }

    #[test]
    fn sampling_fields_serialize_when_set() {
        let req = ChatRequest::new("m", vec![user_msg("q")])
            .with_temperature(0.6)
            .with_top_p(0.95)
            .with_seed(42)
            .with_max_tokens(8192)
            .with_usage_accounting();
        let v: Value = serde_json::to_value(&req).unwrap();
        assert_eq!(v["temperature"], 0.6);
        assert_eq!(v["top_p"], 0.95);
        assert_eq!(v["seed"], 42);
        assert_eq!(v["max_tokens"], 8192);
        assert_eq!(v["usage"]["include"], Value::Bool(true));
    }

    #[test]
    fn effort_levels_round_trip_spellings() {
        for (eff, want) in [
            (ReasoningEffort::Xhigh, "xhigh"),
            (ReasoningEffort::High, "high"),
            (ReasoningEffort::Medium, "medium"),
            (ReasoningEffort::Low, "low"),
            (ReasoningEffort::Minimal, "minimal"),
            (ReasoningEffort::None, "none"),
        ] {
            let p = ReasoningParam::effort(eff);
            let v = serde_json::to_value(p).unwrap();
            assert_eq!(v["effort"], want);
        }
    }
}
