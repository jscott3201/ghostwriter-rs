//! [`ChatRequest`] — the serializable `/chat/completions` request body.
//!
//! An OpenAI-compatible body plus OpenRouter's unified `reasoning` param and its `provider`
//! routing object. `stream` is always `true` (this crate only streams). Sampling fields are
//! `Option`s that omit when `None`.
//!
//! The reasoning param ([`ReasoningParam`]) is the load-bearing knob: it serializes to
//! `{"effort": "xhigh"}` or `{"max_tokens": N}`. `effort` and `max_tokens` are **mutually
//! exclusive** on the wire, which the enum encodes structurally. Per the harness invariant the
//! effort is `"xhigh"`, never `"max"` (which OpenRouter rejects with HTTP 400).
//!
//! [`ProviderRouting`] is the OpenRouter-specific `provider` object (NOT in the OpenAI spec):
//! pin/order/restrict upstream providers for cost, speed, and reproducible provenance.

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
    /// Max completion tokens (the legacy OpenAI `max_tokens` alias, which OpenRouter still
    /// accepts and does not deprecate). On a reasoning model this cap covers the **visible
    /// output PLUS the reasoning tokens** combined — for a precise reasoning-token budget use
    /// the dedicated [`reasoning`](ChatRequest::reasoning) param instead. (C6)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// OpenRouter unified reasoning param. `None` ⇒ omit (provider default).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningParam>,

    /// OpenRouter provider-routing object (an OpenRouter extension, NOT in the OpenAI spec).
    /// `None` ⇒ omit (OpenRouter default routing). See [`ProviderRouting`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderRouting>,

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
            provider: None,
            usage: None,
        }
    }

    /// Set the reasoning param (effort or max-tokens). Chainable.
    #[must_use]
    pub fn with_reasoning(mut self, reasoning: ReasoningParam) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    /// Set the OpenRouter provider-routing object. Chainable.
    ///
    /// Pinning a provider (e.g. via [`ProviderRouting::pin`]) also makes the captured
    /// `served_by` / provenance deterministic, since OpenRouter then routes to exactly that
    /// upstream rather than its default fallback set.
    #[must_use]
    pub fn with_provider(mut self, routing: ProviderRouting) -> Self {
        self.provider = Some(routing);
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

    /// Set the max completion tokens (the legacy OpenAI `max_tokens` alias). Note that on a
    /// reasoning model this cap covers visible output PLUS reasoning tokens; for a precise
    /// reasoning budget use [`with_reasoning`](Self::with_reasoning) with
    /// [`ReasoningParam::max_tokens`]. Chainable. (C6)
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

/// OpenRouter's `provider` routing object — an **OpenRouter extension**, not part of the OpenAI
/// chat-completions spec.
///
/// Controls which upstream provider serves the request. Pinning a provider (see
/// [`ProviderRouting::pin`]) gives deterministic routing, which makes the captured
/// `served_by` / provenance reproducible and stabilizes cost/latency. Every field is omitted
/// from the JSON when empty, so a default [`ProviderRouting`] serializes to `{}` and an unset
/// `ChatRequest.provider` emits no `provider` key at all.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ProviderRouting {
    /// Preferred provider slugs, tried in order, e.g. `["novita"]`. Empty ⇒ omitted.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub order: Vec<String>,
    /// Restrict routing to ONLY these providers. `None` ⇒ omitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub only: Option<Vec<String>>,
    /// Whether to fall through to other providers if the preferred ones fail. `None` ⇒
    /// omitted (OpenRouter default is `true`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_fallbacks: Option<bool>,
    /// Sort strategy: `"price"` | `"throughput"` | `"latency"`. `None` ⇒ omitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sort: Option<String>,
}

impl ProviderRouting {
    /// The canonical **hard pin**: prefer exactly `slug` with fallbacks disabled
    /// (`{"order":[slug],"allow_fallbacks":false}`). Use this to pin, e.g.,
    /// `minimax/minimax-m3` to `"novita"` for cost/speed and reproducible provenance.
    #[must_use]
    pub fn pin(slug: impl Into<String>) -> Self {
        Self {
            order: vec![slug.into()],
            allow_fallbacks: Some(false),
            ..Self::default()
        }
    }

    /// Prefer the given provider slugs in order, leaving fallbacks at OpenRouter's default.
    #[must_use]
    pub fn ordered<I, S>(slugs: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            order: slugs.into_iter().map(Into::into).collect(),
            ..Self::default()
        }
    }

    /// Route by a sort strategy (`"price"` | `"throughput"` | `"latency"`).
    #[must_use]
    pub fn sorted(strategy: impl Into<String>) -> Self {
        Self {
            sort: Some(strategy.into()),
            ..Self::default()
        }
    }
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

    // --- E. provider routing (OpenRouter extension) ----------------------------------------

    #[test]
    fn pin_serializes_to_order_and_no_fallbacks() {
        let req = ChatRequest::new("minimax/minimax-m3", vec![user_msg("q")])
            .with_provider(ProviderRouting::pin("novita"));
        // Exact wire shape: order + allow_fallbacks, nothing else.
        let s = serde_json::to_string(&req).unwrap();
        assert!(s.contains("\"provider\":{\"order\":[\"novita\"],\"allow_fallbacks\":false}"));
        let v: Value = serde_json::to_value(&req).unwrap();
        assert_eq!(v["provider"]["order"][0], "novita");
        assert_eq!(v["provider"]["allow_fallbacks"], Value::Bool(false));
        // `only` and `sort` are omitted, not null.
        assert!(v["provider"].get("only").is_none());
        assert!(v["provider"].get("sort").is_none());
    }

    #[test]
    fn request_without_provider_emits_no_provider_key() {
        let req = ChatRequest::new("m", vec![user_msg("q")]);
        let v: Value = serde_json::to_value(&req).unwrap();
        assert!(v.get("provider").is_none());
    }

    #[test]
    fn empty_routing_serializes_to_empty_object() {
        // A default ProviderRouting omits every field → `{}`.
        let v = serde_json::to_value(ProviderRouting::default()).unwrap();
        assert_eq!(v, serde_json::json!({}));
    }

    #[test]
    fn sorted_serializes_sort_only() {
        let req = ChatRequest::new("m", vec![user_msg("q")])
            .with_provider(ProviderRouting::sorted("throughput"));
        let v: Value = serde_json::to_value(&req).unwrap();
        assert_eq!(v["provider"]["sort"], "throughput");
        assert!(v["provider"].get("order").is_none());
        assert!(v["provider"].get("allow_fallbacks").is_none());
    }

    #[test]
    fn only_serializes_as_array() {
        let routing = ProviderRouting {
            only: Some(vec!["parasail".into(), "deepseek".into()]),
            ..ProviderRouting::default()
        };
        let req = ChatRequest::new("m", vec![user_msg("q")]).with_provider(routing);
        let v: Value = serde_json::to_value(&req).unwrap();
        assert_eq!(v["provider"]["only"][0], "parasail");
        assert_eq!(v["provider"]["only"][1], "deepseek");
        // order empty → omitted.
        assert!(v["provider"].get("order").is_none());
    }

    #[test]
    fn ordered_serializes_order_without_fallback_override() {
        let req = ChatRequest::new("m", vec![user_msg("q")])
            .with_provider(ProviderRouting::ordered(["novita", "parasail"]));
        let v: Value = serde_json::to_value(&req).unwrap();
        assert_eq!(v["provider"]["order"][0], "novita");
        assert_eq!(v["provider"]["order"][1], "parasail");
        // ordered() leaves fallbacks at OpenRouter default → omitted.
        assert!(v["provider"].get("allow_fallbacks").is_none());
    }
}
