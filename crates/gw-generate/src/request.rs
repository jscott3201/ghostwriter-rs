//! [`TeacherCall`] — the invariant-enforcing builder for a teacher / user-synth request.
//!
//! `gw-providers` owns the wire [`ChatRequest`] and already makes the "effort vs max_tokens"
//! state structurally unrepresentable in [`ReasoningParam`]. This module is the layer ABOVE that:
//! it captures the harness reasoning policy ([`ReasoningPolicy`]) and the sampling preset
//! ([`SamplingPreset`]) and emits a `ChatRequest` ONLY after asserting the two load-bearing
//! generation invariants at the data-entry seam, so a violation fails loud here rather than
//! silently producing a budget-less or double-reasoning request that OpenRouter would reject.
//!
//! ## Invariants enforced here (INVARIANT-g)
//!
//! - **`max_tokens` is ALWAYS set** on a built request. [`TeacherCall::build`] returns
//!   [`GenerateError::Invariant`] if the cap is zero/unset — there is no path to a budget-less
//!   teacher request. On a reasoning model this cap covers visible output PLUS reasoning tokens,
//!   so it MUST be generous (the spec calls for 8K–16K on thinking turns) to avoid the
//!   `<|channel>thought` truncation hazard.
//! - **`effort` and `reasoning_max_tokens` are MUTUALLY EXCLUSIVE.** [`ReasoningPolicy`] is an
//!   enum so the "both at once" state is unrepresentable; the build path maps it to exactly one of
//!   `ReasoningParam::Effort` / `ReasoningParam::MaxTokens`. For CoT the effort is `"xhigh"`,
//!   never `"max"` (the shared [`ReasoningEffort`] has no `Max` variant by construction).

use gw_providers::{ChatRequest, ProviderRouting, ReasoningParam};
use gw_schema::{Generation, Message, ReasoningEffort};

use crate::error::{GenerateError, Result};

/// The reasoning knob for a teacher call. The two arms are mutually exclusive on the wire
/// (`{"effort":…}` XOR `{"max_tokens":…}`); making them an enum means the illegal "send both"
/// state is unrepresentable, enforcing INVARIANT-g structurally rather than by a runtime check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningPolicy {
    /// `{"effort": <level>}`. The CoT default is [`ReasoningEffort::Xhigh`]; `"max"` is
    /// unrepresentable (the enum has no `Max` variant).
    Effort(ReasoningEffort),
    /// `{"max_tokens": N}` — an explicit reasoning-token budget, mutually exclusive with effort.
    MaxTokens(u32),
}

impl ReasoningPolicy {
    /// The harness default for full chain-of-thought capture: `{"effort":"xhigh"}`.
    #[must_use]
    pub fn xhigh() -> Self {
        ReasoningPolicy::Effort(ReasoningEffort::Xhigh)
    }

    /// Convert to the wire [`ReasoningParam`]. Total — every policy maps to exactly one arm.
    fn to_param(self) -> ReasoningParam {
        match self {
            ReasoningPolicy::Effort(effort) => ReasoningParam::effort(effort),
            ReasoningPolicy::MaxTokens(budget) => ReasoningParam::max_tokens(budget),
        }
    }

    /// The `(reasoning_effort, reasoning_max_tokens)` pair for the [`Generation`] provenance
    /// block — exactly one is `Some`, mirroring the mutual exclusion (INVARIANT-g).
    fn provenance_fields(self) -> (Option<ReasoningEffort>, Option<u32>) {
        match self {
            ReasoningPolicy::Effort(effort) => (Some(effort), None),
            ReasoningPolicy::MaxTokens(budget) => (None, Some(budget)),
        }
    }
}

/// A sampling preset (CONFIG §8). The two harness presets are [`SamplingPreset::official`]
/// (temperature 1.0, the Gemma-4 model-card default and the V1 baseline for CoT generation) and
/// [`SamplingPreset::precise`] (temperature 0.6, the lower-variance A/B alternative). The `seed`
/// is threaded for best-of-k sibling variation and reproducibility.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingPreset {
    /// Sampling temperature (Official = 1.0, Precise = 0.6).
    pub temperature: f64,
    /// Nucleus sampling top-p. `None` ⇒ provider default.
    pub top_p: Option<f64>,
    /// Deterministic seed. `None` ⇒ provider default (non-reproducible).
    pub seed: Option<i64>,
}

impl SamplingPreset {
    /// The "Official" preset: temperature 1.0, top_p 0.95 (the Gemma-4 model-card default; the
    /// V1 baseline and the default for CoT trace generation). Seed unset.
    #[must_use]
    pub fn official() -> Self {
        Self {
            temperature: 1.0,
            top_p: Some(0.95),
            seed: None,
        }
    }

    /// The "Precise" preset: temperature 0.6, top_p 0.95 — the lower-variance A/B alternative.
    #[must_use]
    pub fn precise() -> Self {
        Self {
            temperature: 0.6,
            top_p: Some(0.95),
            seed: None,
        }
    }

    /// This preset with an explicit `seed` (chainable). Used by the best-of-k fan-out to vary
    /// per-sibling sampling deterministically.
    #[must_use]
    pub fn with_seed(mut self, seed: i64) -> Self {
        self.seed = Some(seed);
        self
    }
}

impl Default for SamplingPreset {
    /// Official (temperature 1.0) is the default for CoT trace generation.
    fn default() -> Self {
        Self::official()
    }
}

/// An invariant-enforcing teacher / user-synth call descriptor. Build a wire [`ChatRequest`] via
/// [`build`](TeacherCall::build); the build path is the SINGLE seam that asserts `max_tokens` is
/// set and that reasoning is exactly one of effort / budget (INVARIANT-g).
#[derive(Debug, Clone, PartialEq)]
pub struct TeacherCall {
    /// The model slug, e.g. `"z-ai/glm-5.2"`.
    pub model: String,
    /// The conversation to complete.
    pub messages: Vec<Message>,
    /// The reasoning knob (effort XOR budget). For CoT this is [`ReasoningPolicy::xhigh`].
    pub reasoning: ReasoningPolicy,
    /// The sampling preset (temperature / top_p / seed).
    pub sampling: SamplingPreset,
    /// The combined-output token cap. ALWAYS set on the built request; a zero is rejected
    /// (INVARIANT-g). Must be generous on reasoning models to dodge the truncation hazard.
    pub max_tokens: u32,
    /// Optional OpenRouter provider routing (e.g. a hard pin for reproducible provenance).
    pub routing: Option<ProviderRouting>,
}

impl TeacherCall {
    /// A teacher call with the CoT defaults: `effort=xhigh`, the Official sampling preset, and the
    /// given combined-output `max_tokens` cap. Layer routing / a different preset with the
    /// setters.
    #[must_use]
    pub fn new(model: impl Into<String>, messages: Vec<Message>, max_tokens: u32) -> Self {
        Self {
            model: model.into(),
            messages,
            reasoning: ReasoningPolicy::xhigh(),
            sampling: SamplingPreset::official(),
            max_tokens,
            routing: None,
        }
    }

    /// Override the reasoning policy. Chainable.
    #[must_use]
    pub fn with_reasoning(mut self, reasoning: ReasoningPolicy) -> Self {
        self.reasoning = reasoning;
        self
    }

    /// Override the sampling preset. Chainable.
    #[must_use]
    pub fn with_sampling(mut self, sampling: SamplingPreset) -> Self {
        self.sampling = sampling;
        self
    }

    /// Set OpenRouter provider routing. Chainable.
    #[must_use]
    pub fn with_routing(mut self, routing: ProviderRouting) -> Self {
        self.routing = Some(routing);
        self
    }

    /// Build the wire [`ChatRequest`], enforcing the generation invariants at this seam.
    ///
    /// `usage` accounting is always requested (so the final chunk carries `reasoning_tokens` /
    /// `cost` for the Verify gate and the cost meter). `stream` is fixed `true` by the providers
    /// crate.
    ///
    /// # Errors
    ///
    /// Returns [`GenerateError::Invariant`] when `max_tokens` is zero — there is NO path to a
    /// budget-less teacher request (INVARIANT-g). The effort/budget mutual exclusion is enforced
    /// structurally by [`ReasoningPolicy`], so it cannot fail here.
    pub fn build(&self) -> Result<ChatRequest> {
        if self.max_tokens == 0 {
            return Err(GenerateError::Invariant(format!(
                "max_tokens must always be set (> 0) on a teacher request for `{}`; \
                 a budget-less reasoning request truncates inside the <|channel>thought block",
                self.model
            )));
        }

        let mut req = ChatRequest::new(self.model.clone(), self.messages.clone())
            .with_reasoning(self.reasoning.to_param())
            .with_max_tokens(self.max_tokens)
            .with_temperature(self.sampling.temperature)
            .with_usage_accounting();

        if let Some(top_p) = self.sampling.top_p {
            req = req.with_top_p(top_p);
        }
        if let Some(seed) = self.sampling.seed {
            req = req.with_seed(seed);
        }
        if let Some(routing) = &self.routing {
            req = req.with_provider(routing.clone());
        }

        Ok(req)
    }

    /// Project the call's reproducibility params into a [`Generation`] block. Fills
    /// `seed`/`temperature`/`top_p`/`max_tokens` from the sampling preset + cap, and EXACTLY one of
    /// `reasoning_effort` / `reasoning_max_tokens` from the reasoning policy (the other stays
    /// `None`, mirroring the wire mutual exclusion). `n_completions` / `completion_index` /
    /// `sibling_group_id` / `persona` / `taxonomy_node` / `prompt_template_id` are filled by the
    /// caller (the assembler), not here.
    #[must_use]
    pub fn generation(&self) -> Generation {
        let (reasoning_effort, reasoning_max_tokens) = self.reasoning.provenance_fields();
        Generation {
            seed: self.sampling.seed,
            temperature: Some(self.sampling.temperature),
            top_p: self.sampling.top_p,
            max_tokens: Some(self.max_tokens),
            reasoning_effort,
            reasoning_max_tokens,
            ..Generation::default()
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
    fn build_always_sets_max_tokens_and_usage() {
        // INVARIANT-g: max_tokens is present on the wire and usage accounting is requested.
        let req = TeacherCall::new("z-ai/glm-5.2", vec![user_msg("q")], 8192)
            .build()
            .unwrap();
        let v: Value = serde_json::to_value(&req).unwrap();
        assert_eq!(v["max_tokens"], 8192);
        assert_eq!(v["usage"]["include"], Value::Bool(true));
        assert_eq!(v["stream"], Value::Bool(true));
    }

    #[test]
    fn zero_max_tokens_is_rejected_loud() {
        // INVARIANT-g: there is NO path to a budget-less teacher request.
        let err = TeacherCall::new("m", vec![user_msg("q")], 0)
            .build()
            .unwrap_err();
        assert!(matches!(err, GenerateError::Invariant(_)));
        assert!(err.to_string().contains("max_tokens"));
    }

    #[test]
    fn default_reasoning_is_xhigh_effort_never_max() {
        let req = TeacherCall::new("m", vec![user_msg("q")], 16384)
            .build()
            .unwrap();
        let s = serde_json::to_string(&req).unwrap();
        assert!(s.contains("\"reasoning\":{\"effort\":\"xhigh\"}"));
        assert!(!s.contains("\"max\""));
    }

    #[test]
    fn effort_and_budget_are_mutually_exclusive_on_wire_and_in_provenance() {
        // Effort policy → effort on the wire, effort-only in provenance.
        let effort_call = TeacherCall::new("m", vec![user_msg("q")], 8192);
        let v: Value = serde_json::to_value(effort_call.build().unwrap()).unwrap();
        assert_eq!(v["reasoning"]["effort"], "xhigh");
        assert!(v["reasoning"].get("max_tokens").is_none());
        let g = effort_call.generation();
        assert_eq!(g.reasoning_effort, Some(ReasoningEffort::Xhigh));
        assert_eq!(g.reasoning_max_tokens, None);

        // Budget policy → max_tokens reasoning on the wire, budget-only in provenance.
        let budget_call = TeacherCall::new("m", vec![user_msg("q")], 8192)
            .with_reasoning(ReasoningPolicy::MaxTokens(2000));
        let v: Value = serde_json::to_value(budget_call.build().unwrap()).unwrap();
        assert_eq!(v["reasoning"]["max_tokens"], 2000);
        assert!(v["reasoning"].get("effort").is_none());
        let g = budget_call.generation();
        assert_eq!(g.reasoning_effort, None);
        assert_eq!(g.reasoning_max_tokens, Some(2000));
    }

    #[test]
    fn official_preset_is_temp_1_default() {
        let call = TeacherCall::new("m", vec![user_msg("q")], 8192);
        assert_eq!(call.sampling, SamplingPreset::official());
        let v: Value = serde_json::to_value(call.build().unwrap()).unwrap();
        assert_eq!(v["temperature"], 1.0);
        assert_eq!(v["top_p"], 0.95);
        // No seed unless one is set.
        assert!(v.get("seed").is_none());
    }

    #[test]
    fn precise_preset_lowers_temperature() {
        let call = TeacherCall::new("m", vec![user_msg("q")], 8192)
            .with_sampling(SamplingPreset::precise().with_seed(7));
        let v: Value = serde_json::to_value(call.build().unwrap()).unwrap();
        assert_eq!(v["temperature"], 0.6);
        assert_eq!(v["seed"], 7);
        // And the seed lands in the reproducibility block.
        assert_eq!(call.generation().seed, Some(7));
        assert_eq!(call.generation().temperature, Some(0.6));
    }

    #[test]
    fn routing_pin_serializes_when_set() {
        let call = TeacherCall::new("minimax/minimax-m3", vec![user_msg("q")], 8192)
            .with_routing(ProviderRouting::pin("novita"));
        let v: Value = serde_json::to_value(call.build().unwrap()).unwrap();
        assert_eq!(v["provider"]["order"][0], "novita");
        assert_eq!(v["provider"]["allow_fallbacks"], Value::Bool(false));
    }
}
