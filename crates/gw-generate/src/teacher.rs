//! [`Teacher`] selection + routing: the four fixed teacher slugs and the per-area choice.
//!
//! The harness pins a closed set of four teachers (ARCHITECTURE §0, CONFIG §8). Teacher choice is
//! **empirically best PER training area** (OpenThoughts3: best-benchmark ≠ best-teacher), so this
//! module models the closed set as [`Teacher`] and exposes a [`TeacherSelector`] seam that maps a
//! `training_area` to a teacher + its routing. The default selector pins a single teacher; a richer
//! per-area table is the engine's to configure.
//!
//! ## Mapping `gw_schema::TeacherRouting` → the wire request
//!
//! [`routing_to_provider`] maps the serde-able [`TeacherRouting`] config onto the OpenRouter
//! `provider` object the providers crate models ([`ProviderRouting`]): `provider_only` →
//! `provider.order`, AND — load-bearing — the D5 redistribution posture `data_collection` +
//! `require_parameters` (config defaults `deny` + `true`) is ALWAYS emitted, EVEN WHEN no provider
//! is pinned, so the deny posture rides EVERY teacher request and the captured traces stay legally
//! redistributable (ARCHITECTURE D5 line 646 / §3.2 line 304 / OSS-hygiene line 720).
//! `reasoning_effort` / `temperature` / `top_p` from the routing are surfaced as a
//! [`ReasoningPolicy`] + [`SamplingPreset`] for the `TeacherCall` builder.

use gw_providers::ProviderRouting;
use gw_schema::{TeacherRef, TeacherRouting};

use crate::request::{ReasoningPolicy, SamplingPreset};

/// The closed set of four teacher models the harness draws CoT traces from (ARCHITECTURE §0,
/// CONFIG §8). The slugs are OpenRouter model slugs; `provider` is always `"openrouter"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Teacher {
    /// `z-ai/glm-5.2`.
    Glm52,
    /// `minimax/minimax-m3`.
    MinimaxM3,
    /// `deepseek/deepseek-v4-pro`.
    DeepseekV4Pro,
    /// `google/gemma-4-31b-it`.
    Gemma431bIt,
}

impl Teacher {
    /// All four teachers, in declaration order. Useful for a union/round-robin policy.
    pub const ALL: [Teacher; 4] = [
        Teacher::Glm52,
        Teacher::MinimaxM3,
        Teacher::DeepseekV4Pro,
        Teacher::Gemma431bIt,
    ];

    /// The OpenRouter model slug.
    #[must_use]
    pub fn slug(self) -> &'static str {
        match self {
            Teacher::Glm52 => "z-ai/glm-5.2",
            Teacher::MinimaxM3 => "minimax/minimax-m3",
            Teacher::DeepseekV4Pro => "deepseek/deepseek-v4-pro",
            Teacher::Gemma431bIt => "google/gemma-4-31b-it",
        }
    }

    /// Resolve a slug back to a [`Teacher`], or `None` if it is outside the closed set.
    #[must_use]
    pub fn from_slug(slug: &str) -> Option<Teacher> {
        Teacher::ALL.into_iter().find(|t| t.slug() == slug)
    }

    /// A [`TeacherRef`] for this teacher with `provider = "openrouter"` and `served_by` unset
    /// (the upstream provider is captured from the streamed response, not known up front).
    #[must_use]
    pub fn teacher_ref(self) -> TeacherRef {
        TeacherRef {
            provider: "openrouter".to_string(),
            slug: self.slug().to_string(),
            served_by: None,
            model_card_revision: None,
        }
    }
}

/// Maps a `training_area` to the [`Teacher`] + its [`TeacherRouting`] that should serve it.
///
/// A seam (not a concrete table) because per-area assignment is empirical and configured by the
/// engine. The default impl ([`FixedTeacher`]) pins one teacher for every area; the engine can
/// supply a richer mapping without this crate growing a config dependency.
pub trait TeacherSelector {
    /// The teacher to use for `training_area`, plus the routing/sampling defaults to apply.
    fn select(&self, training_area: &str) -> (Teacher, TeacherRouting);
}

/// A [`TeacherSelector`] that always returns one teacher with one routing — the simplest policy,
/// used by tests and as a sane default before per-area data exists.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedTeacher {
    teacher: Teacher,
    routing: TeacherRouting,
}

impl FixedTeacher {
    /// Pin `teacher` for every area, using the default [`TeacherRouting`] (effort `xhigh`,
    /// `data_collection=deny`, `require_parameters=true`).
    #[must_use]
    pub fn new(teacher: Teacher) -> Self {
        Self {
            teacher,
            routing: TeacherRouting::default(),
        }
    }

    /// Pin `teacher` for every area with an explicit `routing`.
    #[must_use]
    pub fn with_routing(teacher: Teacher, routing: TeacherRouting) -> Self {
        Self { teacher, routing }
    }
}

impl TeacherSelector for FixedTeacher {
    fn select(&self, _training_area: &str) -> (Teacher, TeacherRouting) {
        (self.teacher, self.routing.clone())
    }
}

/// Map a config [`TeacherRouting`] onto the wire [`ProviderRouting`] the providers crate models.
///
/// ALWAYS returns `Some` (never `None`): the D5 redistribution posture
/// (`data_collection` + `require_parameters`, config defaults `deny` + `true`) MUST ride every
/// teacher request, so even an unpinned routing emits a `provider` object carrying the posture.
/// `provider_only` (a hard pin to specific upstreams), when present and non-empty, additionally
/// sets `provider.order`.
#[must_use]
pub fn routing_to_provider(routing: &TeacherRouting) -> ProviderRouting {
    let base = match &routing.provider_only {
        Some(slugs) if !slugs.is_empty() => ProviderRouting::ordered(slugs.iter().cloned()),
        _ => ProviderRouting::default(),
    };
    base.with_data_posture(routing.data_collection, routing.require_parameters)
}

/// Derive the [`ReasoningPolicy`] for a teacher call from a config [`TeacherRouting`]: its
/// `reasoning_effort` (default `xhigh`). The routing models effort only (no per-call reasoning
/// budget), so this is always an `Effort` policy.
#[must_use]
pub fn routing_to_reasoning(routing: &TeacherRouting) -> ReasoningPolicy {
    ReasoningPolicy::Effort(routing.reasoning_effort)
}

/// Derive a [`SamplingPreset`] from a config [`TeacherRouting`], starting from `base` (the area's
/// preset) and overlaying any explicit `temperature` / `top_p` the routing pins. `seed` is left to
/// the caller (the best-of-k fan-out sets it per sibling).
#[must_use]
pub fn routing_to_sampling(routing: &TeacherRouting, base: SamplingPreset) -> SamplingPreset {
    SamplingPreset {
        temperature: routing.temperature.unwrap_or(base.temperature),
        top_p: routing.top_p.or(base.top_p),
        seed: base.seed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gw_schema::ReasoningEffort;

    #[test]
    fn the_four_teacher_slugs_are_exact() {
        assert_eq!(Teacher::Glm52.slug(), "z-ai/glm-5.2");
        assert_eq!(Teacher::MinimaxM3.slug(), "minimax/minimax-m3");
        assert_eq!(Teacher::DeepseekV4Pro.slug(), "deepseek/deepseek-v4-pro");
        assert_eq!(Teacher::Gemma431bIt.slug(), "google/gemma-4-31b-it");
        assert_eq!(Teacher::ALL.len(), 4);
    }

    #[test]
    fn slug_round_trips_and_rejects_outsiders() {
        for t in Teacher::ALL {
            assert_eq!(Teacher::from_slug(t.slug()), Some(t));
        }
        assert_eq!(Teacher::from_slug("openai/gpt-4"), None);
    }

    #[test]
    fn teacher_ref_is_openrouter_with_unset_served_by() {
        let r = Teacher::Glm52.teacher_ref();
        assert_eq!(r.provider, "openrouter");
        assert_eq!(r.slug, "z-ai/glm-5.2");
        assert!(r.served_by.is_none());
    }

    #[test]
    fn fixed_selector_pins_one_teacher_for_every_area() {
        let sel = FixedTeacher::new(Teacher::DeepseekV4Pro);
        let (t1, _) = sel.select("rust-async");
        let (t2, _) = sel.select("sql-analytics");
        assert_eq!(t1, Teacher::DeepseekV4Pro);
        assert_eq!(t2, Teacher::DeepseekV4Pro);
    }

    #[test]
    fn default_routing_yields_xhigh_and_deny_posture_without_a_pin() {
        let routing = TeacherRouting::default();
        assert_eq!(
            routing_to_reasoning(&routing),
            ReasoningPolicy::Effort(ReasoningEffort::Xhigh)
        );
        // D5: even an UNPINNED routing carries the deny posture (no order, but a provider object).
        let prov = routing_to_provider(&routing);
        assert!(prov.order.is_empty());
        assert_eq!(prov.data_collection, Some(gw_schema::DataCollection::Deny));
        assert_eq!(prov.require_parameters, Some(true));
        assert!(!prov.is_empty());
    }

    #[test]
    fn provider_only_maps_to_order_pin_and_keeps_deny_posture() {
        let routing = TeacherRouting {
            provider_only: Some(vec!["novita".into(), "parasail".into()]),
            ..TeacherRouting::default()
        };
        let prov = routing_to_provider(&routing);
        assert_eq!(
            prov.order,
            vec!["novita".to_string(), "parasail".to_string()]
        );
        // The pin does NOT drop the posture.
        assert_eq!(prov.data_collection, Some(gw_schema::DataCollection::Deny));
        assert_eq!(prov.require_parameters, Some(true));
    }

    #[test]
    fn empty_provider_only_still_carries_deny_posture() {
        let routing = TeacherRouting {
            provider_only: Some(vec![]),
            ..TeacherRouting::default()
        };
        let prov = routing_to_provider(&routing);
        assert!(prov.order.is_empty());
        assert_eq!(prov.data_collection, Some(gw_schema::DataCollection::Deny));
    }

    #[test]
    fn routing_temperature_overrides_base_preset() {
        let routing = TeacherRouting {
            temperature: Some(0.3),
            ..TeacherRouting::default()
        };
        let s = routing_to_sampling(&routing, SamplingPreset::official());
        assert_eq!(s.temperature, 0.3);
        // top_p falls back to the base preset when the routing doesn't pin one.
        assert_eq!(s.top_p, Some(0.95));
    }
}
