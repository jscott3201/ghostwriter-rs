//! `gw-generate` — the teacher-orchestration **producer**.
//!
//! Owns three stages of trace synthesis: the pre-teacher-spend USER-turn **QC gate**
//! ([`UserTurnVerdict`](gw_schema::UserTurnVerdict)), **ASSISTANT-turn generation** (full
//! chain-of-thought captured via teacher models behind the injected `gw-providers` [`Provider`]
//! trait), and **[`TrainingRecord`](gw_schema::TrainingRecord) assembly** at
//! [`assistant_generated`](gw_schema::LifecycleState::AssistantGenerated). It accepts an
//! already-elicited candidate USER turn: the user-turn **elicitation strategy** itself
//! (MAGPIE / SeedExpand / Evolve / Persona prompt construction) is **engine-driven** and a tracked
//! follow-up — gw-generate gates and records a candidate, it does not own the prompt-template engine.
//!
//! It is a **pure producer**: it depends only on `gw-schema`, `gw-providers`, and `gw-format`; it
//! does NO I/O of its own beyond the injected provider/embedder, persists NOTHING, and computes NO
//! content hashes (`gw-storage::put` is authoritative for `record_hash`/`prompt_hash`, so
//! `TrainingRecord.hashes` is left at default).
//!
//! ## Pipeline
//!
//! ```text
//! seed ─▶ synthesize_user_turn ─▶ [UserTurnVerdict gate] ─▶ generate_assistant ─▶ TrainingRecord
//!                                       │ all four bools?        │ (teacher spend)
//!                                       └─ false ▶ blocked       └─ best-of-k fan-out (sibling)
//! ```
//!
//! - **USER synthesis + QC gate** ([`user_synth`]). A candidate USER turn is gated on four bools
//!   (`answerable`, `difficulty_targeted`, `diverse`, `in_scope_safe`) BEFORE any teacher tokens
//!   are spent (the spend guard). `diverse` is an embedding cosine-dedup via the injected
//!   [`Embedder`] seam (no ML dep here). `in_scope_safe` is forced true for
//!   adversarial-by-construction (RefusalExpected) prompts.
//! - **Teacher request building** ([`request`]). The [`TeacherCall`] builder ALWAYS sets
//!   `max_tokens`, requests usage accounting, and emits reasoning as exactly one of
//!   effort (`xhigh`, never `max`) / a reasoning-token budget (mutually exclusive).
//! - **ASSISTANT generation** ([`assistant`]). Streams the teacher via the [`Provider`], ingests
//!   the response through `gw-format` so reasoning stays a sibling of clean content (INVARIANT-a),
//!   and FAILS LOUD on the `<|channel>thought` truncation hazard (`finish_reason == "length"` mid-CoT).
//! - **best-of-k** ([`sibling`]). Fans one prompt into `k` siblings with `completion_index` `0..k`,
//!   `n_completions = k`, and DISTINCT per-sibling seeds (never `k` identical greedy samples).
//! - **Assembly** ([`assemble`]). Folds everything into a `TrainingRecord` at `assistant_generated`.
//!
//! ## Invariants enforced in code (see the cited `file.rs:fn`)
//!
//! - **INVARIANT-a** (reasoning a sibling, never inlined): [`assistant::AccumulatedStream::into_turn`]
//!   reuses `gw-format`'s [`ingest_openrouter`](gw_format::ingest_openrouter).
//! - **INVARIANT-g** (`max_tokens` always set; effort XOR reasoning_max_tokens): [`request::TeacherCall::build`]
//!   + the [`request::ReasoningPolicy`] enum.
//! - **UserTurnVerdict gate** (no teacher spend unless all four true): [`user_synth::GatedUserTurn::passed`]
//!   + [`generate_assistant`] (refuses on a failed gate).
//! - **best-of-k** (distinct per-sibling sampling): [`sibling::plan_group`].
//!
//! ## Hermetic testing
//!
//! Every model call is behind the [`Provider`] trait and the [`Embedder`] seam, so unit + integration
//! tests run with a fake in-memory provider replaying canned `StreamDelta`s and a deterministic
//! embedder — NO network. Any live test is `#[ignore]` + env-key-gated (mirroring
//! `gw-providers/tests/live_openrouter.rs`).

mod assemble;
mod assistant;
mod error;
mod request;
mod sibling;
mod teacher;
mod user_synth;

pub use assemble::{RecordContext, assemble};
pub use assistant::{AccumulatedStream, AssistantTurn, accumulate, generate_turn};
pub use error::{GenerateError, Result};
pub use request::{ReasoningPolicy, SamplingPreset, TeacherCall};
pub use sibling::{DEFAULT_K, SiblingPlan, plan_group, seeds_are_distinct};
pub use teacher::{
    FixedTeacher, Teacher, TeacherSelector, routing_to_provider, routing_to_reasoning,
    routing_to_sampling,
};
pub use user_synth::{
    DEFAULT_COSINE_THRESHOLD, Embedder, GatedUserTurn, NullEmbedder, UserSeed, UserTurnCandidate,
    cosine, gate, user_message,
};

use gw_providers::Provider;

/// Run a candidate USER turn through the QC gate (USER-SYNTHESIS §9) WITHOUT spending teacher
/// tokens. The returned [`GatedUserTurn`] carries the verdict; the orchestrator MUST check
/// [`GatedUserTurn::passed`] before proceeding — [`generate_assistant`] enforces this too.
///
/// `prior_embeddings` are the already-admitted USER-turn vectors for the `diverse` dedup. Gating at
/// the default cosine threshold ([`DEFAULT_COSINE_THRESHOLD`]).
///
/// # Errors
/// Returns [`GenerateError::Embed`] if the injected embedder fails on the candidate.
pub fn synthesize_user_turn<E: Embedder + ?Sized>(
    candidate: UserTurnCandidate,
    embedder: &E,
    prior_embeddings: &[Vec<f32>],
) -> Result<GatedUserTurn> {
    gate(candidate, embedder, prior_embeddings)
}

/// Generate the ASSISTANT turn for a GATED user turn by calling the teacher — the single function
/// that spends teacher tokens, and only ever on a candidate that PASSED the QC gate.
///
/// This is the load-bearing spend guard at the orchestration level: it returns
/// [`GenerateError::Invariant`] (without touching the provider) if `gated.passed()` is false, so
/// there is NO code path that calls the teacher for a turn the four-bool gate rejected.
///
/// On success it builds the request (enforcing INVARIANT-g via [`TeacherCall::build`]), streams the
/// teacher (capturing CoT as a sibling of content via [`generate_turn`]), and returns the clean
/// [`AssistantTurn`] for [`assemble`] to fold into a record.
///
/// # Errors
/// - [`GenerateError::Invariant`] if the gate was not passed, or `max_tokens` is unset.
/// - [`GenerateError::Provider`] / [`GenerateError::TruncatedReasoning`] / [`GenerateError::EmptyResponse`]
///   / [`GenerateError::Format`] from the teacher call (see [`generate_turn`]).
pub async fn generate_assistant<P: Provider + ?Sized>(
    provider: &P,
    gated: &GatedUserTurn,
    call: &TeacherCall,
) -> Result<AssistantTurn> {
    if !gated.passed() {
        return Err(GenerateError::Invariant(format!(
            "refusing to spend teacher tokens: user-turn QC gate failed \
             (answerable={}, difficulty_targeted={}, diverse={}, in_scope_safe={})",
            gated.verdict.answerable,
            gated.verdict.difficulty_targeted,
            gated.verdict.diverse,
            gated.verdict.in_scope_safe,
        )));
    }
    let request = call.build()?;
    generate_turn(provider, request).await
}
