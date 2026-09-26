//! Assemble a [`TrainingRecord`] from a gated user turn + a generated assistant turn.
//!
//! This is the producer's final step: fold the synthesized USER message, the teacher ASSISTANT
//! turn, and the call metadata into a `TrainingRecord` landed at `lifecycle.state =
//! assistant_generated` — the state `gw-generate` hands off to the engine (ARCHITECTURE §5).
//!
//! ## What gw-generate fills — and what it deliberately leaves at default
//!
//! Filled: `messages` (user + assistant), `provenance` (teacher, user_synth_model, the mirrored
//! `user_turn_kind` / `in_scope_safe`, harness_version, run_id), `generation` (the reproducibility
//! params + best-of-k indices + seed inputs), `cost` (token accounting from the stream), and
//! `lifecycle` (history through `assistant_generated`).
//!
//! Left at DEFAULT (a downstream stage owns them): `hashes` — `gw-storage::put` is AUTHORITATIVE
//! for `record_hash` / `prompt_hash` and RECOMPUTES them; gw-generate has no storage dep and MUST
//! NOT compute content hashes. `verification` / `judging` / `reasoning_quality` belong to the
//! Verifier + JudgePanel rails. `sibling_group_id` is left `None` for the engine to fill from the
//! canonical `prompt_hash` (see [`crate::sibling`]).

use gw_schema::{
    Cost, Generation, Lifecycle, LifecycleState, Provenance, StateTransition, TeacherRef,
    TrainingRecord, UserTurnVerdict,
};

use crate::assistant::AssistantTurn;
use crate::sibling::SiblingPlan;
use crate::user_synth::{GatedUserTurn, UserSeed};

/// The schema version this producer emits (pinned `"1.0.0"` for v1).
const SCHEMA_VERSION: (u64, u64, u64) = (1, 0, 0);

/// Inputs that the engine supplies per record — values gw-generate cannot mint itself without a
/// clock / id dependency (kept injected so unit tests are hermetic and reproducible).
#[derive(Debug, Clone, PartialEq)]
pub struct RecordContext {
    /// Time-sortable record id (ULID / UUIDv7), minted by the engine.
    pub record_id: String,
    /// The run this record belongs to.
    pub run_id: String,
    /// The declarative bundle name (e.g. `"rust-async"`).
    pub training_area: String,
    /// `harness_version` for provenance (e.g. the crate version).
    pub harness_version: String,
    /// Optional git commit for provenance.
    pub git_commit: Option<String>,
    /// RFC 3339 timestamp for the lifecycle transitions (injected clock — no `time` dep here).
    pub now_rfc3339: String,
    /// Model that synthesized the USER turn (the harness synthesizes BOTH roles). `None` if the
    /// user turn was not model-synthesized (e.g. a literal seed).
    pub user_synth_model: Option<String>,
}

/// Assemble the final [`TrainingRecord`] at `assistant_generated`.
///
/// `gated` carries the USER turn + its passed verdict; `turn` is the generated ASSISTANT turn;
/// `teacher` / `generation` describe the call; `sibling` (when `Some`) stamps the best-of-k indices.
///
/// PRECONDITION: `gated.passed()` MUST be true — assembling a record from a turn that failed the QC
/// gate would mean teacher tokens were spent on a blocked candidate. The orchestrator enforces this
/// upstream (it never calls the teacher otherwise); [`assemble`] additionally encodes the verdict's
/// `user_turn_kind` / `in_scope_safe` into provenance so the downstream judge can read them.
#[must_use]
pub fn assemble(
    ctx: &RecordContext,
    gated: &GatedUserTurn,
    turn: AssistantTurn,
    teacher: TeacherRef,
    mut generation: Generation,
    sibling: Option<SiblingPlan>,
) -> TrainingRecord {
    // Fold the seed inputs + best-of-k indices into the reproducibility block.
    apply_seed(&mut generation, &gated.candidate.seed);
    if let Some(s) = sibling {
        generation.n_completions = Some(s.n_completions);
        generation.completion_index = Some(s.completion_index);
        // sibling_group_id is LEFT None: it == prompt_hash, which gw-storage computes (see sibling).
    }

    // Build the reference-only blocks before moving the assistant message out of `turn`.
    let provenance = build_provenance(ctx, &gated.verdict, &gated.candidate, teacher, &turn);
    let cost = build_cost(&turn);
    let lifecycle = build_lifecycle(&ctx.now_rfc3339);
    let messages = vec![gated.candidate.message.clone(), turn.message];

    TrainingRecord {
        record_id: ctx.record_id.clone(),
        schema_version: semver::Version::new(SCHEMA_VERSION.0, SCHEMA_VERSION.1, SCHEMA_VERSION.2),
        dataset_version: None,
        training_area: ctx.training_area.clone(),
        tags: Vec::new(),
        messages,
        tools: None,
        provenance,
        generation,
        // Carry the user-turn verification contract onto the record so the engine's Verify rail can
        // run the answer-correctness check (incl. on a crash-resume of the verify edge). `Oracle::None`
        // contracts are kept verbatim — the rail treats a None oracle as judge-only, not a silent pass.
        verification_contract: Some(gated.candidate.contract.clone()),
        // A freshly assembled candidate carries NO precomputed execution report: the report is
        // produced against the finished candidate's content, so the engine's verify edge resolves it
        // by key (and persists it) rather than the seed claiming one up front.
        execution_evidence: None,
        verification: Default::default(),
        judging: Default::default(),
        reasoning_quality: None,
        lifecycle,
        // hashes LEFT at default: gw-storage::put is authoritative and recomputes record_hash.
        hashes: Default::default(),
        cost,
    }
}

/// Copy the user-synth seed inputs into the [`Generation`] block (persona / taxonomy_node /
/// prompt_template_id), leaving anything already set untouched.
fn apply_seed(generation: &mut Generation, seed: &UserSeed) {
    if generation.persona.is_none() {
        generation.persona = seed.persona.clone();
    }
    if generation.taxonomy_node.is_none() {
        generation.taxonomy_node = seed.taxonomy_node.clone();
    }
    if generation.prompt_template_id.is_none() {
        generation.prompt_template_id = seed.prompt_template_id.clone();
    }
}

/// Build the [`Provenance`], mirroring the user-turn verdict's classification so the judge can read
/// it: `user_turn_kind` is the serialized [`VerificationKind`](gw_schema::VerificationKind) variant
/// and `in_scope_safe` mirrors the verdict bool (DATA-SCHEMA B7). `served_by` is captured from the
/// teacher stream.
fn build_provenance(
    ctx: &RecordContext,
    verdict: &UserTurnVerdict,
    candidate: &crate::user_synth::UserTurnCandidate,
    mut teacher: TeacherRef,
    turn: &AssistantTurn,
) -> Provenance {
    teacher.served_by = turn.served_by.clone();
    Provenance {
        run_id: ctx.run_id.clone(),
        parent_ids: Vec::new(),
        teacher,
        user_synth_model: ctx.user_synth_model.clone(),
        // The serialized VerificationKind variant (e.g. "refusal_expected"), gw-judge-readable.
        user_turn_kind: Some(verification_kind_str(candidate.contract.kind)),
        // Structurally ALWAYS `Some(true)` here: assemble's precondition is `gated.passed()`, which
        // requires `verdict.in_scope_safe == true`. We mirror the verdict field verbatim (rather
        // than hard-coding `true`) so the value tracks the verdict if the precondition ever loosens.
        in_scope_safe: Some(verdict.in_scope_safe),
        judge_models: Vec::new(),
        harness_version: ctx.harness_version.clone(),
        git_commit: ctx.git_commit.clone(),
    }
}

/// The serialized snake_case spelling of a [`VerificationKind`](gw_schema::VerificationKind), via
/// serde so it always matches the wire enum (never a hand-maintained string table).
fn verification_kind_str(kind: gw_schema::VerificationKind) -> String {
    // The enum serializes to a bare JSON string; strip the surrounding quotes.
    serde_json::to_value(kind)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

/// Build the [`Cost`] from the teacher stream's usage block (token counts truncated to `u32`,
/// matching the schema; `reasoning_tokens > 0` is part of the Verify gate).
///
/// m6 (observability): when a turn that DID emit reasoning lands with NO `reasoning_tokens` (or no
/// `cost`) — usually because `gw-providers` coerced a type-drifted `usage` block to `None` — the
/// missing count is stamped as `0` here via `unwrap_or`, which could later fail-CLOSE the Verify
/// gate (`reasoning_tokens > 0`) on a genuinely good CoT. We emit a `tracing::warn!` so the
/// silent-zero is visible in the run log. (A record-level None-vs-zero distinction needs a schema
/// change and is out of scope.)
fn build_cost(turn: &AssistantTurn) -> Cost {
    let has_reasoning = turn
        .message
        .reasoning
        .as_deref()
        .is_some_and(|r| !r.is_empty())
        || turn
            .message
            .reasoning_details
            .as_ref()
            .is_some_and(|d| !d.is_empty());
    if has_reasoning && turn.reasoning_tokens.is_none() {
        tracing::warn!(
            generation_id = turn.generation_id.as_deref().unwrap_or("?"),
            "assistant turn has reasoning but usage.reasoning_tokens is missing; \
             cost.reasoning_tokens will be stamped 0 (may fail-close the Verify gate)"
        );
    }
    if has_reasoning && turn.cost.is_none() {
        tracing::warn!(
            generation_id = turn.generation_id.as_deref().unwrap_or("?"),
            "assistant turn has reasoning but usage.cost is missing; cost.usd will be stamped 0.0"
        );
    }
    Cost {
        prompt_tokens: turn.prompt_tokens.unwrap_or(0).min(u64::from(u32::MAX)) as u32,
        completion_tokens: turn.completion_tokens.unwrap_or(0).min(u64::from(u32::MAX)) as u32,
        reasoning_tokens: turn.reasoning_tokens.unwrap_or(0).min(u64::from(u32::MAX)) as u32,
        usd: turn.cost.unwrap_or(0.0),
        latency_ms: 0,
    }
}

/// Build the [`Lifecycle`] landed at `assistant_generated`, with a history that records the
/// `user_synthesized` → `assistant_generated` path (attempt 0). `at` is the injected timestamp.
fn build_lifecycle(at: &str) -> Lifecycle {
    Lifecycle {
        state: LifecycleState::AssistantGenerated,
        history: vec![
            StateTransition {
                state: LifecycleState::UserSynthesized,
                at: at.to_string(),
                attempt: 0,
            },
            StateTransition {
                state: LifecycleState::AssistantGenerated,
                at: at.to_string(),
                attempt: 0,
            },
        ],
        error: None,
        attempts: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::user_synth::{UserSeed, UserTurnCandidate, user_message};
    use gw_schema::{
        Content, Message, Oracle, ReasoningEffort, Role, VerificationContract, VerificationKind,
    };

    fn ctx() -> RecordContext {
        RecordContext {
            record_id: "01J8RECORD".into(),
            run_id: "run-1".into(),
            training_area: "rust-async".into(),
            harness_version: "0.1.0".into(),
            git_commit: Some("abc123".into()),
            now_rfc3339: "2026-06-21T00:00:00Z".into(),
            user_synth_model: Some("z-ai/glm-5.2".into()),
        }
    }

    fn gated(kind: VerificationKind, in_scope_safe: bool) -> GatedUserTurn {
        GatedUserTurn {
            candidate: UserTurnCandidate {
                message: user_message("What is 12 * 8?"),
                seed: UserSeed {
                    persona: Some("curious_user".into()),
                    taxonomy_node: Some("math.arithmetic".into()),
                    prompt_template_id: Some("magpie_v1".into()),
                    difficulty: Some("easy".into()),
                },
                contract: VerificationContract {
                    kind,
                    oracle: Oracle::None,
                    answer_marker: None,
                },
                answerable: true,
                difficulty_targeted: true,
                in_scope: in_scope_safe,
            },
            verdict: UserTurnVerdict {
                answerable: true,
                difficulty_targeted: true,
                diverse: true,
                in_scope_safe,
                notes: None,
            },
        }
    }

    fn turn() -> AssistantTurn {
        AssistantTurn {
            message: Message {
                role: Role::Assistant,
                content: Content::Text("96".into()),
                reasoning: Some("12*8=96".into()),
                reasoning_details: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            refusal: None,
            finish_reason: Some("stop".into()),
            served_by: Some("Parasail".into()),
            generation_id: Some("gen-xyz".into()),
            prompt_tokens: Some(20),
            completion_tokens: Some(200),
            reasoning_tokens: Some(150),
            cost: Some(0.0042),
        }
    }

    fn generation() -> Generation {
        Generation {
            reasoning_effort: Some(ReasoningEffort::Xhigh),
            temperature: Some(1.0),
            max_tokens: Some(8192),
            ..Default::default()
        }
    }

    #[test]
    fn lands_at_assistant_generated_with_full_history() {
        let g = gated(VerificationKind::NumericMatch, true);
        let rec = assemble(
            &ctx(),
            &g,
            turn(),
            TeacherRef {
                provider: "openrouter".into(),
                slug: "z-ai/glm-5.2".into(),
                served_by: None,
                model_card_revision: None,
            },
            generation(),
            None,
        );
        assert_eq!(rec.lifecycle.state, LifecycleState::AssistantGenerated);
        assert_eq!(rec.lifecycle.history.len(), 2);
        assert_eq!(
            rec.lifecycle.history[0].state,
            LifecycleState::UserSynthesized
        );
        assert_eq!(
            rec.lifecycle.history[1].state,
            LifecycleState::AssistantGenerated
        );
    }

    #[test]
    fn messages_are_user_then_assistant_with_reasoning_sibling() {
        let g = gated(VerificationKind::NumericMatch, true);
        let rec = assemble(&ctx(), &g, turn(), teacher_ref(), generation(), None);
        assert_eq!(rec.messages.len(), 2);
        assert_eq!(rec.messages[0].role, Role::User);
        assert_eq!(rec.messages[1].role, Role::Assistant);
        // INVARIANT-a: reasoning is a sibling, content is clean.
        assert_eq!(rec.messages[1].content, Content::Text("96".into()));
        assert_eq!(rec.messages[1].reasoning.as_deref(), Some("12*8=96"));
    }

    /// The producer's final step must MOVE the ingested turn, not rebuild it — otherwise a tool
    /// trajectory's `tool_calls` / result link would be silently dropped between ingest and the
    /// stored record (the "field added but never populated" failure mode).
    #[test]
    fn an_ingested_tool_turn_reaches_the_record_with_its_calls_intact() {
        let g = gated(VerificationKind::NumericMatch, true);
        let mut t = turn();
        t.message = Message {
            role: Role::Assistant,
            content: Content::Null,
            reasoning: Some("two reads needed".into()),
            reasoning_details: None,
            tool_calls: Some(vec![gw_schema::ToolCall {
                id: Some("read-a".into()),
                function: gw_schema::FunctionCall {
                    name: "read_file".into(),
                    arguments: serde_json::json!({"start_line": 1}),
                    raw_arguments: Some("{\"start_line\": 1}".into()),
                },
            }]),
            tool_call_id: None,
            name: None,
        };
        let rec = assemble(&ctx(), &g, t, teacher_ref(), generation(), None);
        let assistant = &rec.messages[1];
        assert_eq!(assistant.content, Content::Null);
        let calls = assistant
            .tool_calls
            .as_ref()
            .expect("calls survive assembly");
        assert_eq!(calls[0].id.as_deref(), Some("read-a"));
        assert_eq!(
            calls[0].function.raw_arguments.as_deref(),
            Some("{\"start_line\": 1}"),
            "the retained raw wire text survives assembly"
        );
    }

    #[test]
    fn provenance_mirrors_user_turn_kind_and_in_scope_safe() {
        let g = gated(VerificationKind::RefusalExpected, true);
        let rec = assemble(&ctx(), &g, turn(), teacher_ref(), generation(), None);
        // The serialized VerificationKind variant is mirrored (gw-judge reads this).
        assert_eq!(
            rec.provenance.user_turn_kind.as_deref(),
            Some("refusal_expected")
        );
        assert_eq!(rec.provenance.in_scope_safe, Some(true));
        assert_eq!(
            rec.provenance.user_synth_model.as_deref(),
            Some("z-ai/glm-5.2")
        );
        // served_by is captured from the stream onto the teacher ref.
        assert_eq!(
            rec.provenance.teacher.served_by.as_deref(),
            Some("Parasail")
        );
    }

    #[test]
    fn hashes_left_at_default_storage_is_authoritative() {
        let g = gated(VerificationKind::NumericMatch, true);
        let rec = assemble(&ctx(), &g, turn(), teacher_ref(), generation(), None);
        assert_eq!(rec.hashes, gw_schema::Hashes::default());
        assert!(rec.hashes.record_hash.is_empty());
        assert!(rec.hashes.prompt_hash.is_empty());
    }

    #[test]
    fn cost_and_seed_inputs_are_folded_in() {
        let g = gated(VerificationKind::NumericMatch, true);
        let rec = assemble(&ctx(), &g, turn(), teacher_ref(), generation(), None);
        assert_eq!(rec.cost.prompt_tokens, 20);
        assert_eq!(rec.cost.completion_tokens, 200);
        assert_eq!(rec.cost.reasoning_tokens, 150);
        assert_eq!(rec.cost.usd, 0.0042);
        // Seed inputs land in the reproducibility block.
        assert_eq!(rec.generation.persona.as_deref(), Some("curious_user"));
        assert_eq!(
            rec.generation.taxonomy_node.as_deref(),
            Some("math.arithmetic")
        );
        assert_eq!(
            rec.generation.prompt_template_id.as_deref(),
            Some("magpie_v1")
        );
    }

    #[test]
    fn sibling_stamps_indices_but_leaves_group_id_none() {
        let g = gated(VerificationKind::NumericMatch, true);
        let plan = SiblingPlan {
            completion_index: 2,
            n_completions: 4,
            sampling: crate::request::SamplingPreset::official().with_seed(2),
        };
        let rec = assemble(&ctx(), &g, turn(), teacher_ref(), generation(), Some(plan));
        assert_eq!(rec.generation.n_completions, Some(4));
        assert_eq!(rec.generation.completion_index, Some(2));
        // The group id is the engine's to fill from the canonical prompt_hash.
        assert_eq!(rec.generation.sibling_group_id, None);
    }

    #[test]
    fn record_round_trips_through_serde() {
        let g = gated(VerificationKind::NumericMatch, true);
        let rec = assemble(&ctx(), &g, turn(), teacher_ref(), generation(), None);
        let s = serde_json::to_string(&rec).unwrap();
        let back: TrainingRecord = serde_json::from_str(&s).unwrap();
        assert_eq!(rec, back);
    }

    fn teacher_ref() -> TeacherRef {
        TeacherRef {
            provider: "openrouter".into(),
            slug: "z-ai/glm-5.2".into(),
            served_by: None,
            model_card_revision: None,
        }
    }
}
