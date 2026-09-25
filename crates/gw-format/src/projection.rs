//! `projection` — project an admitted [`TrainingRecord`] down to a trainer's export shape.
//!
//! Two projections, both at the LAST moment (the envelope keeps full provenance; only here is it
//! reduced to a trainer's columns):
//!
//! - [`project_sft`] renders the record's messages into a `target` SFT shape under a [`CotPolicy`]
//!   and a [`MultiTurnLoss`], returning a [`SftProjection`] that pairs the rendered output with
//!   the loss/masking metadata a trainer needs.
//! - [`project_preference`] builds a [`PreferenceRecord`] (`{prompt, chosen, rejected}`) from an
//!   admitted record and a rejected sibling that shares the same `prompt_hash`.
//!
//! ## CotPolicy + MultiTurnLoss
//!
//! [`CotPolicy`] controls the reasoning region in the RENDERED bytes (see [`crate::render()`]).
//! [`MultiTurnLoss`] and the `Masked` case of [`CotPolicy`] are trainer-side LABEL concerns: v1
//! does not alter the rendered bytes for them, but [`SftProjection`] records them so the trainer /
//! manifest can apply the mask. This keeps the renderer a pure function of
//! `(clean messages + reasoning + target template)`.

use serde::{Deserialize, Serialize};

use gw_schema::{
    Content, CotPolicy, MultiTurnLoss, PreferenceRecord, PreferenceSide, Role, TrainingRecord,
    TrlFormat,
};

use crate::error::{FormatError, Result};
use crate::render::render;

/// An SFT projection: the rendered training text plus the loss/masking metadata a trainer needs.
///
/// The rendered bytes already reflect [`CotPolicy::Stripped`] (reasoning dropped) and render
/// `Supervised`/`Masked` identically; `cot_masked` and `multi_turn_loss` carry the trainer-side
/// label intent that v1 does NOT bake into the bytes (label masking is the trainer's job).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SftProjection {
    /// The target template this was rendered for.
    pub target: TrlFormat,
    /// The rendered output (a raw prompt for token targets; a JSON document for structured ones).
    pub rendered: String,
    /// The CoT policy applied to the reasoning region.
    pub cot_policy: CotPolicy,
    /// `true` when `cot_policy == Masked`: reasoning IS rendered but must be masked OUT of loss by
    /// the trainer. (v1 renders identically to `Supervised`.)
    pub cot_masked: bool,
    /// Which assistant turns enter the loss region (a trainer-side label concern).
    pub multi_turn_loss: MultiTurnLoss,
}

/// Project an admitted `record` into `target`'s SFT shape under `cot` + `multi_turn_loss`.
///
/// # Errors
///
/// Propagates [`crate::render()`] errors — including the tool guard
/// ([`FormatError::UnsupportedToolCalls`]: a tool trajectory projected onto a target with no tool
/// representation fails closed here, the same place it would fail at a direct `render` call) and the
/// prompt-completion final-turn guard — and returns [`FormatError::Projection`] if:
/// - the record has no assistant turn to supervise (nothing to learn from); or
/// - `target` is [`TrlFormat::TrlPromptCompletion`] with `multi_turn_loss == AllAssistant` and the
///   record has MORE THAN ONE assistant turn — prompt-completion can only supervise the FINAL
///   assistant turn, so `AllAssistant` is unsatisfiable for that shape and intermediate assistant
///   turns would silently fall outside loss. Use [`MultiTurnLoss::FinalTurnOnly`] instead.
pub fn project_sft(
    record: &TrainingRecord,
    target: TrlFormat,
    cot: CotPolicy,
    multi_turn_loss: MultiTurnLoss,
) -> Result<SftProjection> {
    let assistant_turns = record
        .messages
        .iter()
        .filter(|m| m.role == Role::Assistant)
        .count();
    if assistant_turns == 0 {
        return Err(FormatError::Projection(
            "record has no assistant turn to project".into(),
        ));
    }
    if target == TrlFormat::TrlPromptCompletion
        && multi_turn_loss == MultiTurnLoss::AllAssistant
        && assistant_turns > 1
    {
        return Err(FormatError::Projection(format!(
            "prompt-completion supervises only the final assistant turn, but \
             MultiTurnLoss::AllAssistant was requested with {assistant_turns} assistant turns \
             (use MultiTurnLoss::FinalTurnOnly)"
        )));
    }
    let rendered = render(&record.messages, target, cot)?;
    Ok(SftProjection {
        target,
        rendered,
        cot_policy: cot,
        cot_masked: cot == CotPolicy::Masked,
        multi_turn_loss,
    })
}

/// Build a DPO [`PreferenceRecord`] from an admitted `chosen` record and a `rejected` sibling.
///
/// Both records MUST share the same `hashes.prompt_hash` (the pairing key / sibling-group id) and
/// have a final assistant turn. The shared prompt is taken from `chosen` (the turns before its
/// final assistant turn). Each side's `content` is the clean final answer; `reasoning` rides as an
/// optional sibling under `cot` ([`CotPolicy::Stripped`] drops it), NEVER inlined (INVARIANT-a).
///
/// # Errors
///
/// Returns [`FormatError::Projection`] if the two records do not share a non-empty `prompt_hash`,
/// or if either lacks a final assistant turn.
pub fn project_preference(
    chosen: &TrainingRecord,
    rejected: &TrainingRecord,
    cot: CotPolicy,
) -> Result<PreferenceRecord> {
    let prompt_hash = &chosen.hashes.prompt_hash;
    if prompt_hash.is_empty() || *prompt_hash != rejected.hashes.prompt_hash {
        return Err(FormatError::Projection(format!(
            "chosen/rejected must share a non-empty prompt_hash (chosen={:?}, rejected={:?})",
            chosen.hashes.prompt_hash, rejected.hashes.prompt_hash
        )));
    }

    let (chosen_idx, chosen_side) = final_assistant_side(chosen, cot)?;
    let (_, rejected_side) = final_assistant_side(rejected, cot)?;

    let prompt = chosen.messages[..chosen_idx].to_vec();

    Ok(PreferenceRecord {
        prompt,
        chosen: chosen_side,
        rejected: rejected_side,
        prompt_hash: prompt_hash.clone(),
    })
}

/// The index of the final assistant turn and its [`PreferenceSide`] (clean content + sibling
/// reasoning under `cot` + the record id + the bias-corrected aggregate).
fn final_assistant_side(
    record: &TrainingRecord,
    cot: CotPolicy,
) -> Result<(usize, PreferenceSide)> {
    let idx = record
        .messages
        .iter()
        .rposition(|m| m.role == Role::Assistant)
        .ok_or_else(|| {
            FormatError::Projection(format!("record {} has no assistant turn", record.record_id))
        })?;
    let msg = &record.messages[idx];
    let reasoning = match cot {
        CotPolicy::Stripped => None,
        CotPolicy::Supervised | CotPolicy::Masked => msg.reasoning.clone(),
    };
    let side = PreferenceSide {
        content: content_text(&msg.content),
        reasoning,
        record_id: record.record_id.clone(),
        aggregate: record.judging.aggregate,
    };
    Ok((idx, side))
}

/// The clean text of a [`Content`] (mirrors [`crate::render()`]'s flattening). An explicitly
/// absent value ([`Content::Null`]) flattens to the empty string, matching the render targets.
fn content_text(content: &Content) -> String {
    match content {
        Content::Text(s) => s.clone(),
        Content::Parts(parts) => parts
            .iter()
            .filter_map(|p| match p {
                gw_schema::ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(""),
        Content::Null => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gw_schema::{
        FunctionCall, Generation, Hashes, Judging, Lifecycle, Message, Provenance, TeacherRef,
        ToolCall, Verification,
    };

    fn msg(role: Role, content: &str, reasoning: Option<&str>) -> Message {
        Message {
            role,
            content: Content::Text(content.into()),
            reasoning: reasoning.map(str::to_owned),
            reasoning_details: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    fn record(
        id: &str,
        prompt_hash: &str,
        answer: &str,
        reasoning: &str,
        agg: f64,
    ) -> TrainingRecord {
        TrainingRecord {
            record_id: id.into(),
            schema_version: semver::Version::new(1, 0, 0),
            dataset_version: None,
            training_area: "t".into(),
            tags: vec![],
            messages: vec![
                msg(Role::User, "q", None),
                msg(Role::Assistant, answer, Some(reasoning)),
            ],
            tools: None,
            provenance: Provenance {
                run_id: "r".into(),
                parent_ids: vec![],
                teacher: TeacherRef {
                    provider: "openrouter".into(),
                    slug: "z-ai/glm-5.2".into(),
                    served_by: None,
                    model_card_revision: None,
                },
                user_synth_model: None,
                user_turn_kind: None,
                in_scope_safe: None,
                judge_models: vec![],
                harness_version: "0.1.0".into(),
                git_commit: None,
            },
            generation: Generation::default(),
            verification_contract: None,
            verification: Verification::default(),
            judging: Judging {
                aggregate: Some(agg),
                ..Default::default()
            },
            reasoning_quality: None,
            lifecycle: Lifecycle::default(),
            hashes: Hashes {
                prompt_hash: prompt_hash.into(),
                ..Default::default()
            },
            cost: Default::default(),
        }
    }

    #[test]
    fn sft_projection_carries_masking_metadata() {
        let rec = record("a", "h1", "96", "12*8", 0.9);
        let p = project_sft(
            &rec,
            TrlFormat::Gemma4,
            CotPolicy::Masked,
            MultiTurnLoss::AllAssistant,
        )
        .unwrap();
        assert!(p.cot_masked);
        assert_eq!(p.cot_policy, CotPolicy::Masked);
        assert!(p.rendered.contains("<|channel>thought"));
    }

    #[test]
    fn sft_stripped_drops_reasoning() {
        let rec = record("a", "h1", "96", "12*8", 0.9);
        let p = project_sft(
            &rec,
            TrlFormat::ChatML,
            CotPolicy::Stripped,
            MultiTurnLoss::default(),
        )
        .unwrap();
        assert!(!p.rendered.contains("<think>"));
        assert!(!p.cot_masked);
    }

    #[test]
    fn preference_pairs_on_shared_prompt_hash() {
        let chosen = record("a", "h1", "96", "good cot", 0.9);
        let rejected = record("b", "h1", "97", "bad cot", 0.3);
        let pref = project_preference(&chosen, &rejected, CotPolicy::Supervised).unwrap();
        assert_eq!(pref.prompt_hash, "h1");
        assert_eq!(pref.chosen.content, "96");
        assert_eq!(pref.rejected.content, "97");
        assert_eq!(pref.chosen.reasoning.as_deref(), Some("good cot"));
        assert_eq!(pref.chosen.aggregate, Some(0.9));
        assert_eq!(pref.prompt.len(), 1);
        assert_eq!(pref.prompt[0].role, Role::User);
    }

    #[test]
    fn preference_stripped_drops_reasoning() {
        let chosen = record("a", "h1", "96", "good cot", 0.9);
        let rejected = record("b", "h1", "97", "bad cot", 0.3);
        let pref = project_preference(&chosen, &rejected, CotPolicy::Stripped).unwrap();
        assert!(pref.chosen.reasoning.is_none());
        assert!(pref.rejected.reasoning.is_none());
    }

    #[test]
    fn preference_mismatched_prompt_hash_errors() {
        let chosen = record("a", "h1", "96", "c", 0.9);
        let rejected = record("b", "h2", "97", "c", 0.3);
        assert!(matches!(
            project_preference(&chosen, &rejected, CotPolicy::Supervised),
            Err(FormatError::Projection(_))
        ));
    }

    /// A two-assistant-turn record (user/assistant/user/assistant), prompt-completion-shaped.
    fn two_turn_record() -> TrainingRecord {
        let mut rec = record("a", "h1", "first", "r1", 0.9);
        rec.messages = vec![
            msg(Role::User, "q1", None),
            msg(Role::Assistant, "first", Some("r1")),
            msg(Role::User, "q2", None),
            msg(Role::Assistant, "second", Some("r2")),
        ];
        rec
    }

    #[test]
    fn sft_prompt_completion_all_assistant_multi_turn_errors() {
        // AllAssistant (the schema default) is unsatisfiable for prompt-completion with >1
        // assistant turn — it can only supervise the FINAL assistant turn.
        let rec = two_turn_record();
        let err = project_sft(
            &rec,
            TrlFormat::TrlPromptCompletion,
            CotPolicy::Supervised,
            MultiTurnLoss::AllAssistant,
        )
        .unwrap_err();
        assert!(matches!(err, FormatError::Projection(_)));
        assert!(err.to_string().contains("final assistant turn"));
    }

    #[test]
    fn sft_prompt_completion_final_turn_only_multi_turn_ok() {
        // FinalTurnOnly is exactly what prompt-completion supports.
        let rec = two_turn_record();
        let p = project_sft(
            &rec,
            TrlFormat::TrlPromptCompletion,
            CotPolicy::Supervised,
            MultiTurnLoss::FinalTurnOnly,
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&p.rendered).unwrap();
        assert_eq!(v["prompt"].as_array().unwrap().len(), 3);
        assert_eq!(v["completion"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn sft_conversational_all_assistant_multi_turn_ok() {
        // The guard is prompt-completion-specific: conversational targets supervise all
        // assistant turns natively, so AllAssistant + multi-turn is fine.
        let rec = two_turn_record();
        assert!(
            project_sft(
                &rec,
                TrlFormat::OpenAiMessages,
                CotPolicy::Supervised,
                MultiTurnLoss::AllAssistant,
            )
            .is_ok()
        );
    }

    /// The real pipeline route: an admitted record with a tool trajectory, projected for export.
    /// A target that cannot represent it must fail closed HERE, not hand a trainer bytes in which
    /// the tool turn has become prose.
    #[test]
    fn sft_projection_fails_closed_on_a_tool_trajectory_for_a_dropping_target() {
        let mut rec = record("a", "h1", "96", "12*8", 0.9);
        rec.messages = vec![
            msg(Role::User, "read two regions", None),
            Message {
                role: Role::Assistant,
                content: Content::Null,
                reasoning: Some("two reads".into()),
                reasoning_details: None,
                tool_calls: Some(vec![ToolCall {
                    id: Some("read-a".into()),
                    function: FunctionCall {
                        name: "read_file".into(),
                        arguments: serde_json::json!({"start_line": 1}),
                        raw_arguments: None,
                    },
                }]),
                tool_call_id: None,
                name: None,
            },
            Message {
                role: Role::Tool,
                content: Content::Text("ok".into()),
                reasoning: None,
                reasoning_details: None,
                tool_calls: None,
                tool_call_id: Some("read-a".into()),
                name: Some("read_file".into()),
            },
            msg(Role::Assistant, "the second read failed", Some("report it")),
        ];
        for target in [
            TrlFormat::Gemma4,
            TrlFormat::ChatML,
            TrlFormat::ShareGpt,
            TrlFormat::Harmony,
        ] {
            let err = project_sft(
                &rec,
                target,
                CotPolicy::Supervised,
                MultiTurnLoss::AllAssistant,
            )
            .unwrap_err();
            assert!(
                matches!(
                    err,
                    FormatError::UnsupportedToolCalls {
                        target: t,
                        ref signals,
                        index: 1,
                        ..
                    } if t == target
                        && signals.contains(&crate::ToolSignal::ToolCalls)
                        && signals.contains(&crate::ToolSignal::ToolRole)
                ),
                "{target:?}: {err:?}"
            );
        }
        // The same record still projects for a tool-faithful target.
        assert!(
            project_sft(
                &rec,
                TrlFormat::OpenAiMessages,
                CotPolicy::Supervised,
                MultiTurnLoss::AllAssistant,
            )
            .is_ok()
        );
    }
}
