//! `projection` — project an admitted [`TrainingRecord`] down to a trainer's export shape.
//!
//! [`project_sft`] renders messages at the export boundary while the source envelope keeps full
//! provenance. Its [`SftProjection`] also carries the trainer's loss and masking intent.
//! Preference structural projection lives in [`crate::project_preference_messages`]; complete
//! evidence validation is owned by the engine preparation API.
//!
//! ## CotPolicy + MultiTurnLoss
//!
//! [`CotPolicy`] controls the reasoning region in the RENDERED bytes (see [`crate::render()`]).
//! [`MultiTurnLoss`] and the `Masked` case of [`CotPolicy`] are trainer-side LABEL concerns: v1
//! does not alter the rendered bytes for them, but [`SftProjection`] records them so the trainer /
//! manifest can apply the mask. This keeps the renderer a pure function of
//! `(clean messages + reasoning + target template)`.

use serde::{Deserialize, Serialize};

use gw_schema::{CotPolicy, MultiTurnLoss, Role, TrainingRecord, TrlFormat};

use crate::error::{FormatError, Result};
use crate::render::render;

/// One complete emitted training example ending at an assistant selected for loss. This pure
/// prefix layout matches `assistant_prefix_v1`; final-only contributes only the final target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SftTrainingUnit {
    /// Index of the selected assistant in the original ordered conversation.
    pub target_index: usize,
    /// Canonical renderer output for the entire prefix including that target.
    pub rendered: String,
}

/// Render each selected complete training unit through the shared renderer. This does not
/// tokenize, assign labels, certify model-specific supervision, or check record admission.
///
/// # Errors
/// Rejects an empty/nonterminal conversation or any prefix rejected by the canonical renderer.
pub fn project_sft_units(
    record: &TrainingRecord,
    target: TrlFormat,
    cot: CotPolicy,
    turns: MultiTurnLoss,
) -> Result<Vec<SftTrainingUnit>> {
    let indices = gw_schema::sft_source_targets(&record.messages, target, turns)?;
    indices
        .into_iter()
        .map(|target_index| {
            Ok(SftTrainingUnit {
                target_index,
                rendered: render(&record.messages[..=target_index], target, cot)?,
            })
        })
        .collect()
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use gw_schema::{
        Content, FunctionCall, Generation, Hashes, Judging, Lifecycle, Message, Provenance,
        TeacherRef, ToolCall, Verification,
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
            task_provenance: None,
            verification_contract: None,
            execution_evidence: None,
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
