//! `render` — project clean `{messages, reasoning}` turns into model-specific chat templates.
//!
//! [`render`] is the single entry point: it dispatches on [`TrlFormat`] and applies the
//! [`CotPolicy`] (whether `reasoning` enters the loss/thought region) uniformly across targets.
//!
//! ## The round-trip contract (INVARIANT-a)
//!
//! The stored `content` is ALWAYS clean final-answer text. Channel tokens (`<think>`,
//! `<|channel>`, `<turn|>`, …) are PRODUCED here by the renderer from the separate `reasoning`
//! field — never read out of `content`. Before rendering ANY target, [`render`] calls
//! [`validate_clean`](crate::validate) and FAILS LOUD with [`FormatError::ControlTokenInContent`]
//! if a clean field already contains a control token (an upstream leak that would not round-trip).
//!
//! The render→ingest **identity** holds for the TOKEN-STREAM targets only
//! ([`Gemma4`](TrlFormat::Gemma4), [`ChatML`](TrlFormat::ChatML), [`Harmony`](TrlFormat::Harmony)):
//! there, [`crate::ingest`] re-extracts the reasoning the renderer framed into `content`. The
//! STRUCTURED targets ([`ShareGpt`](TrlFormat::ShareGpt),
//! [`OpenAiMessages`](TrlFormat::OpenAiMessages),
//! [`TrlPromptCompletion`](TrlFormat::TrlPromptCompletion)) carry `reasoning` in a CLEAN sibling
//! key (lossless by construction, not via the channel-stripping ingest path). The identity also
//! excludes clean content that itself begins with a channel marker — but `validate_clean` rejects
//! such content up front, so that case never reaches a renderer.
//!
//! ## CotPolicy semantics (uniform across targets)
//!
//! - [`Supervised`](CotPolicy::Supervised): render `reasoning` into the loss/thought region.
//! - [`Masked`](CotPolicy::Masked): render `reasoning` IDENTICALLY to `Supervised`. Label
//!   masking is a trainer-side concern; the [`crate::projection`] manifest records that the
//!   region is masked. v1 does not change the rendered bytes for `Masked`.
//! - [`Stripped`](CotPolicy::Stripped): drop `reasoning`; render the answer only (an empty
//!   thought wrapper for Gemma-4).
//!
//! ## The tool boundary: fail closed, never drop silently (INVARIANT i)
//!
//! Four targets have no representation for a tool trajectory: [`Gemma4`](TrlFormat::Gemma4),
//! [`ChatML`](TrlFormat::ChatML), [`ShareGpt`](TrlFormat::ShareGpt) and
//! [`Harmony`](TrlFormat::Harmony). Rendering one there would emit prose in which the call, its
//! arguments and — worst — WHICH call each tool result answers have all been dropped: a training
//! set whose tool turn is indistinguishable from an answer. So [`render`] refuses FIRST, before
//! any target dispatch, with [`FormatError::UnsupportedToolCalls`] naming the route, the dropped
//! signal, the first offending message and the recovery path.
//!
//! The two tool-faithful routes are [`OpenAiMessages`](TrlFormat::OpenAiMessages) (the OpenAI wire
//! shape) and [`TrlPromptCompletion`](TrlFormat::TrlPromptCompletion) (whose `prompt` /
//! `completion` turns are OpenAI message objects). Both keep `tool_calls`, `tool_call_id` and
//! `name` verbatim, so a tool trajectory round-trips through them with its result links intact.
//! Identity admission for a complete trajectory stays the separate, explicit
//! [`validate_tool_links`](crate::validate_tool_links) step — the guard here is a render-boundary
//! check, not a substitute for it.

mod chatml;
mod gemma4;
mod harmony;
mod openai;
mod prompt_completion;
mod sharegpt;

use gw_schema::{Content, CotPolicy, Message, Role, TrlFormat};

use crate::error::{FormatError, Result, ToolCallRecovery, ToolSignal};

/// The targets that carry every tool signal VERBATIM: the OpenAI message objects and the TRL
/// prompt/completion shape, whose `prompt` / `completion` turns ARE OpenAI message objects. They
/// are never refused by the tool guard.
///
/// The remaining targets ([`Gemma4`](TrlFormat::Gemma4), [`ChatML`](TrlFormat::ChatML),
/// [`ShareGpt`](TrlFormat::ShareGpt), [`Harmony`](TrlFormat::Harmony)) have no slot for
/// `tool_calls`, no `tool_call_id` and no tool-turn semantics, so a tool trajectory must not be
/// rendered into them.
const fn preserves_tool_signals(target: TrlFormat) -> bool {
    matches!(
        target,
        TrlFormat::OpenAiMessages | TrlFormat::TrlPromptCompletion
    )
}

/// The first message carrying a tool signal, plus the FULL set of signals the conversation carries.
///
/// Returns `None` for a conversation that declares no tool fields at all (the text-only case, which
/// must keep rendering exactly as before on every target).
fn first_tool_violation(messages: &[Message]) -> Option<(usize, Vec<ToolSignal>)> {
    let mut first: Option<usize> = None;
    let mut signals: Vec<ToolSignal> = Vec::new();
    for (index, msg) in messages.iter().enumerate() {
        let mut here: Vec<ToolSignal> = Vec::new();
        if msg.tool_calls.is_some() {
            here.push(ToolSignal::ToolCalls);
        }
        if msg.tool_call_id.is_some() {
            here.push(ToolSignal::ToolCallId);
        }
        if msg.role == Role::Tool {
            // A result turn exists only to answer a call. On a dropping target it is emitted as an
            // ordinary text turn, so which call it answers stops being recoverable.
            here.push(ToolSignal::ToolRole);
        }
        for signal in here {
            first.get_or_insert(index);
            if !signals.contains(&signal) {
                signals.push(signal);
            }
        }
    }
    first.map(|index| (index, signals))
}

/// Render `messages` into `target`, applying `cot` to the reasoning region.
///
/// The output is a single `String`: a raw template-rendered prompt for the token-stream targets
/// ([`Gemma4`](TrlFormat::Gemma4), [`ChatML`](TrlFormat::ChatML), [`Harmony`](TrlFormat::Harmony))
/// and a pretty-printed JSON document for the structured targets
/// ([`ShareGpt`](TrlFormat::ShareGpt), [`OpenAiMessages`](TrlFormat::OpenAiMessages),
/// [`TrlPromptCompletion`](TrlFormat::TrlPromptCompletion)).
///
/// # Fail closed on a tool trajectory (INVARIANT i)
///
/// The guard runs BEFORE any target dispatch. If `target` is not a tool-faithful route
/// (`OpenAiMessages` / `TrlPromptCompletion`) and any message declares `tool_calls`, carries a
/// `tool_call_id`, or is a [`Role::Tool`] turn, this returns
/// [`crate::FormatError::UnsupportedToolCalls`] instead of emitting a training target in which the
/// tool trajectory has been flattened into prose. A text-only conversation declares no tool fields,
/// so every target still renders exactly as before.
///
/// # Content policy (per target, uniform in `content_text`)
///
/// - [`Content::Text`] is the body as-is.
/// - [`Content::Parts`] flattens to its concatenated TEXT segments; an image/audio-only turn has
///   no training text and renders an empty body.
/// - [`Content::Null`] flattens to the empty string: every target frames content as a string. A
///   tool-calling turn (whose only output is `tool_calls`) is the case this matters for — it is
///   exactly what the tool guard above refuses on a dropping target, and on
///   [`OpenAiMessages`](TrlFormat::OpenAiMessages) it renders an empty body while the calls ride
///   alongside it. The null-vs-empty distinction is preserved in the stored record and in the
///   canonical `messages_json` export column, not in the rendered bytes.
/// - A `system` / `developer` turn is rendered by each target per its own contract (only
///   [`Gemma4`](TrlFormat::Gemma4) folds it, into the following `user` turn); the other targets
///   keep it as a distinct role turn.
///
/// # Errors
///
/// Returns [`crate::FormatError::UnsupportedToolCalls`] if `target` has no tool representation and
/// `messages` carries a tool signal (see above), [`crate::FormatError::ControlTokenInContent`] if a
/// message's clean `content`/`reasoning` already contains a chat control token (an upstream leak;
/// validated BEFORE any rendering), [`crate::FormatError::Template`] if the Gemma-4 template fails
/// to render, [`crate::FormatError::Serde`] if a structured target fails to serialize, and
/// [`crate::FormatError::UnsupportedRole`] if a message carries a role the target cannot place.
pub fn render(messages: &[Message], target: TrlFormat, cot: CotPolicy) -> Result<String> {
    crate::validate::validate_clean(messages)?;
    if !preserves_tool_signals(target)
        && let Some((index, signals)) = first_tool_violation(messages)
    {
        return Err(FormatError::UnsupportedToolCalls {
            target,
            signals,
            index,
            recovery: ToolCallRecovery::CanonicalExportAndOfficialTemplate,
        });
    }
    match target {
        TrlFormat::Gemma4 => gemma4::render(messages, cot),
        TrlFormat::ChatML => chatml::render(messages, cot),
        TrlFormat::ShareGpt => sharegpt::render(messages, cot),
        TrlFormat::OpenAiMessages => openai::render(messages, cot),
        TrlFormat::Harmony => harmony::render(messages, cot),
        TrlFormat::TrlPromptCompletion => prompt_completion::render(messages, cot),
    }
}

/// The clean text of a [`Content`]. `Parts` are flattened to their concatenated text segments
/// (image/audio refs carry no training text), so callers always see leak-free final-answer text.
/// [`Content::Null`] flattens to the empty string: every render target frames content as a
/// string, so a tool-calling turn (whose only output is `tool_calls`) renders an empty body. The
/// null-vs-empty distinction is preserved in the stored record, not in the rendered bytes.
pub(crate) fn content_text(content: &Content) -> String {
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

/// The reasoning that should be rendered for `msg` under `cot`.
///
/// - [`Stripped`](CotPolicy::Stripped) always yields `None` (answer-only).
/// - [`Supervised`](CotPolicy::Supervised) / [`Masked`](CotPolicy::Masked) yield the message's
///   `reasoning` (v1 renders `Masked` identically to `Supervised`; the masking is recorded in the
///   projection manifest, not the bytes).
///
/// An empty-string `reasoning` is treated as present (so a deliberately-empty thought channel
/// round-trips); only `None` and `Stripped` collapse to "no reasoning".
pub(crate) fn effective_reasoning(msg: &Message, cot: CotPolicy) -> Option<&str> {
    match cot {
        CotPolicy::Stripped => None,
        CotPolicy::Supervised | CotPolicy::Masked => msg.reasoning.as_deref(),
    }
}

/// `true` if `role` is an assistant turn (`assistant`). Tool/user/system are not.
pub(crate) fn is_assistant(role: Role) -> bool {
    role == Role::Assistant
}

#[cfg(test)]
mod tests {
    use super::*;
    use gw_schema::ContentPart;

    #[test]
    fn content_text_flattens_parts() {
        let c = Content::Parts(vec![
            ContentPart::Text { text: "a".into() },
            ContentPart::ImageUrl {
                image_url: "x".into(),
            },
            ContentPart::Text { text: "b".into() },
        ]);
        assert_eq!(content_text(&c), "ab");
    }

    #[test]
    fn effective_reasoning_respects_policy() {
        let msg = Message {
            role: Role::Assistant,
            content: Content::Text("ans".into()),
            reasoning: Some("cot".into()),
            reasoning_details: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        assert_eq!(
            effective_reasoning(&msg, CotPolicy::Supervised),
            Some("cot")
        );
        assert_eq!(effective_reasoning(&msg, CotPolicy::Masked), Some("cot"));
        assert_eq!(effective_reasoning(&msg, CotPolicy::Stripped), None);
    }

    /// A text-only conversation declares no tool fields, so the guard must find nothing.
    #[test]
    fn a_text_only_conversation_has_no_violation() {
        let msgs = vec![Message {
            role: Role::Assistant,
            content: Content::Text("96".into()),
            reasoning: Some("12*8".into()),
            reasoning_details: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];
        assert_eq!(first_tool_violation(&msgs), None);
        assert_eq!(first_tool_violation(&[]), None);
    }

    /// Each of the three signals trips the guard on its own — the trigger is their union, and a
    /// bare `Role::Tool` turn (no call, no id) is exactly the case a "tool_calls only" check
    /// would miss while still destroying the pairing.
    #[test]
    fn each_signal_alone_reports_its_own_value_at_its_own_index() {
        let base = Message {
            role: Role::User,
            content: Content::Text("q".into()),
            reasoning: None,
            reasoning_details: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let cases: Vec<(Vec<Message>, usize, ToolSignal)> = vec![
            (
                vec![Message {
                    tool_calls: Some(vec![]),
                    ..base.clone()
                }],
                0,
                ToolSignal::ToolCalls,
            ),
            (
                vec![Message {
                    tool_call_id: Some("read-a".into()),
                    ..base.clone()
                }],
                0,
                ToolSignal::ToolCallId,
            ),
            (
                vec![Message {
                    role: Role::Tool,
                    ..base.clone()
                }],
                0,
                ToolSignal::ToolRole,
            ),
        ];
        for (msgs, index, expected) in cases {
            let (got_index, signals) = first_tool_violation(&msgs).expect("a signal is present");
            assert_eq!(got_index, index);
            assert_eq!(signals, vec![expected]);
        }
    }

    /// The diagnostic reports the FIRST offending message and the FULL signal set, deduplicated
    /// and in a stable order (so the message is reproducible).
    #[test]
    fn the_diagnostic_reports_the_first_index_and_every_signal() {
        let base = Message {
            role: Role::Assistant,
            content: Content::Null,
            reasoning: None,
            reasoning_details: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        let msgs = vec![
            base.clone(),
            Message {
                tool_calls: Some(vec![]),
                ..base.clone()
            },
            Message {
                role: Role::Tool,
                tool_call_id: Some("read-b".into()),
                ..base
            },
        ];
        let (index, signals) = first_tool_violation(&msgs).expect("signals present");
        assert_eq!(index, 1);
        assert_eq!(
            signals,
            vec![
                ToolSignal::ToolCalls,
                ToolSignal::ToolCallId,
                ToolSignal::ToolRole
            ]
        );
    }

    /// Exactly the two OpenAI-message routes are exempt; the other four are not.
    #[test]
    fn only_the_openai_message_routes_preserve_tool_signals() {
        for target in [TrlFormat::OpenAiMessages, TrlFormat::TrlPromptCompletion] {
            assert!(
                preserves_tool_signals(target),
                "{target:?} must stay unguarded"
            );
        }
        for target in [
            TrlFormat::Gemma4,
            TrlFormat::ChatML,
            TrlFormat::ShareGpt,
            TrlFormat::Harmony,
        ] {
            assert!(
                !preserves_tool_signals(target),
                "{target:?} must fail closed on a tool trajectory"
            );
        }
    }
}
