//! [`FormatError`] — the single error type surfaced by every `gw-format` operation.
//!
//! Variants distinguish the failure classes a caller must reason about: a minijinja template
//! compile/render fault (`Template`), JSON (de)serialization of the ingest/export shapes
//! (`Serde`), a role the target template cannot represent (`UnsupportedRole`), a control token
//! leaked into a clean field (`ControlTokenInContent`, the loud round-trip guard), a tool
//! trajectory routed to a target with no tool representation (`UnsupportedToolCalls`, the
//! fail-closed render-boundary guard), a malformed ingest payload (`Ingest`), a projection
//! precondition violation (`Projection`), and a tool-call/result identity violation
//! (`ToolIdentity`). No `anyhow` — this crate surfaces a typed error.

use std::fmt;

use thiserror::Error;

use gw_schema::{Role, TrlFormat};

/// Everything that can go wrong rendering, ingesting, or projecting a record.
///
/// `#[non_exhaustive]` so new variants can be added without a breaking change. `Template` and
/// `Serde` carry their underlying source via `#[from]`; the rest are constructed directly with a
/// human-readable message.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FormatError {
    /// A minijinja template failed to compile or render (e.g. the Gemma-4 template).
    #[error("template error: {0}")]
    Template(#[from] minijinja::Error),

    /// (De)serializing an ingest payload or an export shape (`serde_json::Value` / `String`)
    /// failed.
    #[error("serde_json error: {0}")]
    Serde(#[from] serde_json::Error),

    /// The target template cannot represent this [`Role`] (e.g. a stray `developer` turn a
    /// format does not map). Carries the offending role and the target name.
    #[error("unsupported role `{role:?}` for target `{target}`")]
    UnsupportedRole {
        /// The role the template could not place.
        role: Role,
        /// The target format name (e.g. `"sharegpt"`).
        target: &'static str,
    },

    /// A tool-bearing conversation was routed to a target template that has NO representation for
    /// it: [`Gemma4`](TrlFormat::Gemma4), [`ChatML`](TrlFormat::ChatML),
    /// [`ShareGpt`](TrlFormat::ShareGpt) and [`Harmony`](TrlFormat::Harmony) emit no
    /// `tool_calls`, no `tool_call_id` and no tool-turn semantics, so rendering there would
    /// silently turn a tool trajectory into plain prose — a training target whose tool turn looks
    /// like an answer. Failing loud beats shipping that. The targets that DO carry the signals
    /// ([`OpenAiMessages`](TrlFormat::OpenAiMessages),
    /// [`TrlPromptCompletion`](TrlFormat::TrlPromptCompletion)) are never rejected here.
    ///
    /// Every field is structured so a caller can act on it programmatically: `target` is the
    /// refused route, `signals` names WHAT would have been dropped, `index` is the first message
    /// that carries a signal, and `recovery` is the supported way to obtain a tool-faithful
    /// target. Assert on those fields — not on the rendered message.
    #[error(
        "unsupported tool calls for target `{target:?}`: messages[{index}] carries {signals:?}, \
         which that template cannot represent; {recovery}"
    )]
    UnsupportedToolCalls {
        /// The route that was asked for and refuses the trajectory.
        target: TrlFormat,
        /// Every signal present in the conversation (deduplicated, in a stable order) — what the
        /// render would have dropped.
        signals: Vec<ToolSignal>,
        /// The index of the FIRST message carrying a signal.
        index: usize,
        /// The supported way to get a tool-faithful target instead.
        recovery: ToolCallRecovery,
    },

    /// A message's CLEAN field (`content` or `reasoning`) already contains a chat control token
    /// (e.g. `<think>`, `<|channel>`, `<turn|>`). Per INVARIANT-a these fields must be CLEAN — a
    /// control token there means an upstream stage leaked channel markup and the render would not
    /// round-trip. Failing loud beats silently corrupting the training target. Carries the
    /// offending token and the role of the message it was found on.
    #[error("control token `{token}` found in clean field of `{role:?}` message")]
    ControlTokenInContent {
        /// The control token that leaked into a clean field.
        token: &'static str,
        /// The role of the message carrying the leak.
        role: Role,
    },

    /// An OpenRouter / OpenAI ingest payload was malformed (missing `message`, wrong type, …).
    #[error("ingest error: {0}")]
    Ingest(String),

    /// A projection precondition was violated (e.g. a DPO pair whose sides do not share a
    /// `prompt_hash`, or a record with no assistant turn to project).
    #[error("projection error: {0}")]
    Projection(String),

    /// A tool-call / tool-result identity violation (INVARIANT i): a dangling or duplicate call
    /// id, a result arriving before its call, or a result whose link is ambiguous because several
    /// same-named calls are declared. Carries the offending message index and the reason — the
    /// check never guesses a link from a function name.
    #[error("tool identity error: {0}")]
    ToolIdentity(String),
}

/// One conversation signal a tool-dropping target cannot represent (INVARIANT i).
///
/// These are the three independent carriers of a tool trajectory, kept as distinct values so the
/// diagnostic says exactly what the refused render would have lost — and so a test can assert the
/// trigger set rather than a prose sentence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolSignal {
    /// An assistant turn declares `tool_calls` (the call itself: name + arguments).
    ToolCalls,
    /// A turn carries a `tool_call_id` — the explicit result link (INVARIANT i).
    ToolCallId,
    /// A turn is a [`Role::Tool`] result turn, which exists only to answer a call. On a dropping
    /// target it becomes an ordinary `user` / `tool` text turn, so the pairing is gone.
    ToolRole,
}

/// How to obtain a tool-faithful training target when a target template refuses the trajectory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallRecovery {
    /// Export the canonical `messages_json` conversation — the lossless column that keeps
    /// `tool_calls`, the `tool_call_id` links, `content: null` and `name` — and apply the model's
    /// OFFICIAL chat template in an external consumer that owns that template.
    CanonicalExportAndOfficialTemplate,
}

impl fmt::Display for ToolCallRecovery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CanonicalExportAndOfficialTemplate => f.write_str(
                "export the canonical `messages_json` conversation (tool calls and result links \
                 intact) and apply the model's official chat template in an external consumer",
            ),
        }
    }
}

/// Convenience alias for results returned by `gw-format` operations.
pub type Result<T> = std::result::Result<T, FormatError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_role_names_role_and_target() {
        let e = FormatError::UnsupportedRole {
            role: Role::Developer,
            target: "sharegpt",
        };
        let msg = e.to_string();
        assert!(msg.contains("Developer"));
        assert!(msg.contains("sharegpt"));
    }

    #[test]
    fn serde_error_converts() {
        let bad = serde_json::from_str::<i32>("not json").unwrap_err();
        let e: FormatError = bad.into();
        assert!(matches!(e, FormatError::Serde(_)));
    }

    #[test]
    fn control_token_names_token_and_role() {
        let e = FormatError::ControlTokenInContent {
            token: "<think>",
            role: Role::Assistant,
        };
        let msg = e.to_string();
        assert!(msg.contains("<think>"));
        assert!(msg.contains("Assistant"));
    }

    #[test]
    fn ingest_and_projection_render_message() {
        assert!(
            FormatError::Ingest("no message".into())
                .to_string()
                .contains("no message")
        );
        assert!(
            FormatError::Projection("prompt_hash mismatch".into())
                .to_string()
                .contains("mismatch")
        );
    }

    #[test]
    fn tool_identity_renders_message() {
        let e = FormatError::ToolIdentity("messages[2] is a dangling result link".into());
        assert!(e.to_string().contains("tool identity error"));
        assert!(e.to_string().contains("messages[2]"));
    }

    #[test]
    fn unsupported_tool_calls_exposes_structured_fields_and_a_human_message() {
        let e = FormatError::UnsupportedToolCalls {
            target: TrlFormat::Gemma4,
            signals: vec![ToolSignal::ToolCalls, ToolSignal::ToolRole],
            index: 2,
            recovery: ToolCallRecovery::CanonicalExportAndOfficialTemplate,
        };
        // The fields carry the actionable facts, so a caller need not parse prose.
        match &e {
            FormatError::UnsupportedToolCalls {
                target,
                signals,
                index,
                recovery,
            } => {
                assert_eq!(*target, TrlFormat::Gemma4);
                assert_eq!(signals, &vec![ToolSignal::ToolCalls, ToolSignal::ToolRole]);
                assert_eq!(*index, 2);
                assert_eq!(
                    *recovery,
                    ToolCallRecovery::CanonicalExportAndOfficialTemplate
                );
            }
            other => panic!("expected UnsupportedToolCalls, got {other:?}"),
        }
        // The rendered message still names the route, the position and the way out.
        let msg = e.to_string();
        assert!(msg.contains("Gemma4"), "{msg}");
        assert!(msg.contains("messages[2]"), "{msg}");
        assert!(msg.contains("ToolCalls"), "{msg}");
        assert!(msg.contains("messages_json"), "{msg}");
    }
}
