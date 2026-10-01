//! Tool-call / tool-result identity validation (INVARIANT i).
//!
//! A tool trajectory is a RELATION, not a list of independent turns: an assistant turn declares N
//! [`ToolCall`](gw_schema::ToolCall)s and each is answered by exactly one
//! [`gw_schema::Role::Tool`] turn. The only faithful binding is the explicit
//! [`Message::tool_call_id`] result link — a function *name* is a weak fallback that breaks the
//! moment a trajectory calls the same tool twice.
//!
//! [`validate_tool_links`] is the admission check for a COMPLETE trajectory. It NEVER guesses:
//! an ambiguous or unresolvable result is an error (quarantine at the caller's discretion), not
//! a best-effort name match. It is deliberately NOT wired into [`render`](crate::render()) — a
//! text-only conversation declares no tool fields at all and must keep validating exactly as
//! before, so this check is an explicit, separate admission step rather than a global gate.
//!
//! It is also not the same check as the render boundary's tool guard: that guard only asks "can
//! this template represent the signals at all", while this asks "is the pairing itself sound". A
//! trajectory can be unrepresentable for a target (refused by
//! [`FormatError::UnsupportedToolCalls`]) while also having a sound identity, and vice versa.

use crate::error::{FormatError, Result};
use gw_schema::Message;

/// Check every tool-result link in one conversation.
///
/// The scope is the whole `messages` slice: that slice is one conversation, and call ids are
/// unique within it. (Multi-agent capture needs an explicit agent/session scope on top of this —
/// ids are NOT assumed unique across conversations.)
///
/// # What is checked
///
/// - **Duplicate call ids** — two declared calls sharing an id make every link to that id
///   ambiguous, so the whole trajectory is rejected.
/// - **Dangling result id** — a result whose [`Message::tool_call_id`] matches no declared call.
/// - **Partial trajectory** — a result that arrives BEFORE the message declaring its call. A
///   resume boundary or a truncated capture produces this shape and it must not be admitted as
///   if it were complete.
/// - **Duplicate result** — two results answering the same declared call.
/// - **Ambiguous missing id** — a result with NO [`Message::tool_call_id`] is accepted only when
///   exactly one declared call carries the same [`Message::name`] AND that call has an id (so a
///   link is *possible*, just not recorded). Zero same-name calls is a dangling name; two or more
///   is precisely the "repeated name" case this invariant exists to refuse to guess. The
///   validator never synthesizes a link in either case.
///
/// # What is deliberately NOT checked
///
/// A declared call with no result is allowed: a teacher that has just emitted a call is a live
/// boundary, not a corrupt record. Only the result side is load-bearing for identity.
///
/// A conversation with no tool fields at all (the historical text-only case) validates `Ok` and is
/// never rejected by this function.
///
/// # Errors
///
/// Returns [`FormatError::ToolIdentity`] naming the offending message index and the reason.
pub fn validate_tool_links(messages: &[Message]) -> Result<()> {
    gw_schema::validate_tool_links(messages).map_err(FormatError::ToolIdentity)
}

#[cfg(test)]
mod tests {
    use gw_schema::{Content, FunctionCall, Role, ToolCall};

    use super::*;

    fn call(id: Option<&str>, name: &str, args: &str) -> ToolCall {
        ToolCall {
            id: id.map(str::to_owned),
            function: FunctionCall {
                name: name.into(),
                arguments: serde_json::json!({ "path": args }),
                raw_arguments: None,
            },
        }
    }

    fn assistant_calls(calls: Vec<ToolCall>) -> Message {
        Message {
            role: Role::Assistant,
            content: Content::Null,
            reasoning: None,
            reasoning_details: None,
            tool_calls: Some(calls),
            tool_call_id: None,
            name: None,
        }
    }

    fn result(id: Option<&str>, name: &str) -> Message {
        Message {
            role: Role::Tool,
            content: Content::Text("ok".into()),
            reasoning: None,
            reasoning_details: None,
            tool_calls: None,
            tool_call_id: id.map(str::to_owned),
            name: Some(name.into()),
        }
    }

    fn text_turns() -> Vec<Message> {
        vec![
            Message {
                role: Role::User,
                content: Content::Text("hi".into()),
                reasoning: None,
                reasoning_details: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            Message {
                role: Role::Assistant,
                content: Content::Text("42".into()),
                reasoning: Some("6*7".into()),
                reasoning_details: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ]
    }

    #[test]
    fn text_only_conversation_is_never_rejected() {
        assert!(validate_tool_links(&text_turns()).is_ok());
        assert!(validate_tool_links(&[]).is_ok());
    }

    #[test]
    fn two_same_name_calls_with_explicit_links_resolve() {
        let msgs = vec![
            assistant_calls(vec![
                call(Some("a"), "read_file", "toy.py"),
                call(Some("b"), "read_file", "toy.py"),
            ]),
            // Reversed arrival order is fine — the ID, not the position, binds them.
            result(Some("b"), "read_file"),
            result(Some("a"), "read_file"),
        ];
        assert!(validate_tool_links(&msgs).is_ok());
    }

    #[test]
    fn a_single_declared_call_left_unanswered_is_allowed() {
        // A teacher that just emitted a call is a live boundary, not a corrupt record.
        let msgs = vec![assistant_calls(vec![call(
            Some("a"),
            "read_file",
            "toy.py",
        )])];
        assert!(validate_tool_links(&msgs).is_ok());
    }

    #[test]
    fn dangling_result_id_is_rejected() {
        let msgs = vec![
            assistant_calls(vec![call(Some("a"), "read_file", "toy.py")]),
            result(Some("nope"), "read_file"),
        ];
        let err = validate_tool_links(&msgs).unwrap_err();
        assert!(err.to_string().contains("nope"), "{err}");
    }

    #[test]
    fn duplicate_call_ids_are_rejected() {
        let msgs = vec![
            assistant_calls(vec![
                call(Some("a"), "read_file", "x"),
                call(Some("a"), "read_file", "y"),
            ]),
            result(Some("a"), "read_file"),
        ];
        let err = validate_tool_links(&msgs).unwrap_err();
        assert!(err.to_string().contains("duplicate tool call id"), "{err}");
    }

    #[test]
    fn ambiguous_missing_id_across_repeated_names_is_rejected() {
        let msgs = vec![
            assistant_calls(vec![
                call(Some("a"), "read_file", "x"),
                call(Some("b"), "read_file", "y"),
            ]),
            result(None, "read_file"),
        ];
        let err = validate_tool_links(&msgs).unwrap_err();
        assert!(err.to_string().contains("refusing to guess"), "{err}");
    }

    #[test]
    fn missing_id_with_a_unique_named_call_is_accepted() {
        // Unambiguous by name, and the call HAS an id so a link is recordable — the validator does
        // not synthesize one, it just does not quarantine an unambiguous trajectory.
        let msgs = vec![
            assistant_calls(vec![call(Some("a"), "read_file", "x")]),
            result(None, "read_file"),
        ];
        assert!(validate_tool_links(&msgs).is_ok());
    }

    #[test]
    fn missing_id_whose_only_call_has_no_id_is_rejected() {
        let msgs = vec![
            assistant_calls(vec![call(None, "read_file", "x")]),
            result(None, "read_file"),
        ];
        let err = validate_tool_links(&msgs).unwrap_err();
        assert!(err.to_string().contains("declares no id"), "{err}");
    }

    #[test]
    fn result_with_no_matching_name_is_rejected() {
        let msgs = vec![
            assistant_calls(vec![call(Some("a"), "read_file", "x")]),
            result(None, "write_file"),
        ];
        let err = validate_tool_links(&msgs).unwrap_err();
        assert!(err.to_string().contains("cannot be linked"), "{err}");
    }

    #[test]
    fn result_before_its_call_is_rejected_as_partial() {
        let msgs = vec![
            result(Some("a"), "read_file"),
            assistant_calls(vec![call(Some("a"), "read_file", "x")]),
        ];
        let err = validate_tool_links(&msgs).unwrap_err();
        assert!(err.to_string().contains("partial"), "{err}");
    }

    #[test]
    fn two_results_for_one_call_are_rejected() {
        let msgs = vec![
            assistant_calls(vec![call(Some("a"), "read_file", "x")]),
            result(Some("a"), "read_file"),
            result(Some("a"), "read_file"),
        ];
        let err = validate_tool_links(&msgs).unwrap_err();
        assert!(err.to_string().contains("second result"), "{err}");
    }
}
