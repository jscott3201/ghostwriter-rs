//! Tool-call / tool-result identity validation (INVARIANT i).
//!
//! A tool trajectory is a RELATION, not a list of independent turns: an assistant turn declares N
//! [`ToolCall`](gw_schema::ToolCall)s and each is answered by exactly one
//! [`Role::Tool`] turn. The only faithful binding is the explicit
//! [`Message::tool_call_id`] result link — a function *name* is a weak fallback that breaks the
//! moment a trajectory calls the same tool twice.
//!
//! [`validate_tool_links`] is the admission check for a COMPLETE trajectory. It NEVER guesses:
//! an ambiguous or unresolvable result is an error (quarantine at the caller's discretion), not
//! a best-effort name match. It is deliberately NOT wired into [`render`](crate::render()) — a
//! text-only conversation declares no tool fields at all and must keep validating exactly as
//! before, so this check is an explicit, separate admission step rather than a global gate.

use std::collections::BTreeSet;

use gw_schema::{Message, Role};

use crate::error::{FormatError, Result};

/// One declared call: its function name, its optional id, and the index of the message that
/// declared it (so ordering can be checked without cloning).
struct DeclaredCall<'a> {
    name: &'a str,
    id: Option<&'a str>,
    message_index: usize,
}

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
    let declared = declared_calls(messages);
    assert_ids_unique(&declared)?;

    let mut answered: BTreeSet<&str> = BTreeSet::new();
    for (index, message) in messages.iter().enumerate() {
        if message.role != Role::Tool {
            continue;
        }
        match message.tool_call_id.as_deref() {
            Some(id) => check_explicit_link(&declared, &answered, index, id)?,
            None => check_implicit_link(&declared, index, message)?,
        }
        if let Some(id) = message.tool_call_id.as_deref() {
            answered.insert(id);
        }
    }
    Ok(())
}

/// Every declared call in the conversation, in message order.
fn declared_calls(messages: &[Message]) -> Vec<DeclaredCall<'_>> {
    messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| {
            let calls = message.tool_calls.as_ref()?;
            Some(calls.iter().map(move |call| DeclaredCall {
                name: &call.function.name,
                id: call.id.as_deref(),
                message_index: index,
            }))
        })
        .flatten()
        .collect()
}

/// Reject two declared calls sharing one id — the link would be ambiguous for every result.
fn assert_ids_unique(declared: &[DeclaredCall<'_>]) -> Result<()> {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for call in declared {
        if let Some(id) = call.id
            && !seen.insert(id)
        {
            return Err(FormatError::ToolIdentity(format!(
                "duplicate tool call id `{id}`: the result link for that id is ambiguous"
            )));
        }
    }
    Ok(())
}

/// A result carrying an explicit id: it must resolve to exactly one call declared EARLIER, and no
/// earlier result may have answered it already.
fn check_explicit_link<'a>(
    declared: &[DeclaredCall<'a>],
    answered: &BTreeSet<&'a str>,
    index: usize,
    id: &'a str,
) -> Result<()> {
    let Some(call) = declared.iter().find(|c| c.id == Some(id)) else {
        return Err(FormatError::ToolIdentity(format!(
            "messages[{index}] (tool result) declares tool_call_id `{id}`, \
             which no assistant tool_call in this conversation declares"
        )));
    };
    if call.message_index >= index {
        return Err(FormatError::ToolIdentity(format!(
            "messages[{index}] (tool result) answers tool_call_id `{id}` before the call is \
             declared — the captured trajectory is partial, not merely reordered"
        )));
    }
    if answered.contains(id) {
        return Err(FormatError::ToolIdentity(format!(
            "messages[{index}] is a second result for tool_call_id `{id}`; \
             each declared call is answered exactly once"
        )));
    }
    Ok(())
}

/// A result with NO explicit id: acceptable only if exactly one declared call shares its name AND
/// that call has an id. Zero candidates is a dangling name; two or more is the repeated-name case
/// this invariant exists to refuse to guess.
fn check_implicit_link(
    declared: &[DeclaredCall<'_>],
    index: usize,
    message: &Message,
) -> Result<()> {
    let name = message.name.as_deref().unwrap_or_default();
    let candidates: Vec<&DeclaredCall<'_>> = declared.iter().filter(|c| c.name == name).collect();
    match candidates.as_slice() {
        [] => Err(FormatError::ToolIdentity(format!(
            "messages[{index}] is a tool result with no tool_call_id and no `name` (or a name \
             matching no call), so it cannot be linked to any call in this conversation"
        ))),
        [only] if only.id.is_none() => Err(FormatError::ToolIdentity(format!(
            "messages[{index}] is a tool result with no tool_call_id; the only `{name}` call in \
             this conversation declares no id, so no result link exists to record"
        ))),
        [only] if only.message_index >= index => Err(FormatError::ToolIdentity(format!(
            "messages[{index}] is a tool result for `{name}` that appears before the call is \
             declared — the captured trajectory is partial"
        ))),
        [_only] => Ok(()),
        many => Err(FormatError::ToolIdentity(format!(
            "messages[{index}] is a tool result with no tool_call_id, but {} calls named `{name}` \
             are declared — refusing to guess which one it answers",
            many.len()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use gw_schema::{Content, FunctionCall, ToolCall};

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
