//! Pure tool-call/result identity predicates shared by formatting and screened verification.
//! Calls without results remain valid; these checks do not infer a stricter trajectory contract.
use crate::{Message, Role};
use std::collections::BTreeSet;

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
/// Returns the existing diagnostic naming the offending message index and the reason.
pub fn validate_tool_links(messages: &[Message]) -> Result<(), String> {
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
fn assert_ids_unique(declared: &[DeclaredCall<'_>]) -> Result<(), String> {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for call in declared {
        if let Some(id) = call.id
            && !seen.insert(id)
        {
            return Err(format!(
                "duplicate tool call id `{id}`: the result link for that id is ambiguous"
            ));
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
) -> Result<(), String> {
    let Some(call) = declared.iter().find(|c| c.id == Some(id)) else {
        return Err(format!(
            "messages[{index}] (tool result) declares tool_call_id `{id}`, \
             which no assistant tool_call in this conversation declares"
        ));
    };
    if call.message_index >= index {
        return Err(format!(
            "messages[{index}] (tool result) answers tool_call_id `{id}` before the call is \
             declared — the captured trajectory is partial, not merely reordered"
        ));
    }
    if answered.contains(id) {
        return Err(format!(
            "messages[{index}] is a second result for tool_call_id `{id}`; \
             each declared call is answered exactly once"
        ));
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
) -> Result<(), String> {
    let name = message.name.as_deref().unwrap_or_default();
    let candidates: Vec<&DeclaredCall<'_>> = declared.iter().filter(|c| c.name == name).collect();
    match candidates.as_slice() {
        [] => Err(format!(
            "messages[{index}] is a tool result with no tool_call_id and no `name` (or a name \
             matching no call), so it cannot be linked to any call in this conversation"
        )),
        [only] if only.id.is_none() => Err(format!(
            "messages[{index}] is a tool result with no tool_call_id; the only `{name}` call in \
             this conversation declares no id, so no result link exists to record"
        )),
        [only] if only.message_index >= index => Err(format!(
            "messages[{index}] is a tool result for `{name}` that appears before the call is \
             declared — the captured trajectory is partial"
        )),
        [_only] => Ok(()),
        many => Err(format!(
            "messages[{index}] is a tool result with no tool_call_id, but {} calls named `{name}` \
             are declared — refusing to guess which one it answers",
            many.len()
        )),
    }
}
