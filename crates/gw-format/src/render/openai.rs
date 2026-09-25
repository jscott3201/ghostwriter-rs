//! OpenAI-messages renderer: `{"messages":[{"role","content",…}]}` conversational.
//!
//! This is the `sft_modal.py` ingest shape: one clean `content` per turn, with `reasoning` a
//! sibling key on assistant turns (rendered under [`CotPolicy`]; NEVER inlined into `content`,
//! INVARIANT-a). `tool_calls`, `tool_call_id` and `name` are preserved verbatim, so a tool
//! trajectory round-trips through this target with its result links intact (INVARIANT i). Output is
//! a pretty-printed JSON document.
//!
//! This is one of the two TOOL-FAITHFUL routes: [`crate::render()`] never refuses a tool
//! trajectory here. Note the documented content policy — a `Content::Null` body (the tool-calling
//! turn) renders as `""` while the calls ride alongside it, so the null-vs-empty distinction lives
//! in the stored record and the canonical export column, not in these bytes.

use serde_json::{Map, Value, json};

use gw_schema::{CotPolicy, Message, Role};

use crate::error::Result;
use crate::render::{content_text, effective_reasoning, is_assistant};

/// The OpenAI wire role for a message. `developer` is preserved (a valid OpenAI role); every
/// other role passes through unchanged.
fn role_token(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::Developer => "developer",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

/// Build the `{messages:[…]}` JSON value for `messages` under `cot`.
pub(crate) fn build(messages: &[Message], cot: CotPolicy) -> Value {
    let out: Vec<Value> = messages
        .iter()
        .map(|msg| {
            let mut obj = Map::new();
            obj.insert("role".into(), json!(role_token(msg.role)));
            obj.insert("content".into(), json!(content_text(&msg.content)));
            if is_assistant(msg.role)
                && let Some(reasoning) = effective_reasoning(msg, cot)
            {
                obj.insert("reasoning".into(), json!(reasoning));
            }
            if let Some(name) = &msg.name {
                obj.insert("name".into(), json!(name));
            }
            if let Some(tool_calls) = &msg.tool_calls {
                obj.insert(
                    "tool_calls".into(),
                    serde_json::to_value(tool_calls).unwrap_or(Value::Null),
                );
            }
            // The result link (INVARIANT i) rides the wire verbatim, so a downstream consumer sees
            // WHICH call each tool result answers instead of re-inferring it from the function name.
            if let Some(tool_call_id) = &msg.tool_call_id {
                obj.insert("tool_call_id".into(), json!(tool_call_id));
            }
            Value::Object(obj)
        })
        .collect();
    json!({ "messages": out })
}

/// Render `messages` into a pretty-printed OpenAI-messages JSON document under `cot`.
pub(crate) fn render(messages: &[Message], cot: CotPolicy) -> Result<String> {
    Ok(serde_json::to_string_pretty(&build(messages, cot))?)
}
