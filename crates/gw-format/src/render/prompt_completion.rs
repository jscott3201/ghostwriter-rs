//! TRL prompt-completion renderer: `{"prompt":[…], "completion":[…]}`.
//!
//! The split point is the LAST assistant turn: every message before it is the `prompt`, and the
//! final assistant turn (plus any trailing assistant turns) is the `completion`. Each side is an
//! array of OpenAI-message objects (`{role, content, reasoning?, …}`) built by
//! [`crate::render::openai`], so `reasoning` rides as a sibling key on assistant turns under
//! [`CotPolicy`] and `content` stays clean (INVARIANT-a).

use serde_json::{Value, json};

use gw_schema::{CotPolicy, Message, Role};

use crate::error::Result;
use crate::render::openai;

/// The index of the last assistant turn, or `None` if there is no assistant turn (the whole
/// conversation is then the prompt with an empty completion).
fn last_assistant_index(messages: &[Message]) -> Option<usize> {
    messages.iter().rposition(|m| m.role == Role::Assistant)
}

/// Render `messages` into a pretty-printed prompt-completion JSON document under `cot`.
pub(crate) fn render(messages: &[Message], cot: CotPolicy) -> Result<String> {
    let split = last_assistant_index(messages).unwrap_or(messages.len());
    let prompt = openai_messages_array(&messages[..split], cot);
    let completion = openai_messages_array(&messages[split..], cot);
    let doc = json!({ "prompt": prompt, "completion": completion });
    Ok(serde_json::to_string_pretty(&doc)?)
}

/// The bare `messages` array (no `{messages:…}` wrapper) for a slice, reusing the OpenAI builder.
fn openai_messages_array(messages: &[Message], cot: CotPolicy) -> Value {
    // openai::build wraps in {messages: […]}; lift the inner array out.
    match openai::build(messages, cot) {
        Value::Object(mut map) => map.remove("messages").unwrap_or(Value::Array(vec![])),
        _ => Value::Array(vec![]),
    }
}
