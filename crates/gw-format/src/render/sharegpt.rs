//! ShareGPT renderer: `{"conversations":[{"from","value"}]}`.
//!
//! Roles map `system → system`, `user → human`, `assistant → gpt`, `tool → tool`. ShareGPT has
//! no native reasoning slot; this renderer uses the lossless **option 1** from the spec: a
//! sibling `reasoning` key on the `gpt` turn (when `reasoning` is present under [`CotPolicy`]).
//! `value` stays CLEAN final-answer text (INVARIANT-a). Output is a pretty-printed JSON document.

use serde_json::{Map, Value, json};

use gw_schema::{CotPolicy, Message, Role};

use crate::error::Result;
use crate::render::{content_text, effective_reasoning, is_assistant};

/// The ShareGPT `from` token for a role. `developer` collapses to `system`.
fn from_token(role: Role) -> &'static str {
    match role {
        Role::System | Role::Developer => "system",
        Role::User => "human",
        Role::Assistant => "gpt",
        Role::Tool => "tool",
    }
}

/// Build the `{conversations:[…]}` JSON value for `messages` under `cot`.
pub(crate) fn build(messages: &[Message], cot: CotPolicy) -> Value {
    let conversations: Vec<Value> = messages
        .iter()
        .map(|msg| {
            let mut turn = Map::new();
            turn.insert("from".into(), json!(from_token(msg.role)));
            turn.insert("value".into(), json!(content_text(&msg.content)));
            if is_assistant(msg.role)
                && let Some(reasoning) = effective_reasoning(msg, cot)
            {
                turn.insert("reasoning".into(), json!(reasoning));
            }
            Value::Object(turn)
        })
        .collect();
    json!({ "conversations": conversations })
}

/// Render `messages` into a pretty-printed ShareGPT JSON document under `cot`.
pub(crate) fn render(messages: &[Message], cot: CotPolicy) -> Result<String> {
    Ok(serde_json::to_string_pretty(&build(messages, cot))?)
}
