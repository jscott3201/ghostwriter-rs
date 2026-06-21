//! TRL prompt-completion renderer: `{"prompt":[…], "completion":[…]}`.
//!
//! The conversation MUST end on the assistant turn being supervised. The split point is that final
//! assistant turn: every message before it is the `prompt`, and the final assistant turn is the
//! `completion`. Each side is an array of OpenAI-message objects (`{role, content, reasoning?, …}`)
//! built by [`crate::render::openai`], so `reasoning` rides as a sibling key on assistant turns
//! under [`CotPolicy`] and `content` stays clean (INVARIANT-a).
//!
//! ## Fail loud on a non-assistant final turn
//!
//! TRL supervises the WHOLE `completion` array as model output. If the conversation does not end on
//! an assistant turn, a contiguous tail-split would leak trailing user/tool turns into `completion`
//! and supervise them as if the model wrote them — silent corruption. So this renderer requires the
//! final message to be an assistant turn and returns [`FormatError::Projection`] otherwise.

use serde_json::{Value, json};

use gw_schema::{CotPolicy, Message, Role};

use crate::error::{FormatError, Result};
use crate::render::openai;

/// Render `messages` into a pretty-printed prompt-completion JSON document under `cot`.
///
/// # Errors
///
/// Returns [`FormatError::Projection`] if `messages` is empty or does not end on an assistant turn
/// (prompt-completion can only supervise a final assistant turn).
pub(crate) fn render(messages: &[Message], cot: CotPolicy) -> Result<String> {
    let split = match messages.last() {
        Some(m) if m.role == Role::Assistant => messages.len() - 1,
        _ => {
            return Err(FormatError::Projection(
                "prompt-completion requires the conversation to end on an assistant turn".into(),
            ));
        }
    };
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
