//! Gemma-4 byte-exact renderer (the initial fine-tune target).
//!
//! # Token-byte provenance (MUST diff-verify before production SFT)
//!
//! The Gemma-4 token bytes this module emits are transcribed from the `_research` spec
//! (`07-data-schema-storage-pipeline.md` §2.4 and `05-gemma4-target-bed-and-sft-format.md` §2),
//! which read them from the pinned `google/gemma-4-12B-it` `chat_template.jinja`:
//!
//! | Concept       | Bytes                          | Note                                  |
//! |---------------|--------------------------------|---------------------------------------|
//! | BOS           | `<bos>`                        | emitted by the template               |
//! | turn OPEN     | `<\|turn>{role}\n`             | ASYMMETRIC — NOT `<start_of_turn>`    |
//! | turn CLOSE    | `<turn\|>\n`                   | ASYMMETRIC — NOT `<end_of_turn>`      |
//! | thought OPEN  | `<\|channel>thought\n`         | ASYMMETRIC — NOT symmetric `<\|channel\|>` |
//! | thought CLOSE | `<channel\|>`                  | ASYMMETRIC                            |
//!
//! Roles map to `user` and `model` only (`assistant → model`); there is no system role. A
//! leading or mid-conversation `system`/`developer` message is folded into the NEXT `user` turn
//! that follows it (and a trailing system message with no following user turn becomes a lone user
//! turn) — a harness simplification of the upstream conditional-system-turn behavior. Every
//! `model` turn carries a thought wrapper: it holds the reasoning when
//! [`CotPolicy::Supervised`]/[`CotPolicy::Masked`], or is EMPTY (`<|channel>thought\n<channel|>`)
//! when [`CotPolicy::Stripped`].
//!
//! ## Deliberate omissions (v1)
//!
//! - The upstream `<|think|>` thinking-enable marker (injected at the first system turn) is **NOT**
//!   emitted: this renderer folds the system turn away entirely, so there is no system turn to
//!   carry it. Thinking is instead expressed per-`model`-turn via the thought wrapper.
//! - The Gemma-4 tool-call DSL is out of scope for v1. A conversation that declares `tool_calls`, a
//!   `tool_call_id` link, or a [`Role::Tool`] turn does NOT reach this renderer at all:
//!   [`crate::render()`] fails closed with
//!   [`FormatError::UnsupportedToolCalls`] because this
//!   template has nowhere to put them and would silently render a tool trajectory as prose. The
//!   `Role::Tool` arm below is therefore unreachable through the public entry point; it remains a
//!   defensive mapping, not a data path.
//! - [`Content::Parts`](gw_schema::Content::Parts) is flattened to its text segments; an
//!   image/audio-only turn renders as empty content. Noted for a future multimodal corpus.
//!
//! TODO(gemma4-verify): these bytes are per the `_research` spec and are golden-file tested
//! here, but they MUST be diff-verified byte-for-byte against the official pinned
//! `google/gemma-4-12B-it` `chat_template.jinja` before production SFT — including reconciling the
//! omitted `<|think|>` system-turn marker (this renderer's system-folding choice diverges from
//! upstream's conditional system turn). Do NOT fetch the template at runtime or in tests (no
//! network).

use std::sync::OnceLock;

use minijinja::{Environment, context};
use serde::Serialize;

use gw_schema::{CotPolicy, Message, Role};

use crate::error::{FormatError, Result};
use crate::render::{content_text, effective_reasoning};

/// The embedded Gemma-4 template (authored to the `_research` spec; see module docs).
const GEMMA4_TEMPLATE: &str = include_str!("../../templates/gemma4.jinja");

/// One render-ready Gemma-4 turn. `role` is the remapped Gemma role (`user` | `model`);
/// `reasoning` is `None` for user turns and for stripped model turns.
#[derive(Serialize)]
struct Gemma4Turn {
    role: &'static str,
    content: String,
    reasoning: Option<String>,
}

/// Lazily-compiled environment holding the single Gemma-4 template, so repeated renders do not
/// recompile it. The template source is `'static`, so the environment is too.
fn environment() -> &'static Environment<'static> {
    static ENV: OnceLock<Environment<'static>> = OnceLock::new();
    ENV.get_or_init(|| {
        let mut env = Environment::new();
        // The template is a vetted constant; a compile failure here is a build-time bug.
        env.add_template("gemma4", GEMMA4_TEMPLATE)
            .expect("embedded gemma4.jinja must compile");
        env
    })
}

/// Render `messages` into the byte-exact Gemma-4 turn format under `cot`.
pub(crate) fn render(messages: &[Message], cot: CotPolicy) -> Result<String> {
    let turns = build_turns(messages, cot)?;
    let tmpl = environment()
        .get_template("gemma4")
        .map_err(FormatError::Template)?;
    Ok(tmpl.render(context! { turns => turns })?)
}

/// Lower `messages` to Gemma-4 turns: map roles, fold a leading system/developer turn into the
/// first user turn, and resolve each turn's reasoning under `cot`.
fn build_turns(messages: &[Message], cot: CotPolicy) -> Result<Vec<Gemma4Turn>> {
    let mut turns = Vec::with_capacity(messages.len());
    // System/developer content waiting to be folded into the next user turn.
    let mut pending_system: Option<String> = None;

    for msg in messages {
        let text = content_text(&msg.content);
        match msg.role {
            Role::System | Role::Developer => {
                // Accumulate; fold into the next user turn (Gemma-4 has no system role here).
                match &mut pending_system {
                    Some(acc) => {
                        acc.push_str("\n\n");
                        acc.push_str(&text);
                    }
                    None => pending_system = Some(text),
                }
            }
            Role::User | Role::Tool => {
                // A `Role::Tool` turn never arrives here through `crate::render()` (the tool guard
                // refuses the whole conversation first); this mapping is defensive only. Tool
                // results are surfaced as user-side content.
                let content = match pending_system.take() {
                    Some(sys) if !sys.is_empty() => format!("{sys}\n\n{text}"),
                    _ => text,
                };
                turns.push(Gemma4Turn {
                    role: "user",
                    content,
                    reasoning: None,
                });
            }
            Role::Assistant => {
                let reasoning = effective_reasoning(msg, cot).map(str::to_owned);
                turns.push(Gemma4Turn {
                    role: "model",
                    content: text,
                    reasoning,
                });
            }
        }
    }

    // A trailing system turn with no following user turn becomes a lone user turn (rare, but the
    // system content must not be silently dropped).
    if let Some(sys) = pending_system {
        turns.push(Gemma4Turn {
            role: "user",
            content: sys,
            reasoning: None,
        });
    }

    Ok(turns)
}
