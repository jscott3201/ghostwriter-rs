//! `gw-format` — chat-template rendering and SFT / preference export projection.
//!
//! Renders the canonical `{messages, reasoning}` record into model-specific chat templates and
//! projects admitted [`gw_schema::TrainingRecord`]s down to a trainer's SFT or preference (DPO)
//! export shape. Depends only on `gw-schema` (+ `minijinja`, `serde`, `serde_json`); no I/O, no
//! network, no `unsafe`.
//!
//! ## The load-bearing invariant (INVARIANT-a)
//!
//! The stored `content` is ALWAYS clean final-answer text. Raw channel tokens (`<think>`,
//! `<|channel>`, `<turn|>`, …) live ONLY inside the `reasoning` text or are PRODUCED by the
//! renderer — never stored in `content`. [`render()`] enforces this up front: it FAILS LOUD with
//! [`FormatError::ControlTokenInContent`] if a clean field already carries a control token, rather
//! than silently double-framing and corrupting the target.
//!
//! For the TOKEN-STREAM targets ([`Gemma4`](gw_schema::TrlFormat::Gemma4),
//! [`ChatML`](gw_schema::TrlFormat::ChatML), [`Harmony`](gw_schema::TrlFormat::Harmony)), format
//! A↔B conversion is a pure function of `(clean messages + reasoning + target template)`:
//! [`render()`] and [`ingest_openrouter`] / [`strip_channel_tokens`] are inverses, so
//! `render → ingest` recovers the clean messages + reasoning. The STRUCTURED targets
//! ([`ShareGpt`](gw_schema::TrlFormat::ShareGpt),
//! [`OpenAiMessages`](gw_schema::TrlFormat::OpenAiMessages),
//! [`TrlPromptCompletion`](gw_schema::TrlFormat::TrlPromptCompletion)) carry `reasoning` in a CLEAN
//! sibling key — lossless by construction, NOT round-tripped through the channel-stripping ingest
//! path.
//!
//! ## Layout
//!
//! - **error** — [`FormatError`], the single typed error (no `anyhow`).
//! - **render** — [`render()`]: dispatch one `&[Message]` into a [`gw_schema::TrlFormat`] under a
//!   [`gw_schema::CotPolicy`]. Byte-exact Gemma-4 (via an embedded `minijinja` template), ChatML,
//!   ShareGPT, OpenAI-messages, Harmony, and TRL prompt-completion.
//! - **ingest** — [`ingest_openrouter`] (one OpenAI/OpenRouter assistant message →
//!   [`gw_schema::Message`]) and [`strip_channel_tokens`] (the round-trip stripper).
//! - **projection** — [`project_sft`] (admitted record → [`SftProjection`]) and
//!   [`project_preference`] (admitted + rejected sibling → [`gw_schema::PreferenceRecord`]).
//!
//! ## Gemma-4 token bytes (MUST diff-verify)
//!
//! The Gemma-4 renderer emits asymmetric turn/channel tokens (`<|turn>…<turn|>`,
//! `<|channel>thought…<channel|>`) transcribed from the `_research` spec. They are golden-file
//! tested, but **must be diff-verified byte-for-byte against the official pinned
//! `google/gemma-4-12B-it` `chat_template.jinja` before production SFT** (no network). See
//! [`render()`] / the `render::gemma4` module docs.
//!
//! ```
//! use gw_format::render;
//! use gw_schema::{Content, CotPolicy, Message, Role, TrlFormat};
//!
//! let messages = vec![
//!     Message {
//!         role: Role::User,
//!         content: Content::Text("What is 12 * 8?".into()),
//!         reasoning: None,
//!         reasoning_details: None,
//!         tool_calls: None,
//!         name: None,
//!     },
//!     Message {
//!         role: Role::Assistant,
//!         content: Content::Text("96".into()),
//!         reasoning: Some("10*8=80, 2*8=16, 80+16=96".into()),
//!         reasoning_details: None,
//!         tool_calls: None,
//!         name: None,
//!     },
//! ];
//! let out = render(&messages, TrlFormat::Gemma4, CotPolicy::Supervised).unwrap();
//! assert!(out.starts_with("<bos><|turn>user\n"));
//! assert!(out.contains("<|channel>thought\n10*8=80, 2*8=16, 80+16=96\n<channel|>96"));
//! ```

mod error;
mod ingest;
mod projection;
mod render;
mod validate;

pub use error::{FormatError, Result};
pub use ingest::{ingest_openrouter, strip_channel_tokens};
pub use projection::{SftProjection, project_preference, project_sft};
pub use render::render;
