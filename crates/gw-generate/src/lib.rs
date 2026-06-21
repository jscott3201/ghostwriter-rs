//! `gw-generate` — trace synthesis.
//!
//! Synthesizes BOTH the user turn and the assistant turn (with full chain-of-thought) via
//! teacher models, with prompt templating and per-area teacher selection. Depends on
//! `gw-schema`, `gw-providers`, and `gw-format`.
