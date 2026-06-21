//! `gw-format` — chat-template rendering and export projection.
//!
//! Renders the canonical `{messages, reasoning}` record into model-specific chat templates
//! (byte-exact Gemma-4 via minijinja) and projects admitted records to SFT / preference
//! (DPO) export shapes. Depends only on `gw-schema`.
