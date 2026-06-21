//! `gw-providers` — the model I/O layer.
//!
//! **Load-bearing piece:** a hand-rolled `reqwest` + SSE streaming client with a custom delta
//! type that captures `content` AND `reasoning`/`reasoning_details`. `async-openai` 0.41.1's
//! typed stream delta silently drops the reasoning field, so the teacher path owns its own
//! response parsing (capturing full chain-of-thought is the entire reason this harness exists).
//! Also: a `governor` GCRA rate limiter per provider/model and transient-error retries.
