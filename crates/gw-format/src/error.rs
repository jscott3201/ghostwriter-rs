//! [`FormatError`] — the single error type surfaced by every `gw-format` operation.
//!
//! Variants distinguish the failure classes a caller must reason about: a minijinja template
//! compile/render fault (`Template`), JSON (de)serialization of the ingest/export shapes
//! (`Serde`), a role the target template cannot represent (`UnsupportedRole`), a control token
//! leaked into a clean field (`ControlTokenInContent`, the loud round-trip guard), a malformed
//! ingest payload (`Ingest`), and a projection precondition violation (`Projection`). No
//! `anyhow` — this crate surfaces a typed error.

use thiserror::Error;

use gw_schema::Role;

/// Everything that can go wrong rendering, ingesting, or projecting a record.
///
/// `#[non_exhaustive]` so new variants can be added without a breaking change. `Template` and
/// `Serde` carry their underlying source via `#[from]`; the rest are constructed directly with a
/// human-readable message.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FormatError {
    /// A minijinja template failed to compile or render (e.g. the Gemma-4 template).
    #[error("template error: {0}")]
    Template(#[from] minijinja::Error),

    /// (De)serializing an ingest payload or an export shape (`serde_json::Value` / `String`)
    /// failed.
    #[error("serde_json error: {0}")]
    Serde(#[from] serde_json::Error),

    /// The target template cannot represent this [`Role`] (e.g. a stray `developer` turn a
    /// format does not map). Carries the offending role and the target name.
    #[error("unsupported role `{role:?}` for target `{target}`")]
    UnsupportedRole {
        /// The role the template could not place.
        role: Role,
        /// The target format name (e.g. `"sharegpt"`).
        target: &'static str,
    },

    /// A message's CLEAN field (`content` or `reasoning`) already contains a chat control token
    /// (e.g. `<think>`, `<|channel>`, `<turn|>`). Per INVARIANT-a these fields must be CLEAN — a
    /// control token there means an upstream stage leaked channel markup and the render would not
    /// round-trip. Failing loud beats silently corrupting the training target. Carries the
    /// offending token and the role of the message it was found on.
    #[error("control token `{token}` found in clean field of `{role:?}` message")]
    ControlTokenInContent {
        /// The control token that leaked into a clean field.
        token: &'static str,
        /// The role of the message carrying the leak.
        role: Role,
    },

    /// An OpenRouter / OpenAI ingest payload was malformed (missing `message`, wrong type, …).
    #[error("ingest error: {0}")]
    Ingest(String),

    /// A projection precondition was violated (e.g. a DPO pair whose sides do not share a
    /// `prompt_hash`, or a record with no assistant turn to project).
    #[error("projection error: {0}")]
    Projection(String),
}

/// Convenience alias for results returned by `gw-format` operations.
pub type Result<T> = std::result::Result<T, FormatError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_role_names_role_and_target() {
        let e = FormatError::UnsupportedRole {
            role: Role::Developer,
            target: "sharegpt",
        };
        let msg = e.to_string();
        assert!(msg.contains("Developer"));
        assert!(msg.contains("sharegpt"));
    }

    #[test]
    fn serde_error_converts() {
        let bad = serde_json::from_str::<i32>("not json").unwrap_err();
        let e: FormatError = bad.into();
        assert!(matches!(e, FormatError::Serde(_)));
    }

    #[test]
    fn control_token_names_token_and_role() {
        let e = FormatError::ControlTokenInContent {
            token: "<think>",
            role: Role::Assistant,
        };
        let msg = e.to_string();
        assert!(msg.contains("<think>"));
        assert!(msg.contains("Assistant"));
    }

    #[test]
    fn ingest_and_projection_render_message() {
        assert!(
            FormatError::Ingest("no message".into())
                .to_string()
                .contains("no message")
        );
        assert!(
            FormatError::Projection("prompt_hash mismatch".into())
                .to_string()
                .contains("mismatch")
        );
    }
}
