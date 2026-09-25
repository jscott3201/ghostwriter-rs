//! Control-token validation — the loud round-trip guard (INVARIANT-a).
//!
//! A training-data renderer must FAIL LOUD on un-round-trippable input rather than silently
//! corrupt the target. Per INVARIANT-a, a message's `content` and `reasoning` are supposed to be
//! CLEAN: they carry NO chat control tokens (those are PRODUCED by the renderer / live only in the
//! reasoning the renderer frames). If a clean field already contains a control token, an upstream
//! stage leaked channel markup — rendering it would double-frame and break the round-trip — so we
//! reject it with [`FormatError::ControlTokenInContent`].

use gw_schema::{Content, Message, Role};

use crate::error::{FormatError, Result};

/// Every chat control token any target template (or recognized provider) emits. If one of these
/// appears in a CLEAN field (`content` / `reasoning`), an upstream stage leaked channel markup.
///
/// Order matters only for the reported token when several overlap (longer/more-specific first), so
/// the most descriptive marker is named.
pub(crate) const CONTROL_TOKENS: &[&str] = &[
    // ChatML / Qwen / DeepSeek
    "<think>",
    "</think>",
    "<|im_start|>",
    "<|im_end|>",
    // Gemma-4 (asymmetric)
    "<|channel>",
    "<channel|>",
    "<|channel|>",
    "<|turn>",
    "<turn|>",
    "<|think|>",
    "<bos>",
    // Harmony
    "<|start|>",
    "<|end|>",
    "<|message|>",
    "<|return|>",
];

/// The first control token contained in `text`, if any.
pub(crate) fn first_control_token(text: &str) -> Option<&'static str> {
    CONTROL_TOKENS
        .iter()
        .copied()
        .find(|tok| text.contains(tok))
}

/// Validate that every message's clean fields (`content` text + `reasoning`) are free of control
/// tokens, BEFORE rendering any target.
///
/// # Errors
///
/// Returns [`FormatError::ControlTokenInContent`] (naming the token + the role) on the first leak.
pub(crate) fn validate_clean(messages: &[Message]) -> Result<()> {
    for msg in messages {
        check_field(content_str(&msg.content).as_deref(), msg.role)?;
        if let Some(reasoning) = msg.reasoning.as_deref() {
            check_field(Some(reasoning), msg.role)?;
        }
    }
    Ok(())
}

/// Reject `field` if it contains any control token.
fn check_field(field: Option<&str>, role: Role) -> Result<()> {
    if let Some(text) = field
        && let Some(token) = first_control_token(text)
    {
        return Err(FormatError::ControlTokenInContent { token, role });
    }
    Ok(())
}

/// The concatenated clean text of a [`Content`] (text parts only), for scanning. An explicitly
/// absent value ([`Content::Null`]) has no text to scan, so it yields `None`.
pub(crate) fn content_str(content: &Content) -> Option<String> {
    match content {
        Content::Text(s) => Some(s.clone()),
        Content::Parts(parts) => Some(
            parts
                .iter()
                .filter_map(|p| match p {
                    gw_schema::ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        ),
        Content::Null => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: Role, content: &str, reasoning: Option<&str>) -> Message {
        Message {
            role,
            content: Content::Text(content.into()),
            reasoning: reasoning.map(str::to_owned),
            reasoning_details: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    #[test]
    fn clean_messages_pass() {
        let m = vec![
            msg(Role::User, "What is 2+2?", None),
            msg(Role::Assistant, "4", Some("2+2=4")),
        ];
        assert!(validate_clean(&m).is_ok());
    }

    #[test]
    fn control_token_in_reasoning_fails_for_every_delimiter() {
        for &tok in CONTROL_TOKENS {
            let m = vec![msg(Role::Assistant, "ans", Some(&format!("x{tok}y")))];
            let err = validate_clean(&m).unwrap_err();
            match err {
                FormatError::ControlTokenInContent { role, .. } => {
                    assert_eq!(role, Role::Assistant)
                }
                other => panic!("expected ControlTokenInContent for {tok}, got {other:?}"),
            }
        }
    }

    #[test]
    fn control_token_in_content_fails() {
        let m = vec![msg(Role::User, "hi <|im_end|> there", None)];
        assert!(matches!(
            validate_clean(&m),
            Err(FormatError::ControlTokenInContent {
                token: "<|im_end|>",
                role: Role::User
            })
        ));
    }
}
