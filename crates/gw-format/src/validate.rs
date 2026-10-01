//! Control-token validation — the loud round-trip guard (INVARIANT-a).
//!
//! A training-data renderer must FAIL LOUD on un-round-trippable input rather than silently
//! corrupt the target. Per INVARIANT-a, a message's `content` and `reasoning` are supposed to be
//! CLEAN: they carry NO chat control tokens (those are PRODUCED by the renderer / live only in the
//! reasoning the renderer frames). If a clean field already contains a control token, an upstream
//! stage leaked channel markup — rendering it would double-frame and break the round-trip — so we
//! reject it with [`FormatError::ControlTokenInContent`].

use crate::error::{FormatError, Result};
use gw_schema::Message;
pub(crate) use gw_schema::{content_text_parts as content_str, first_control_token};

/// Validate the shared clean-field prerequisites before formatting.
pub(crate) fn validate_clean(messages: &[Message]) -> Result<()> {
    if let Some((role, token)) = gw_schema::first_clean_field_violation(messages) {
        return Err(FormatError::ControlTokenInContent { token, role });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gw_schema::{CLEAN_FIELD_CONTROL_TOKENS as CONTROL_TOKENS, Content, Role};

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
