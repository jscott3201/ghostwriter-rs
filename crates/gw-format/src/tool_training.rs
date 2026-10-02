//! Strict complete tool sources, separate from permissive canonical capture.
use crate::{FormatError, Result};
use gw_schema::Message;
use serde_json::Value;

/// Validate complete text/function source structure and explicit call/result linkage.
/// See [`gw_schema::validate_tool_training_source`] for the supported schema contract.
///
/// # Errors
/// Returns [`FormatError::ToolTrainingSource`] for unsupported or incomplete sources.
pub fn validate_tool_training_source(messages: &[Message], tools: Option<&[Value]>) -> Result<()> {
    gw_schema::validate_tool_training_source(messages, tools)
        .map_err(FormatError::ToolTrainingSource)
}

/// Reject consumer control delimiters in projected fields, excluding retained raw arguments.
///
/// # Errors
/// Returns [`FormatError::ToolTrainingSource`] for empty delimiters or ambiguous source text.
pub fn validate_tool_projection_delimiters(
    messages: &[Message],
    tools: Option<&[Value]>,
    delimiters: &[&str],
) -> Result<()> {
    gw_schema::validate_tool_projection_delimiters(messages, tools, delimiters)
        .map_err(FormatError::ToolTrainingSource)
}

#[cfg(test)]
#[path = "tool_training_tests.rs"]
mod tests;
