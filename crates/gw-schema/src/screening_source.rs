//! Pure source-field presence and supported-shape classification for lexical screening.
use crate::{
    Content, ContentPart, Message, MultiTurnLoss, ReasoningDetail, Role, ScreeningField, TrlFormat,
};
use serde_json::Value;
use std::collections::BTreeSet;

/// Source requirements derived without rendering, lexical matching, tokenization or I/O.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScreeningSourceShape {
    /// Every text-field class present, including explicitly present empty text, in canonical order.
    pub required_fields: Vec<ScreeningField>,
    /// Unsupported source-shape reason codes, deduplicated and canonically ordered.
    pub unsupported_reasons: Vec<&'static str>,
}

// Object keys are text even when their values are numeric. Empty objects/arrays have no text;
// empty strings do. This exactly matches the lexical collector's independent JSON text segments.
fn has_json_text(value: &Value) -> bool {
    match value {
        Value::String(_) => true,
        Value::Array(values) => values.iter().any(has_json_text),
        Value::Object(values) => !values.is_empty(),
        _ => false,
    }
}

/// Classify every source message and optional top-level tool definition using the lexical
/// collector's presence rules. It distinguishes null/absent fields from present empty strings;
/// object keys, string leaves and retained raw arguments each require their declared field class.
/// This base classification does not require a complete training conversation or tool links.
#[must_use]
pub fn classify_screening_source(
    messages: &[Message],
    tools: Option<&[Value]>,
) -> ScreeningSourceShape {
    let mut fields = BTreeSet::new();
    let mut unsupported = BTreeSet::new();
    for message in messages {
        let content_field = if message.role == Role::Tool {
            ScreeningField::ToolResult
        } else {
            ScreeningField::Content
        };
        match &message.content {
            Content::Text(_) => {
                fields.insert(content_field);
            }
            Content::Null => {}
            Content::Parts(parts) => {
                for part in parts {
                    match part {
                        ContentPart::Text { .. } => {
                            fields.insert(content_field);
                        }
                        _ => {
                            unsupported.insert("unsupported_media");
                        }
                    }
                }
            }
        }
        if message.reasoning.is_some() {
            fields.insert(ScreeningField::Reasoning);
        }
        for detail in message.reasoning_details.iter().flatten() {
            match detail {
                ReasoningDetail::Text { .. } | ReasoningDetail::Summary { .. } => {
                    fields.insert(ScreeningField::ReasoningDetail);
                }
                ReasoningDetail::Encrypted { .. } => {
                    unsupported.insert("encrypted_reasoning");
                }
            }
        }
        if message.name.is_some() {
            fields.insert(ScreeningField::ToolName);
        }
        for call in message.tool_calls.iter().flatten() {
            fields.insert(ScreeningField::ToolName);
            if !call.function.arguments.is_object() {
                unsupported.insert("nonobject_tool_arguments");
            }
            if has_json_text(&call.function.arguments) || call.function.raw_arguments.is_some() {
                fields.insert(ScreeningField::ToolArguments);
            }
        }
    }
    if tools.is_some_and(|definitions| definitions.iter().any(has_json_text)) {
        fields.insert(ScreeningField::ToolDefinition);
    }
    ScreeningSourceShape {
        required_fields: fields.into_iter().collect(),
        unsupported_reasons: unsupported.into_iter().collect(),
    }
}

/// Add the existing complete-tool-link and selected-SFT-prefix prerequisites to source field
/// classification. Calls without results retain the tool validator's existing acceptance behavior.
/// Rendering still belongs to the format crate; this shares only its pure source predicates.
#[must_use]
pub fn classify_screening_training_source(
    messages: &[Message],
    tools: Option<&[Value]>,
    target: TrlFormat,
    turns: MultiTurnLoss,
) -> ScreeningSourceShape {
    let mut shape = classify_screening_source(messages, tools);
    if crate::validate_tool_links(messages).is_err() {
        shape.unsupported_reasons.push("invalid_tool_links");
    }
    if crate::sft_source_targets(messages, target, turns).is_err() {
        shape
            .unsupported_reasons
            .push("unsupported_training_projection");
    }
    shape.unsupported_reasons.sort_unstable();
    shape.unsupported_reasons.dedup();
    shape
}

#[cfg(test)]
#[path = "screening_source_tests.rs"]
mod tests;
