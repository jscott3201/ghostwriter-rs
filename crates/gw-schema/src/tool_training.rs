//! Strict complete tool sources, separate from permissive canonical capture.
use std::collections::{BTreeMap, BTreeSet};

use crate::{Content, Message, Role};
use serde_json::Value;

type Result<T> = std::result::Result<T, String>;

fn invalid(message: &str) -> String {
    message.into()
}

fn name(value: Option<&Value>) -> Result<&str> {
    value
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| invalid("function names must be nonempty strings"))
}

/// Validate a complete text/function training source without altering its canonical evidence.
///
/// Requires OpenAI `type: function` wrappers, object parameter schemas and arguments, unique
/// definition names and call IDs, and exactly one explicitly linked later result for every call.
/// Parallel calls, including repeated function names and reordered results, are supported here.
/// A model consumer must separately check its own trajectory and template restrictions.
/// `raw_arguments` is retained evidence only; this validator uses parsed `arguments`.
///
/// Supported schema keywords are `type`, `description`, `properties`, `required`, `items`,
/// `enum`, and boolean `additionalProperties`. References, compositions and multimodal content
/// are rejected. This checks schema structure and source linkage, not argument conformance to
/// the schema. It makes no tokenizer, loss-label or student-quality claim.
///
/// # Errors
/// Returns a descriptive error for unsupported or incomplete sources.
pub fn validate_tool_training_source(messages: &[Message], tools: Option<&[Value]>) -> Result<()> {
    let mut definitions = BTreeSet::new();
    for tool in tools.unwrap_or_default() {
        let wrapper = tool
            .as_object()
            .ok_or_else(|| invalid("tool definition must be an object"))?;
        if wrapper.len() != 2 || wrapper.get("type").and_then(Value::as_str) != Some("function") {
            return Err(invalid("only type/function tool wrappers are supported"));
        }
        let function = wrapper
            .get("function")
            .and_then(Value::as_object)
            .ok_or_else(|| invalid("function definition must be an object"))?;
        if function.keys().any(|key| {
            !matches!(
                key.as_str(),
                "name" | "description" | "parameters" | "strict"
            )
        }) || function.get("description").is_some_and(|v| !v.is_string())
            || function.get("strict").is_some_and(|v| !v.is_boolean())
        {
            return Err(invalid("unsupported function definition field"));
        }
        if !definitions.insert(name(function.get("name"))?) {
            return Err(invalid("duplicate function definition name"));
        }
        let parameters = function
            .get("parameters")
            .ok_or_else(|| invalid("function parameters are required"))?;
        if parameters.get("type").and_then(Value::as_str) != Some("object") {
            return Err(invalid("function parameters must declare object type"));
        }
        validate_schema(parameters, 0)?;
    }
    let mut calls = BTreeMap::new();
    let mut replies = BTreeSet::new();
    for message in messages {
        if matches!(message.content, Content::Parts(_)) {
            return Err(invalid("only text or null content is supported"));
        }
        if let Some(declared) = &message.tool_calls {
            if message.role != Role::Assistant {
                return Err(invalid("only assistant messages may declare calls"));
            }
            for call in declared {
                let id = call
                    .id
                    .as_deref()
                    .filter(|id| !id.trim().is_empty())
                    .ok_or_else(|| invalid("call ID must be explicit and nonempty"))?;
                if !definitions.contains(call.function.name.as_str()) {
                    return Err(invalid("call must resolve to a function definition"));
                }
                if !call.function.arguments.is_object() {
                    return Err(invalid("call arguments must be an object without coercion"));
                }
                if calls.insert(id, call.function.name.as_str()).is_some() {
                    return Err(invalid("duplicate call ID"));
                }
            }
        }
        if message.role == Role::Tool {
            let id = message
                .tool_call_id
                .as_deref()
                .filter(|id| !id.trim().is_empty())
                .ok_or_else(|| invalid("result must carry an explicit nonempty call ID"))?;
            let called = calls
                .get(id)
                .ok_or_else(|| invalid("result is dangling or precedes its call"))?;
            if !replies.insert(id) {
                return Err(invalid("each call requires exactly one result"));
            }
            if message.name.as_deref().is_some_and(|name| name != *called) {
                return Err(invalid("result name contradicts its linked function"));
            }
        } else if message.tool_call_id.is_some() || message.name.is_some() {
            return Err(invalid("result identity fields require the tool role"));
        }
    }
    if calls.len() != replies.len() {
        return Err(invalid("every call requires a later explicit result"));
    }
    Ok(())
}

fn validate_schema(value: &Value, depth: usize) -> Result<()> {
    if depth > 64 {
        return Err(invalid("parameter schema nesting exceeds supported depth"));
    }
    let schema = value
        .as_object()
        .ok_or_else(|| invalid("parameter schema must be an object"))?;
    if schema.keys().any(|key| {
        !matches!(
            key.as_str(),
            "type"
                | "description"
                | "properties"
                | "required"
                | "items"
                | "enum"
                | "additionalProperties"
        )
    }) {
        return Err(invalid("unsupported parameter schema keyword or reference"));
    }
    let kind = schema
        .get("type")
        .and_then(Value::as_str)
        .filter(|kind| {
            matches!(
                *kind,
                "object" | "array" | "string" | "number" | "integer" | "boolean" | "null"
            )
        })
        .ok_or_else(|| invalid("schema type must be one supported explicit string"))?;
    if schema.get("description").is_some_and(|v| !v.is_string())
        || schema
            .get("additionalProperties")
            .is_some_and(|v| kind != "object" || !v.is_boolean())
        || schema
            .get("enum")
            .is_some_and(|v| v.as_array().is_none_or(Vec::is_empty))
    {
        return Err(invalid("malformed schema annotation or constraint"));
    }
    if let Some(properties) = schema.get("properties") {
        let properties = properties
            .as_object()
            .filter(|_| kind == "object")
            .ok_or_else(|| invalid("properties require an object schema"))?;
        for property in properties.values() {
            validate_schema(property, depth + 1)?;
        }
    }
    if let Some(required) = schema.get("required") {
        let required = required
            .as_array()
            .filter(|_| kind == "object")
            .ok_or_else(|| invalid("required must be an array on an object schema"))?;
        let mut seen = BTreeSet::new();
        for entry in required {
            let key = entry
                .as_str()
                .ok_or_else(|| invalid("required entries must be strings"))?;
            if !seen.insert(key)
                || schema
                    .get("properties")
                    .and_then(Value::as_object)
                    .is_none_or(|p| !p.contains_key(key))
            {
                return Err(invalid(
                    "required entries must name distinct declared properties",
                ));
            }
        }
    }
    if let Some(items) = schema.get("items") {
        if kind != "array" {
            return Err(invalid("items require an array schema"));
        }
        validate_schema(items, depth + 1)?;
    }
    Ok(())
}

/// Check a separate projection prerequisite: no selected template delimiter may occur anywhere
/// in message fields, reasoning, nested parsed arguments, or definition keys and values.
///
/// Supply the consumer's actual special-token spellings. Canonical publication does not invoke
/// this check and remains lossless. Retained `raw_arguments` is excluded because it is never a
/// training target; parsed arguments are checked instead.
///
/// # Errors
/// Returns a descriptive error for an empty delimiter or ambiguous source text.
pub fn validate_tool_projection_delimiters(
    messages: &[Message],
    tools: Option<&[Value]>,
    delimiters: &[&str],
) -> Result<()> {
    if delimiters.iter().any(|delimiter| delimiter.is_empty()) {
        return Err(invalid("projection delimiters must be nonempty"));
    }
    let mut projected = messages.to_vec();
    for message in &mut projected {
        for call in message.tool_calls.iter_mut().flatten() {
            call.function.raw_arguments = None;
        }
    }
    fn ambiguous(value: &Value, delimiters: &[&str]) -> bool {
        let contains = |text: &str| delimiters.iter().any(|delimiter| text.contains(delimiter));
        match value {
            Value::String(text) => contains(text),
            Value::Array(values) => values.iter().any(|value| ambiguous(value, delimiters)),
            Value::Object(values) => values
                .iter()
                .any(|(key, value)| contains(key) || ambiguous(value, delimiters)),
            _ => false,
        }
    }
    if ambiguous(
        &serde_json::to_value((&projected, tools)).map_err(|error| error.to_string())?,
        delimiters,
    ) {
        return Err(invalid("source contains a template control delimiter"));
    }
    Ok(())
}
