//! Complete serial source validation and independently reconstructed native ownership.
use crate::{
    Content, Message, PreparedSftExample, PreparedSftProfileName, PreparedSftRecipe, Role,
};
use serde_json::Value;
use std::collections::BTreeSet;
type Result<T> = std::result::Result<T, &'static str>;

pub(super) fn is_profile(recipe: &PreparedSftRecipe) -> bool {
    recipe
        .preparation_profile
        .as_ref()
        .is_some_and(|p| p.name == PreparedSftProfileName::Gemma4ThirtyOneBToolsV1)
}
fn keys(value: &Value, allowed: &[&str]) -> Result<()> {
    if value
        .as_object()
        .is_none_or(|m| m.keys().any(|k| !allowed.contains(&k.as_str())))
    {
        return Err("unsupported serial source fields");
    }
    Ok(())
}
fn identifier(text: &str) -> Result<()> {
    if text.is_empty()
        || !text.chars().all(|c| c.is_alphanumeric() || c == '_')
        || text
            .chars()
            .next()
            .is_none_or(|c| !c.is_alphabetic() && c != '_')
    {
        return Err("unescaped serial tool names/keys require identifiers");
    }
    Ok(())
}
fn argument(value: &Value, depth: usize) -> Result<()> {
    if depth > 64 {
        return Err("serial arguments too deeply nested");
    }
    match value {
        Value::Object(map) => {
            let mut seen = BTreeSet::new();
            for (key, value) in map {
                identifier(key)?;
                if !seen.insert(key.to_lowercase()) {
                    return Err("case-folded serial keys ambiguous");
                }
                argument(value, depth + 1)?;
            }
        }
        Value::Array(values) => {
            for value in values {
                argument(value, depth + 1)?;
            }
        }
        Value::Number(n) if n.is_f64() => {
            crate::prepared_gemma31b_render::argument(value, false)?;
        }
        _ => (),
    }
    Ok(())
}
fn schema(value: &Value, root: bool, depth: usize) -> Result<()> {
    if depth > 64 {
        return Err("serial schema too deeply nested");
    }
    let kind = value["type"]
        .as_str()
        .ok_or("serial schema requires type")?;
    let allowed: &[&str] = if root {
        &["type", "properties", "required"]
    } else {
        match kind {
            "object" => &["type", "description", "properties", "required"],
            "array" => &["type", "description", "items"],
            "string" => &["type", "description", "enum"],
            _ => &["type", "description"],
        }
    };
    keys(value, allowed)?;
    if kind == "object" {
        let properties = value["properties"]
            .as_object()
            .ok_or("serial objects require explicit properties")?;
        let mut seen = BTreeSet::new();
        for (key, value) in properties {
            identifier(key)?;
            if !seen.insert(key.to_lowercase()) {
                return Err("case-folded properties ambiguous");
            }
            schema(value, false, depth + 1)?;
        }
    }
    if kind == "array" {
        schema(&value["items"], false, depth + 1)?;
    }
    if let Some(values) = value.get("enum")
        && values
            .as_array()
            .is_none_or(|v| v.is_empty() || v.iter().any(|v| !v.is_string()))
    {
        return Err("serial string enum requires text entries");
    }
    Ok(())
}
fn source(
    example: &PreparedSftExample,
    recipe: &PreparedSftRecipe,
) -> Result<(Vec<Value>, Vec<Value>)> {
    let messages: Vec<Value> = serde_json::from_value(crate::prepared_gemma31b_json::parse(
        &example.source.messages_json,
    )?)
    .map_err(|_| "invalid serial messages")?;
    let typed: Vec<Message> = serde_json::from_str(&example.source.messages_json)
        .map_err(|_| "invalid serial typed messages")?;
    let tools = match &example.source.tools_json {
        Some(Some(raw)) => {
            serde_json::from_value::<Vec<Value>>(crate::prepared_gemma31b_json::parse(raw)?)
                .map_err(|_| "invalid serial definitions")?
        }
        Some(None) => vec![],
        None => return Err("serial tools require complete v5 source"),
    };
    crate::validate_tool_training_source(&typed, Some(&tools))
        .map_err(|_| "invalid complete serial source")?;
    let literals: Vec<_> = recipe.tokenizer_policy["added_tokens"]
        .as_array()
        .ok_or("missing serial controls")?
        .iter()
        .filter_map(|t| t["content"].as_str())
        .collect();
    crate::validate_tool_projection_delimiters(&typed, Some(&tools), &literals)
        .map_err(|_| "ambiguous serial control literal")?;
    for tool in &tools {
        let function = &tool["function"];
        keys(function, &["name", "description", "parameters"])?;
        identifier(
            function["name"]
                .as_str()
                .ok_or("missing serial function name")?,
        )?;
        schema(&function["parameters"], true, 0)?;
    }
    let mut phase = Role::User;
    let mut pending = None;
    for (index, (raw, m)) in messages.iter().zip(&typed).enumerate() {
        keys(
            raw,
            &[
                "role",
                "content",
                "reasoning",
                "reasoning_details",
                "tool_calls",
                "tool_call_id",
                "name",
            ],
        )?;
        if let Some(calls) = raw["tool_calls"].as_array() {
            for call in calls {
                keys(call, &["id", "type", "function"])?;
                if call.get("type").is_some_and(|t| t != "function") {
                    return Err("unsupported serial call type");
                }
                keys(&call["function"], &["name", "arguments", "raw_arguments"])?;
            }
        }
        if let Some(details) = raw.get("reasoning_details").filter(|v| !v.is_null()) {
            let details = details
                .as_array()
                .ok_or("serial reasoning details require a list")?;
            let flat = m
                .reasoning
                .as_deref()
                .ok_or("serial details require flat reasoning")?;
            let mut joined = String::new();
            let mut previous = None;
            for detail in details {
                keys(
                    detail,
                    &["type", "text", "index", "signature", "id", "format"],
                )?;
                let index = detail["index"]
                    .as_u64()
                    .filter(|n| *n <= u64::from(u32::MAX))
                    .ok_or("serial detail index invalid")?;
                if detail["type"] != "reasoning.text" || previous.is_some_and(|p| p >= index) {
                    return Err("serial details require ordered text");
                }
                joined.push_str(
                    detail["text"]
                        .as_str()
                        .ok_or("serial detail text required")?,
                );
                previous = Some(index);
            }
            if joined != flat {
                return Err("serial reasoning details differ from flat text");
            }
        }
        if m.role != Role::Assistant && (m.reasoning.is_some() || m.reasoning_details.is_some()) {
            return Err("serial reasoning requires assistant");
        }
        if index == 0 && m.role == Role::System {
            if !matches!(m.content, Content::Text(_)) || m.tool_calls.is_some() {
                return Err("serial system requires text");
            }
        } else if m.role == Role::User && phase == Role::User {
            if !matches!(m.content, Content::Text(_)) || m.tool_calls.is_some() {
                return Err("serial user requires text");
            }
            phase = Role::Assistant;
        } else if m.role == Role::Assistant && phase == Role::Assistant {
            if let Some(calls) = &m.tool_calls {
                if calls.len() != 1
                    || !(matches!(&m.content,Content::Text(s) if s.is_empty())
                        || matches!(m.content, Content::Null))
                {
                    return Err("serial call requires one call and empty content");
                }
                argument(&calls[0].function.arguments, 0)?;
                pending = calls[0].id.as_deref();
                phase = Role::Tool;
            } else {
                if !matches!(&m.content,Content::Text(s) if !s.trim().is_empty()) {
                    return Err("serial answer requires text");
                }
                phase = Role::User;
            }
        } else if m.role == Role::Tool && phase == Role::Tool {
            if !matches!(m.content, Content::Text(_))
                || m.tool_call_id.as_deref() != pending
                || m.tool_calls.is_some()
            {
                return Err("serial result requires immediate explicit link");
            }
            phase = Role::Assistant;
        } else {
            return Err("unsupported serial message order");
        }
    }
    if phase != Role::User
        || typed
            .last()
            .is_none_or(|m| m.role != Role::Assistant || m.tool_calls.is_some())
    {
        return Err("complete serial source must end in answer");
    }
    Ok((messages, tools))
}
pub(super) fn validate(example: &PreparedSftExample, recipe: &PreparedSftRecipe) -> Result<()> {
    let (messages, tools) = source(example, recipe)?;
    let target =
        usize::try_from(example.target_index).map_err(|_| "serial target outside source")?;
    let message = messages
        .get(target)
        .filter(|m| m["role"] == "assistant")
        .ok_or("serial target must name assistant")?;
    let call = message["tool_calls"].is_array();
    let controls = &recipe
        .preparation_profile
        .as_ref()
        .ok_or("serial profile missing")?
        .controls;
    let (rendered, spans) = crate::prepared_gemma31b_render::render(
        &messages[..=target],
        &tools,
        recipe.cot_policy,
        controls,
    )?;
    if rendered != example.rendered || spans != example.spans {
        return Err("serial rendering/ownership differs from captured canonical source");
    }
    if example.target_kind.as_deref() != Some(if call { "tool_call" } else { "text_answer" })
        || example.shifted_call_token_indices.is_none()
        || (call && !example.shifted_answer_token_indices.is_empty())
        || (!call
            && example
                .shifted_call_token_indices
                .as_ref()
                .is_some_and(|v| !v.is_empty()))
        || example.input_ids.first() != Some(&2)
        || example.labels.first() != Some(&-100)
        || example.offset_mapping.first() != Some(&[0, 5])
        || (call && (example.input_ids.last() != Some(&50) || example.labels.last() != Some(&50)))
        || (!call
            && (!example.input_ids.ends_with(&[106, 107])
                || !example.labels.ends_with(&[106, -100])))
    {
        return Err("serial call/answer ending or token evidence disagrees");
    }
    Ok(())
}
