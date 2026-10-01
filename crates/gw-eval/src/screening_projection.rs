//! Structured source prefixes and independently bounded text segments. Rendering remains owned
//! by gw-format; these shapes retain distinctions a particular renderer may collapse.
use crate::screening_lexical::TextIndex;
use gw_schema::*;
use serde_json::{Value, json};

pub(crate) struct PromptUnit {
    pub id: String,
    pub shape: Value,
    pub segments: Vec<usize>,
    pub has_user_text: bool,
}
pub(crate) struct TrainingUnit {
    pub prompt: PromptUnit,
    pub rendered: String,
}
pub(crate) struct RecordProjection {
    pub units: Vec<TrainingUnit>,
    pub segments: Vec<usize>,
    pub unsupported: Vec<&'static str>,
}
pub(crate) struct ProtectedProjection {
    pub set_id: String,
    pub item_id: String,
    pub prompt: PromptUnit,
    pub segments: Vec<usize>,
    pub unsupported: Vec<&'static str>,
}
struct MessageShape {
    shape: Value,
    segments: Vec<usize>,
    user_text: bool,
}

struct Collector<'a> {
    owner: &'a str,
    index: &'a mut TextIndex,
    limits: &'a ScreeningLimits,
}
impl Collector<'_> {
    fn text(
        &mut self,
        local: &str,
        field: ScreeningField,
        text: &str,
        segments: &mut Vec<usize>,
    ) -> Result<Value, &'static str> {
        segments.push(self.index.add(
            format!("{}:{local}", self.owner),
            field,
            text,
            self.limits,
        )?);
        // Local field coordinates are retained in the structure, independently of owner IDs.
        Ok(json!({"text_segment":local}))
    }
    fn json_text(
        &mut self,
        value: &Value,
        path: &str,
        field: ScreeningField,
        segments: &mut Vec<usize>,
    ) -> Result<Value, &'static str> {
        Ok(match value {
            Value::String(text) => self.text(path, field, text, segments)?,
            Value::Array(values) => Value::Array(
                values
                    .iter()
                    .enumerate()
                    .map(|(i, v)| self.json_text(v, &format!("{path}/{i}"), field, segments))
                    .collect::<Result<_, _>>()?,
            ),
            Value::Object(values) => {
                let mut members: Vec<_> = values.iter().collect();
                members.sort_by_key(|(key, _)| *key);
                Value::Object(
                    members
                        .into_iter()
                        .enumerate()
                        .map(|(i, (key, value))| {
                            // Keys remain exact structural identities and separate bounded text
                            // segments. Evidence uses sorted member coordinates, never key text.
                            let local = format!("{path}/members/{i}");
                            let _ = self.text(&format!("{local}/key"), field, key, segments)?;
                            self.json_text(value, &format!("{local}/value"), field, segments)
                                .map(|value| (key.clone(), value))
                        })
                        .collect::<Result<_, _>>()?,
                )
            }
            other => other.clone(),
        })
    }
    fn message(
        &mut self,
        message: &Message,
        position: usize,
    ) -> Result<MessageShape, &'static str> {
        let path = format!("messages/{position}");
        let mut shape = serde_json::to_value(message).map_err(|_| "message_serialization")?;
        let mut segments = vec![];
        let content_field = if message.role == Role::Tool {
            ScreeningField::ToolResult
        } else {
            ScreeningField::Content
        };
        let mut user_text = false;
        shape["content"] = match &message.content {
            Content::Text(text) => {
                let value = self.text(
                    &format!("{path}/content"),
                    content_field,
                    text,
                    &mut segments,
                )?;
                user_text = message.role == Role::User
                    && !self.index.segments[*segments.last().expect("added segment")]
                        .tokens
                        .is_empty();
                json!({"kind":"text","value":value})
            }
            Content::Null => json!({"kind":"null"}),
            Content::Parts(parts) => {
                let mut shapes = vec![];
                for (part_index, part) in parts.iter().enumerate() {
                    match part {
                        ContentPart::Text { text } => {
                            let value = self.text(
                                &format!("{path}/parts/{part_index}/text"),
                                content_field,
                                text,
                                &mut segments,
                            )?;
                            user_text |= message.role == Role::User
                                && !self.index.segments[*segments.last().expect("added segment")]
                                    .tokens
                                    .is_empty();
                            shapes.push(json!({"type":"text","value":value}));
                        }
                        _ => {
                            shapes.push(
                                serde_json::to_value(part).map_err(|_| "part_serialization")?,
                            );
                        }
                    }
                }
                json!({"kind":"parts","value":shapes})
            }
        };
        if let Some(text) = &message.reasoning {
            shape["reasoning"] = self.text(
                &format!("{path}/reasoning"),
                ScreeningField::Reasoning,
                text,
                &mut segments,
            )?;
        }
        if let Some(details) = &message.reasoning_details {
            for (detail_index, detail) in details.iter().enumerate() {
                let local = format!("{path}/reasoning_details/{detail_index}");
                match detail {
                    ReasoningDetail::Text { text, .. } => {
                        shape["reasoning_details"][detail_index]["text"] = self.text(
                            &format!("{local}/text"),
                            ScreeningField::ReasoningDetail,
                            text,
                            &mut segments,
                        )?
                    }
                    ReasoningDetail::Summary { summary, .. } => {
                        shape["reasoning_details"][detail_index]["summary"] = self.text(
                            &format!("{local}/summary"),
                            ScreeningField::ReasoningDetail,
                            summary,
                            &mut segments,
                        )?
                    }
                    ReasoningDetail::Encrypted { .. } => {}
                }
            }
        }
        if let Some(name) = &message.name {
            // Names/links remain exact structural identities as well as screened text fields.
            let _ = self.text(
                &format!("{path}/name"),
                ScreeningField::ToolName,
                name,
                &mut segments,
            )?;
        }
        if let Some(calls) = &message.tool_calls {
            for (call_index, call) in calls.iter().enumerate() {
                let local = format!("{path}/tool_calls/{call_index}/function");
                let _ = self.text(
                    &format!("{local}/name"),
                    ScreeningField::ToolName,
                    &call.function.name,
                    &mut segments,
                )?;
                shape["tool_calls"][call_index]["function"]["arguments"] = self.json_text(
                    &call.function.arguments,
                    &format!("{local}/arguments"),
                    ScreeningField::ToolArguments,
                    &mut segments,
                )?;
                if let Some(raw) = &call.function.raw_arguments {
                    shape["tool_calls"][call_index]["function"]["raw_arguments"] = self.text(
                        &format!("{local}/raw_arguments"),
                        ScreeningField::ToolArguments,
                        raw,
                        &mut segments,
                    )?;
                }
            }
        }
        Ok(MessageShape {
            shape,
            segments,
            user_text,
        })
    }
}

pub(crate) fn project_record(
    record: &TrainingRecord,
    owner: &str,
    policy: &ScreeningPolicy,
    classification: ScreeningSourceShape,
    index: &mut TextIndex,
) -> Result<RecordProjection, &'static str> {
    let mut collect = Collector {
        owner,
        index,
        limits: &policy.limits,
    };
    let messages = record
        .messages
        .iter()
        .enumerate()
        .map(|(i, m)| collect.message(m, i))
        .collect::<Result<Vec<_>, _>>()?;
    let mut tool_segments = vec![];
    let tools = match &record.tools {
        None => Value::Null,
        Some(tools) => collect.json_text(
            &json!(tools),
            "tools",
            ScreeningField::ToolDefinition,
            &mut tool_segments,
        )?,
    };
    let mut unsupported = classification.unsupported_reasons;
    let rendered = match gw_format::project_sft_units(
        record,
        policy.target,
        policy.cot_policy,
        policy.multi_turn_loss,
    ) {
        Ok(units) => units,
        Err(_) => {
            if !unsupported.contains(&"unsupported_training_projection") {
                unsupported.push("unsupported_training_projection");
            }
            vec![]
        }
    };
    let units=rendered.into_iter().map(|unit| {
        let prefix=&messages[..unit.target_index];
        let mut segments:Vec<_>=prefix.iter().flat_map(|m|m.segments.iter().copied()).collect();
        segments.extend(tool_segments.iter().copied());
        TrainingUnit {
            prompt:PromptUnit {
                id:format!("{owner}:target/{}",unit.target_index),
                shape:json!({"messages":prefix.iter().map(|m|&m.shape).collect::<Vec<_>>(),"tools":tools}),
                segments,
                has_user_text:prefix.iter().any(|m|m.user_text),
            },
            rendered:unit.rendered,
        }
    }).collect();
    let mut segments: Vec<_> = messages.into_iter().flat_map(|m| m.segments).collect();
    segments.extend(tool_segments);
    let actual_fields: std::collections::BTreeSet<_> = segments
        .iter()
        .map(|&index| collect.index.segments[index].field)
        .collect();
    if actual_fields.into_iter().collect::<Vec<_>>() != classification.required_fields {
        return Err("source_field_classification_mismatch");
    }
    Ok(RecordProjection {
        units,
        segments,
        unsupported,
    })
}

pub(crate) fn project_protected(
    set_id: &str,
    item: &ProtectedScreeningItem,
    owner: &str,
    policy: &ScreeningPolicy,
    index: &mut TextIndex,
) -> Result<ProtectedProjection, &'static str> {
    let mut all = item.prompt.clone();
    all.extend(item.responses.iter().cloned());
    let mut unsupported = classify_screening_source(&all, None).unsupported_reasons;
    let mut collect = Collector {
        owner,
        index,
        limits: &policy.limits,
    };
    let prompt = item
        .prompt
        .iter()
        .enumerate()
        .map(|(i, m)| collect.message(m, i))
        .collect::<Result<Vec<_>, _>>()?;
    let prompt_segments: Vec<_> = prompt
        .iter()
        .flat_map(|m| m.segments.iter().copied())
        .collect();
    let unit = PromptUnit {
        id: format!("{owner}:prompt"),
        shape: json!({"messages":prompt.iter().map(|m|&m.shape).collect::<Vec<_>>(),"tools":Value::Null}),
        segments: prompt_segments.clone(),
        has_user_text: prompt.iter().any(|m| m.user_text),
    };
    if !unit.has_user_text {
        unsupported.push("incomplete_protected_prompt");
    }
    let mut segments = prompt_segments;
    for (i, message) in item.responses.iter().enumerate() {
        // Response fields have disjoint global segment identities from the complete prompt.
        let response = collect.message(message, item.prompt.len() + i)?;
        segments.extend(response.segments);
    }
    if gw_schema::validate_tool_links(&all).is_err() {
        unsupported.push("invalid_protected_tool_links");
    }
    Ok(ProtectedProjection {
        set_id: set_id.into(),
        item_id: item.item_id.clone(),
        prompt: unit,
        segments,
        unsupported,
    })
}
