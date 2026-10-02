//! Assemble one response's indexed function fragments without executing tools.
use crate::{GenerateError, Result};
use gw_providers::ToolCallDelta;
use gw_schema::{FunctionCall, ToolCall};
use std::collections::{BTreeMap, BTreeSet};

/// Accumulated native call fragments for one physical response.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct StreamedToolCalls {
    calls: BTreeMap<u32, PartialCall>,
    invalid: Option<&'static str>,
}
#[derive(Debug, Default, Clone, PartialEq)]
struct PartialCall {
    id: Option<String>,
    name: String,
    arguments: String,
}
impl StreamedToolCalls {
    /// Append indexed fragments in arrival order. Invalid evidence remains sticky until finish.
    pub fn push(&mut self, fragments: Vec<ToolCallDelta>) {
        for fragment in fragments {
            if fragment
                .kind
                .as_deref()
                .is_some_and(|kind| kind != "function")
            {
                self.invalid = Some("unsupported tool-call kind");
            }
            let call = self.calls.entry(fragment.index).or_default();
            if let Some(id) = fragment.id {
                if call.id.as_ref().is_some_and(|prior| prior != &id) {
                    self.invalid = Some("conflicting tool-call identity");
                } else {
                    call.id = Some(id);
                }
            }
            if let Some(function) = fragment.function {
                if let Some(name) = function.name {
                    call.name.push_str(&name);
                }
                if let Some(arguments) = function.arguments {
                    call.arguments.push_str(&arguments);
                }
            }
        }
    }
    /// Whether no call fragments have been received.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }

    pub(crate) fn finish(&self, finish: Option<&str>) -> Result<Option<Vec<ToolCall>>> {
        if let Some(reason) = self.invalid {
            return Err(invalid(reason));
        }
        if self.calls.is_empty() {
            if finish == Some("tool_calls") {
                return Err(invalid("tool_calls finish without calls"));
            }
            return Ok(None);
        }
        if finish != Some("tool_calls") {
            return Err(invalid("tool calls lack a supported terminal finish"));
        }
        let mut ids = BTreeSet::new();
        let mut calls = Vec::new();
        for (expected, (&index, call)) in self.calls.iter().enumerate() {
            if u32::try_from(expected).ok() != Some(index) {
                return Err(invalid("noncontiguous tool-call indices"));
            }
            let id = call
                .id
                .as_ref()
                .filter(|id| !id.trim().is_empty())
                .ok_or_else(|| invalid("missing tool-call identity"))?;
            if !ids.insert(id) || call.name.trim().is_empty() {
                return Err(invalid("duplicate identity or missing function name"));
            }
            let arguments: serde_json::Value = serde_json::from_str(&call.arguments)
                .map_err(|_| invalid("incomplete or malformed tool arguments"))?;
            if !arguments.is_object() {
                return Err(invalid("tool arguments must be an object"));
            }
            calls.push(ToolCall {
                id: Some(id.clone()),
                function: FunctionCall {
                    name: call.name.clone(),
                    arguments,
                    raw_arguments: Some(call.arguments.clone()),
                },
            });
        }
        Ok(Some(calls))
    }
}
fn invalid(detail: &str) -> GenerateError {
    GenerateError::InvalidCompletion(detail.into())
}

pub(crate) fn validate_selection(
    message: &gw_schema::Message,
    tools: Option<&gw_providers::ToolConfig>,
) -> Result<()> {
    use gw_providers::{ToolChoice, ToolDefinition};
    let calls = message.tool_calls.as_deref().unwrap_or_default();
    let Some(config) = tools else {
        return if calls.is_empty() {
            Ok(())
        } else {
            Err(invalid("unrequested tool calls"))
        };
    };
    if (config.parallel_tool_calls == Some(false) && calls.len() > 1)
        || (matches!(config.tool_choice, Some(ToolChoice::None)) && !calls.is_empty())
        || (matches!(
            config.tool_choice,
            Some(ToolChoice::Required | ToolChoice::Function(_))
        ) && calls.is_empty())
    {
        return Err(invalid("response violates requested tool selection"));
    }
    for call in calls {
        if !config
            .tools
            .iter()
            .any(|ToolDefinition::Function { function }| function.name == call.function.name)
            || matches!(&config.tool_choice, Some(ToolChoice::Function(name)) if name != &call.function.name)
        {
            return Err(invalid(
                "response called an undeclared or unselected function",
            ));
        }
    }
    Ok(())
}
