//! Native function tools and the separate canonical-to-wire message mapping.
use gw_schema::{Message, Role};
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Value, json};

/// One native function definition. The parameter schema is transmitted unchanged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// Nonempty function name, unique within a request.
    pub name: String,
    /// Human-readable function description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON object describing the function's parameters.
    pub parameters: Value,
    /// Optional provider strict-schema control; no support is inferred from its presence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// Supported native tool definition. Other tool kinds are not silently downgraded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolDefinition {
    /// An OpenAI-compatible function tool.
    Function {
        /// Function signature and parameter schema.
        function: FunctionDefinition,
    },
}

/// Native tool selection, including an explicit named function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    /// Permit a text answer or a tool call.
    Auto,
    /// Disallow tool calls for this turn.
    None,
    /// Require at least one tool call.
    Required,
    /// Require this exact declared function.
    Function(String),
}
impl Serialize for ToolChoice {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Auto => serializer.serialize_str("auto"),
            Self::None => serializer.serialize_str("none"),
            Self::Required => serializer.serialize_str("required"),
            Self::Function(name) => {
                json!({"type":"function","function":{"name":name}}).serialize(serializer)
            }
        }
    }
}

/// Native tool controls applied together, so an omitted capability cannot leave a dangling choice.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ToolConfig {
    /// Nonempty function definitions with unique names.
    pub tools: Vec<ToolDefinition>,
    /// Requested selection behavior; absent means the server's default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// Explicit parallel-call policy; absent means the server's default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
}
impl ToolConfig {
    /// Validate shape and named selection without claiming server/model support.
    pub fn validate(&self) -> Result<(), crate::ProviderError> {
        let mut names = std::collections::BTreeSet::new();
        for ToolDefinition::Function { function } in &self.tools {
            if function.name.trim().is_empty()
                || !function.parameters.is_object()
                || !names.insert(&function.name)
            {
                return Err(invalid());
            }
        }
        if names.is_empty()
            || matches!(&self.tool_choice, Some(ToolChoice::Function(name)) if !names.contains(name))
        {
            return Err(invalid());
        }
        Ok(())
    }
}
fn invalid() -> crate::ProviderError {
    crate::ProviderError::Config("invalid native function-tool contract".into())
}

/// Convert persisted canonical messages to OpenAI-compatible wire values.
/// Structured arguments remain unchanged in storage; only the wire uses a JSON string.
/// Raw argument evidence is reused only when it decodes to the canonical value.
///
/// # Errors
/// Rejects invalid argument objects, missing call identities, and invalid tool-role linkage.
pub fn wire_messages(messages: &[Message]) -> Result<Vec<Value>, crate::ProviderError> {
    gw_schema::validate_tool_links(messages).map_err(|_| invalid())?;
    messages.iter().map(wire_message).collect()
}
fn wire_message(message: &Message) -> Result<Value, crate::ProviderError> {
    let mut value = serde_json::to_value(message).map_err(|_| invalid())?;
    if message.role == Role::Tool
        && message
            .tool_call_id
            .as_ref()
            .is_none_or(|id| id.trim().is_empty())
    {
        return Err(invalid());
    }
    if let Some(calls) = &message.tool_calls {
        if message.role != Role::Assistant || calls.is_empty() {
            return Err(invalid());
        }
        let mut ids = std::collections::BTreeSet::new();
        let mut wire = Vec::new();
        for call in calls {
            let id = call
                .id
                .as_ref()
                .filter(|id| !id.trim().is_empty())
                .ok_or_else(invalid)?;
            if !ids.insert(id)
                || call.function.name.trim().is_empty()
                || !call.function.arguments.is_object()
            {
                return Err(invalid());
            }
            let arguments = match &call.function.raw_arguments {
                Some(raw)
                    if serde_json::from_str::<Value>(raw).ok().as_ref()
                        == Some(&call.function.arguments) =>
                {
                    raw.clone()
                }
                _ => serde_json::to_string(&call.function.arguments).map_err(|_| invalid())?,
            };
            wire.push(json!({"id":id,"type":"function","function":{"name":call.function.name,"arguments":arguments}}));
        }
        value["tool_calls"] = Value::Array(wire);
    }
    Ok(value)
}

pub(crate) fn serialize_messages<S: Serializer>(
    messages: &[Message],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    wire_messages(messages)
        .map_err(serde::ser::Error::custom)?
        .serialize(serializer)
}

/// One indexed native tool-call fragment from a streaming delta.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallDelta {
    /// Zero-based call index, stable for the duration of one physical response.
    pub index: u32,
    /// Call identifier, normally supplied on the initial fragment.
    pub id: Option<String>,
    /// Tool kind, when supplied. Only `function` is supported by the assembler.
    #[serde(rename = "type")]
    pub kind: Option<String>,
    /// Incremental function name and JSON-encoded arguments.
    pub function: Option<FunctionDelta>,
}

/// Incremental strings are appended in arrival order within their call index.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FunctionDelta {
    /// Function-name fragment.
    pub name: Option<String>,
    /// Raw JSON argument fragment; parsed only after terminal completion.
    pub arguments: Option<String>,
}
