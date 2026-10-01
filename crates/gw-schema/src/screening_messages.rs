//! Strict protected-manifest message decoding. Ordinary record parsing is deliberately unchanged.
use crate::{Content, ContentPart, FunctionCall, Message, ReasoningDetail, Role, ToolCall};
use serde::{Deserialize, Deserializer};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StrictMessage {
    role: Role,
    content: StrictContent,
    reasoning: Option<String>,
    reasoning_details: Option<Vec<StrictReasoning>>,
    tool_calls: Option<Vec<StrictToolCall>>,
    tool_call_id: Option<String>,
    name: Option<String>,
}
#[derive(Deserialize)]
#[serde(untagged)]
enum StrictContent {
    Text(String),
    Parts(Vec<StrictPart>),
    Null,
}
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum StrictPart {
    Text {
        text: String,
    },
    ImageUrl {
        image_url: String,
    },
    InputAudio {
        audio_url: Option<String>,
        format: Option<String>,
    },
}
#[derive(Deserialize)]
#[serde(tag = "type", deny_unknown_fields)]
enum StrictReasoning {
    #[serde(rename = "reasoning.text")]
    Text {
        text: String,
        signature: Option<String>,
        id: Option<String>,
        format: Option<String>,
        index: u32,
    },
    #[serde(rename = "reasoning.summary")]
    Summary {
        summary: String,
        id: Option<String>,
        format: Option<String>,
        index: u32,
    },
    #[serde(rename = "reasoning.encrypted")]
    Encrypted {
        data: String,
        id: Option<String>,
        format: Option<String>,
        index: u32,
    },
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StrictToolCall {
    id: Option<String>,
    function: StrictFunction,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StrictFunction {
    name: String,
    arguments: serde_json::Value,
    raw_arguments: Option<String>,
}

impl From<StrictContent> for Content {
    fn from(value: StrictContent) -> Self {
        match value {
            StrictContent::Text(text) => Self::Text(text),
            StrictContent::Null => Self::Null,
            StrictContent::Parts(parts) => Self::Parts(
                parts
                    .into_iter()
                    .map(|part| match part {
                        StrictPart::Text { text } => ContentPart::Text { text },
                        StrictPart::ImageUrl { image_url } => ContentPart::ImageUrl { image_url },
                        StrictPart::InputAudio { audio_url, format } => {
                            ContentPart::InputAudio { audio_url, format }
                        }
                    })
                    .collect(),
            ),
        }
    }
}
impl From<StrictReasoning> for ReasoningDetail {
    fn from(value: StrictReasoning) -> Self {
        match value {
            StrictReasoning::Text {
                text,
                signature,
                id,
                format,
                index,
            } => Self::Text {
                text,
                signature,
                id,
                format,
                index,
            },
            StrictReasoning::Summary {
                summary,
                id,
                format,
                index,
            } => Self::Summary {
                summary,
                id,
                format,
                index,
            },
            StrictReasoning::Encrypted {
                data,
                id,
                format,
                index,
            } => Self::Encrypted {
                data,
                id,
                format,
                index,
            },
        }
    }
}

pub(super) fn deserialize<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<Message>, D::Error> {
    Ok(Vec::<StrictMessage>::deserialize(deserializer)?
        .into_iter()
        .map(|value| Message {
            role: value.role,
            content: value.content.into(),
            reasoning: value.reasoning,
            reasoning_details: value
                .reasoning_details
                .map(|details| details.into_iter().map(Into::into).collect()),
            tool_calls: value.tool_calls.map(|calls| {
                calls
                    .into_iter()
                    .map(|call| ToolCall {
                        id: call.id,
                        function: FunctionCall {
                            name: call.function.name,
                            arguments: call.function.arguments,
                            raw_arguments: call.function.raw_arguments,
                        },
                    })
                    .collect()
            }),
            tool_call_id: value.tool_call_id,
            name: value.name,
        })
        .collect())
}
