//! Crate-private wire wrappers for one OpenRouter SSE `data:` chunk.
//!
//! These mirror the OpenAI/OpenRouter chunk shape just enough to flatten in
//! [`crate::delta::parse_chunk`]. They are deliberately **liberal in what they accept**
//! (Postel's law): unknown fields are ignored, `reasoning_details` is captured as a raw
//! `serde_json::Value` (so a non-array or malformed shape can never abort the chunk), and each
//! reasoning fragment is re-parsed/converted leniently via [`WireReasoningDetail`].

use gw_schema::ReasoningDetail;
use serde::Deserialize;

use crate::delta::Usage;

/// The top-level shape of one OpenRouter streaming chunk. Only the fields the harness needs are
/// modeled; everything else is ignored.
#[derive(Debug, Deserialize)]
pub(crate) struct RawChunk {
    #[serde(default)]
    pub(crate) choices: Vec<RawChoice>,
    /// Generation id (e.g. `"gen-..."`), present on every chunk. (C5)
    #[serde(default)]
    pub(crate) id: Option<String>,
    /// Resolved model slug, present on most chunks (OpenRouter echoes it).
    #[serde(default)]
    pub(crate) model: Option<String>,
    /// Upstream provider name, present on the final chunk.
    #[serde(default)]
    pub(crate) provider: Option<String>,
    /// Token accounting, present on the final chunk when `usage` was requested.
    #[serde(default)]
    pub(crate) usage: Option<Usage>,
}

/// One element of `choices`. The harness streams `n=1`, so only the first is read.
#[derive(Debug, Deserialize)]
pub(crate) struct RawChoice {
    #[serde(default)]
    pub(crate) delta: RawDelta,
    #[serde(default)]
    pub(crate) finish_reason: Option<String>,
    /// The upstream provider's raw stop reason (may differ from the normalized one). (C3)
    #[serde(default)]
    pub(crate) native_finish_reason: Option<String>,
}

/// The `choices[0].delta` object — where the CoT lives.
///
/// `reasoning_details` is captured as a raw `serde_json::Value` (NOT `Vec<Value>`), so a
/// non-array shape (`{...}` / `"str"` / `42`) cannot fail chunk deserialization and drop a
/// co-located `content`. The array case is converted per-fragment in
/// [`crate::delta::parse_chunk`].
///
/// `tool_calls` / `function_call` are intentionally NOT modeled: the harness streams `n=1`
/// with no tools, so OpenRouter never emits them on the teacher path. (C7)
#[derive(Debug, Default, Deserialize)]
pub(crate) struct RawDelta {
    #[serde(default)]
    pub(crate) content: Option<String>,
    #[serde(default)]
    pub(crate) reasoning: Option<String>,
    #[serde(default)]
    pub(crate) reasoning_details: Option<serde_json::Value>,
    /// A model refusal string, when present. (C4)
    #[serde(default)]
    pub(crate) refusal: Option<String>,
}

/// A LENIENT wire view of one `reasoning_details[]` fragment.
///
/// Internally tagged on `type`, every payload field optional, so a fragment that omits `index`,
/// sends a null/absent `text`, or carries a brand-new `type` still deserializes instead of
/// killing the stream. Converted into the STRICT [`gw_schema::ReasoningDetail`] by
/// [`WireReasoningDetail::into_strict`]; the [`WireReasoningDetail::Unknown`] catch-all (novel
/// `type` tags) is dropped (the flat `reasoning` string still carries the human-readable CoT).
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum WireReasoningDetail {
    #[serde(rename = "reasoning.text")]
    Text {
        #[serde(default)]
        text: Option<String>,
        #[serde(default)]
        signature: Option<String>,
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        format: Option<String>,
        #[serde(default)]
        index: Option<u32>,
    },
    #[serde(rename = "reasoning.summary")]
    Summary {
        #[serde(default)]
        summary: Option<String>,
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        format: Option<String>,
        #[serde(default)]
        index: Option<u32>,
    },
    #[serde(rename = "reasoning.encrypted")]
    Encrypted {
        #[serde(default)]
        data: Option<String>,
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        format: Option<String>,
        #[serde(default)]
        index: Option<u32>,
    },
    /// Any `type` the schema does not (yet) model. Dropped on conversion.
    #[serde(other)]
    Unknown,
}

impl WireReasoningDetail {
    /// Convert into the strict stored type, filling absent `index` with `0` and absent
    /// text/summary/data with `""`. Returns `None` for the [`WireReasoningDetail::Unknown`]
    /// catch-all, which is dropped.
    fn into_strict(self) -> Option<ReasoningDetail> {
        match self {
            WireReasoningDetail::Text {
                text,
                signature,
                id,
                format,
                index,
            } => Some(ReasoningDetail::Text {
                text: text.unwrap_or_default(),
                signature,
                id,
                format,
                index: index.unwrap_or(0),
            }),
            WireReasoningDetail::Summary {
                summary,
                id,
                format,
                index,
            } => Some(ReasoningDetail::Summary {
                summary: summary.unwrap_or_default(),
                id,
                format,
                index: index.unwrap_or(0),
            }),
            WireReasoningDetail::Encrypted {
                data,
                id,
                format,
                index,
            } => Some(ReasoningDetail::Encrypted {
                data: data.unwrap_or_default(),
                id,
                format,
                index: index.unwrap_or(0),
            }),
            WireReasoningDetail::Unknown => None,
        }
    }
}

/// Convert a raw `reasoning_details` JSON value into strict [`ReasoningDetail`]s.
///
/// Only a JSON **array** carries fragments; a non-array shape (`object`/`string`/`number`/
/// `null`) is logged at `debug` and treated as "no details" — it is NEVER an error (so a
/// co-located `content` in the same chunk survives). Within an array, each fragment that fails
/// to parse or is an unknown tag is skipped with a `debug` log; the rest are kept.
pub(crate) fn convert_reasoning_details(
    raw: Option<serde_json::Value>,
) -> Option<Vec<ReasoningDetail>> {
    let array = match raw {
        Some(serde_json::Value::Array(v)) => v,
        Some(_) => {
            tracing::debug!("reasoning_details was not an array; ignoring");
            return None;
        }
        None => return None,
    };

    let mut out = Vec::with_capacity(array.len());
    for value in array {
        match serde_json::from_value::<WireReasoningDetail>(value) {
            Ok(wire) => match wire.into_strict() {
                Some(detail) => out.push(detail),
                None => tracing::debug!("skipping reasoning_details fragment with unknown type"),
            },
            Err(e) => {
                tracing::debug!(error = %e, "skipping unparseable reasoning_details fragment")
            }
        }
    }
    if out.is_empty() { None } else { Some(out) }
}
