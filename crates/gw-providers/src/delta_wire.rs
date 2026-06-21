//! Crate-private wire wrappers for one OpenRouter SSE `data:` chunk.
//!
//! These mirror the OpenAI/OpenRouter chunk shape just enough to flatten in
//! [`crate::delta::parse_chunk`]. They are deliberately **liberal in what they accept**
//! (Postel's law): unknown fields are ignored, `reasoning_details` is captured as a raw
//! `serde_json::Value` (so a non-array or malformed shape can never abort the chunk), and each
//! reasoning fragment is re-parsed/converted leniently via [`WireReasoningDetail`].

use gw_schema::ReasoningDetail;
use serde::Deserialize;

use crate::delta::{CompletionTokensDetails, Usage};

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
    /// Token accounting, present on the final chunk when `usage` was requested. Captured as a
    /// raw `Value` (NOT a typed `Usage`) so a type-drifted field can never abort the terminal
    /// chunk; coerced tolerantly via [`convert_usage`]. (F)
    #[serde(default)]
    pub(crate) usage: Option<serde_json::Value>,
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

/// Coerce a JSON value into `u64`, tolerating provider type-drift: accept an integer, a float
/// (truncated toward zero), or a numeric string (e.g. `"42"` / `"42.0"`). Anything else → `None`.
fn coerce_u64(v: Option<&serde_json::Value>) -> Option<u64> {
    match v {
        Some(serde_json::Value::Number(n)) => n
            .as_u64()
            .or_else(|| n.as_i64().filter(|i| *i >= 0).map(|i| i as u64))
            .or_else(|| {
                n.as_f64()
                    .filter(|f| *f >= 0.0 && f.is_finite())
                    .map(|f| f as u64)
            }),
        Some(serde_json::Value::String(s)) => {
            let s = s.trim();
            s.parse::<u64>().ok().or_else(|| {
                s.parse::<f64>()
                    .ok()
                    .filter(|f| *f >= 0.0 && f.is_finite())
                    .map(|f| f as u64)
            })
        }
        _ => None,
    }
}

/// Coerce a JSON value into `f64`, tolerating type-drift: accept a number or a numeric string.
/// Anything else (or a non-finite value) → `None`.
fn coerce_f64(v: Option<&serde_json::Value>) -> Option<f64> {
    match v {
        Some(serde_json::Value::Number(n)) => n.as_f64().filter(|f| f.is_finite()),
        Some(serde_json::Value::String(s)) => {
            s.trim().parse::<f64>().ok().filter(|f| f.is_finite())
        }
        _ => None,
    }
}

/// Tolerantly coerce a raw `usage` JSON value into a [`Usage`], **never** erroring.
///
/// Each field is extracted independently and leniently (numbers, floats, and numeric strings all
/// accepted; see [`coerce_u64`] / [`coerce_f64`]); an un-coercible field simply becomes `None`.
/// A non-object `usage` (`"nope"`, an array, a number) or a fully-empty one degrades to `None`,
/// while a partially-bad object keeps every field it can coerce. This mirrors the lenient
/// `reasoning_details` path so a type-drifted usage field can never abort the terminal chunk
/// and discard `finish_reason` / provenance / the reasoning-token count. (F)
pub(crate) fn convert_usage(raw: Option<serde_json::Value>) -> Option<Usage> {
    let obj = match raw {
        Some(serde_json::Value::Object(m)) => m,
        Some(_) => {
            tracing::debug!("usage was not an object; ignoring");
            return None;
        }
        None => return None,
    };

    let reasoning_tokens = obj
        .get("completion_tokens_details")
        .and_then(serde_json::Value::as_object)
        .and_then(|d| coerce_u64(d.get("reasoning_tokens")));
    let completion_tokens_details = reasoning_tokens.map(|rt| CompletionTokensDetails {
        reasoning_tokens: Some(rt),
    });

    let usage = Usage {
        prompt_tokens: coerce_u64(obj.get("prompt_tokens")),
        completion_tokens: coerce_u64(obj.get("completion_tokens")),
        total_tokens: coerce_u64(obj.get("total_tokens")),
        completion_tokens_details,
        cost: coerce_f64(obj.get("cost")),
    };
    if usage.is_empty() { None } else { Some(usage) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn coerce_u64_accepts_int_float_and_numeric_string() {
        assert_eq!(coerce_u64(Some(&json!(42))), Some(42));
        assert_eq!(coerce_u64(Some(&json!(42.9))), Some(42)); // truncates toward zero
        assert_eq!(coerce_u64(Some(&json!("52"))), Some(52));
        assert_eq!(coerce_u64(Some(&json!("52.0"))), Some(52));
        // rejects: negatives, non-numeric strings, bools, null, absent.
        assert_eq!(coerce_u64(Some(&json!(-1))), None);
        assert_eq!(coerce_u64(Some(&json!("abc"))), None);
        assert_eq!(coerce_u64(Some(&json!(true))), None);
        assert_eq!(coerce_u64(Some(&serde_json::Value::Null)), None);
        assert_eq!(coerce_u64(None), None);
    }

    #[test]
    fn coerce_f64_accepts_number_and_numeric_string() {
        assert_eq!(coerce_f64(Some(&json!(0))), Some(0.0));
        assert_eq!(coerce_f64(Some(&json!(0.00043))), Some(0.00043));
        assert_eq!(coerce_f64(Some(&json!("0.00043"))), Some(0.00043));
        assert_eq!(coerce_f64(Some(&json!("nan"))), None);
        assert_eq!(coerce_f64(Some(&json!([1, 2]))), None);
        assert_eq!(coerce_f64(None), None);
    }

    #[test]
    fn convert_usage_non_object_is_none() {
        assert!(convert_usage(Some(json!("nope"))).is_none());
        assert!(convert_usage(Some(json!(42))).is_none());
        assert!(convert_usage(Some(json!([1, 2]))).is_none());
        assert!(convert_usage(None).is_none());
    }

    #[test]
    fn convert_usage_partial_keeps_coercible() {
        let u = convert_usage(Some(json!({
            "prompt_tokens": 7, "completion_tokens": "abc", "cost": "0.5"
        })))
        .expect("partially coercible");
        assert_eq!(u.prompt_tokens, Some(7));
        assert_eq!(u.completion_tokens, None);
        assert_eq!(u.cost, Some(0.5));
    }
}
