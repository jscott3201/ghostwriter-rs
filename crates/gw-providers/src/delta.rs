//! [`StreamDelta`] — the custom streaming delta that captures chain-of-thought.
//!
//! This is the reason `gw-providers` hand-rolls its SSE client. `async-openai` 0.41.1's typed
//! `ChatCompletionStreamResponseDelta` has exactly `{content, function_call, tool_calls, role,
//! refusal}` — **no** `reasoning` / `reasoning_details` field and no `#[serde(flatten)]` catch
//! map — so deserializing an OpenRouter streaming delta through it silently discards the CoT
//! (the entire payload this harness exists to capture). The types here deserialize one
//! OpenRouter SSE `data:` chunk and surface `content`, `reasoning`, and the structured
//! [`gw_schema::ReasoningDetail`] blocks. Unknown fields are ignored (serde's default).

use gw_schema::ReasoningDetail;
use serde::Deserialize;

/// One streaming delta: the per-chunk slice of an assistant turn.
///
/// Each field defaults to `None` (an OpenRouter chunk carries only what changed), so a chunk
/// bearing only reasoning, only content, or only a `finish_reason` all deserialize cleanly.
/// `reasoning` is the flat plaintext CoT some providers emit; `reasoning_details` is the
/// structured form (`reasoning.text` / `.summary` / `.encrypted`) stored verbatim.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct StreamDelta {
    /// Incremental final-answer text. Concatenate across chunks for the answer.
    #[serde(default)]
    pub content: Option<String>,

    /// Incremental flat plaintext chain-of-thought. Concatenate across chunks for the CoT.
    #[serde(default)]
    pub reasoning: Option<String>,

    /// Incremental structured reasoning blocks, accumulated by `index` for faithful
    /// multi-turn passback and the Verify gate.
    #[serde(default)]
    pub reasoning_details: Option<Vec<ReasoningDetail>>,

    /// Set on the terminal chunk of a choice: `"stop"`, `"length"`, `"content_filter"`, etc.
    /// Hoisted up from `choices[0].finish_reason` for convenience.
    #[serde(default)]
    pub finish_reason: Option<String>,

    /// Provenance captured from the final chunk: which upstream provider served the request
    /// (`served_by`) and the resolved model slug. Populated only when the chunk carried them.
    #[serde(default)]
    pub provenance: Option<ChunkProvenance>,

    /// Token accounting from the final chunk's `usage` block, when present.
    #[serde(default)]
    pub usage: Option<Usage>,
}

impl StreamDelta {
    /// `true` when this delta carries no incremental payload (no content, no reasoning, no
    /// structured reasoning) — e.g. a role-only opening chunk or a bare keep-alive shape. The
    /// caller may skip emitting such deltas to the UI.
    #[must_use]
    pub fn is_empty_payload(&self) -> bool {
        self.content.is_none() && self.reasoning.is_none() && self.reasoning_details.is_none()
    }
}

/// Per-request provenance surfaced on the final SSE chunk → drives `Provenance.served_by`.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct ChunkProvenance {
    /// The upstream provider OpenRouter routed to (its `provider` field), e.g. `"Parasail"`.
    #[serde(default)]
    pub served_by: Option<String>,
    /// The resolved model slug OpenRouter reports (its `model` field).
    #[serde(default)]
    pub model: Option<String>,
}

impl ChunkProvenance {
    fn is_empty(&self) -> bool {
        self.served_by.is_none() && self.model.is_none()
    }
}

/// Token accounting mirrored from an OpenRouter `usage` block (final chunk only).
#[derive(Debug, Clone, Copy, Default, PartialEq, Deserialize)]
pub struct Usage {
    /// Prompt (input) tokens billed.
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
    /// Completion (output, incl. reasoning) tokens billed.
    #[serde(default)]
    pub completion_tokens: Option<u64>,
    /// `prompt + completion`.
    #[serde(default)]
    pub total_tokens: Option<u64>,
}

impl Usage {
    fn is_empty(&self) -> bool {
        self.prompt_tokens.is_none()
            && self.completion_tokens.is_none()
            && self.total_tokens.is_none()
    }
}

// --- wire wrappers: one OpenRouter SSE `data:` chunk ---------------------------------------

/// The top-level shape of one OpenRouter streaming chunk. Only the fields the harness needs are
/// modeled; everything else is ignored. Lives crate-private; [`parse_chunk`] flattens it.
#[derive(Debug, Deserialize)]
struct RawChunk {
    #[serde(default)]
    choices: Vec<RawChoice>,
    /// Resolved model slug, present on most chunks (OpenRouter echoes it).
    #[serde(default)]
    model: Option<String>,
    /// Upstream provider name, present on the final chunk.
    #[serde(default)]
    provider: Option<String>,
    /// Token accounting, present on the final chunk when `usage` was requested.
    #[serde(default)]
    usage: Option<Usage>,
}

/// One element of `choices`. The harness streams `n=1`, so only the first is read.
#[derive(Debug, Deserialize)]
struct RawChoice {
    #[serde(default)]
    delta: RawDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

/// The `choices[0].delta` object — where the CoT lives.
///
/// `reasoning_details` is deserialized as raw `serde_json::Value`s, **not** straight into the
/// strict [`gw_schema::ReasoningDetail`], so one malformed/novel fragment cannot abort the
/// whole chunk. Each fragment is then leniently re-parsed and converted per-fragment in
/// [`parse_chunk`] (Postel's law: be liberal in what the wire decoder accepts).
#[derive(Debug, Default, Deserialize)]
struct RawDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    reasoning_details: Option<Vec<serde_json::Value>>,
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
enum WireReasoningDetail {
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

/// Leniently convert a vec of raw reasoning fragments into strict [`ReasoningDetail`]s,
/// skipping (with a `debug` log) any fragment that fails to parse or is an unknown tag. Never
/// errors — a bad fragment must not abort the chunk or drop the rest of the CoT.
fn convert_reasoning_details(raw: Vec<serde_json::Value>) -> Option<Vec<ReasoningDetail>> {
    let mut out = Vec::with_capacity(raw.len());
    for value in raw {
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

/// Parse one OpenRouter SSE `data:` JSON payload into a [`StreamDelta`], flattening
/// `choices[0].delta`, hoisting `finish_reason`, and capturing `model`/`provider`/`usage`.
/// Reasoning fragments are converted leniently (see [`convert_reasoning_details`]).
///
/// # Errors
/// Returns the underlying `serde_json` error if the payload is not a valid chunk object. A
/// malformed *reasoning fragment* never errors here — it is skipped, not propagated.
pub(crate) fn parse_chunk(json: &str) -> Result<StreamDelta, serde_json::Error> {
    let raw: RawChunk = serde_json::from_str(json)?;
    let first = raw.choices.into_iter().next();
    let (delta, finish_reason) = match first {
        Some(c) => (c.delta, c.finish_reason),
        None => (RawDelta::default(), None),
    };

    let provenance = {
        let p = ChunkProvenance {
            served_by: raw.provider,
            model: raw.model,
        };
        if p.is_empty() { None } else { Some(p) }
    };
    let usage = raw.usage.filter(|u| !u.is_empty());
    let reasoning_details = delta.reasoning_details.and_then(convert_reasoning_details);

    Ok(StreamDelta {
        content: delta.content,
        reasoning: delta.reasoning,
        reasoning_details,
        finish_reason,
        provenance,
        usage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_only_chunk() {
        let j = r#"{"choices":[{"delta":{"content":"hi"}}]}"#;
        let d = parse_chunk(j).unwrap();
        assert_eq!(d.content.as_deref(), Some("hi"));
        assert!(d.reasoning.is_none());
        assert!(!d.is_empty_payload());
    }

    #[test]
    fn flat_reasoning_chunk() {
        let j = r#"{"choices":[{"delta":{"reasoning":"let me think"}}]}"#;
        let d = parse_chunk(j).unwrap();
        assert_eq!(d.reasoning.as_deref(), Some("let me think"));
        assert!(d.content.is_none());
    }

    #[test]
    fn structured_reasoning_details_chunk() {
        let j = r#"{"choices":[{"delta":{"reasoning_details":[
            {"type":"reasoning.text","text":"step one","index":0}]}}]}"#;
        let d = parse_chunk(j).unwrap();
        let details = d.reasoning_details.expect("details present");
        assert_eq!(details.len(), 1);
        match &details[0] {
            ReasoningDetail::Text { text, index, .. } => {
                assert_eq!(text, "step one");
                assert_eq!(*index, 0);
            }
            other => panic!("expected reasoning.text, got {other:?}"),
        }
    }

    #[test]
    fn unknown_fields_ignored() {
        let j = r#"{"id":"gen-1","object":"chat.completion.chunk","created":1,
            "choices":[{"index":0,"delta":{"role":"assistant","content":"x"},
            "logprobs":null,"native_finish_reason":"stop"}],"system_fingerprint":"fp"}"#;
        let d = parse_chunk(j).unwrap();
        assert_eq!(d.content.as_deref(), Some("x"));
    }

    #[test]
    fn role_only_chunk_is_empty_payload() {
        let j = r#"{"choices":[{"delta":{"role":"assistant"}}]}"#;
        let d = parse_chunk(j).unwrap();
        assert!(d.is_empty_payload());
        assert!(d.finish_reason.is_none());
    }

    #[test]
    fn final_chunk_captures_provenance_and_usage() {
        let j = r#"{"model":"z-ai/glm-5.2","provider":"Parasail",
            "choices":[{"delta":{},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":10,"completion_tokens":42,"total_tokens":52}}"#;
        let d = parse_chunk(j).unwrap();
        assert_eq!(d.finish_reason.as_deref(), Some("stop"));
        let prov = d.provenance.expect("provenance present");
        assert_eq!(prov.served_by.as_deref(), Some("Parasail"));
        assert_eq!(prov.model.as_deref(), Some("z-ai/glm-5.2"));
        let usage = d.usage.expect("usage present");
        assert_eq!(usage.completion_tokens, Some(42));
        assert_eq!(usage.total_tokens, Some(52));
    }

    #[test]
    fn empty_choices_yields_default_delta() {
        let j = r#"{"choices":[],"model":"m"}"#;
        let d = parse_chunk(j).unwrap();
        assert!(d.is_empty_payload());
        // model alone still produces provenance.
        assert_eq!(d.provenance.unwrap().model.as_deref(), Some("m"));
    }

    // --- lenient wire reasoning parse (Postel's law) ---------------------------------------

    #[test]
    fn reasoning_text_fragment_missing_index_still_decodes() {
        // A spec-valid fragment that OMITS `index` must not abort the chunk; it fills index=0.
        let j = r#"{"choices":[{"delta":{"content":"after",
            "reasoning_details":[{"type":"reasoning.text","text":"no index here"}]}}]}"#;
        let d = parse_chunk(j).unwrap();
        // Subsequent content still streams in the SAME chunk.
        assert_eq!(d.content.as_deref(), Some("after"));
        let details = d.reasoning_details.expect("details present");
        match &details[0] {
            ReasoningDetail::Text { text, index, .. } => {
                assert_eq!(text, "no index here");
                assert_eq!(*index, 0);
            }
            other => panic!("expected reasoning.text, got {other:?}"),
        }
    }

    #[test]
    fn unknown_reasoning_type_is_skipped_not_aborted() {
        // A novel `type` ("reasoning.foo") is dropped, but the chunk (and other fields) survive.
        let j = r#"{"choices":[{"delta":{"reasoning":"flat cot here",
            "reasoning_details":[{"type":"reasoning.foo","blah":1,"index":0}]}}]}"#;
        let d = parse_chunk(j).unwrap();
        // Flat reasoning still carries the human-readable CoT.
        assert_eq!(d.reasoning.as_deref(), Some("flat cot here"));
        // The unknown fragment was dropped → no strict details.
        assert!(d.reasoning_details.is_none());
    }

    #[test]
    fn null_or_absent_text_does_not_abort() {
        // Explicit null text → empty string, not a decode failure.
        let j = r#"{"choices":[{"delta":{"reasoning_details":
            [{"type":"reasoning.text","text":null,"index":2}]}}]}"#;
        let d = parse_chunk(j).unwrap();
        let details = d.reasoning_details.expect("details present");
        match &details[0] {
            ReasoningDetail::Text { text, index, .. } => {
                assert_eq!(text, "");
                assert_eq!(*index, 2);
            }
            other => panic!("expected reasoning.text, got {other:?}"),
        }
    }

    #[test]
    fn mixed_bad_and_good_fragment_keeps_the_good_one() {
        // One unparseable/unknown fragment + one valid: keep the valid one, drop the bad.
        let j = r#"{"choices":[{"delta":{"reasoning_details":[
            {"type":"reasoning.weird","x":true},
            {"type":"reasoning.text","text":"keep me","index":1}]}}]}"#;
        let d = parse_chunk(j).unwrap();
        let details = d.reasoning_details.expect("the good fragment survives");
        assert_eq!(details.len(), 1);
        match &details[0] {
            ReasoningDetail::Text { text, index, .. } => {
                assert_eq!(text, "keep me");
                assert_eq!(*index, 1);
            }
            other => panic!("expected reasoning.text, got {other:?}"),
        }
    }

    #[test]
    fn summary_and_encrypted_fragments_decode_leniently() {
        let j = r#"{"choices":[{"delta":{"reasoning_details":[
            {"type":"reasoning.summary","summary":"sum"},
            {"type":"reasoning.encrypted","data":"abc"}]}}]}"#;
        let d = parse_chunk(j).unwrap();
        let details = d.reasoning_details.expect("details present");
        assert_eq!(details.len(), 2);
        assert!(
            matches!(&details[0], ReasoningDetail::Summary { summary, index, .. }
            if summary == "sum" && *index == 0)
        );
        assert!(
            matches!(&details[1], ReasoningDetail::Encrypted { data, index, .. }
            if data == "abc" && *index == 0)
        );
    }

    #[test]
    fn all_fragments_bad_yields_none_not_error() {
        let j = r#"{"choices":[{"delta":{"reasoning_details":[
            {"type":"reasoning.foo"},{"type":"reasoning.bar"}]}}]}"#;
        let d = parse_chunk(j).unwrap();
        assert!(d.reasoning_details.is_none());
    }
}
