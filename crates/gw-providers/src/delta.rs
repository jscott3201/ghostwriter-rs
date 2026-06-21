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

use crate::delta_wire::{RawChunk, RawDelta, convert_reasoning_details, convert_usage};

/// One streaming delta: the per-chunk slice of an assistant turn.
///
/// Each field defaults to `None` (an OpenRouter chunk carries only what changed), so a chunk
/// bearing only reasoning, only content, or only a `finish_reason` all deserialize cleanly.
/// `reasoning` is the flat plaintext CoT some providers emit; `reasoning_details` is the
/// structured form (`reasoning.text` / `.summary` / `.encrypted`) stored verbatim.
///
/// `tool_calls` / `function_call` are intentionally **not** modeled: the harness streams `n=1`
/// with no tools, so OpenRouter never emits them on the teacher path. (C7)
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

    /// A model refusal, when the turn declines the request. First-class for refusal grading,
    /// so a refusal-only chunk is **not** considered empty (see [`is_empty_payload`]). (C4)
    ///
    /// [`is_empty_payload`]: StreamDelta::is_empty_payload
    #[serde(default)]
    pub refusal: Option<String>,

    /// Set on the terminal chunk of a choice: `"stop"`, `"length"`, `"content_filter"`, etc.
    /// Hoisted up from `choices[0].finish_reason` for convenience.
    #[serde(default)]
    pub finish_reason: Option<String>,

    /// OpenRouter's raw upstream stop reason from `choices[0].native_finish_reason` (the
    /// provider's own code, which may differ from the normalized `finish_reason`). (C3)
    #[serde(default)]
    pub native_finish_reason: Option<String>,

    /// Provenance captured from the final chunk: which upstream provider served the request
    /// (`served_by`), the resolved model slug, and the generation `id`. Populated only when the
    /// chunk carried them.
    #[serde(default)]
    pub provenance: Option<ChunkProvenance>,

    /// Token accounting from the final chunk's `usage` block, when present.
    #[serde(default)]
    pub usage: Option<Usage>,
}

impl StreamDelta {
    /// `true` when this delta carries no incremental **display** payload — no content, no
    /// reasoning, no structured reasoning, no refusal — e.g. a role-only opening chunk or a
    /// bare keep-alive shape. A caller may skip emitting such deltas to the UI.
    ///
    /// This reflects only the absence of incremental display payload. Callers MUST still
    /// consume a terminal chunk's `finish_reason` / `native_finish_reason` / `usage` /
    /// `provenance` even when `is_empty_payload()` is `true` (the final chunk often has no
    /// display text but carries the stop reason and token accounting). (D)
    #[must_use]
    pub fn is_empty_payload(&self) -> bool {
        self.content.is_none()
            && self.reasoning.is_none()
            && self.reasoning_details.is_none()
            && self.refusal.is_none()
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
    /// OpenRouter's generation id (its top-level `id`, e.g. `"gen-..."`) — the record's
    /// primary key. Captured verbatim; `created` is intentionally not captured. (C5)
    #[serde(default)]
    pub id: Option<String>,
}

impl ChunkProvenance {
    pub(crate) fn is_empty(&self) -> bool {
        self.served_by.is_none() && self.model.is_none() && self.id.is_none()
    }
}

/// Token accounting mirrored from an OpenRouter `usage` block (final chunk only).
///
/// This struct is **not** the direct wire deserialization target: the usage block rides the
/// terminal chunk, so a type-drifted field (e.g. `cost` as a string, a token count as a float)
/// must not abort the whole chunk and discard `finish_reason` / provenance. The wire `usage` is
/// captured as a raw `serde_json::Value` and coerced **tolerantly** into this struct by the
/// crate-private `convert_usage`, which never errors. (The `Deserialize` derive is retained for
/// downstream convenience, not used on the hot wire path.)
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
    /// Nested completion-token breakdown; carries `reasoning_tokens` on the final usage chunk
    /// for reasoning teachers (feeds the Verify gate). (C1)
    #[serde(default)]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    /// OpenRouter's authoritative per-generation cost (USD) on the usage block, when present.
    /// This is the source of truth for record cost (a future PR maps it into `gw-schema::Cost`;
    /// not wired here to avoid baking in silent-zero defaults). (C2)
    #[serde(default)]
    pub cost: Option<f64>,
}

impl Usage {
    pub(crate) fn is_empty(&self) -> bool {
        self.prompt_tokens.is_none()
            && self.completion_tokens.is_none()
            && self.total_tokens.is_none()
            && self.completion_tokens_details.is_none()
            && self.cost.is_none()
    }

    /// The reasoning-token count from `completion_tokens_details.reasoning_tokens`, if the
    /// usage block carried it. (C1)
    #[must_use]
    pub fn reasoning_tokens(&self) -> Option<u64> {
        self.completion_tokens_details
            .and_then(|d| d.reasoning_tokens)
    }
}

/// The `usage.completion_tokens_details` sub-object. Kept `Copy` so [`Usage`] stays `Copy`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Deserialize)]
pub struct CompletionTokensDetails {
    /// Reasoning tokens included in `completion_tokens` (the CoT's token cost). (C1)
    #[serde(default)]
    pub reasoning_tokens: Option<u64>,
}

/// Parse one OpenRouter SSE `data:` JSON payload into a [`StreamDelta`], flattening
/// `choices[0].delta`, hoisting `finish_reason` / `native_finish_reason`, and capturing
/// `id`/`model`/`provider`/`usage`. Reasoning fragments are converted leniently
/// (see [`convert_reasoning_details`]).
///
/// # Errors
/// Returns the underlying `serde_json` error only if the payload is not a valid JSON chunk
/// object at all. A malformed `reasoning_details` (non-array, or a bad fragment) and a
/// type-drifted `usage` (string cost, float token counts, garbage shape) never error here —
/// they are coerced or skipped, not propagated, so a terminal chunk's `finish_reason` /
/// provenance / coercible usage fields always survive.
pub(crate) fn parse_chunk(json: &str) -> Result<StreamDelta, serde_json::Error> {
    let raw: RawChunk = serde_json::from_str(json)?;
    let first = raw.choices.into_iter().next();
    let (delta, finish_reason, native_finish_reason) = match first {
        Some(c) => (c.delta, c.finish_reason, c.native_finish_reason),
        None => (RawDelta::default(), None, None),
    };

    let provenance = {
        let p = ChunkProvenance {
            served_by: raw.provider,
            model: raw.model,
            id: raw.id,
        };
        if p.is_empty() { None } else { Some(p) }
    };
    let usage = convert_usage(raw.usage);
    let reasoning_details = convert_reasoning_details(delta.reasoning_details);

    Ok(StreamDelta {
        content: delta.content,
        reasoning: delta.reasoning,
        reasoning_details,
        refusal: delta.refusal,
        finish_reason,
        native_finish_reason,
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

    // --- A. non-array reasoning_details container must NEVER abort the chunk ----------------

    #[test]
    fn reasoning_details_object_shape_does_not_abort() {
        // A `{...}` instead of an array: ignore reasoning_details, keep co-located content.
        let j = r#"{"choices":[{"delta":{"content":"keep me",
            "reasoning_details":{"oops":"object not array"}}}]}"#;
        let d = parse_chunk(j).unwrap();
        assert_eq!(d.content.as_deref(), Some("keep me"));
        assert!(d.reasoning_details.is_none());
    }

    #[test]
    fn reasoning_details_string_shape_does_not_abort() {
        let j = r#"{"choices":[{"delta":{"content":"keep me",
            "reasoning_details":"a bare string"}}]}"#;
        let d = parse_chunk(j).unwrap();
        assert_eq!(d.content.as_deref(), Some("keep me"));
        assert!(d.reasoning_details.is_none());
    }

    #[test]
    fn reasoning_details_number_shape_does_not_abort() {
        let j = r#"{"choices":[{"delta":{"content":"keep me","reasoning_details":42}}]}"#;
        let d = parse_chunk(j).unwrap();
        assert_eq!(d.content.as_deref(), Some("keep me"));
        assert!(d.reasoning_details.is_none());
    }

    // --- C. spec-conformance wire capture --------------------------------------------------

    #[test]
    fn reasoning_tokens_parsed_from_completion_tokens_details() {
        // C1: final usage chunk carries completion_tokens_details.reasoning_tokens.
        let j = r#"{"choices":[{"delta":{},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":10,"completion_tokens":200,"total_tokens":210,
            "completion_tokens_details":{"reasoning_tokens":150}}}"#;
        let d = parse_chunk(j).unwrap();
        let usage = d.usage.expect("usage present");
        assert_eq!(usage.reasoning_tokens(), Some(150));
        assert_eq!(
            usage.completion_tokens_details.unwrap().reasoning_tokens,
            Some(150)
        );
    }

    #[test]
    fn usage_without_details_has_no_reasoning_tokens() {
        let j = r#"{"choices":[{"delta":{}}],
            "usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}"#;
        let d = parse_chunk(j).unwrap();
        assert_eq!(d.usage.unwrap().reasoning_tokens(), None);
    }

    #[test]
    fn top_level_cost_parsed_from_usage() {
        // C2: OpenRouter cost on the usage block.
        let j = r#"{"choices":[{"delta":{},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30,"cost":0.0042}}"#;
        let d = parse_chunk(j).unwrap();
        assert_eq!(d.usage.unwrap().cost, Some(0.0042));
    }

    #[test]
    fn native_finish_reason_captured_on_finish_chunk() {
        // C3: raw upstream stop reason surfaced alongside the normalized one.
        let j = r#"{"choices":[{"delta":{},"finish_reason":"stop",
            "native_finish_reason":"end_turn"}]}"#;
        let d = parse_chunk(j).unwrap();
        assert_eq!(d.finish_reason.as_deref(), Some("stop"));
        assert_eq!(d.native_finish_reason.as_deref(), Some("end_turn"));
    }

    #[test]
    fn refusal_captured_and_refusal_only_chunk_not_empty() {
        // C4: refusal is first-class; a refusal-only chunk is NOT empty payload.
        let j = r#"{"choices":[{"delta":{"refusal":"I can't help with that."}}]}"#;
        let d = parse_chunk(j).unwrap();
        assert_eq!(d.refusal.as_deref(), Some("I can't help with that."));
        assert!(!d.is_empty_payload());
    }

    #[test]
    fn generation_id_captured_into_provenance() {
        // C5: top-level `id` flows into ChunkProvenance.id.
        let j = r#"{"id":"gen-abc123","model":"z-ai/glm-5.2","provider":"Parasail",
            "choices":[{"delta":{},"finish_reason":"stop"}]}"#;
        let d = parse_chunk(j).unwrap();
        let prov = d.provenance.expect("provenance present");
        assert_eq!(prov.id.as_deref(), Some("gen-abc123"));
        assert_eq!(prov.served_by.as_deref(), Some("Parasail"));
        assert_eq!(prov.model.as_deref(), Some("z-ai/glm-5.2"));
    }

    #[test]
    fn id_alone_produces_provenance() {
        // `id` is enough for provenance even without provider/model.
        let j = r#"{"id":"gen-xyz","choices":[{"delta":{"content":"x"}}]}"#;
        let d = parse_chunk(j).unwrap();
        assert_eq!(d.provenance.unwrap().id.as_deref(), Some("gen-xyz"));
    }

    #[test]
    fn role_only_chunk_still_empty_with_new_fields() {
        // A role-only opening chunk remains empty payload (no content/reasoning/refusal).
        let j = r#"{"choices":[{"delta":{"role":"assistant"}}]}"#;
        let d = parse_chunk(j).unwrap();
        assert!(d.is_empty_payload());
        assert!(d.refusal.is_none());
        assert!(d.native_finish_reason.is_none());
    }

    // --- F. lenient usage coercion: a type-drifted usage field must NEVER abort the chunk ----

    #[test]
    fn cost_as_string_does_not_abort_terminal_chunk() {
        // cost arrives as a JSON string; finish_reason + token counts must still be captured.
        let j = r#"{"choices":[{"delta":{},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":10,"completion_tokens":42,"cost":"0.00043"}}"#;
        let d = parse_chunk(j).unwrap();
        assert_eq!(d.finish_reason.as_deref(), Some("stop"));
        let usage = d.usage.expect("usage present");
        assert_eq!(usage.cost, Some(0.00043));
        assert_eq!(usage.completion_tokens, Some(42));
        assert_eq!(usage.prompt_tokens, Some(10));
    }

    #[test]
    fn completion_tokens_as_float_is_truncated() {
        let j = r#"{"choices":[{"delta":{},"finish_reason":"stop"}],
            "usage":{"completion_tokens":42.0}}"#;
        let d = parse_chunk(j).unwrap();
        assert_eq!(d.usage.unwrap().completion_tokens, Some(42));
    }

    #[test]
    fn reasoning_tokens_as_float_is_coerced() {
        let j = r#"{"choices":[{"delta":{},"finish_reason":"stop"}],
            "usage":{"completion_tokens":200,"completion_tokens_details":{"reasoning_tokens":150.0}}}"#;
        let d = parse_chunk(j).unwrap();
        assert_eq!(d.usage.unwrap().reasoning_tokens(), Some(150));
    }

    #[test]
    fn garbage_usage_string_yields_none_but_keeps_finish_reason() {
        // usage is a bare string ("nope"): degrade usage to None, never error, keep finish_reason.
        let j = r#"{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":"nope"}"#;
        let d = parse_chunk(j).unwrap();
        assert_eq!(d.finish_reason.as_deref(), Some("stop"));
        assert!(d.usage.is_none());
    }

    #[test]
    fn numeric_string_token_count_is_parsed() {
        let j = r#"{"choices":[{"delta":{}}],"usage":{"total_tokens":"52"}}"#;
        let d = parse_chunk(j).unwrap();
        assert_eq!(d.usage.unwrap().total_tokens, Some(52));
    }

    #[test]
    fn integer_zero_cost_is_preserved() {
        // cost:0 (an int) coerces to Some(0.0), not dropped.
        let j = r#"{"choices":[{"delta":{},"finish_reason":"stop"}],
            "usage":{"completion_tokens":1,"cost":0}}"#;
        let d = parse_chunk(j).unwrap();
        assert_eq!(d.usage.unwrap().cost, Some(0.0));
    }

    #[test]
    fn partially_bad_usage_keeps_coercible_fields() {
        // completion_tokens is a non-numeric string (dropped); the rest survives.
        let j = r#"{"choices":[{"delta":{},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":7,"completion_tokens":"abc","cost":0.001}}"#;
        let d = parse_chunk(j).unwrap();
        let usage = d
            .usage
            .expect("usage present (prompt_tokens + cost coercible)");
        assert_eq!(usage.prompt_tokens, Some(7));
        assert_eq!(usage.completion_tokens, None);
        assert_eq!(usage.cost, Some(0.001));
    }

    #[test]
    fn fully_uncoercible_usage_object_degrades_to_none() {
        let j = r#"{"choices":[{"delta":{}}],
            "usage":{"prompt_tokens":"x","completion_tokens":true,"cost":[1,2]}}"#;
        let d = parse_chunk(j).unwrap();
        assert!(d.usage.is_none());
    }
}
