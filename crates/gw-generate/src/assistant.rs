//! Assistant-turn generation: stream a teacher completion, ingest it into a clean [`Message`].
//!
//! This is the core teacher-call path. Given an injected [`Provider`] and a built [`ChatRequest`],
//! it streams the SSE deltas, accumulates `content` / `reasoning` / `reasoning_details` / the
//! terminal `finish_reason` / provenance / usage, and produces a clean [`AssistantTurn`].
//!
//! ## INVARIANT-a — reasoning is a sibling of content, never inlined
//!
//! The streamed `content` is run through `gw-format`'s [`ingest_openrouter`](gw_format::ingest_openrouter) so any channel tokens
//! a provider inlined (`<think>…</think>`, Gemma-4 `<|channel>thought…`, Harmony) are stripped OUT
//! of `content` INTO `reasoning`, and a control token surviving in the clean content fails loud
//! (the ingest crate owns this). We reuse that path rather than re-deriving reasoning extraction.
//! The provider's flat `reasoning` and structured `reasoning_details` (captured directly from the
//! stream, since the providers crate already separates them) win over anything peeled from content.
//!
//! ## The `<|channel>thought` truncation hazard
//!
//! A reasoning teacher whose CoT is cut off by `max_tokens` returns `finish_reason == "length"`
//! with an unterminated thought channel. [`accumulate`] flags this; [`generate_turn`] FAILS LOUD
//! with [`GenerateError::TruncatedReasoning`] when reasoning was being emitted and the stream ended
//! on `length`, so a truncated CoT is never handed back for admission. The caller retries with a
//! larger budget or routes to `revising`.
//!
//! ## Boundary: the live teacher stream is still text-only
//!
//! [`AccumulatedStream::ingest_payload`] hands ingest `content` + the flat `reasoning` only,
//! because the streaming delta type deliberately does not model `tool_calls` / `tool_call_id`
//! (see `gw-providers::delta`, which streams `n=1`). So a turn produced by THIS path is text-only
//! and its `content` is always the explicit string the provider streamed — never
//! [`Content::Null`]. The tool fields are populated on the non-streaming ingest boundary
//! (`gw_format::ingest_openrouter`, which does parse them), so a captured/imported trajectory keeps
//! its identity; making the live teacher emit tool calls is a provider-delta change, deliberately
//! out of scope here.

use futures::StreamExt;
use serde_json::json;

use gw_providers::{ChatRequest, Provider};
use gw_schema::{Content, Message, ReasoningDetail};

use crate::error::{GenerateError, Result};

/// `true` when a [`Message`]'s content carries no text (the ingested shape for a content-less
/// turn). An explicitly absent value ([`Content::Null`]) counts as content-less too: a structured
/// refusal arrives with a null body and must still be folded into the output. Multimodal parts are
/// never treated as content-less (they carry an image/audio payload a refusal fold would erase).
fn message_content_is_empty(message: &Message) -> bool {
    match &message.content {
        Content::Text(t) => t.is_empty(),
        Content::Null => true,
        Content::Parts(_) => false,
    }
}

/// The accumulated, still-raw result of draining a teacher stream. [`into_turn`] converts it into a
/// clean [`AssistantTurn`] (running the ingest path + the truncation check).
///
/// [`into_turn`]: AccumulatedStream::into_turn
#[derive(Debug, Default, Clone, PartialEq)]
pub struct AccumulatedStream {
    /// Concatenated final-answer text (may still carry inlined channel tokens until ingest).
    pub content: String,
    /// Concatenated flat plaintext chain-of-thought.
    pub reasoning: String,
    /// Structured reasoning blocks, accumulated across chunks in arrival order.
    pub reasoning_details: Vec<ReasoningDetail>,
    /// A model refusal, if the turn declined (first-class; not an error).
    pub refusal: Option<String>,
    /// The terminal `finish_reason` (`"stop"`, `"length"`, `"content_filter"`, …).
    pub finish_reason: Option<String>,
    /// The raw upstream stop reason (`native_finish_reason`).
    pub native_finish_reason: Option<String>,
    /// Upstream provider that served the request (`served_by`), from the final chunk.
    pub served_by: Option<String>,
    /// The resolved model slug OpenRouter reported.
    pub resolved_model: Option<String>,
    /// OpenRouter's generation id (`gen-…`).
    pub generation_id: Option<String>,
    /// Prompt (input) tokens billed.
    pub prompt_tokens: Option<u64>,
    /// Completion (output, incl. reasoning) tokens billed.
    pub completion_tokens: Option<u64>,
    /// Reasoning tokens (from `completion_tokens_details.reasoning_tokens`).
    pub reasoning_tokens: Option<u64>,
    /// OpenRouter's authoritative per-generation cost (USD).
    pub cost: Option<f64>,
}

impl AccumulatedStream {
    /// `true` when the stream ended because it hit the token cap (`finish_reason == "length"`).
    /// This is the signal of the `<|channel>thought` truncation hazard.
    #[must_use]
    pub fn hit_length_cap(&self) -> bool {
        self.finish_reason.as_deref() == Some("length")
    }

    /// `true` when chain-of-thought was emitted in either form (flat or structured).
    #[must_use]
    pub fn has_reasoning(&self) -> bool {
        !self.reasoning.is_empty() || !self.reasoning_details.is_empty()
    }
}

/// A clean assistant turn produced from a teacher stream: the canonical [`Message`] (with reasoning
/// a sibling of content) plus the captured provenance / usage the assembler folds into a record.
#[derive(Debug, Clone, PartialEq)]
pub struct AssistantTurn {
    /// The clean assistant [`Message`]: `content` is the final answer, `reasoning` /
    /// `reasoning_details` carry the CoT (INVARIANT-a). For a structured-`refusal` turn, the
    /// refusal text IS the content (the refusal is the assistant output — USER-SYNTHESIS §3.4),
    /// and [`refusal`](AssistantTurn::refusal) additionally carries it for downstream visibility.
    pub message: Message,
    /// The provider's structured `refusal` text, when the turn declined via the OpenRouter
    /// `refusal` delta field (rather than via plain `content`). `None` for a normal turn or a
    /// refusal expressed in `content`. When set, it has also been folded into
    /// [`message`](AssistantTurn::message)`.content` so a refusal turn is never an empty output.
    pub refusal: Option<String>,
    /// The terminal `finish_reason`.
    pub finish_reason: Option<String>,
    /// The upstream provider that served the request (→ `Provenance.served_by`).
    pub served_by: Option<String>,
    /// OpenRouter's generation id (→ the record's primary key).
    pub generation_id: Option<String>,
    /// Prompt tokens billed (→ `Cost.prompt_tokens`).
    pub prompt_tokens: Option<u64>,
    /// Completion tokens billed (→ `Cost.completion_tokens`).
    pub completion_tokens: Option<u64>,
    /// Reasoning tokens (→ `Cost.reasoning_tokens`; part of the Verify gate).
    pub reasoning_tokens: Option<u64>,
    /// Per-generation cost in USD (→ `Cost.usd`).
    pub cost: Option<f64>,
}

impl AccumulatedStream {
    /// Convert the accumulated stream into a clean [`AssistantTurn`].
    ///
    /// Runs the streamed `content` through `gw-format`'s [`ingest_openrouter`](gw_format::ingest_openrouter) to strip any inlined
    /// channel tokens into reasoning (INVARIANT-a) and to fail loud on a control-token leak. The
    /// provider's directly-streamed flat `reasoning` and structured `reasoning_details` take
    /// precedence over anything peeled from content (they are the higher-fidelity capture).
    ///
    /// # Errors
    /// - [`GenerateError::TruncatedReasoning`] if the stream hit the token cap
    ///   (`finish_reason == "length"`) WHILE reasoning was being emitted — the truncation hazard.
    /// - [`GenerateError::EmptyResponse`] if no content AND no reasoning AND no refusal was
    ///   produced at all (a degenerate completion).
    /// - [`GenerateError::Format`] if the ingest path rejects the content (control-token leak).
    pub fn into_turn(self) -> Result<AssistantTurn> {
        // The truncation hazard: a CoT cut off mid-channel must never be admitted.
        if self.hit_length_cap() && self.has_reasoning() {
            return Err(GenerateError::TruncatedReasoning {
                detail: format!(
                    "{} reasoning tokens emitted then finish_reason=length \
                     (CoT cut off inside the <|channel>thought block)",
                    self.reasoning_tokens.unwrap_or_default()
                ),
                cost_usd: self.cost,
            });
        }

        if self.content.is_empty() && !self.has_reasoning() && self.refusal.is_none() {
            return Err(GenerateError::EmptyResponse(
                "teacher stream yielded no content, reasoning, or refusal".into(),
            ));
        }

        // Ingest the (possibly channel-tokened) content into a clean Message. We hand the
        // directly-streamed reasoning alongside so the ingest path uses the high-fidelity provider
        // reasoning rather than re-peeling it from content.
        let payload = self.ingest_payload();
        let mut message = gw_format::ingest_openrouter(&payload)?;

        // A structured-refusal turn (content was null, refusal carried the decline) must NOT yield
        // an empty assistant output: the refusal IS the training signal for a RefusalExpected turn
        // (USER-SYNTHESIS §3.4), so fold the refusal text into `content`. A refusal expressed via
        // plain `content` (content non-empty) is left untouched.
        if message_content_is_empty(&message)
            && let Some(refusal) = &self.refusal
        {
            message.content = Content::Text(refusal.clone());
        }

        // Prefer the structured reasoning_details captured directly off the stream (the ingest
        // payload only carries flat reasoning text).
        if !self.reasoning_details.is_empty() {
            message.reasoning_details = Some(self.reasoning_details);
        }

        Ok(AssistantTurn {
            message,
            refusal: self.refusal,
            finish_reason: self.finish_reason,
            served_by: self.served_by,
            generation_id: self.generation_id,
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            reasoning_tokens: self.reasoning_tokens,
            cost: self.cost,
        })
    }

    /// The bare assistant-message JSON handed to `ingest_openrouter`: `content` plus the flat
    /// `reasoning` (when any was streamed). Structured `reasoning_details` are re-attached after
    /// ingest from the higher-fidelity stream capture.
    fn ingest_payload(&self) -> serde_json::Value {
        let mut msg = json!({ "role": "assistant", "content": self.content });
        if !self.reasoning.is_empty() {
            msg["reasoning"] = json!(self.reasoning);
        }
        msg
    }
}

/// Merge one streamed [`ReasoningDetail`] fragment into the accumulated sequence, concatenating by
/// `(variant, index)` (DATA-SCHEMA §2.3: "concatenate `delta.reasoning_details[].text` by index").
///
/// Providers stream the SAME logical reasoning block as many fragments that all share an `index`
/// (e.g. the `together.sse` capture emits 14 `reasoning.text` fragments at index 0 forming one
/// continuous sentence). Storing them as separate blocks corrupts the CoT and breaks the
/// reasoning-quality segmenter's dense/monotonic-index rule. So: when an arriving Text/Summary
/// fragment matches the variant AND index of the LAST accumulated block, its `text`/`summary` is
/// appended onto that block (the first block's non-`None` `signature`/`id`/`format` are preserved);
/// any other case (different index, different variant, or an Encrypted blob whose `data` must not be
/// string-spliced) starts a new block. Arrival order across distinct indices is preserved (§2.3:
/// the sequence must not be rearranged).
fn merge_reasoning_detail(acc: &mut Vec<ReasoningDetail>, fragment: ReasoningDetail) {
    match (acc.last_mut(), &fragment) {
        (
            Some(ReasoningDetail::Text { text, index, .. }),
            ReasoningDetail::Text {
                text: add,
                index: idx,
                ..
            },
        ) if index == idx => text.push_str(add),
        (
            Some(ReasoningDetail::Summary { summary, index, .. }),
            ReasoningDetail::Summary {
                summary: add,
                index: idx,
                ..
            },
        ) if index == idx => summary.push_str(add),
        // Different index/variant, or an Encrypted blob: keep as a distinct block (preserve order).
        _ => acc.push(fragment),
    }
}

/// Drain a [`DeltaStream`](gw_providers::DeltaStream) into an [`AccumulatedStream`], stopping at the
/// first provider error (which is returned, not swallowed — INVARIANT: a stream reset is a real
/// failure, never a silently-truncated record).
///
/// # Errors
/// Returns [`GenerateError::Provider`] on the first streamed [`ProviderError`](gw_providers::ProviderError).
pub async fn accumulate(mut stream: gw_providers::DeltaStream) -> Result<AccumulatedStream> {
    let mut acc = AccumulatedStream::default();
    while let Some(item) = stream.next().await {
        let delta = item?;
        if let Some(c) = delta.content {
            acc.content.push_str(&c);
        }
        if let Some(r) = delta.reasoning {
            acc.reasoning.push_str(&r);
        }
        if let Some(d) = delta.reasoning_details {
            for fragment in d {
                merge_reasoning_detail(&mut acc.reasoning_details, fragment);
            }
        }
        if let Some(refusal) = delta.refusal {
            acc.refusal = Some(refusal);
        }
        if let Some(f) = delta.finish_reason {
            acc.finish_reason = Some(f);
        }
        if let Some(nf) = delta.native_finish_reason {
            acc.native_finish_reason = Some(nf);
        }
        if let Some(p) = delta.provenance {
            // These ride the final chunk; first non-None wins (first == last in practice).
            acc.served_by = acc.served_by.or(p.served_by);
            acc.resolved_model = acc.resolved_model.or(p.model);
            acc.generation_id = acc.generation_id.or(p.id);
        }
        if let Some(u) = delta.usage {
            acc.prompt_tokens = acc.prompt_tokens.or(u.prompt_tokens);
            acc.completion_tokens = acc.completion_tokens.or(u.completion_tokens);
            acc.reasoning_tokens = acc.reasoning_tokens.or(u.reasoning_tokens());
            acc.cost = acc.cost.or(u.cost);
        }
    }
    Ok(acc)
}

/// Call the teacher: open the streamed completion via the injected [`Provider`], drain it, and
/// produce a clean [`AssistantTurn`]. This is the single entry that spends teacher tokens.
///
/// # Errors
/// Propagates [`GenerateError::Provider`] (open / stream failure), [`GenerateError::TruncatedReasoning`]
/// (the hazard), [`GenerateError::EmptyResponse`], or [`GenerateError::Format`] (ingest leak).
pub async fn generate_turn<P: Provider + ?Sized>(
    provider: &P,
    request: ChatRequest,
) -> Result<AssistantTurn> {
    let stream = provider.stream_chat(request).await?;
    let acc = accumulate(stream).await?;
    acc.into_turn()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gw_schema::{Content, Role};

    #[test]
    fn clean_content_and_reasoning_become_siblings() {
        // INVARIANT-a: streamed reasoning stays out of content.
        let acc = AccumulatedStream {
            content: "The answer is 42.".into(),
            reasoning: "Work it through.".into(),
            finish_reason: Some("stop".into()),
            reasoning_tokens: Some(120),
            ..Default::default()
        };
        let turn = acc.into_turn().unwrap();
        assert_eq!(turn.message.role, Role::Assistant);
        assert_eq!(
            turn.message.content,
            Content::Text("The answer is 42.".into())
        );
        assert_eq!(turn.message.reasoning.as_deref(), Some("Work it through."));
    }

    #[test]
    fn inlined_channel_tokens_are_stripped_into_reasoning() {
        // INVARIANT-a: a provider that inlined a <think> block into content must round-trip clean,
        // reusing the gw-format ingest path (not re-derived here).
        let acc = AccumulatedStream {
            content: "<think>3*4</think>12".into(),
            finish_reason: Some("stop".into()),
            ..Default::default()
        };
        let turn = acc.into_turn().unwrap();
        assert_eq!(turn.message.content, Content::Text("12".into()));
        assert_eq!(turn.message.reasoning.as_deref(), Some("3*4"));
    }

    #[test]
    fn structured_details_from_stream_take_precedence() {
        let acc = AccumulatedStream {
            content: "ok".into(),
            reasoning: "flat".into(),
            reasoning_details: vec![ReasoningDetail::Text {
                text: "structured step".into(),
                signature: None,
                id: None,
                format: None,
                index: 0,
            }],
            finish_reason: Some("stop".into()),
            ..Default::default()
        };
        let turn = acc.into_turn().unwrap();
        let details = turn.message.reasoning_details.expect("details present");
        assert_eq!(details.len(), 1);
        // Flat reasoning is still carried as the human-readable CoT.
        assert_eq!(turn.message.reasoning.as_deref(), Some("flat"));
    }

    #[test]
    fn truncated_reasoning_fails_loud() {
        // The <|channel>thought hazard: reasoning emitted + finish_reason=length → reject.
        let acc = AccumulatedStream {
            content: String::new(),
            reasoning: "a long unfinished thought".into(),
            finish_reason: Some("length".into()),
            reasoning_tokens: Some(16000),
            cost: Some(0.0123),
            ..Default::default()
        };
        let err = acc.into_turn().unwrap_err();
        let msg = err.to_string();
        let GenerateError::TruncatedReasoning { cost_usd, .. } = err else {
            panic!("expected truncated reasoning error");
        };
        assert_eq!(cost_usd, Some(0.0123));
        assert!(msg.contains("16000"));
    }

    #[test]
    fn length_cap_without_reasoning_is_not_truncation() {
        // finish_reason=length but NO reasoning (a non-thinking turn that ran long) is content,
        // not a truncated CoT — it must not be misclassified as the hazard.
        let acc = AccumulatedStream {
            content: "a long answer".into(),
            finish_reason: Some("length".into()),
            ..Default::default()
        };
        let turn = acc.into_turn().unwrap();
        assert_eq!(turn.message.content, Content::Text("a long answer".into()));
    }

    #[test]
    fn empty_completion_is_rejected() {
        let acc = AccumulatedStream {
            finish_reason: Some("stop".into()),
            ..Default::default()
        };
        let err = acc.into_turn().unwrap_err();
        assert!(matches!(err, GenerateError::EmptyResponse(_)));
    }

    #[test]
    fn structured_refusal_becomes_the_assistant_output() {
        // M1: a structured-refusal turn (content null, refusal set) must put the refusal text into
        // content — the refusal IS the RefusalExpected training signal, never an empty output.
        let acc = AccumulatedStream {
            refusal: Some("I can't help with that.".into()),
            finish_reason: Some("stop".into()),
            ..Default::default()
        };
        let turn = acc.into_turn().unwrap();
        assert_eq!(
            turn.message.content,
            Content::Text("I can't help with that.".into())
        );
        // And the raw refusal is also carried for downstream visibility.
        assert_eq!(turn.refusal.as_deref(), Some("I can't help with that."));
    }

    #[test]
    fn refusal_expressed_via_content_is_left_unchanged() {
        // A refusal the teacher wrote into plain `content` (finish_reason=stop, no refusal field)
        // must pass through verbatim — the refusal-folding only fires for the structured field.
        let acc = AccumulatedStream {
            content: "I won't do that, and here is why.".into(),
            finish_reason: Some("stop".into()),
            ..Default::default()
        };
        let turn = acc.into_turn().unwrap();
        assert_eq!(
            turn.message.content,
            Content::Text("I won't do that, and here is why.".into())
        );
        assert!(turn.refusal.is_none());
    }

    fn text(s: &str, index: u32) -> ReasoningDetail {
        ReasoningDetail::Text {
            text: s.into(),
            signature: None,
            id: None,
            format: None,
            index,
        }
    }

    #[test]
    fn merge_concatenates_text_fragments_at_the_same_index() {
        // M2: the together.sse shape — many Text fragments all at index 0 form ONE block.
        let mut acc: Vec<ReasoningDetail> = Vec::new();
        for frag in ["Let ", "me ", "think ", "step ", "by ", "step."] {
            merge_reasoning_detail(&mut acc, text(frag, 0));
        }
        assert_eq!(
            acc.len(),
            1,
            "same-index Text fragments must merge to one block"
        );
        match &acc[0] {
            ReasoningDetail::Text { text, index, .. } => {
                assert_eq!(text, "Let me think step by step.");
                assert_eq!(*index, 0);
            }
            other => panic!("expected merged reasoning.text, got {other:?}"),
        }
    }

    #[test]
    fn merge_starts_a_new_block_on_index_change_preserving_order() {
        // §2.3: distinct indices stay distinct AND in arrival order (no rearrangement).
        let mut acc: Vec<ReasoningDetail> = Vec::new();
        merge_reasoning_detail(&mut acc, text("a", 0));
        merge_reasoning_detail(&mut acc, text("b", 0));
        merge_reasoning_detail(&mut acc, text("c", 1));
        merge_reasoning_detail(&mut acc, text("d", 1));
        assert_eq!(acc.len(), 2);
        match (&acc[0], &acc[1]) {
            (
                ReasoningDetail::Text {
                    text: t0,
                    index: i0,
                    ..
                },
                ReasoningDetail::Text {
                    text: t1,
                    index: i1,
                    ..
                },
            ) => {
                assert_eq!((t0.as_str(), *i0), ("ab", 0));
                assert_eq!((t1.as_str(), *i1), ("cd", 1));
            }
            other => panic!("expected two Text blocks, got {other:?}"),
        }
    }

    #[test]
    fn merge_preserves_first_blocks_signature_and_id() {
        // The first fragment's non-None signature/id/format survive the concatenation.
        let mut acc = vec![ReasoningDetail::Text {
            text: "first".into(),
            signature: Some("sig-1".into()),
            id: Some("id-1".into()),
            format: Some("fmt".into()),
            index: 0,
        }];
        merge_reasoning_detail(&mut acc, text(" second", 0));
        match &acc[0] {
            ReasoningDetail::Text {
                text,
                signature,
                id,
                format,
                ..
            } => {
                assert_eq!(text, "first second");
                assert_eq!(signature.as_deref(), Some("sig-1"));
                assert_eq!(id.as_deref(), Some("id-1"));
                assert_eq!(format.as_deref(), Some("fmt"));
            }
            other => panic!("expected merged Text, got {other:?}"),
        }
    }

    #[test]
    fn merge_keeps_encrypted_blobs_distinct() {
        // Encrypted `data` is opaque — never string-spliced, even at the same index.
        let mut acc: Vec<ReasoningDetail> = Vec::new();
        for data in ["blobA", "blobB"] {
            merge_reasoning_detail(
                &mut acc,
                ReasoningDetail::Encrypted {
                    data: data.into(),
                    id: None,
                    format: None,
                    index: 0,
                },
            );
        }
        assert_eq!(acc.len(), 2, "encrypted blobs must not be concatenated");
    }

    fn fake_stream(deltas: Vec<gw_providers::StreamDelta>) -> gw_providers::DeltaStream {
        use futures::stream;
        Box::pin(stream::iter(deltas.into_iter().map(
            Ok::<gw_providers::StreamDelta, gw_providers::ProviderError>,
        )))
    }

    #[tokio::test]
    async fn accumulate_merges_same_index_text_across_chunks() {
        // M2 at the stream level: 3 chunks, each a Text fragment at index 0 → one merged block.
        use gw_providers::StreamDelta;
        let deltas = vec![
            StreamDelta {
                reasoning_details: Some(vec![text("alpha ", 0)]),
                ..Default::default()
            },
            StreamDelta {
                reasoning_details: Some(vec![text("beta ", 0)]),
                ..Default::default()
            },
            StreamDelta {
                reasoning_details: Some(vec![text("gamma", 0)]),
                ..Default::default()
            },
        ];
        let acc = accumulate(fake_stream(deltas)).await.unwrap();
        assert_eq!(acc.reasoning_details.len(), 1);
        match &acc.reasoning_details[0] {
            ReasoningDetail::Text { text, .. } => assert_eq!(text, "alpha beta gamma"),
            other => panic!("expected one merged Text block, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn accumulate_usage_and_provenance_are_first_non_none_wins() {
        // m7: two chunks both carry usage + provenance with DIFFERENT values; the FIRST non-None
        // must win (cost is the authoritative billing figure — must not silently drift to last).
        use gw_providers::{ChunkProvenance, StreamDelta, Usage};
        let deltas = vec![
            StreamDelta {
                provenance: Some(ChunkProvenance {
                    served_by: Some("Parasail".into()),
                    model: Some("z-ai/glm-5.2".into()),
                    id: Some("gen-first".into()),
                }),
                usage: Some(Usage {
                    prompt_tokens: Some(10),
                    completion_tokens: Some(100),
                    total_tokens: Some(110),
                    completion_tokens_details: None,
                    cost: Some(0.001),
                }),
                ..Default::default()
            },
            StreamDelta {
                provenance: Some(ChunkProvenance {
                    served_by: Some("Wafer".into()),
                    model: Some("other".into()),
                    id: Some("gen-second".into()),
                }),
                usage: Some(Usage {
                    prompt_tokens: Some(999),
                    completion_tokens: Some(999),
                    total_tokens: Some(999),
                    completion_tokens_details: None,
                    cost: Some(9.999),
                }),
                ..Default::default()
            },
        ];
        let acc = accumulate(fake_stream(deltas)).await.unwrap();
        assert_eq!(acc.served_by.as_deref(), Some("Parasail"));
        assert_eq!(acc.generation_id.as_deref(), Some("gen-first"));
        assert_eq!(acc.prompt_tokens, Some(10));
        assert_eq!(acc.cost, Some(0.001), "first cost must win, not last");
    }
}
