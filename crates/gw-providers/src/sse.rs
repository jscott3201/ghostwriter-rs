//! The hand-rolled SSE line decoder — the load-bearing piece.
//!
//! OpenAI/OpenRouter SSE is line-oriented: payload lines are `data: {json}`, the stream ends at
//! the sentinel `data: [DONE]`, and `:`-prefixed comment lines (OpenRouter emits
//! `: OPENROUTER PROCESSING` keep-alives) plus `event:` / blank lines are ignored. The catch is
//! that `reqwest`'s `bytes_stream()` yields arbitrary byte chunks that split lines — and even
//! the JSON — at any boundary. [`decode_sse`] buffers partial lines across chunk boundaries,
//! splits on `\n`, and yields one parsed [`StreamDelta`] per `data:` payload line.
//!
//! A clean end-of-stream is `[DONE]`. If the byte stream ends *without* `[DONE]`, the decoder
//! yields a terminal [`ProviderError::StreamReset`] so the retry layer can react.

use std::collections::VecDeque;

use futures::stream::{Stream, StreamExt};

use crate::delta::{StreamDelta, parse_chunk};
use crate::error::ProviderError;

/// The SSE sentinel that marks a clean end-of-stream.
const DONE_SENTINEL: &str = "[DONE]";

/// Decode a stream of raw byte chunks into a stream of parsed [`StreamDelta`]s.
///
/// `byte_stream` is typically `response.bytes_stream()`. Each yielded item is either a decoded
/// delta or a [`ProviderError`]: a [`ProviderError::Decode`] for a malformed `data:` payload, a
/// [`ProviderError::Transport`] for a read error from the byte stream, or a terminal
/// [`ProviderError::StreamReset`] if the bytes end before the `[DONE]` sentinel. After `[DONE]`
/// (or an error) the returned stream is exhausted.
///
/// The decoder is allocation-light: it keeps a single growing line buffer and a small queue of
/// ready items between polls. It performs no network I/O of its own.
pub fn decode_sse<S, B>(byte_stream: S) -> impl Stream<Item = Result<StreamDelta, ProviderError>>
where
    S: Stream<Item = reqwest::Result<B>> + Unpin,
    B: AsRef<[u8]>,
{
    // `Option<Decoder>` so we can fuse: once None, the stream is done.
    futures::stream::unfold(Some(Decoder::new(byte_stream)), |state| async move {
        let mut dec = state?;
        match dec.next_item().await {
            Some(item) => {
                // On a terminal item (DONE or error) `next_item` returns the item but the
                // decoder marks itself done; reflect that by dropping it from the state.
                let next_state = if dec.finished { None } else { Some(dec) };
                Some((item, next_state))
            }
            None => None,
        }
    })
}

/// Internal stateful decoder driving one byte stream.
struct Decoder<S> {
    stream: S,
    /// Bytes received but not yet terminated by a `\n`.
    line_buf: Vec<u8>,
    /// Parsed deltas ready to yield (a single chunk can contain several complete lines).
    ready: VecDeque<Result<StreamDelta, ProviderError>>,
    /// Set once the upstream byte stream has ended.
    stream_ended: bool,
    /// Set once `[DONE]` was seen — a clean, expected end-of-stream.
    saw_done: bool,
    /// Set once the post-stream terminal flush has run (so it runs exactly once).
    flushed: bool,
    /// Set once this decoder must yield nothing further.
    finished: bool,
}

impl<S, B> Decoder<S>
where
    S: Stream<Item = reqwest::Result<B>> + Unpin,
    B: AsRef<[u8]>,
{
    fn new(stream: S) -> Self {
        Self {
            stream,
            line_buf: Vec::with_capacity(4096),
            ready: VecDeque::new(),
            stream_ended: false,
            saw_done: false,
            flushed: false,
            finished: false,
        }
    }

    /// Produce the next item (delta or error), pulling more bytes as needed. Returns `None`
    /// only once the stream is fully drained and no terminal error is owed.
    async fn next_item(&mut self) -> Option<Result<StreamDelta, ProviderError>> {
        loop {
            if self.finished {
                return None;
            }
            if let Some(item) = self.ready.pop_front() {
                if item.is_err() {
                    self.finished = true;
                }
                return Some(item);
            }
            // `[DONE]` is a clean end-of-stream: stop here without another upstream poll, and
            // never yield any post-`[DONE]` data (drained above; the rest is discarded).
            if self.saw_done {
                self.finished = true;
                return None;
            }
            if self.stream_ended {
                if self.flushed {
                    // Trailing flush already produced its item(s) and they have been drained.
                    self.finished = true;
                    return None;
                }
                // No more bytes will arrive. Flush any trailing item, else owe a StreamReset.
                self.flush_terminal();
                self.flushed = true;
                continue; // re-enter the loop to drain `ready` / decide termination
            }

            // Pull the next byte chunk and turn its complete lines into ready items.
            match self.stream.next().await {
                Some(Ok(bytes)) => self.ingest(bytes.as_ref()),
                Some(Err(e)) => {
                    // A transport read error mid-stream. Surface it as the terminal item.
                    self.ready
                        .push_back(Err(ProviderError::Transport(e.to_string())));
                    self.stream_ended = true;
                }
                None => self.stream_ended = true,
            }
        }
    }

    /// Once the byte stream has fully ended, queue the final owed item(s): a trailing
    /// newline-less `data:` line (if any) AND, when no `[DONE]` was seen, a terminal
    /// [`ProviderError::StreamReset`]. This is idempotent — it marks the buffer drained so a
    /// re-entry does not re-queue. Critically, a flushed trailing delta does NOT suppress the
    /// StreamReset: a partial response must still look incomplete to the retry layer.
    fn flush_terminal(&mut self) {
        // Flush any trailing partial line that lacked a final `\n`.
        if !self.line_buf.is_empty() {
            let line = std::mem::take(&mut self.line_buf);
            if let Some(item) = self.process_line(&line) {
                let is_err = item.is_err();
                self.ready.push_back(item);
                if is_err {
                    // A decode error is itself terminal; don't also append a StreamReset.
                    return;
                }
            }
        }
        // `[DONE]` (possibly set by the trailing line just flushed) ⇒ clean end; otherwise the
        // stream was cut short and the consumer must be told so it can re-dispatch.
        if !self.saw_done {
            self.ready.push_back(Err(ProviderError::StreamReset(
                "byte stream ended before `[DONE]` sentinel".to_string(),
            )));
        }
    }

    /// Append a byte chunk to the buffer and drain every complete (`\n`-terminated) line into
    /// `ready`. Partial trailing bytes remain buffered for the next chunk.
    fn ingest(&mut self, bytes: &[u8]) {
        self.line_buf.extend_from_slice(bytes);
        while let Some(pos) = self.line_buf.iter().position(|&b| b == b'\n') {
            // Split off the line (without the `\n`); keep the remainder.
            let mut line: Vec<u8> = self.line_buf.drain(..=pos).collect();
            line.pop(); // drop '\n'
            if line.last() == Some(&b'\r') {
                line.pop(); // tolerate CRLF
            }
            if let Some(item) = self.process_line(&line) {
                let is_err = item.is_err();
                self.ready.push_back(item);
                if is_err || self.saw_done {
                    break;
                }
            }
            if self.saw_done {
                break;
            }
        }
    }

    /// Classify and (if it is a `data:` payload) parse one logical SSE line.
    ///
    /// Returns `Some(item)` for a `data:` JSON payload (or the DONE sentinel, which sets
    /// `saw_done` and returns `None`), and `None` for comment / blank / `event:` lines.
    fn process_line(&mut self, raw: &[u8]) -> Option<Result<StreamDelta, ProviderError>> {
        // Lines are UTF-8 JSON; lossily decode for classification (parse still uses &str).
        let line = String::from_utf8_lossy(raw);
        let line = line.trim_end();

        if line.is_empty() || line.starts_with(':') || line.starts_with("event:") {
            return None; // keep-alive comment / blank / event-type line
        }
        let payload = match line.strip_prefix("data:") {
            Some(rest) => rest.trim_start(),
            None => return None, // unknown field line (id:, retry:, …) — ignore
        };
        // An empty `data:` / `data: ` payload is a heartbeat, not JSON — ignore it BEFORE the
        // DONE/parse path so it can never become a terminal Decode error.
        if payload.is_empty() {
            return None;
        }
        if payload == DONE_SENTINEL {
            self.saw_done = true;
            return None;
        }
        Some(parse_chunk(payload).map_err(|e| ProviderError::Decode(e.to_string())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    /// Build a byte stream from owned chunks for the decoder. Uses `Vec<u8>` chunks (the
    /// decoder is generic over `AsRef<[u8]>`, so this exercises the same path as
    /// `reqwest`'s `bytes::Bytes`).
    fn byte_stream(chunks: Vec<&str>) -> impl Stream<Item = reqwest::Result<Vec<u8>>> + Unpin {
        let owned: Vec<Vec<u8>> = chunks.into_iter().map(|s| s.as_bytes().to_vec()).collect();
        Box::pin(stream::iter(owned.into_iter().map(Ok)))
    }

    /// Collect a decoded stream into a Vec for assertions.
    async fn collect(chunks: Vec<&str>) -> Vec<Result<StreamDelta, ProviderError>> {
        decode_sse(byte_stream(chunks)).collect().await
    }

    #[tokio::test]
    async fn single_line_per_chunk() {
        let out = collect(vec![
            "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
            "data: [DONE]\n\n",
        ])
        .await;
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].as_ref().unwrap().content.as_deref(), Some("a"));
    }

    #[tokio::test]
    async fn multi_line_in_one_chunk() {
        let blob = "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\
                    data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\
                    data: [DONE]\n";
        let out = collect(vec![blob]).await;
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].as_ref().unwrap().content.as_deref(), Some("a"));
        assert_eq!(out[1].as_ref().unwrap().content.as_deref(), Some("b"));
    }

    #[tokio::test]
    async fn split_mid_line_across_chunks() {
        // The `data:` line is split arbitrarily, including mid-JSON.
        let out = collect(vec![
            "data: {\"choi",
            "ces\":[{\"delta\":{\"con",
            "tent\":\"hello\"}}]}\n\ndata: [DONE]\n\n",
        ])
        .await;
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].as_ref().unwrap().content.as_deref(), Some("hello"));
    }

    #[tokio::test]
    async fn split_mid_json_byte_for_byte() {
        // The pathological case: every single byte arrives in its own chunk.
        let full = "data: {\"choices\":[{\"delta\":{\"reasoning\":\"hi\"}}]}\n\ndata: [DONE]\n";
        let chunks: Vec<&str> = full.split("").filter(|s| !s.is_empty()).collect();
        let out = collect(chunks).await;
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].as_ref().unwrap().reasoning.as_deref(), Some("hi"));
    }

    #[tokio::test]
    async fn ignores_comment_keepalive_and_blank_lines() {
        let out = collect(vec![
            ": OPENROUTER PROCESSING\n\n",
            "\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n",
            ": OPENROUTER PROCESSING\n",
            "data: [DONE]\n",
        ])
        .await;
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].as_ref().unwrap().content.as_deref(), Some("x"));
    }

    #[tokio::test]
    async fn reasoning_details_chunk_decodes() {
        let line = "data: {\"choices\":[{\"delta\":{\"reasoning_details\":\
            [{\"type\":\"reasoning.text\",\"text\":\"step\",\"index\":0}]}}]}\n\n";
        let out = collect(vec![line, "data: [DONE]\n\n"]).await;
        assert_eq!(out.len(), 1);
        let d = out[0].as_ref().unwrap();
        let details = d.reasoning_details.as_ref().unwrap();
        match &details[0] {
            gw_schema::ReasoningDetail::Text { text, index, .. } => {
                assert_eq!(text, "step");
                assert_eq!(*index, 0);
            }
            other => panic!("expected reasoning.text, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn crlf_line_endings_tolerated() {
        let out = collect(vec![
            "data: {\"choices\":[{\"delta\":{\"content\":\"c\"}}]}\r\n\r\n",
            "data: [DONE]\r\n",
        ])
        .await;
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].as_ref().unwrap().content.as_deref(), Some("c"));
    }

    #[tokio::test]
    async fn missing_done_yields_stream_reset() {
        // No `[DONE]` sentinel and the stream just ends.
        let out = collect(vec![
            "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
        ])
        .await;
        assert_eq!(out.len(), 2);
        assert!(out[0].is_ok());
        match &out[1] {
            Err(ProviderError::StreamReset(_)) => {}
            other => panic!("expected StreamReset, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn malformed_json_is_decode_error_and_terminates() {
        let out = collect(vec![
            "data: {not json}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"unreached\"}}]}\n\n",
        ])
        .await;
        // The decode error is terminal: nothing after it is yielded.
        assert_eq!(out.len(), 1);
        match &out[0] {
            Err(ProviderError::Decode(_)) => {}
            other => panic!("expected Decode error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn done_without_trailing_newline_is_clean() {
        // Final `data: [DONE]` arrives without a terminating newline.
        let out = collect(vec![
            "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
            "data: [DONE]",
        ])
        .await;
        assert_eq!(out.len(), 1);
        assert!(out[0].is_ok());
    }

    #[tokio::test]
    async fn good_chunk_then_end_decodes_then_resets() {
        // One good chunk, then the byte stream ends WITHOUT `[DONE]`: a clean delta followed by
        // a terminal StreamReset. (Manufacturing a real `reqwest::Error` for the mid-stream
        // transport path is awkward; that mapping is unit-tested in `error.rs`.)
        let good: Vec<u8> = b"data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n".to_vec();
        let s = stream::iter(vec![Ok::<_, reqwest::Error>(good)]);
        let out: Vec<_> = decode_sse(Box::pin(s)).collect().await;
        assert_eq!(out.len(), 2);
        assert!(out[0].is_ok());
        assert!(matches!(out[1], Err(ProviderError::StreamReset(_))));
    }

    #[tokio::test]
    async fn newline_less_complete_line_without_done_resets() {
        // PRE-MERGE 2: a complete, valid-JSON `data:` line with NO trailing newline and NO
        // `[DONE]`. The flushed trailing delta must NOT mask the truncation — the consumer must
        // still see a terminal StreamReset so the retry layer re-dispatches.
        let out = collect(vec![
            "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}",
        ])
        .await;
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].as_ref().unwrap().content.as_deref(), Some("a"));
        assert!(matches!(out[1], Err(ProviderError::StreamReset(_))));
    }

    #[tokio::test]
    async fn data_after_done_is_not_yielded() {
        // Folded followup: once `[DONE]` is seen the decoder stops; later data is discarded.
        let out = collect(vec![
            "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
            "data: [DONE]\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"ghost\"}}]}\n\n",
        ])
        .await;
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].as_ref().unwrap().content.as_deref(), Some("a"));
    }

    #[tokio::test]
    async fn empty_data_heartbeat_is_ignored_not_decode_error() {
        // Folded followup: empty `data:` / `data: ` payloads are heartbeats, not JSON.
        let out = collect(vec![
            "data:\n\n",
            "data: \n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n",
            "data: [DONE]\n\n",
        ])
        .await;
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].as_ref().unwrap().content.as_deref(), Some("x"));
    }
}
