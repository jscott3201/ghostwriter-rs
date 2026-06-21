//! Offline replay of **real** OpenRouter SSE captures through the actual [`decode_sse`] decoder.
//!
//! Unlike the live smoke test, this runs in CI with no network: the fixtures in
//! `tests/fixtures/sse/*.sse` are verbatim streamed responses captured from four distinct
//! `minimax/minimax-m3` backends (Novita, Together, AtlasCloud, Minimax). They are a real-world
//! regression corpus — if a future change breaks decoding of any provider's actual wire shape
//! (reasoning capture, the lenient `usage` block, the `[DONE]` framing), one of these fails.
//!
//! Each fixture is replayed twice: as one whole chunk, and split into 7-byte pieces — the second
//! deliberately fragments lines (and JSON) mid-byte to exercise the decoder's partial-line
//! buffering against genuine payloads.

use futures::{StreamExt, stream};
use gw_providers::{ProviderError, StreamDelta, decode_sse};

/// All four curated provider captures. Every one must decode identically.
const FIXTURES: &[&str] = &[
    "novita.sse",
    "together.sse",
    "atlas-cloud_fp8.sse",
    "minimax_fp8.sse",
];

fn fixture_bytes(name: &str) -> Vec<u8> {
    let path = format!("{}/tests/fixtures/sse/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("read fixture {path}: {e}"))
}

/// Feed `bytes` through `decode_sse` split into `chunk_size`-byte pieces.
async fn decode_chunked(
    bytes: &[u8],
    chunk_size: usize,
) -> Vec<Result<StreamDelta, ProviderError>> {
    let chunks: Vec<Vec<u8>> = bytes.chunks(chunk_size).map(<[u8]>::to_vec).collect();
    let byte_stream = stream::iter(chunks.into_iter().map(Ok::<Vec<u8>, reqwest::Error>));
    decode_sse(Box::pin(byte_stream)).collect().await
}

#[derive(Default)]
struct Captured {
    reasoning: String,
    content: String,
    detail_fragments: usize,
    reasoning_tokens: Option<u64>,
    cost: Option<f64>,
    finish_reason: Option<String>,
    errors: Vec<String>,
}

fn accumulate(items: Vec<Result<StreamDelta, ProviderError>>) -> Captured {
    let mut c = Captured::default();
    for item in items {
        match item {
            Ok(delta) => {
                if let Some(s) = delta.content {
                    c.content.push_str(&s);
                }
                if let Some(s) = delta.reasoning {
                    c.reasoning.push_str(&s);
                }
                if let Some(d) = delta.reasoning_details {
                    c.detail_fragments += d.len();
                }
                if let Some(f) = delta.finish_reason {
                    c.finish_reason = Some(f);
                }
                if let Some(u) = delta.usage {
                    c.reasoning_tokens = u.reasoning_tokens().or(c.reasoning_tokens);
                    c.cost = u.cost.or(c.cost);
                }
            }
            Err(e) => c.errors.push(e.to_string()),
        }
    }
    c
}

/// Every fixture must decode cleanly (no errors, ends on `[DONE]`), capture chain-of-thought in
/// both forms, capture the final answer, and surface the lenient `usage` accounting — identically
/// whether delivered in one chunk or fragmented 7 bytes at a time.
async fn assert_decodes(name: &str) {
    let bytes = fixture_bytes(name);
    for chunk_size in [bytes.len().max(1), 7] {
        let c = accumulate(decode_chunked(&bytes, chunk_size).await);
        assert!(
            c.errors.is_empty(),
            "{name} @chunk={chunk_size}: decode errors {:?}",
            c.errors
        );
        assert!(
            !c.reasoning.is_empty() || c.detail_fragments > 0,
            "{name} @chunk={chunk_size}: no reasoning captured (flat or structured)"
        );
        assert!(
            !c.content.is_empty(),
            "{name} @chunk={chunk_size}: no content captured"
        );
        assert_eq!(
            c.finish_reason.as_deref(),
            Some("stop"),
            "{name} @chunk={chunk_size}: unexpected finish_reason"
        );
        assert!(
            c.reasoning_tokens.is_some(),
            "{name} @chunk={chunk_size}: usage.completion_tokens_details.reasoning_tokens missing"
        );
        assert!(
            c.cost.is_some(),
            "{name} @chunk={chunk_size}: usage.cost missing"
        );
    }
}

#[tokio::test]
async fn all_provider_fixtures_decode_cleanly() {
    for name in FIXTURES {
        assert_decodes(name).await;
    }
}

/// Sanity: the curated set actually covers multiple distinct backends.
#[test]
fn fixture_set_is_present() {
    for name in FIXTURES {
        assert!(
            !fixture_bytes(name).is_empty(),
            "fixture {name} is missing or empty"
        );
    }
}
