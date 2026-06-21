//! Live smoke test against the real OpenRouter API. **Never runs in CI.**
//!
//! Gated two independent ways:
//!   1. `#[ignore]` — `cargo nextest run` and `--profile ci` pass no `--ignored`, so the offline
//!      pipeline skips it entirely.
//!   2. a runtime `OPENROUTER_API_KEY` presence check — so even an explicit `--ignored` run
//!      no-ops gracefully (prints a skip note and returns) when no key is configured.
//!
//! Run it manually with a key:
//!
//! ```sh
//! set -a; source .env; set +a   # load OPENROUTER_API_KEY into the env (never committed)
//! cargo test -p gw-providers --test live_openrouter -- --ignored --nocapture
//! ```
//!
//! It is the one round-trip synthetic fixtures cannot fake: it proves the hand-rolled SSE
//! decoder captures real chain-of-thought from a live reasoning teacher
//! (`minimax/minimax-m3`, pinned to the `novita` provider for cost/speed and reproducible
//! provenance).

use std::time::Duration;

use futures::StreamExt;
use gw_providers::{ChatRequest, OpenRouterProvider, Provider, ProviderRouting, ReasoningParam};
use gw_schema::{Content, Message, ReasoningEffort, Role};

/// Model + provider under test (owner-approved): MiniMax-M3, hard-pinned to Novita.
const MODEL: &str = "minimax/minimax-m3";
const PROVIDER_SLUG: &str = "novita";
/// Upper bound on the whole streamed exchange, so a hung connection fails fast instead of
/// blocking the (manual) test run forever.
const STREAM_BUDGET: Duration = Duration::from_secs(120);

#[tokio::test]
#[ignore = "live network: requires OPENROUTER_API_KEY; run with `-- --ignored`"]
async fn minimax_m3_via_novita_streams_reasoning() {
    if std::env::var("OPENROUTER_API_KEY").is_err() {
        eprintln!("OPENROUTER_API_KEY not set — skipping live smoke test (this is expected in CI)");
        return;
    }

    let provider = OpenRouterProvider::from_env().expect("build provider from OPENROUTER_API_KEY");

    let req = ChatRequest::new(
        MODEL,
        vec![Message {
            role: Role::User,
            content: Content::Text(
                "What is 17 * 23? Reason through it step by step, then state the final number."
                    .into(),
            ),
            reasoning: None,
            reasoning_details: None,
            tool_calls: None,
            name: None,
        }],
    )
    .with_reasoning(ReasoningParam::effort(ReasoningEffort::High))
    .with_provider(ProviderRouting::pin(PROVIDER_SLUG))
    .with_usage_accounting();

    let mut stream = provider
        .stream_chat(req)
        .await
        .expect("open the streamed response");

    let mut content = String::new();
    let mut reasoning = String::new();
    let mut detail_fragments = 0usize;
    let mut served_by = None;
    let mut model = None;
    let mut generation_id = None;
    let mut finish_reason = None;
    let mut native_finish_reason = None;
    let mut reasoning_tokens = None;
    let mut completion_tokens = None;
    let mut cost = None;

    let drain = async {
        while let Some(item) = stream.next().await {
            let delta = item.expect("each stream item decodes without a provider error");
            if let Some(c) = &delta.content {
                content.push_str(c);
            }
            if let Some(r) = &delta.reasoning {
                reasoning.push_str(r);
            }
            if let Some(d) = &delta.reasoning_details {
                detail_fragments += d.len();
            }
            if let Some(f) = &delta.finish_reason {
                finish_reason = Some(f.clone());
            }
            if let Some(nf) = &delta.native_finish_reason {
                native_finish_reason = Some(nf.clone());
            }
            if let Some(p) = &delta.provenance {
                // First non-None wins (these appear on the final chunk; first == last).
                if served_by.is_none() {
                    served_by = p.served_by.clone();
                }
                if model.is_none() {
                    model = p.model.clone();
                }
                if generation_id.is_none() {
                    generation_id = p.id.clone();
                }
            }
            if let Some(u) = &delta.usage {
                if reasoning_tokens.is_none() {
                    reasoning_tokens = u.reasoning_tokens();
                }
                if completion_tokens.is_none() {
                    completion_tokens = u.completion_tokens;
                }
                if cost.is_none() {
                    cost = u.cost;
                }
            }
        }
    };

    tokio::time::timeout(STREAM_BUDGET, drain)
        .await
        .expect("stream completed within the budget");

    eprintln!("\n--- live smoke result: {MODEL} @ {PROVIDER_SLUG} ---");
    eprintln!("served_by                = {served_by:?}");
    eprintln!("model                    = {model:?}");
    eprintln!("generation id            = {generation_id:?}");
    eprintln!("finish_reason            = {finish_reason:?}");
    eprintln!("native_finish_reason     = {native_finish_reason:?}");
    eprintln!("content chars            = {}", content.len());
    eprintln!("reasoning chars (flat)   = {}", reasoning.len());
    eprintln!("reasoning_details frags  = {detail_fragments}");
    eprintln!("reasoning_tokens (usage) = {reasoning_tokens:?}");
    eprintln!("completion_tokens        = {completion_tokens:?}");
    eprintln!("cost usd                 = {cost:?}");
    eprintln!(
        "reasoning (head)         = {}",
        reasoning.chars().take(240).collect::<String>()
    );
    eprintln!(
        "content (head)           = {}",
        content.chars().take(160).collect::<String>()
    );

    // The whole point of the crate: we captured chain-of-thought from a live teacher.
    assert!(
        !reasoning.is_empty() || detail_fragments > 0,
        "expected to capture reasoning (flat `reasoning` or structured `reasoning_details`) \
         from {MODEL}; got none"
    );
    // And a final answer alongside the CoT (reasoning is a sibling of content, never inlined).
    assert!(!content.is_empty(), "expected a non-empty final answer");
    // The provider pin took effect → deterministic provenance.
    if let Some(sb) = &served_by {
        assert!(
            sb.to_lowercase().contains(PROVIDER_SLUG),
            "pinned provider `{PROVIDER_SLUG}` but served_by was {sb:?}"
        );
    }
}
