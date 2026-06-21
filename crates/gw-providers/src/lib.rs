//! `gw-providers` — the model I/O layer.
//!
//! **Load-bearing piece:** a hand-rolled `reqwest` + SSE streaming client with a custom delta
//! type that captures `content` AND `reasoning`/`reasoning_details`. `async-openai` 0.41.1's
//! typed stream delta silently drops the reasoning field, so the teacher path owns its own
//! response parsing (capturing full chain-of-thought is the entire reason this harness exists).
//! Also: a `governor` GCRA rate limiter per provider/model and transient-error retries.
//!
//! ## Layout
//!
//! - `error` — [`ProviderError`], the single error type + retryability classification.
//! - `delta` — [`StreamDelta`], the custom delta that captures CoT, and the SSE chunk
//!   wire-types.
//! - `sse` — [`decode_sse`], the hand-rolled SSE line decoder that buffers partial lines
//!   across arbitrary byte-chunk boundaries and ends on `[DONE]`.
//! - `request` — [`ChatRequest`] + [`ReasoningParam`] (`{"effort":"xhigh"}` /
//!   `{"max_tokens":N}`), serialized with `stream: true`.
//! - `limiter` — [`RateLimiter`], a thin `governor` GCRA wrapper with an async `until_ready`.
//! - `retry` — [`retry()`], hand-rolled exponential backoff honoring `Retry-After`.
//! - `client` — [`OpenRouterProvider`], the concrete [`Provider`] impl.
//!
//! ## The [`Provider`] trait
//!
//! [`Provider::stream_chat`] returns a `Pin<Box<dyn Stream<…> + Send>>` of decoded deltas. The
//! crate uses **no `async-trait`** dependency: the trait method is `async fn` returning a boxed
//! stream, which keeps the trait usable as `&dyn Provider` from the engine without adding a
//! macro crate. Each yielded item is a [`StreamDelta`] or a [`ProviderError`]; a mid-stream
//! reset arrives as a terminal [`ProviderError::StreamReset`].
//!
//! ```no_run
//! use futures::StreamExt;
//! use gw_providers::{ChatRequest, OpenRouterProvider, Provider, ReasoningParam};
//! use gw_schema::{Content, Message, Role};
//!
//! # async fn run() -> Result<(), gw_providers::ProviderError> {
//! let provider = OpenRouterProvider::from_env()?; // reads OPENROUTER_API_KEY
//! let req = ChatRequest::new(
//!     "z-ai/glm-5.2",
//!     vec![Message {
//!         role: Role::User,
//!         content: Content::Text("Prove sqrt(2) is irrational.".into()),
//!         reasoning: None,
//!         reasoning_details: None,
//!         tool_calls: None,
//!         name: None,
//!     }],
//! )
//! .with_reasoning(ReasoningParam::xhigh())
//! .with_usage_accounting();
//!
//! let mut stream = provider.stream_chat(req).await?;
//! while let Some(delta) = stream.next().await {
//!     let delta = delta?;
//!     if let Some(cot) = delta.reasoning {
//!         print!("{cot}");
//!     }
//! }
//! # Ok(())
//! # }
//! ```

mod client;
mod delta;
mod delta_wire;
mod error;
mod limiter;
mod request;
mod retry;
mod sse;

use std::pin::Pin;

use futures::stream::Stream;

pub use client::{
    DEFAULT_API_KEY_ENV, DEFAULT_BASE_URL, OpenRouterProvider, OpenRouterProviderBuilder,
};
pub use delta::{ChunkProvenance, CompletionTokensDetails, StreamDelta, Usage};
pub use error::ProviderError;
pub use limiter::RateLimiter;
pub use request::{ChatRequest, ProviderRouting, ReasoningParam, UsageRequest};
pub use retry::{RetryPolicy, retry, retry_with};
pub use sse::decode_sse;

/// The boxed, `Send` stream of decoded deltas returned by [`Provider::stream_chat`].
pub type DeltaStream = Pin<Box<dyn Stream<Item = Result<StreamDelta, ProviderError>> + Send>>;

/// The boxed, `Send` future returned by [`Provider::stream_chat`]. Boxing (rather than RPITIT
/// `impl Future`) keeps the trait **dyn-compatible** so it can be used as `&dyn Provider`.
pub type StreamChatFuture<'a> =
    Pin<Box<dyn std::future::Future<Output = Result<DeltaStream, ProviderError>> + Send + 'a>>;

/// A streaming chat provider over an OpenAI-compatible `/chat/completions` endpoint.
///
/// Implemented by [`OpenRouterProvider`]. The single method streams a chat completion as a
/// sequence of [`StreamDelta`]s, each carrying incremental `content` and/or
/// `reasoning`/`reasoning_details` (the captured chain-of-thought). The returned future
/// resolves once a response is in hand (after rate-limiting + retries); per-delta errors and a
/// terminal [`ProviderError::StreamReset`] arrive on the stream itself.
///
/// The method returns a boxed future ([`StreamChatFuture`]) rather than RPITIT `impl Future`,
/// so the trait is **dyn-compatible** — the engine can hold a `&dyn Provider`. The `Send + Sync`
/// supertrait bound means a bare `Box<dyn Provider>` is itself `Send + Sync`, so it can be moved
/// into a spawned task / shared across threads (e.g. a `tokio::spawn`-ed generation worker)
/// without the caller having to spell out the markers.
pub trait Provider: Send + Sync {
    /// Stream a chat completion. The future completes once the (retried, rate-limited) HTTP
    /// response is established; the body is then decoded lazily as the returned stream is
    /// polled.
    ///
    /// # Errors
    /// Returns a [`ProviderError`] if the request cannot be established (config error, or all
    /// retries exhausted on a transient fault). Errors encountered *while streaming* are
    /// yielded as `Err` items on the returned [`DeltaStream`].
    fn stream_chat(&self, req: ChatRequest) -> StreamChatFuture<'_>;
}

/// Compile-time assertion that [`Provider`] is dyn-compatible (object-safe). If a future change
/// reintroduces RPITIT, this stops compiling.
#[allow(dead_code)]
fn _assert_provider_dyn_compatible(p: &dyn Provider) -> &dyn Provider {
    p
}

/// Compile-time assertion that `Box<dyn Provider>` is `Send + Sync` (so it is spawnable /
/// shareable). If the supertrait bound is dropped, this stops compiling.
#[allow(dead_code)]
fn _assert_provider_send_sync() {
    fn req<T: Send + Sync>() {}
    req::<Box<dyn Provider>>();
}
