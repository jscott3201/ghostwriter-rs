//! `gw-eval` — the OFF-PATH, headless, GPU-free, read-only evaluation/diagnostics crate.
//!
//! Two v1 instruments, both model-free and reading only fields the harness already persists:
//!
//! - [`separation`] (A1) — a selector-vs-random **separation diagnostic** plus the
//!   `decidable_fraction` judge-budget skip lever. Proves, model-free, that argmax-aggregate
//!   selection beats random at equal budget ON THE REASONING-QUALITY AXIS, and reports the
//!   fraction of sibling groups the deterministic verifier can actually decide (so panel judge
//!   tokens can be skipped on the rest).
//! - [`promote`] (A3 + ITEM 9) — a variance-aware **promotion gate**. Promotes a candidate
//!   fine-tune only if a capability-drift probe is clean AND a `k·σ`-noise-band A/B compare shows
//!   no regression with at least one win. The decision is re-derivable at a new `k` without
//!   re-running eval.
//!
//! No model calls, no network, no GPU, no terminal. Dependency graph:
//! `{gw-schema, gw-storage} → gw-eval`. The pure analysis cores ([`separation::analyze`],
//! [`promote::promote`]) are infallible and unit-testable without async; thin async wrappers
//! (e.g. [`separation::analyze_store`]) read a [`gw_storage::Store`].
//!
//! ## Error handling
//!
//! Library code never uses `anyhow`; every fallible path returns the typed [`EvalError`].

mod error;
pub mod promote;
pub mod separation;

#[cfg(test)]
mod test_support;

pub use error::{EvalError, Result};
pub use promote::{
    BenchmarkOutcome, EvalResults, PromoteConfig, PromotionReport, promote as promote_gate,
};
pub use separation::{SeparationConfig, SeparationReport, analyze, analyze_store};
