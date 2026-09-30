//! `gw-eval` — the OFF-PATH, headless, GPU-free, read-only evaluation/diagnostics crate.
//!
//! Two model-free instruments reading persisted records and externally supplied evidence:
//!
//! - [`separation`] — descriptive verifier mixedness and judge-score spread, with optional
//!   independent [`outcomes`] for a frozen corpus. Qualification applies to the declared
//!   selection rule, reference metric and independent-prompt assumption; it does not establish
//!   the engine's full admission quality or downstream learning benefit.
//! - [`promote`] (A3 + ITEM 9) — a variance-aware **promotion gate**. Promotes a candidate
//!   fine-tune only with complete, finite supplied evidence, a clean capability-drift probe, and
//!   a `k·σ`-noise-band A/B comparison showing
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
mod outcome_analysis;
pub mod outcomes;
pub mod promote;
pub mod separation;

#[cfg(test)]
mod test_support;

pub use error::{EvalError, Result};
pub use outcomes::{OutcomeEvidence, OutcomeReport, OutcomeStatus};
pub use promote::{
    BenchmarkOutcome, EvalResults, EvaluationSide, EvidenceIssue, PromoteConfig, PromotionReport,
    promote as promote_gate,
};
pub use separation::{
    SeparationConfig, SeparationDiagnostics, SeparationReport, analyze, analyze_store,
};
