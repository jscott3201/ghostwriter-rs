//! `gw-judge` — two-rail grading and admission, and the design-effect consensus.
//!
//! Decides which synthesized `(user, assistant+CoT)` traces are **admitted**, **revised**,
//! **rejected**, or **escalated** for human/verifier review — and records *why*, so every
//! admission is re-derivable and auditable. It owns the grading *algorithm* and the live
//! trait-bearing rail types; `gw-schema` owns only the *persisted* envelope. Depends on
//! `gw-schema` (envelope types), `gw-providers` (the injected [`Provider`](gw_providers::Provider)
//! for judge calls), and `gw-storage` (the never-re-spend call cache).
//!
//! ## The two rails + the composition
//!
//! ```text
//! HybridGrader
//! ├── Verifier  rail (verifier.rs) — deterministic, ground-truth HARD GATE; pure/local
//! │                                  (Oracle::SandboxExecution reaches an injected SandboxOracle seam)
//! ├── JudgePanel rail (panel.rs)   — LLM-as-jury via the Provider trait, blind sealed first pass,
//! │                                  every call wrapped by the never-re-spend cache (cache.rs)
//! └── consensus     (consensus.rs) — the calibration-weighted, correlation-adjusted design effect
//! ```
//!
//! ## The load-bearing piece — consensus (INVARIANT-f)
//!
//! Admission aggregation is **NEVER a plain mean or majority vote**. The aggregate is the
//! calibration-weighted consensus ([`weighted_aggregate`]); whether that aggregate may be *trusted*
//! is gated by the **weighted correlation design-effect** ([`effective_n`]):
//!
//! ```text
//! n_eff = (Σ wᵢ)² / (wᵀ R w)
//! ```
//!
//! where `w` are the per-judge calibration weights ([`calibration_weights`]) and `R` is the
//! inter-judge correlation matrix. This unifies BOTH failure modes — weight concentration AND
//! inter-judge correlation — in one number. Negative off-diagonals are clipped to 0 and `n_eff` is
//! capped at `k`, so equally-weighted perfectly-correlated judges collapse to `n_eff ≈ 1` (NOT `k`)
//! and equally-weighted independent judges give `n_eff ≈ k`. A too-correlated panel
//! (`n_eff/k < min_n_eff_ratio`) escalates rather than self-reinforcing its shared error.
//!
//! ## Invariants enforced in code (see the cited `file.rs:fn`)
//!
//! - **INVARIANT-f** (never a plain mean): the aggregate is `consensus::weighted_aggregate` gated by
//!   `consensus::effective_n`; the n_eff clip + `≤ k` cap live in `consensus::effective_n`.
//! - **Verifier authoritative** over the panel: `grader::HybridGrader::grade` short-circuits to
//!   `Decision::Reject` when `VerifierGrade::is_hard_reject`, regardless of any panel score.
//! - **Never re-spend**: `cache::grade_one_cached` checks `Store::cache_get` before spending and
//!   `cache_put` after, with `temperature_bits` folded into the key via `cache::folded_rubric_key`.
//! - **Re-derivable admission**: `Judging.threshold_at_decision` is stored and
//!   `grader::rederive_verdict` re-derives Admit/Reject/Revise/Escalate from a stored panel WITHOUT
//!   re-judging.
//! - **Verify hard gate** (DATA-SCHEMA INVARIANT b): `verifier::reasoning_present_check` hard-fails a
//!   required-CoT record with empty / summary-only / encrypted-only reasoning or
//!   `reasoning_tokens == 0`.
//!
//! ## Hermetic testing
//!
//! Every judge model call is behind the [`Provider`](gw_providers::Provider) trait, so unit tests
//! inject a fake in-memory provider returning canned judge JSON — NO network. The verifier rail is
//! pure; the cache uses `Store::open_in_memory`. Any live test is `#[ignore]` + env-key-gated.

mod cache;
mod calibration;
mod consensus;
mod decision;
mod error;
mod grader;
mod panel;
mod verifier;

pub use cache::{JUDGE_CACHE_KIND, folded_rubric_key, grade_one_cached, grade_panel_cached};
pub use calibration::{
    CalibrationParams, DEFAULT_BETA, DEFAULT_GAMMA, calibration_weights, uniform_weights,
};
pub use consensus::{
    CorrelationMatrix, agreement, effective_n, kish_effective_n, weighted_aggregate,
};
pub use decision::{Decision, DecisionReason, EscalateTo, Verdict};
pub use error::{JudgeError, Result};
pub use grader::{AreaThresholds, GradeOutcome, HybridGrader, rederive_verdict};
pub use panel::{
    Grade, JudgeSampling, JudgeScoring, PanelJudge, build_judge_request, grade_one, grade_panel,
};
pub use verifier::{
    NullSandboxOracle, SandboxOracle, VerifierGrade, VerifierInput, reasoning_present_check,
    run_verifier, verifier_reject_decision,
};
