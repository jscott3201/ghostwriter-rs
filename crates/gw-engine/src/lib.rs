//! `gw-engine` — the headless orchestrator.
//!
//! The convergence point of the pipeline: it drives every record through the per-record state machine
//! (generate → verify → judge → admit/reject/revise/escalate → format → export), persisting after
//! EVERY transition, with a concurrency-capped sharded executor, per-job cancellation, content-hash
//! call caching (never re-spend), a run-wide budget cap, and crash-recovery. Runnable WITHOUT a
//! terminal; the TUI and the CLI are interchangeable consumers of its [`EngineEvent`] stream. Depends
//! on `gw-schema`, `gw-generate`, `gw-judge`, `gw-storage`, `gw-format`, and `gw-providers`.
//!
//! ## The shape of a run
//!
//! ```text
//! Engine::run(run_id, seed_source, cancel)
//!   └─ for each shard (concurrent, Semaphore-capped):
//!        load resume cursor ─▶ for each un-committed seed item:
//!          run_group (best-of-k):
//!            ├─ generate k siblings (teacher spend, content-hash cached, budget-gated)
//!            ├─ drive each: AssistantGenerated→Verified→Judged→{Admitted|Rejected|Revising|NeedsReview}
//!            │             →Formatted→Exported            (persist after EVERY transition)
//!            ├─ admit the best by judging.aggregate (verifier gate must pass); RETAIN the rest
//!            └─ bounded single revise for any Revising member (no second revising)
//!          commit shard cursor (checkpoint)
//! ```
//!
//! ## Invariants enforced in code (see the cited `file.rs:fn`)
//!
//! - **persist-after-every-transition** — every edge calls `Store::put` + `advance_lifecycle`:
//!   `step::persist_envelope_and_advance`.
//! - **idempotent crash-resume** — `step` reads `lifecycle.state` and advances from there; a record
//!   persisted mid-flight re-enters at its last state, not from `Seeded`: `step::step` + `step::drive`.
//! - **never re-spend** — a sibling/retry already persisted is driven, not re-generated:
//!   `sibling::run_group` / `revise::revise_once`; judge calls go through `grade_panel_cached`.
//! - **best-of-k retain siblings** — the best is selected by `judging.aggregate`; rejected siblings
//!   stay at their terminal state (never deleted): `sibling::select_and_finalize`.
//! - **R-prior never identity** — the engine builds `uniform_offdiagonal(k, rho)` and FAILS LOUD on an
//!   identity R for a `k > 1` panel: `grade::correlation_prior`.
//! - **single-bounded revise** — a second `Revise` on the `attempt = 1` retry is downgraded to
//!   `Rejected` inside `step::reconcile` (the retry is tagged `step::REVISE_RETRY_TAG`), so there is
//!   never a second `revising`; one retry per original: `revise::revise_once`.
//! - **verdict → lifecycle** — `gw-judge::Decision::to_lifecycle` is applied at reconcile:
//!   `step::reconcile`.
//! - **needs_review leaves the pipeline** — `NeedsReview` is terminal-for-`step` and never formatted /
//!   exported / counted admitted: `step::step` (terminal arm) + `executor::Engine::tally`.
//! - **budget cutoff** — no new teacher work once the cap is reached: `budget::BudgetMeter` +
//!   `executor::Engine::run_shard`. The in-memory meter is REHYDRATED from persisted `cost.usd` at run
//!   start so a restart does not re-grant the cap: `executor::Engine::persisted_spend` +
//!   `budget::BudgetMeter::reset_to`.
//! - **deterministic answer rail** — the per-record `VerificationContract` is carried on the envelope
//!   and threaded into the Verify rail, so a wrong answer / complied-with adversarial prompt is caught
//!   on the deterministic rail (not silently panel-admitted): `step::verify`.
//! - **no lost work** — a budget-gated bounded revise does NOT commit the shard cursor past its item
//!   (it re-drives under fresh budget) and `Revising` is a counted non-terminal bucket:
//!   `executor::Engine::process_item` / `run_shard` / `tally`.
//! - **per-record fault isolation** — a record-level fault (bad teacher/judge/format) parks ONE record
//!   at `Error` and the run CONTINUES; only infrastructure faults abort the run:
//!   `error::EngineError::is_record_level` + `executor::Engine::park_item_errored`.
//!
//! ## Hermetic testing
//!
//! Every side-effecting client is injected through [`Clients`] (a [`Provider`](gw_providers::Provider)
//! for the teacher + judge rails, an [`Embedder`](gw_generate::Embedder), a
//! [`SandboxOracle`](gw_judge::SandboxOracle), the [`Store`](gw_storage::Store), and a [`SeedSource`]),
//! so unit + integration tests run over fakes + `Store::open_in_memory` — NO network, deterministic. A
//! live test is `#[ignore]` + key-gated.

mod budget;
mod checkpoint;
mod clients;
mod control;
mod error;
mod event;
mod executor;
mod grade;
mod revise;
mod seed;
mod sibling;
mod step;

pub use budget::BudgetMeter;
pub use checkpoint::{ShardCursor, commit_cursor, load_cursor};
pub use clients::{AreaConfig, Clients, DEFAULT_CORRELATION_RHO, DEFAULT_K, DEFAULT_MAX_TOKENS};
pub use control::RunControl;
pub use error::{EngineError, Result};
pub use event::{DEFAULT_EVENT_CAPACITY, EngineEvent, EventSink};
pub use executor::{Engine, ExportSpec, RunReport};
pub use grade::{correlation_prior, decision_from_judging, verifier_grade_from_verification};
pub use seed::{InMemorySeedSource, SeedItem, SeedSource, record_id};
pub use sibling::{GroupOutcome, run_group};
pub use step::{drive, is_terminal, step};

// Re-export the revise entrypoint under a stable path.
pub use revise::revise_once;
