//! `gw-engine` — the headless orchestrator.
//!
//! Owns the job queue, `Semaphore` concurrency, per-job `CancellationToken`, and the admission
//! pipeline (generate → judge → admit/reject/revise → persist). Runnable without a terminal;
//! the TUI is one consumer of its event stream. Depends on `gw-schema`, `gw-generate`,
//! `gw-judge`, `gw-storage`, and `gw-format`.
