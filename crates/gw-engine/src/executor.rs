//! The sharded executor: the concurrency-capped async run loop (ARCHITECTURE §3.3, §5).
//!
//! [`Engine::run`] partitions the seed space into shards (per the injected [`SeedSource`]), drives
//! each shard concurrently under a [`tokio::sync::Semaphore`] (the `max_in_flight` cap), and wires a
//! per-run [`CancellationToken`] so a Ctrl-C / `Abort` breach
//! tears down cleanly. Each shard:
//!
//! 1. reads its resume cursor (crash-recovery: skip already-committed seed items);
//! 2. for each remaining seed item, runs the best-of-k group ([`crate::run_group`]) — which
//!    generates, drives, and selects the best;
//! 3. re-enters generation for any `Revising` member (the bounded single retry,
//!    [`crate::revise_once`]);
//! 4. commits the shard cursor past the item (checkpoint);
//! 5. stops dispatching NEW work when the budget cap is reached (Drain) or the token is cancelled.
//!
//! ## Concurrency model
//!
//! Shards run as spawned tasks; WITHIN a shard, seed items are processed sequentially (the per-item
//! best-of-k fan-out is itself the parallelism knob, and sequential within-shard keeps the resume
//! cursor monotone). The `max_in_flight` semaphore caps how many seed items are in flight ACROSS all
//! shards at once. The [`Store`](gw_storage::Store) is `Clone` (shared pool), the `Clients` bundle is
//! `Clone`, so each shard task owns its own handle.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use gw_schema::{BudgetBreach, CotPolicy, ExportManifest, LifecycleState, TrlFormat};
use gw_storage::{RecordFilter, RunStatus, export_parquet};

use crate::checkpoint::{commit_cursor, load_cursor};
use crate::clients::{AreaConfig, Clients};
use crate::control::RunControl;
use crate::error::{EngineError, Result};
use crate::event::EngineEvent;
use crate::seed::SeedSource;
use crate::sibling::run_group;
use crate::step::is_terminal;

/// Default circuit-breaker threshold (F2): if the first `CIRCUIT_BREAKER_PARKS` items in a run ALL
/// park at `Error` with ZERO successes, the run aborts as infrastructure-fatal regardless of the error
/// class. This is the LOAD-BEARING general backstop — it catches a provider-hard-down / any-systemic
/// fault the error taxonomy ([`EngineError::is_record_level`]) does not itself classify as fatal, so a
/// misconfigured run fails fast instead of churning the whole seed space into per-record `Error`s.
const CIRCUIT_BREAKER_PARKS: usize = 8;

/// The headless orchestrator entrypoint. Holds the injected [`Clients`] + [`AreaConfig`] and drives a
/// run end-to-end. Construct with [`Engine::new`]; drive with [`Engine::run`].
#[derive(Clone)]
pub struct Engine {
    clients: Clients,
    area: AreaConfig,
    max_in_flight: u32,
    on_breach: BudgetBreach,
    export: Option<ExportSpec>,
}

/// Whether a processed seed item is fully settled (safe to commit the cursor past) or still has
/// pending work that must be re-driven on a later relaunch (E2: a budget-gated bounded revise).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ItemOutcome {
    /// Every member reached a terminal state — the shard cursor may advance past this item.
    Settled,
    /// A member is parked at `Revising` because the budget was exhausted before its retry could run.
    /// The cursor MUST NOT advance past this item; a relaunch under fresh budget completes the retry.
    PendingRevise,
    /// Cancellation stopped this item at a persisted transition boundary. The cursor MUST NOT advance
    /// past it; a relaunch resumes from the persisted record state without re-spending.
    Interrupted,
}

/// Whether a processed item produced ANY healthy/decided record, for the circuit-breaker (F2). An
/// item is `Decided` when at least one of its records reached a non-`Error` terminal (a real pipeline
/// decision — `Exported`/`Rejected`/`NeedsReview`, or a pending `Revising`); `AllErrored` when every
/// record (and the k=1 escape-park) landed at `Error`. A run that produces only `AllErrored` items
/// consecutively is systemically broken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ItemHealth {
    /// At least one record reached a non-`Error` outcome — a genuine success; disarms the breaker.
    Decided,
    /// Every record for this item landed at `Error` — counts toward the consecutive-park breaker.
    AllErrored,
}

/// The run-level CIRCUIT-BREAKER backstop (F2): trips when the first `threshold` items ALL park at
/// `Error` with ZERO successes, aborting a systemically-broken run (provider hard-down, anything the
/// error taxonomy misses) regardless of error class. Shared across concurrent shards via `Arc`. Once
/// ANY item is `Decided`, the breaker DISARMS for the rest of the run (a run that produced a real
/// decision is not systemically broken — later isolated errors are legitimate per-record faults).
#[derive(Debug)]
struct CircuitBreaker {
    consecutive_parks: AtomicUsize,
    disarmed: AtomicBool,
    threshold: usize,
}

impl CircuitBreaker {
    fn new(threshold: usize) -> Self {
        Self {
            consecutive_parks: AtomicUsize::new(0),
            disarmed: AtomicBool::new(false),
            threshold: threshold.max(1),
        }
    }

    /// Record one item's health. Returns `true` if the breaker has TRIPPED (the run must abort). A
    /// `Decided` item disarms the breaker permanently; an `AllErrored` item increments the consecutive
    /// count (unless already disarmed) and trips at the threshold.
    fn observe(&self, health: ItemHealth) -> bool {
        match health {
            ItemHealth::Decided => {
                self.disarmed.store(true, Ordering::SeqCst);
                self.consecutive_parks.store(0, Ordering::SeqCst);
                false
            }
            ItemHealth::AllErrored => {
                if self.disarmed.load(Ordering::SeqCst) {
                    return false;
                }
                let n = self.consecutive_parks.fetch_add(1, Ordering::SeqCst) + 1;
                n >= self.threshold
            }
        }
    }
}

/// The terminal summary of a run: per-state record counts + whether it drained cleanly.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RunReport {
    /// Records admitted (reached `Admitted` or beyond — `Admitted`/`Formatted`/`Exported`).
    pub admitted: usize,
    /// Records exported (the terminal-good state).
    pub exported: usize,
    /// Records rejected.
    pub rejected: usize,
    /// Records parked at `NeedsReview` (LEFT the automated pipeline — not exported, not admitted).
    pub needs_review: usize,
    /// Records still at `Revising` — a NON-TERMINAL bucket (E2): a bounded revise that has not yet
    /// completed (e.g. its retry was budget-gated). These are re-driven on a later relaunch; they are
    /// neither admitted nor rejected, and counting them here keeps the in-progress state from being
    /// silently lost from the report.
    pub revising: usize,
    /// Records that errored unrecoverably.
    pub errored: usize,
    /// `true` if every shard drained without hitting the budget cap or a cancellation.
    pub completed: bool,
}

/// Optional end-of-run Parquet shard export configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportSpec {
    /// Destination Parquet shard path.
    pub dst: PathBuf,
    /// The target training-data template recorded in the export manifest.
    pub target: TrlFormat,
    /// Whether reasoning enters the supervised loss region for this export.
    pub cot: CotPolicy,
    /// Optional dataset version recorded in the sidecar manifest.
    pub dataset_version: Option<semver::Version>,
}

impl Engine {
    /// Build an engine over the injected clients + area config, with the `max_in_flight` concurrency
    /// cap (clamped to ≥ 1).
    #[must_use]
    pub fn new(clients: Clients, area: AreaConfig, max_in_flight: u32) -> Self {
        Self {
            clients,
            area,
            max_in_flight: max_in_flight.max(1),
            on_breach: BudgetBreach::Drain,
            export: None,
        }
    }

    /// Set the run-control policy for a budget breach. The default is [`BudgetBreach::Drain`].
    #[must_use]
    pub fn with_on_breach(mut self, policy: BudgetBreach) -> Self {
        self.on_breach = policy;
        self
    }

    /// Return the configured budget-breach policy.
    #[must_use]
    pub fn on_breach(&self) -> BudgetBreach {
        self.on_breach
    }

    /// Enable an end-of-run Parquet shard export. Without this builder, [`Self::run`] skips the export
    /// step entirely.
    #[must_use]
    pub fn with_export(mut self, spec: ExportSpec) -> Self {
        self.export = Some(spec);
        self
    }

    /// Return the configured end-of-run export spec, if one was installed.
    #[must_use]
    pub fn export_spec(&self) -> Option<&ExportSpec> {
        self.export.as_ref()
    }

    /// Run `source`'s seed space end-to-end under `run_id`, honoring `cancel`. Creates/loads the run
    /// row, drives every shard concurrently, and returns the terminal [`RunReport`].
    ///
    /// CRASH-RECOVERY: re-running the SAME `run_id` over the SAME `source` resumes — each shard skips
    /// its committed seed items, and any mid-flight record re-enters at its last persisted state (the
    /// teacher is never re-spent). BUDGET: once the cap is reached, no new teacher work is dispatched
    /// (Drain) or cancels in-flight items at transition boundaries (Abort). CANCELLATION: a cancelled
    /// token stops dispatching new work and leaves in-flight items at their last persisted boundary.
    ///
    /// # Errors
    /// Propagates the first [`EngineError`] from any shard. A shard error halts the run (status
    /// `Failed`); records already persisted stand for a later resume.
    pub async fn run<S: SeedSource + ?Sized>(
        &self,
        run_id: &str,
        source: &S,
        cancel: CancellationToken,
    ) -> Result<RunReport> {
        let shard_count = source.shard_count().max(1);
        let prompts_hash = source.prompts_hash()?;
        self.clients
            .store
            .validate_or_record_run_partition(run_id, shard_count, &prompts_hash)
            .await
            .map_err(|err| match err {
                gw_storage::StorageError::RunPartitionMismatch { .. } => {
                    EngineError::Invariant(err.to_string())
                }
                other => other.into(),
            })?;
        // Snapshot the config + budget cap into the run row (idempotent: re-creating resets to running).
        let config_json = serde_json::to_string(&self.budget_snapshot())?;
        self.clients
            .store
            .create_run(run_id, &config_json, Some(self.clients.budget.cap()))
            .await?;

        // E3: rehydrate the budget meter from spend already persisted for this run (a prior, possibly
        // crashed, launch). The in-memory meter resets to 0 each process, so without this a restart
        // would re-grant the FULL cap and a resumed run could spend up to N×cap across N launches. Sum
        // the persisted `cost.usd` and SET the meter (idempotent across in-process re-runs + restarts),
        // BEFORE any shard dispatches so the gate is correct from the first item.
        let prior_spend = self.persisted_spend(run_id).await?;
        self.clients.budget.reset_to(prior_spend);

        self.seed_embedding_priors(run_id).await?;

        self.clients.events.emit(EngineEvent::RunStarted {
            run_id: run_id.to_string(),
            shards: shard_count,
        });

        let semaphore = Arc::new(Semaphore::new(self.max_in_flight as usize));
        // F2: a run-level circuit-breaker shared across shards — aborts a systemically-broken run.
        let breaker = Arc::new(CircuitBreaker::new(CIRCUIT_BREAKER_PARKS));
        let mut handles = Vec::with_capacity(shard_count);
        for shard in 0..shard_count as i64 {
            let engine = self.clone();
            let run_id = run_id.to_string();
            let items = source.items_for_shard(shard);
            let semaphore = Arc::clone(&semaphore);
            let breaker = Arc::clone(&breaker);
            let cancel = cancel.clone();
            handles.push(tokio::spawn(async move {
                engine
                    .run_shard(&run_id, shard, items, &semaphore, &breaker, &cancel)
                    .await
            }));
        }

        // Join all shards; the first error aborts the run (status Failed, records stand for resume).
        for handle in handles {
            match handle.await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    self.clients
                        .store
                        .set_run_status(run_id, RunStatus::Failed)
                        .await?;
                    return Err(e);
                }
                Err(join_err) => {
                    self.clients
                        .store
                        .set_run_status(run_id, RunStatus::Failed)
                        .await?;
                    return Err(EngineError::Invariant(format!(
                        "shard task panicked/cancelled: {join_err}"
                    )));
                }
            }
        }

        let halted = cancel.is_cancelled() || self.clients.budget.is_exhausted();
        let status = if halted {
            RunStatus::Halted
        } else {
            RunStatus::Completed
        };
        self.clients.store.set_run_status(run_id, status).await?;

        let mut report = self.tally(run_id).await?;
        report.completed = !halted;
        // The gate and export filter both use lifecycle-admitted records, so an exported shard agrees
        // with the run report and best-of-k retained siblings remain excluded.
        if !halted
            && report.admitted > 0
            && let Some(spec) = &self.export
        {
            match self.export_shard(run_id, spec).await {
                Ok(manifest) => self.clients.events.emit(EngineEvent::ShardExported {
                    run_id: run_id.to_string(),
                    manifest,
                }),
                Err(err) => {
                    tracing::warn!(run_id, error = %err, "end-of-run shard export failed");
                    self.clients.events.emit(EngineEvent::ShardExportFailed {
                        run_id: run_id.to_string(),
                        error: err.to_string(),
                    })
                }
            };
        }
        self.clients.events.emit(EngineEvent::RunFinished {
            run_id: run_id.to_string(),
            completed: report.completed,
        });
        Ok(report)
    }

    async fn seed_embedding_priors(&self, run_id: &str) -> Result<()> {
        let records = self
            .clients
            .store
            .scan(&RecordFilter::new().run_id(run_id))
            .await?;
        let mut vectors = Vec::new();
        for record in records.into_iter().filter(|record| {
            matches!(
                record.lifecycle.state,
                LifecycleState::Admitted | LifecycleState::Formatted | LifecycleState::Exported
            )
        }) {
            let Some(text) = crate::priors::user_turn_text(&record) else {
                tracing::warn!(record_id = %record.record_id, "admitted record has no user turn; skipping embedding prior");
                continue;
            };
            // Embed without holding the shared lock; gate snapshots remain short-lived.
            match self.clients.embedder.embed(&text) {
                Ok(vector) => vectors.push(vector),
                Err(error) => tracing::warn!(
                    record_id = %record.record_id,
                    %error,
                    "failed to seed embedding prior; continuing run"
                ),
            }
        }
        crate::priors::replace(&self.clients.priors, vectors);
        Ok(())
    }

    /// Drive one shard's seed items sequentially, resuming from the persisted cursor, under the shared
    /// semaphore + cancellation token. Commits the shard cursor after each item terminates.
    async fn run_shard(
        &self,
        run_id: &str,
        shard: i64,
        items: Vec<crate::seed::SeedItem>,
        semaphore: &Semaphore,
        breaker: &CircuitBreaker,
        cancel: &CancellationToken,
    ) -> Result<()> {
        let cursor = load_cursor(&self.clients.store, run_id, shard).await?;
        let resumed = cursor.next_offset > 0;
        self.clients.events.emit(EngineEvent::ShardStarted {
            run_id: run_id.to_string(),
            shard,
            resumed,
        });

        for item in items {
            // Crash-recovery: skip seed items already committed (below the resume cursor).
            if item.offset < cursor.next_offset {
                continue;
            }
            // Stop dispatching new work on cancellation or budget exhaustion.
            if cancel.is_cancelled() {
                break;
            }
            if !self.clients.budget.may_dispatch() {
                self.clients.events.emit(EngineEvent::BudgetReached {
                    run_id: run_id.to_string(),
                    spent: self.clients.budget.spent(),
                    cap: self.clients.budget.cap(),
                });
                if self.on_breach == BudgetBreach::Abort {
                    cancel.cancel();
                }
                break;
            }

            // Cap concurrent in-flight seed items across all shards. The permit is held for the whole
            // item and ALWAYS released (the `?`-free match below cannot early-return before `drop`), so
            // a record-level error never leaks a permit and deadlocks the pool (E5/E10).
            let permit = semaphore
                .acquire()
                .await
                .map_err(|e| EngineError::Invariant(format!("semaphore closed: {e}")))?;

            let control = RunControl::new(cancel, self.on_breach);
            let processed = self.process_item(run_id, shard, &item, control).await;
            drop(permit);

            let (item_outcome, health) = match processed {
                Ok((outcome, health)) => (outcome, health),
                // E5: a RECORD-LEVEL fault that ESCAPED the group (the k=1 single-trace path, or a fault
                // before any sibling persisted) is ISOLATED here — park the faulting record at `Error`,
                // emit `RecordErrored`, and CONTINUE the shard so other items still complete. (For k>1,
                // `run_group` already isolates a per-sibling fault internally and finalizes the
                // survivors, so a record-level error rarely reaches here for a fan-out group.) The item
                // counts as `AllErrored` toward the circuit-breaker (F2). An INFRASTRUCTURE fault
                // (Storage/Serde/Invariant, or a SYSTEMIC non-retryable auth/config provider fault — F2)
                // is fatal and propagates to abort the run.
                Err(e) if e.is_record_level() => {
                    self.park_item_errored(run_id, shard, &item, &e).await?;
                    // The item is settled-as-errored: advance the cursor past it (do not retry a
                    // deterministically-failing record forever).
                    (ItemOutcome::Settled, Some(ItemHealth::AllErrored))
                }
                Err(e) => return Err(e),
            };

            // F2 circuit-breaker: a systemically-broken run (consecutive items all errored, zero
            // successes) aborts FAST instead of churning the whole seed space into per-record `Error`s —
            // the load-bearing backstop for any systemic fault the error taxonomy did not itself classify
            // as fatal (e.g. a provider hard-down). A `Decided` item disarms it for the rest of the run.
            if let Some(h) = health
                && breaker.observe(h)
            {
                return Err(EngineError::Invariant(format!(
                    "circuit breaker: {} consecutive seed items all errored with zero successful records \
                     — aborting a systemically-broken run (provider down or misconfigured)",
                    breaker.threshold
                )));
            }

            match item_outcome {
                // The item fully terminated; advance the cursor past it.
                ItemOutcome::Settled => {
                    commit_cursor(&self.clients.store, run_id, shard, item.offset, "committed")
                        .await?;
                }
                // E2: a budget-gated revise is still pending. Do NOT commit the cursor past this item;
                // stop dispatching here (the budget is exhausted) so a relaunch re-drives it. Breaking
                // also prevents committing LATER offsets over this un-settled one (monotone cursor).
                ItemOutcome::PendingRevise => {
                    self.clients.events.emit(EngineEvent::BudgetReached {
                        run_id: run_id.to_string(),
                        spent: self.clients.budget.spent(),
                        cap: self.clients.budget.cap(),
                    });
                    break;
                }
                // A cancellation/Abort interrupted this item at a persisted boundary. Do not commit the
                // cursor past it; a relaunch re-drives from the record state and never re-spends.
                ItemOutcome::Interrupted => break,
            }
        }

        self.clients.events.emit(EngineEvent::ShardFinished {
            run_id: run_id.to_string(),
            shard,
        });
        Ok(())
    }

    /// Park the ACTUAL faulting record at [`LifecycleState::Error`] after a RECORD-LEVEL fault (E5/F1),
    /// and emit [`EngineEvent::RecordErrored`].
    ///
    /// ATTRIBUTION (F1): the best-of-k group drives siblings independently, so a fault may strike any
    /// sibling (`c1`/`c2`) while another is already healthy at `Judged`. The faulting sibling's id rides
    /// on the error via [`EngineError::attributed_record`] (stamped by `crate::run_group`); this parks
    /// THAT record — falling back to the item's primary id (`c0`) only for an unattributed fault (e.g.
    /// a k=1 path). If the faulting record never persisted (the fault hit during generation before the
    /// first `put`), a MINIMAL stub is persisted at `Error` so the failure is queryable, counted
    /// (`report.errored`), and auditable.
    ///
    /// NO-CLOBBER (F1, invariant 1): a fault on ONE sibling must NEVER overwrite a DIFFERENT healthy
    /// sibling. So this refuses to park any record already at HEALTHY PROGRESS — not just a terminal
    /// state, but ANY of `AssistantGenerated`/`Verified`/`Judged`/`Admitted`/`Formatted`/`Exported`.
    /// Even a mis-targeted park (a bug in attribution) can then never corrupt a good record; it degrades
    /// to a logged no-op. The error message is recorded (never carries a secret — the wrapped errors are
    /// safe to log).
    async fn park_item_errored(
        &self,
        run_id: &str,
        shard: i64,
        item: &crate::seed::SeedItem,
        err: &EngineError,
    ) -> Result<()> {
        // F1: park the ATTRIBUTED faulting record, not a blindly-assumed c0.
        let attributed = err.attributed_record();
        let rid = attributed
            .map(str::to_string)
            .unwrap_or_else(|| crate::seed::record_id(run_id, shard, item.seed, 0, 0));
        let msg = err.to_string();
        // Advance an existing row, or persist a minimal stub if generation failed before the first put.
        match self.clients.store.get(&rid).await {
            // NO-CLOBBER: never overwrite a DECIDED record (a non-Error terminal or the Revising handoff).
            // The attributed FAULTING record is still terminalized to Error from any in-flight state (a
            // mid-drive fault must not strand it); an un-attributed fallback parks only a pre-gen stub.
            Ok(existing) if !park_allowed(existing.lifecycle.state, attributed.is_some()) => {
                self.clients.events.emit(EngineEvent::RecordErrored {
                    record_id: rid,
                    error: msg,
                });
                return Ok(());
            }
            Ok(_) => {}
            Err(gw_storage::StorageError::NotFound(_)) => {
                let stub = self.error_stub(run_id, &rid, item);
                self.clients.store.put(&stub).await?;
            }
            Err(e) => return Err(e.into()),
        }
        self.clients
            .store
            .advance_lifecycle(&rid, LifecycleState::Error, Some(&msg))
            .await?;
        self.clients.events.emit(EngineEvent::RecordErrored {
            record_id: rid,
            error: msg,
        });
        Ok(())
    }

    /// A minimal `TrainingRecord` stub for a record-level fault that struck BEFORE generation persisted
    /// anything (E5) — delegates to the shared [`error_stub`] (the same builder `crate::sibling` uses to
    /// isolate a pre-persist sibling fault).
    fn error_stub(
        &self,
        run_id: &str,
        rid: &str,
        item: &crate::seed::SeedItem,
    ) -> gw_schema::TrainingRecord {
        error_stub(&self.area, &self.clients, run_id, rid, &item.candidate)
    }

    /// Process one seed item: run its best-of-k group, then the bounded revise for any `Revising`
    /// member. Errors on a member are surfaced (the record's last state stands for resume).
    ///
    /// Returns whether the item is FULLY SETTLED (every member reached a terminal state) and may be
    /// committed past. When a member is left at `Revising` because the budget was exhausted before its
    /// bounded retry could generate (E2), this returns `false` so the shard does NOT commit the cursor
    /// past the item — a relaunch under fresh budget re-drives it and completes the retry. Without this,
    /// the cursor would advance past a stranded `Revising` record and it would be skipped forever.
    async fn process_item(
        &self,
        run_id: &str,
        shard: i64,
        item: &crate::seed::SeedItem,
        control: RunControl<'_>,
    ) -> Result<(ItemOutcome, Option<ItemHealth>)> {
        let outcome = run_group(run_id, shard, item, &self.clients, &self.area, control).await?;
        // F2: the group's circuit-breaker health — did this item produce ANY non-`Error` record? Computed
        // from the group's terminal siblings (an empty group is a budget-Drain skip → neutral `None`).
        let health = group_health(&outcome.siblings);
        if outcome.interrupted {
            return Ok((ItemOutcome::Interrupted, None));
        }
        // The bounded single revise: re-enter generation for any member at `Revising`.
        for sibling in &outcome.siblings {
            if sibling.lifecycle.state == LifecycleState::Revising {
                if control.is_cancelled() {
                    return Ok((ItemOutcome::Interrupted, None));
                }
                match crate::revise::revise_once(
                    run_id,
                    shard,
                    item,
                    sibling,
                    &self.clients,
                    &self.area,
                    control,
                )
                .await
                {
                    Ok(retry) => {
                        // E2: if the revise could not run (budget-gated), the ORIGINAL is still at
                        // `Revising` and the retry was never generated/terminated. The item is NOT
                        // settled — do not commit past it, so a relaunch under fresh budget completes it.
                        if control.is_cancelled() && !is_terminal(retry.lifecycle.state) {
                            return Ok((ItemOutcome::Interrupted, None));
                        }
                        if retry.lifecycle.state == LifecycleState::Revising {
                            if control.is_cancelled() {
                                return Ok((ItemOutcome::Interrupted, None));
                            }
                            return Ok((ItemOutcome::PendingRevise, health));
                        }
                    }
                    // X1/X2: a RECORD-LEVEL fault in the bounded retry. `revise_once` attributes the fault
                    // to the RETRY id, so park the faulting RETRY at `Error` — never the `Revising`
                    // original (its single bounded retry is exhausted; it stays as its audit row). The
                    // item then SETTLES with the GROUP's real health: the `Revising` winner IS a genuine
                    // decision, so a failed retry must not falsely arm the circuit-breaker (and the fault
                    // is honestly recorded at `Error`, not silently swallowed by a mis-targeted c0 park).
                    Err(e) if e.is_record_level() => {
                        self.park_item_errored(run_id, shard, item, &e).await?;
                    }
                    // An INFRASTRUCTURE fault (storage/serde/systemic provider) aborts the run.
                    Err(e) => return Err(e),
                }
            }
        }
        Ok((ItemOutcome::Settled, health))
    }

    /// A JSON snapshot of the budget config pinned into the run row (the cap is the audit-relevant bit).
    fn budget_snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "cap_usd": self.clients.budget.cap(),
            "on_breach": self.on_breach,
            "max_in_flight": self.max_in_flight,
            "training_area": self.area.training_area,
            "teacher_slug": self.area.teacher_slug,
            "k": self.area.k,
        })
    }

    /// Sum the `cost.usd` already persisted across every record in `run_id` — the run's spend so far
    /// (E3). The store is authoritative: `gw-generate::assemble` stamps `cost.usd` on every record at
    /// generation, so this recovers the exact spend a prior launch incurred. Non-finite/negative costs
    /// are treated as 0 (mirroring the meter's charge clamp), so a garbled row never corrupts the total.
    async fn persisted_spend(&self, run_id: &str) -> Result<f64> {
        let records = self
            .clients
            .store
            .scan(&RecordFilter::new().run_id(run_id))
            .await?;
        let total = records
            .iter()
            .map(|r| r.cost.usd)
            .filter(|c| c.is_finite() && *c > 0.0)
            .sum();
        Ok(total)
    }

    /// Export this run's records to the configured Parquet shard and write the adjacent manifest
    /// sidecar.
    ///
    /// The shared Parquet exporter filters by judge verdict; end-of-run export narrows that input to
    /// lifecycle-admitted records first so shard rows agree with [`RunReport::admitted`], then restores
    /// manifest `n_records` to the whole scanned run population. The configured `dataset_version` is set
    /// only on the sidecar manifest, not on record rows.
    ///
    /// # Errors
    /// Returns an engine error if scanning records, writing the Parquet shard, serializing the manifest,
    /// or writing the manifest sidecar fails.
    pub async fn export_shard(&self, run_id: &str, spec: &ExportSpec) -> Result<ExportManifest> {
        let mut records = self
            .clients
            .store
            .scan(&RecordFilter::new().run_id(run_id))
            .await?;
        let n_records_total = records.len() as u64;
        records.retain(|record| {
            matches!(
                record.lifecycle.state,
                LifecycleState::Admitted | LifecycleState::Formatted | LifecycleState::Exported
            )
        });
        let mut manifest = export_parquet(&records, spec.target, spec.cot, &spec.dst).await?;
        manifest.n_records = n_records_total;
        manifest.dataset_version = spec.dataset_version.clone();
        let sidecar = manifest_sidecar_path(&spec.dst);
        let body = serde_json::to_vec_pretty(&manifest)?;
        tokio::fs::write(&sidecar, body)
            .await
            .map_err(gw_storage::StorageError::from)?;
        Ok(manifest)
    }

    /// Tally the run's records by terminal lifecycle state for the [`RunReport`].
    async fn tally(&self, run_id: &str) -> Result<RunReport> {
        let mut report = RunReport::default();
        let count = |state| {
            let store = &self.clients.store;
            async move {
                store
                    .scan(&RecordFilter::new().run_id(run_id).lifecycle_state(state))
                    .await
                    .map(|v| v.len())
            }
        };
        let exported = count(LifecycleState::Exported).await?;
        let admitted_only = count(LifecycleState::Admitted).await?;
        let formatted = count(LifecycleState::Formatted).await?;
        report.exported = exported;
        // "admitted" = anything that passed admission (Admitted/Formatted/Exported).
        report.admitted = exported + admitted_only + formatted;
        report.rejected = count(LifecycleState::Rejected).await?;
        report.needs_review = count(LifecycleState::NeedsReview).await?;
        // E2: a non-terminal `Revising` bucket — a bounded revise still in flight (e.g. budget-gated).
        report.revising = count(LifecycleState::Revising).await?;
        report.errored = count(LifecycleState::Error).await?;
        Ok(report)
    }
}

fn manifest_sidecar_path(dst: &Path) -> PathBuf {
    let mut path = dst.as_os_str().to_owned();
    path.push(".manifest.json");
    PathBuf::from(path)
}

/// `true` when a record at `state` is at a DECIDED outcome a fault must NEVER overwrite (NO-CLOBBER): a
/// non-`Error` terminal (`Admitted`/`Formatted`/`Exported`/`Rejected`/`NeedsReview`) or the in-flight
/// revise handoff (`Revising`). A record at any IN-FLIGHT pre-decision state
/// (`Seeded`/`UserSynthesized`/`AssistantGenerated`/`Verified`/`Judged`) is NOT decided — the faulting
/// record on its own id is still terminalized to `Error` from there (otherwise a mid-drive fault would
/// silently strand it). Also defines what counts as a genuine pipeline decision for [`group_health`].
fn is_decided(state: LifecycleState) -> bool {
    matches!(
        state,
        LifecycleState::Admitted
            | LifecycleState::Formatted
            | LifecycleState::Exported
            | LifecycleState::Rejected
            | LifecycleState::NeedsReview
            | LifecycleState::Revising
    )
}

/// `true` when [`Engine::park_item_errored`] may advance the record at `state` to `Error`. For an
/// ATTRIBUTED fault the record IS the one that faulted, so it is parked from any non-[`is_decided`]
/// (in-flight) state — a mid-drive fault must terminalize it, not strand it. For an UN-attributed
/// fallback (the rare escape path that targets `c0`) we stay CONSERVATIVE — only a pre-generation stub
/// state — so a mis-targeted fallback can never clobber a healthy in-flight sibling.
fn park_allowed(state: LifecycleState, attributed: bool) -> bool {
    if attributed {
        !is_decided(state)
    } else {
        matches!(
            state,
            LifecycleState::Seeded | LifecycleState::UserSynthesized | LifecycleState::Error
        )
    }
}

/// The circuit-breaker health of a processed best-of-k group (F2): `Decided` if ANY sibling reached a
/// genuine pipeline DECISION ([`is_decided`] — `Admitted`/`Formatted`/`Exported`/`Rejected`/`NeedsReview`
/// or the `Revising` handoff), which DISARMS the breaker; otherwise `AllErrored` (every sibling errored,
/// or — defensively — none reached a decision). A NON-decision forward state (e.g. a sibling stranded at
/// `Verified`) does NOT count as `Decided`, so a systemic judge outage cannot falsely disarm the breaker.
/// An EMPTY group — the budget gate tripped before any sibling generated (a Drain, not a fault) — is
/// `None`: NEUTRAL, it neither arms nor disarms the breaker.
fn group_health(siblings: &[gw_schema::TrainingRecord]) -> Option<ItemHealth> {
    if siblings.is_empty() {
        return None;
    }
    if siblings.iter().any(|s| is_decided(s.lifecycle.state)) {
        Some(ItemHealth::Decided)
    } else {
        Some(ItemHealth::AllErrored)
    }
}

/// A minimal `TrainingRecord` stub for a record-level fault that struck BEFORE generation persisted
/// anything (E5/F1): just enough envelope to satisfy the store's foreign key + projection so the
/// failure is queryable. Carries the candidate's user turn (the prompt is known) and lands at `Seeded`;
/// the caller then advances it to `Error`. Shared by the executor's `park_item_errored` (the k=1 /
/// escape path) and `crate::sibling`'s per-sibling isolation (the fan-out path).
pub(crate) fn error_stub(
    area: &AreaConfig,
    clients: &Clients,
    run_id: &str,
    rid: &str,
    candidate: &gw_generate::UserTurnCandidate,
) -> gw_schema::TrainingRecord {
    use gw_schema::{Provenance, TeacherRef, TrainingRecord};
    TrainingRecord {
        record_id: rid.to_string(),
        schema_version: semver::Version::new(1, 0, 0),
        dataset_version: None,
        training_area: area.training_area.clone(),
        tags: vec![],
        messages: vec![candidate.message.clone()],
        tools: None,
        provenance: Provenance {
            run_id: run_id.to_string(),
            parent_ids: vec![],
            teacher: TeacherRef {
                provider: "openrouter".to_string(),
                slug: area.teacher_slug.clone(),
                served_by: None,
                model_card_revision: None,
            },
            user_synth_model: None,
            user_turn_kind: None,
            in_scope_safe: None,
            judge_models: vec![],
            harness_version: clients.harness_version.clone(),
            git_commit: clients.git_commit.clone(),
        },
        generation: Default::default(),
        verification_contract: Some(candidate.contract.clone()),
        verification: Default::default(),
        judging: Default::default(),
        reasoning_quality: None,
        lifecycle: Default::default(),
        hashes: Default::default(),
        cost: Default::default(),
    }
}
