//! The application model + the PURE update step (ARCHITECTURE §1.4, §2.2).
//!
//! [`App`] is the entire UI state; [`App::update`] is the ONLY place state mutates. `update` takes an
//! [`Action`] and a `&mut App`, applies the transition, and optionally returns a follow-up `Action` —
//! exactly the Component+Action `update` contract. It touches NO terminal and performs NO I/O, so the
//! whole dashboard is unit-testable by feeding `Action`s and asserting on the model (see the tests at
//! the bottom and the `TestBackend` view tests in `tests/`).
//!
//! Because the engine event channel is BOUNDED + drop-on-full (events are observability; the SQLite
//! ledger is truth), the model is written to TOLERATE GAPS: counters are derived from the per-record
//! state map (an idempotent last-writer-wins map keyed by `record_id`), never from running increments,
//! so a dropped `StateAdvanced` cannot corrupt a count — at worst a row lags until its next event.

use std::collections::BTreeMap;

use ratatui::widgets::TableState;

use gw_schema::LifecycleState;

use crate::action::Action;

/// How many recent error/budget log lines to retain in the log pane (older lines are dropped).
pub const ERROR_LOG_CAPACITY: usize = 200;

/// How many cost samples to retain for the cost sparkline (one per [`Action::CostCharged`]).
pub const COST_HISTORY_CAPACITY: usize = 256;

/// The run-level header state: ids and shard progress, populated from run/shard boundary actions.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RunHeader {
    /// The active run id (empty until [`Action::RunStarted`]).
    pub run_id: String,
    /// The number of shards planned for the run.
    pub shards_planned: usize,
    /// How many shards have started.
    pub shards_started: usize,
    /// How many shards have finished.
    pub shards_finished: usize,
    /// How many of the started shards resumed from a checkpoint.
    pub shards_resumed: usize,
    /// `Some(completed)` once the run finishes: `true` drained cleanly, `false` halted early.
    pub finished: Option<bool>,
}

/// One record's row in the lifecycle table: its current state and how many transitions it has seen
/// (a coarse "last updated" proxy that needs no clock — the model is clock-free for hermetic tests).
#[derive(Debug, Clone, PartialEq)]
pub struct RecordRow {
    /// The current lifecycle state.
    pub state: LifecycleState,
    /// Monotone update counter — incremented on every [`Action::StateAdvanced`] for this record.
    pub updates: u64,
}

/// The cost meter: cumulative spend, the cap (once known), and whether the cap was reached.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CostMeter {
    /// The cumulative USD spent for the run (last `run_total_usd` seen).
    pub spent_usd: f64,
    /// The configured cap, known only once a [`Action::BudgetReached`] arrives (the engine reports the
    /// cap on that event; `None` until then).
    pub cap_usd: Option<f64>,
    /// `true` once the budget cap has been reached.
    pub budget_reached: bool,
}

impl CostMeter {
    /// The spend fraction in `[0.0, 1.0]` for the cost gauge, or `None` if no cap is known yet. Clamped
    /// so a late/overshooting `spent` never produces a ratio outside the gauge's domain.
    #[must_use]
    pub fn fraction(&self) -> Option<f64> {
        let cap = self.cap_usd?;
        if cap <= 0.0 {
            return Some(1.0);
        }
        Some((self.spent_usd / cap).clamp(0.0, 1.0))
    }
}

/// The complete application model. The view is a pure function of this; `update` is the sole mutator.
#[derive(Debug, Default)]
pub struct App {
    /// Run/shard header progress.
    pub header: RunHeader,
    /// Per-record current state, keyed + ORDERED by `record_id` (a `BTreeMap` gives a stable, sorted
    /// table that does not reshuffle as events arrive — selection stays meaningful).
    pub records: BTreeMap<String, RecordRow>,
    /// Selection/scroll state for the lifecycle `Table` — stored in the MODEL (not the render fn) so it
    /// survives across frames (ARCHITECTURE §1.2).
    pub table_state: TableState,
    /// The cost meter (gauge + budget banner source).
    pub cost: CostMeter,
    /// Recent error/budget log lines, newest last; capped at `ERROR_LOG_CAPACITY`.
    pub error_log: Vec<String>,
    /// Cumulative-spend samples for the cost sparkline, capped at `COST_HISTORY_CAPACITY`.
    pub cost_history: Vec<u64>,
    /// `true` once a quit was requested; the event loop checks this to break.
    pub should_quit: bool,
    /// Tick counter (drives any animated UI; incremented on every [`Action::Tick`]).
    pub ticks: u64,
}

impl App {
    /// Construct an empty model.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Count of records currently in `state`. Derived (never a running increment) so dropped events
    /// cannot corrupt it.
    #[must_use]
    pub fn count_in(&self, state: LifecycleState) -> usize {
        self.records.values().filter(|r| r.state == state).count()
    }

    /// Records that passed admission (`Admitted`/`Formatted`/`Exported`), matching the engine's
    /// `RunReport::admitted` definition.
    #[must_use]
    pub fn admitted_total(&self) -> usize {
        self.count_in(LifecycleState::Admitted)
            + self.count_in(LifecycleState::Formatted)
            + self.count_in(LifecycleState::Exported)
    }

    /// The admit rate over records that reached a DECISIVE outcome (admitted-family or `Rejected`),
    /// in `[0.0, 1.0]`; `None` until at least one such decision exists. `NeedsReview`/`Error` and
    /// in-flight states are excluded from the denominator (they are not an admit/reject decision).
    #[must_use]
    pub fn admit_rate(&self) -> Option<f64> {
        let admitted = self.admitted_total();
        let rejected = self.count_in(LifecycleState::Rejected);
        let decided = admitted + rejected;
        if decided == 0 {
            return None;
        }
        Some(admitted as f64 / decided as f64)
    }

    /// The total number of records the dashboard is tracking.
    #[must_use]
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// Apply one [`Action`], mutating the model and optionally emitting a follow-up `Action`. PURE: no
    /// I/O, no terminal. This is the entire business logic of the dashboard.
    pub fn update(&mut self, action: Action) -> Option<Action> {
        match action {
            Action::RunStarted { run_id, shards } => {
                self.header.run_id = run_id;
                self.header.shards_planned = shards;
                self.header.finished = None;
            }
            Action::ShardStarted { resumed, .. } => {
                self.header.shards_started += 1;
                if resumed {
                    self.header.shards_resumed += 1;
                }
            }
            Action::StateAdvanced { record_id, to } => {
                let row = self.records.entry(record_id).or_insert(RecordRow {
                    state: to,
                    updates: 0,
                });
                row.state = to;
                row.updates += 1;
                self.clamp_selection();
            }
            Action::CostCharged { run_total_usd } => {
                // Monotone guard: the cumulative total never decreases, so ignore a stale/out-of-order
                // lower value (a dropped-then-late event must not rewind the gauge).
                if run_total_usd > self.cost.spent_usd {
                    self.cost.spent_usd = run_total_usd;
                }
                self.push_cost_sample();
            }
            Action::BudgetReached { spent, cap } => {
                self.cost.budget_reached = true;
                self.cost.cap_usd = Some(cap);
                if spent > self.cost.spent_usd {
                    self.cost.spent_usd = spent;
                }
                self.push_log(format!(
                    "BUDGET REACHED — spent ${spent:.2} of ${cap:.2} cap"
                ));
            }
            Action::RecordErrored { record_id, error } => {
                self.push_log(format!("ERROR {record_id}: {error}"));
                // The engine also emits a StateAdvanced(Error); if it was dropped, reflect Error here so
                // the table and counters stay consistent with the log.
                let row = self.records.entry(record_id).or_insert(RecordRow {
                    state: LifecycleState::Error,
                    updates: 0,
                });
                row.state = LifecycleState::Error;
                row.updates += 1;
                self.clamp_selection();
            }
            Action::ShardFinished { .. } => {
                self.header.shards_finished += 1;
            }
            Action::RunFinished { completed } => {
                self.header.finished = Some(completed);
            }
            Action::SelectUp => self.move_selection(-1),
            Action::SelectDown => self.move_selection(1),
            Action::SelectFirst => self.select_index(0),
            Action::SelectLast => {
                let last = self.record_count().saturating_sub(1);
                self.select_index(last);
            }
            Action::Tick => self.ticks = self.ticks.wrapping_add(1),
            Action::Resize { .. } => {}
            Action::Quit => self.should_quit = true,
            // DEFERRED (v2 per-token trace viewer): the engine emits no per-token deltas in v1, so these
            // are never produced. Accepted as no-ops so the seam compiles without a v1 behavior.
            Action::ReasoningDelta { .. } | Action::AnswerDelta { .. } => {}
        }
        None
    }

    /// Append a log line, evicting the oldest if at capacity.
    fn push_log(&mut self, line: String) {
        if self.error_log.len() >= ERROR_LOG_CAPACITY {
            self.error_log.remove(0);
        }
        self.error_log.push(line);
    }

    /// Push the current cumulative spend (as whole cents) onto the cost sparkline history.
    fn push_cost_sample(&mut self) {
        if self.cost_history.len() >= COST_HISTORY_CAPACITY {
            self.cost_history.remove(0);
        }
        // Cents keep the sparkline integer-valued and monotone without precision drama.
        let cents = (self.cost.spent_usd * 100.0).round().max(0.0) as u64;
        self.cost_history.push(cents);
    }

    /// Move the selection by `delta` rows (saturating at the ends); a no-op on an empty table.
    fn move_selection(&mut self, delta: i64) {
        let len = self.record_count();
        if len == 0 {
            self.table_state.select(None);
            return;
        }
        let current = self.table_state.selected().unwrap_or(0) as i64;
        let next = (current + delta).clamp(0, len as i64 - 1) as usize;
        self.table_state.select(Some(next));
    }

    /// Select an explicit index, clamped to the table bounds; clears selection on an empty table.
    fn select_index(&mut self, index: usize) {
        let len = self.record_count();
        if len == 0 {
            self.table_state.select(None);
        } else {
            self.table_state.select(Some(index.min(len - 1)));
        }
    }

    /// Keep the selection in-bounds after the row set changes. Selects row 0 on the FIRST row inserted
    /// (so the table starts with a visible highlight), and clamps a now-out-of-range selection.
    fn clamp_selection(&mut self) {
        let len = self.record_count();
        match self.table_state.selected() {
            None if len > 0 => self.table_state.select(Some(0)),
            Some(i) if i >= len && len > 0 => self.table_state.select(Some(len - 1)),
            Some(_) if len == 0 => self.table_state.select(None),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn advance(app: &mut App, id: &str, to: LifecycleState) {
        app.update(Action::StateAdvanced {
            record_id: id.to_string(),
            to,
        });
    }

    #[test]
    fn run_started_sets_header() {
        let mut app = App::new();
        app.update(Action::RunStarted {
            run_id: "run-1".into(),
            shards: 4,
        });
        assert_eq!(app.header.run_id, "run-1");
        assert_eq!(app.header.shards_planned, 4);
        assert_eq!(app.header.finished, None);
    }

    #[test]
    fn shard_started_counts_resumes() {
        let mut app = App::new();
        app.update(Action::ShardStarted {
            shard: 0,
            resumed: false,
        });
        app.update(Action::ShardStarted {
            shard: 1,
            resumed: true,
        });
        assert_eq!(app.header.shards_started, 2);
        assert_eq!(app.header.shards_resumed, 1);
    }

    #[test]
    fn state_advance_is_last_writer_wins_and_counts_derive() {
        let mut app = App::new();
        advance(&mut app, "a", LifecycleState::Seeded);
        advance(&mut app, "a", LifecycleState::AssistantGenerated);
        advance(&mut app, "a", LifecycleState::Admitted);
        advance(&mut app, "b", LifecycleState::Rejected);
        // Only the LAST state of "a" counts — derived, not incremented.
        assert_eq!(app.count_in(LifecycleState::Admitted), 1);
        assert_eq!(app.count_in(LifecycleState::Seeded), 0);
        assert_eq!(app.count_in(LifecycleState::Rejected), 1);
        assert_eq!(app.record_count(), 2);
        assert_eq!(app.records["a"].updates, 3);
    }

    #[test]
    fn dropped_intermediate_event_does_not_corrupt_count() {
        // Simulate a dropped `Seeded`/`Verified`: we only ever see the final `Exported`. The derived
        // count is still correct because counts read the map, not a running tally.
        let mut app = App::new();
        advance(&mut app, "x", LifecycleState::Exported);
        assert_eq!(app.count_in(LifecycleState::Exported), 1);
        assert_eq!(app.admitted_total(), 1);
    }

    #[test]
    fn admitted_total_spans_admit_family() {
        let mut app = App::new();
        advance(&mut app, "a", LifecycleState::Admitted);
        advance(&mut app, "f", LifecycleState::Formatted);
        advance(&mut app, "e", LifecycleState::Exported);
        advance(&mut app, "r", LifecycleState::Rejected);
        assert_eq!(app.admitted_total(), 3);
    }

    #[test]
    fn admit_rate_excludes_nondecisions() {
        let mut app = App::new();
        assert_eq!(app.admit_rate(), None);
        advance(&mut app, "a", LifecycleState::Exported);
        advance(&mut app, "r", LifecycleState::Rejected);
        advance(&mut app, "n", LifecycleState::NeedsReview);
        advance(&mut app, "e", LifecycleState::Error);
        // Decided = 1 admitted + 1 rejected; NeedsReview/Error excluded.
        assert_eq!(app.admit_rate(), Some(0.5));
    }

    #[test]
    fn cost_charged_is_monotone() {
        let mut app = App::new();
        app.update(Action::CostCharged { run_total_usd: 5.0 });
        app.update(Action::CostCharged { run_total_usd: 3.0 }); // stale/out-of-order — ignored
        app.update(Action::CostCharged { run_total_usd: 7.5 });
        assert!((app.cost.spent_usd - 7.5).abs() < f64::EPSILON);
        assert_eq!(app.cost_history.len(), 3);
        assert_eq!(*app.cost_history.last().unwrap(), 750);
    }

    #[test]
    fn budget_reached_sets_banner_and_logs() {
        let mut app = App::new();
        app.update(Action::BudgetReached {
            spent: 10.0,
            cap: 10.0,
        });
        assert!(app.cost.budget_reached);
        assert_eq!(app.cost.cap_usd, Some(10.0));
        assert_eq!(app.cost.fraction(), Some(1.0));
        assert!(app.error_log.last().unwrap().contains("BUDGET REACHED"));
    }

    #[test]
    fn record_errored_logs_and_reflects_error_state() {
        let mut app = App::new();
        app.update(Action::RecordErrored {
            record_id: "rec-9".into(),
            error: "teacher 500".into(),
        });
        assert_eq!(app.count_in(LifecycleState::Error), 1);
        assert!(app.error_log.last().unwrap().contains("rec-9"));
        assert!(app.error_log.last().unwrap().contains("teacher 500"));
    }

    #[test]
    fn error_log_is_capped() {
        let mut app = App::new();
        for i in 0..(ERROR_LOG_CAPACITY + 50) {
            app.update(Action::RecordErrored {
                record_id: format!("r{i}"),
                error: "e".into(),
            });
        }
        assert_eq!(app.error_log.len(), ERROR_LOG_CAPACITY);
    }

    #[test]
    fn run_finished_records_completion() {
        let mut app = App::new();
        app.update(Action::RunFinished { completed: false });
        assert_eq!(app.header.finished, Some(false));
    }

    #[test]
    fn selection_moves_and_clamps_within_bounds() {
        let mut app = App::new();
        // Empty table: movement is a no-op, selection stays None.
        app.update(Action::SelectDown);
        assert_eq!(app.table_state.selected(), None);

        advance(&mut app, "a", LifecycleState::Seeded);
        advance(&mut app, "b", LifecycleState::Seeded);
        advance(&mut app, "c", LifecycleState::Seeded);
        // First insert auto-selects row 0.
        assert_eq!(app.table_state.selected(), Some(0));

        app.update(Action::SelectDown);
        app.update(Action::SelectDown);
        assert_eq!(app.table_state.selected(), Some(2));
        // Saturates at the bottom.
        app.update(Action::SelectDown);
        assert_eq!(app.table_state.selected(), Some(2));

        app.update(Action::SelectFirst);
        assert_eq!(app.table_state.selected(), Some(0));
        // Saturates at the top.
        app.update(Action::SelectUp);
        assert_eq!(app.table_state.selected(), Some(0));

        app.update(Action::SelectLast);
        assert_eq!(app.table_state.selected(), Some(2));
    }

    #[test]
    fn quit_sets_flag() {
        let mut app = App::new();
        assert!(!app.should_quit);
        app.update(Action::Quit);
        assert!(app.should_quit);
    }

    #[test]
    fn tick_increments() {
        let mut app = App::new();
        app.update(Action::Tick);
        app.update(Action::Tick);
        assert_eq!(app.ticks, 2);
    }
}
