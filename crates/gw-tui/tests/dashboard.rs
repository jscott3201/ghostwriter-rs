//! Hermetic dashboard tests: feed engine [`EngineEvent`]s → [`Action`]s → [`App::update`], then render
//! the view into an in-memory [`TestBackend`] buffer and assert on cell content. NO real TTY.
//!
//! These cover the engine⟷tui seam end to end: a stream of lifecycle events lands the right states in
//! the table, the cost gauge reflects `run_total_usd`, a `RecordErrored` shows in the log pane, and the
//! budget banner renders on `BudgetReached`.

use gw_engine::EngineEvent;
use gw_schema::LifecycleState;
use gw_tui::{Action, App, view};
use ratatui::Terminal;
use ratatui::backend::TestBackend;

/// Render `app` into a fresh `w`×`h` `TestBackend` and return the buffer's text content as one string
/// (cell symbols concatenated row-by-row), for substring assertions.
fn render_to_string(app: &mut App, w: u16, h: u16) -> String {
    let backend = TestBackend::new(w, h);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    terminal
        .draw(|frame| view(frame, app))
        .expect("draw into test backend");
    let buffer = terminal.backend().buffer();
    let mut text = String::new();
    for y in 0..buffer.area.height {
        for x in 0..buffer.area.width {
            text.push_str(buffer[(x, y)].symbol());
        }
        text.push('\n');
    }
    text
}

/// Drive a model from a slice of engine events (the real seam: `from_engine_event` → `update`).
fn drive(events: Vec<EngineEvent>) -> App {
    let mut app = App::new();
    for event in events {
        app.update(Action::from_engine_event(event));
    }
    app
}

#[test]
fn header_shows_run_and_shard_progress() {
    let mut app = drive(vec![
        EngineEvent::RunStarted {
            run_id: "run-xyz".into(),
            shards: 3,
        },
        EngineEvent::ShardStarted {
            run_id: "run-xyz".into(),
            shard: 0,
            resumed: false,
        },
        EngineEvent::ShardStarted {
            run_id: "run-xyz".into(),
            shard: 1,
            resumed: true,
        },
        EngineEvent::ShardFinished {
            run_id: "run-xyz".into(),
            shard: 0,
        },
    ]);
    let out = render_to_string(&mut app, 100, 24);
    assert!(out.contains("run-xyz"), "run id in header:\n{out}");
    assert!(out.contains("2/3 started"), "shard started count:\n{out}");
    assert!(out.contains("1 finished"), "shard finished count:\n{out}");
    assert!(out.contains("resumed"), "resumed flag:\n{out}");
}

#[test]
fn table_shows_record_states() {
    let mut app = drive(vec![
        EngineEvent::StateAdvanced {
            record_id: "rec-aaa".into(),
            to: LifecycleState::Admitted,
        },
        EngineEvent::StateAdvanced {
            record_id: "rec-bbb".into(),
            to: LifecycleState::Rejected,
        },
    ]);
    let out = render_to_string(&mut app, 100, 24);
    assert!(out.contains("rec-aaa"), "record id rendered:\n{out}");
    assert!(out.contains("admitted"), "admitted label:\n{out}");
    assert!(out.contains("rejected"), "rejected label:\n{out}");
}

#[test]
fn cost_gauge_reflects_run_total() {
    let mut app = drive(vec![
        EngineEvent::CostCharged {
            record_id: "rec-aaa".into(),
            usd: 2.5,
            run_total_usd: 2.5,
        },
        EngineEvent::CostCharged {
            record_id: "rec-bbb".into(),
            usd: 1.0,
            run_total_usd: 3.5,
        },
    ]);
    let out = render_to_string(&mut app, 100, 24);
    assert!(out.contains("$3.50"), "cost gauge shows total:\n{out}");
}

#[test]
fn budget_banner_and_log_on_budget_reached() {
    let mut app = drive(vec![EngineEvent::BudgetReached {
        run_id: "run-1".into(),
        spent: 9.99,
        cap: 10.0,
    }]);
    let out = render_to_string(&mut app, 100, 24);
    assert!(
        out.contains("BUDGET REACHED"),
        "budget banner in log:\n{out}"
    );
    // With a cap known, the gauge renders the spend/cap pair.
    assert!(out.contains("/ $10.00"), "gauge cap label:\n{out}");
}

#[test]
fn errored_record_appears_in_log_pane() {
    let mut app = drive(vec![EngineEvent::RecordErrored {
        record_id: "rec-zzz".into(),
        error: "teacher http 500".into(),
    }]);
    let out = render_to_string(&mut app, 120, 24);
    assert!(out.contains("rec-zzz"), "errored record id in log:\n{out}");
    assert!(out.contains("ERROR"), "error label in log:\n{out}");
    // The error count also surfaces in the counters pane.
    assert_eq!(app.count_in(LifecycleState::Error), 1);
}

#[test]
fn run_finished_completed_banner() {
    let mut app = drive(vec![EngineEvent::RunFinished {
        run_id: "run-1".into(),
        completed: true,
    }]);
    let out = render_to_string(&mut app, 100, 24);
    assert!(out.contains("RUN COMPLETE"), "completed banner:\n{out}");
}

#[test]
fn run_finished_halted_banner() {
    let mut app = drive(vec![EngineEvent::RunFinished {
        run_id: "run-1".into(),
        completed: false,
    }]);
    let out = render_to_string(&mut app, 100, 24);
    assert!(out.contains("RUN HALTED"), "halted banner:\n{out}");
}

#[test]
fn counters_pane_reflects_terminal_states() {
    let mut app = drive(vec![
        EngineEvent::StateAdvanced {
            record_id: "a".into(),
            to: LifecycleState::Exported,
        },
        EngineEvent::StateAdvanced {
            record_id: "b".into(),
            to: LifecycleState::Exported,
        },
        EngineEvent::StateAdvanced {
            record_id: "c".into(),
            to: LifecycleState::Rejected,
        },
        EngineEvent::StateAdvanced {
            record_id: "d".into(),
            to: LifecycleState::NeedsReview,
        },
    ]);
    let out = render_to_string(&mut app, 120, 24);
    assert!(out.contains("Exported"), "counters label present:\n{out}");
    assert!(out.contains("admit-rate"), "admit-rate label:\n{out}");
    // 2 exported (admitted-family) vs 1 rejected => admit-rate 67%.
    assert!(out.contains("67%"), "admit-rate value:\n{out}");
}

#[test]
fn empty_model_renders_waiting_header_without_panic() {
    let mut app = App::new();
    let out = render_to_string(&mut app, 80, 20);
    assert!(out.contains("waiting for run"), "idle header:\n{out}");
}

#[test]
fn renders_in_a_small_terminal_without_panic() {
    // Guard against layout overflow panics in a cramped terminal.
    let mut app = drive(vec![
        EngineEvent::RunStarted {
            run_id: "r".into(),
            shards: 1,
        },
        EngineEvent::StateAdvanced {
            record_id: "rec".into(),
            to: LifecycleState::Judged,
        },
    ]);
    let _ = render_to_string(&mut app, 20, 8);
}

#[test]
fn selection_highlight_survives_new_rows() {
    // Selection state lives in the model, so it persists across renders and new-row arrivals.
    let mut app = drive(vec![
        EngineEvent::StateAdvanced {
            record_id: "a".into(),
            to: LifecycleState::Seeded,
        },
        EngineEvent::StateAdvanced {
            record_id: "b".into(),
            to: LifecycleState::Seeded,
        },
    ]);
    app.update(Action::SelectDown);
    assert_eq!(app.table_state.selected(), Some(1));
    // A render does not reset selection.
    let _ = render_to_string(&mut app, 100, 24);
    assert_eq!(app.table_state.selected(), Some(1));
    // A new row arriving keeps the selection in-bounds.
    app.update(Action::StateAdvanced {
        record_id: "c".into(),
        to: LifecycleState::Seeded,
    });
    let _ = render_to_string(&mut app, 100, 24);
    assert_eq!(app.table_state.selected(), Some(1));
}
