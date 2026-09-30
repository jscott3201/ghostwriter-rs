//! The view: a PURE function of the [`App`] model into a [`Frame`] (ARCHITECTURE §1.1, §1.3).
//!
//! ratatui is immediate-mode: every frame the whole UI is re-described from the model. [`view`] reads
//! `&App` (and `&mut` only for the `TableState` the stateful `Table` widget needs) and renders — it
//! NEVER mutates business state. The event loop ([`crate::event_loop`]) is the sole owner of the
//! `Frame`; it calls `terminal.draw(|f| view(f, &mut app))`.
//!
//! Layout (top → bottom): a run/shard HEADER, then a body split horizontally into a per-record
//! lifecycle TABLE (left) and COUNTERS + accounting gauge (right), then full-width accounting
//! evidence, an errors/log pane, and a one-line key-hints footer.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Cell, Gauge, List, ListItem, Paragraph, Row, Table, Wrap};

use gw_schema::LifecycleState;

use crate::model::App;

/// Render the whole dashboard from the model. `app` is `&mut` only because the stateful `Table` needs
/// `&mut TableState`; no business state is mutated here.
pub fn view(frame: &mut Frame, app: &mut App) {
    // Critical accounting evidence has reserved space at ordinary terminal sizes. Optional token
    // detail and older log lines expand only when the terminal has enough height.
    let spacious = frame.area().height >= 32;
    let evidence_height = if spacious { 11 } else { 6 };
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // header
            Constraint::Min(6),    // body (table | metrics)
            Constraint::Length(evidence_height),
            Constraint::Length(if spacious { 8 } else { 3 }), // error/log pane
            Constraint::Length(1),                            // footer hints
        ])
        .split(frame.area());

    render_header(frame, app, chunks[0]);

    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(chunks[1]);
    render_table(frame, app, body[0]);
    render_metrics(frame, app, body[1]);

    render_accounting_evidence(frame, app, chunks[2]);
    render_log(frame, app, chunks[3]);
    render_footer(frame, chunks[4]);
}

/// The run/shard progress header, plus a completed/halted banner once the run finishes.
fn render_header(frame: &mut Frame, app: &mut App, area: Rect) {
    let h = &app.header;
    let run = if h.run_id.is_empty() {
        "(waiting for run)"
    } else {
        h.run_id.as_str()
    };
    let mut spans = vec![
        Span::styled("run ", Style::default().fg(Color::DarkGray)),
        Span::styled(run, Style::default().add_modifier(Modifier::BOLD)),
        Span::raw("  "),
        Span::styled(
            format!(
                "shards {}/{} started, {} finished",
                h.shards_started, h.shards_planned, h.shards_finished
            ),
            Style::default().fg(Color::Cyan),
        ),
    ];
    if h.shards_resumed > 0 {
        spans.push(Span::raw("  "));
        spans.push(Span::styled(
            format!("{} resumed", h.shards_resumed),
            Style::default().fg(Color::Yellow),
        ));
    }
    if let Some(completed) = h.finished {
        spans.push(Span::raw("  "));
        let (label, color) = if completed {
            ("RUN COMPLETE", Color::Green)
        } else {
            ("RUN HALTED", Color::Red)
        };
        spans.push(Span::styled(
            label,
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ));
    }
    let para = Paragraph::new(Line::from(spans))
        .block(Block::default().borders(Borders::ALL).title("ghostwriter"));
    frame.render_widget(para, area);
}

/// The per-record lifecycle table: `record_id` → current state → update count. Selection/scroll come
/// from the model's `TableState`.
fn render_table(frame: &mut Frame, app: &mut App, area: Rect) {
    let rows: Vec<Row> = app
        .records
        .iter()
        .map(|(id, row)| {
            let (label, color) = state_style(row.state);
            Row::new(vec![
                Cell::from(truncate(id, 28)),
                Cell::from(Span::styled(label, Style::default().fg(color))),
                Cell::from(row.updates.to_string()),
            ])
        })
        .collect();

    let title = format!("records ({})", app.record_count());
    let table = Table::new(
        rows,
        [
            Constraint::Min(12),
            Constraint::Length(20),
            Constraint::Length(5),
        ],
    )
    .header(
        Row::new(vec!["record_id", "state", "upd"])
            .style(Style::default().add_modifier(Modifier::BOLD)),
    )
    .block(Block::default().borders(Borders::ALL).title(title))
    .row_highlight_style(Style::default().add_modifier(Modifier::REVERSED))
    .highlight_symbol("> ");
    frame.render_stateful_widget(table, area, &mut app.table_state);
}

/// The metrics column: terminal-state counters, admit-rate, and known-cost gauge.
fn render_metrics(frame: &mut Frame, app: &mut App, area: Rect) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(4),    // counters list
            Constraint::Length(3), // cost gauge
        ])
        .split(area);

    render_counters(frame, app, rows[0]);
    render_cost_gauge(frame, app, rows[1]);
}

/// Per-state counters (two per line so they fit a short terminal) + an admit-rate line.
fn render_counters(frame: &mut Frame, app: &App, area: Rect) {
    let admit_rate = app
        .admit_rate()
        .map_or_else(|| "n/a".to_string(), |r| format!("{:.0}%", r * 100.0));
    // Pack the six terminal-state counters two-per-line so all are visible even in a cramped column.
    let counters = [
        ("Exported", LifecycleState::Exported, Color::Green),
        ("Admitted", LifecycleState::Admitted, Color::Green),
        ("Rejected", LifecycleState::Rejected, Color::Red),
        ("NeedsRev", LifecycleState::NeedsReview, Color::Yellow),
        ("Revising", LifecycleState::Revising, Color::Magenta),
        ("Error", LifecycleState::Error, Color::Red),
    ];
    let cell = |label: &str, state: LifecycleState, color: Color| {
        vec![
            Span::styled(format!("{label:<9}"), Style::default().fg(color)),
            Span::styled(
                format!("{:<5}", app.count_in(state)),
                Style::default().add_modifier(Modifier::BOLD),
            ),
        ]
    };
    let mut items: Vec<ListItem> = Vec::new();
    for pair in counters.chunks(2) {
        let mut spans = cell(pair[0].0, pair[0].1, pair[0].2);
        if let Some(second) = pair.get(1) {
            spans.extend(cell(second.0, second.1, second.2));
        }
        items.push(ListItem::new(Line::from(spans)));
    }
    items.push(ListItem::new(Line::from(vec![
        Span::styled("admit-rate ", Style::default().fg(Color::Cyan)),
        Span::styled(admit_rate, Style::default().add_modifier(Modifier::BOLD)),
    ])));
    let list = List::new(items).block(Block::default().borders(Borders::ALL).title("counters"));
    frame.render_widget(list, area);
}

/// A monetary gauge exists only for an effective finite policy with known spend.
fn render_cost_gauge(frame: &mut Frame, app: &App, area: Rect) {
    let Some(snapshot) = &app.accounting else {
        frame.render_widget(Paragraph::new("Accounting: loading / unknown"), area);
        return;
    };
    if let (
        Some(gw_schema::PolicyState {
            policy: gw_schema::AccountingPolicy::FiniteUsd { limit_usd },
            ..
        }),
        Some(known),
    ) = (&snapshot.effective, snapshot.known_usd)
    {
        let ratio = if *limit_usd == 0.0 {
            1.0
        } else {
            (known / limit_usd).clamp(0.0, 1.0)
        };
        frame.render_widget(
            Gauge::default()
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("known USD dispatch threshold"),
                )
                .gauge_style(Style::default().fg(if ratio >= 1.0 {
                    Color::Yellow
                } else {
                    Color::Green
                }))
                .ratio(ratio)
                .label(format!("${known:.4} / ${limit_usd:.4}")),
            area,
        );
    } else {
        let known = snapshot
            .known_usd
            .map_or_else(|| "overflow".into(), |cost| format!("${cost:.4}"));
        let mode = match snapshot.effective.as_ref().map(|state| &state.policy) {
            Some(gw_schema::AccountingPolicy::ObservationOnly) => "Observation only",
            Some(gw_schema::AccountingPolicy::FiniteUsd { .. }) => "Finite USD",
            None => "Policy unknown",
        };
        frame.render_widget(
            Paragraph::new(format!("{mode}: known {known}"))
                .block(Block::default().borders(Borders::ALL).title("accounting")),
            area,
        );
    }
}

/// Critical uncertainty and policy authority precede optional measured tokens and client wall time.
fn render_accounting_evidence(frame: &mut Frame, app: &App, area: Rect) {
    let text = app.accounting.as_ref().map_or_else(|| "Waiting for durable evidence".into(), |s| {
        let history = match s.history {
            gw_schema::AccountingHistory::RecordedFromCreation => "from creation",
            gw_schema::AccountingHistory::Unknown => "unknown",
        };
        format!("unknown cost {}; invalid cost {}; conflicts {}; unresolved {}\nhistory {}; unknown lanes {}\nconfigured {}\neffective {}\ninput {}\noutput {}\ntotal {}\nreasoning {}\nwall ms {}; attempts {}",
            s.unknown_cost_attempts, s.invalid_cost_attempts, s.conflicting_attempts, s.unresolved_attempts,
            history, s.unknown_coverage_lanes, policy_label(s.configured.as_ref()), policy_label(s.effective.as_ref()),
            token_label(&s.prompt_tokens), token_label(&s.completion_tokens), token_label(&s.total_tokens), token_label(&s.reasoning_tokens),
            s.elapsed_ms.map_or_else(|| "overflow".into(), |v| v.to_string()), s.attempts)
    });
    frame.render_widget(
        Paragraph::new(text).wrap(Wrap { trim: true }).block(
            Block::default()
                .borders(Borders::ALL)
                .title(if area.height > 6 {
                    "accounting evidence (token categories overlap)"
                } else {
                    "accounting evidence"
                }),
        ),
        area,
    );
}
fn token_label(value: &gw_schema::TokenEvidence) -> String {
    format!(
        "{}; missing {} / invalid {}",
        value
            .known
            .map_or_else(|| "overflow".into(), |v| v.to_string()),
        value.missing_attempts,
        value.invalid_attempts
    )
}
fn policy_label(value: Option<&gw_schema::PolicyState>) -> String {
    let Some(value) = value else {
        return "unknown".into();
    };
    match value.policy {
        gw_schema::AccountingPolicy::ObservationOnly => {
            format!("observation only, epoch {}", value.epoch)
        }
        gw_schema::AccountingPolicy::FiniteUsd { limit_usd } => {
            format!("finite ${limit_usd:.4}, epoch {}", value.epoch)
        }
    }
}

/// The error log pane: most-recent lines, newest at the bottom.
fn render_log(frame: &mut Frame, app: &App, area: Rect) {
    let inner_height = area.height.saturating_sub(2) as usize; // minus the border rows
    let start = app.error_log.len().saturating_sub(inner_height.max(1));
    let text: Vec<Line> = app.error_log[start..]
        .iter()
        .map(|l| {
            let style = if l.starts_with("BUDGET") {
                Style::default().fg(Color::Yellow).bold()
            } else {
                Style::default().fg(Color::Red)
            };
            Line::from(Span::styled(l.clone(), style))
        })
        .collect();
    let para = Paragraph::new(text)
        .block(Block::default().borders(Borders::ALL).title("errors"))
        .wrap(Wrap { trim: true });
    frame.render_widget(para, area);
}

/// The one-line key hints footer.
fn render_footer(frame: &mut Frame, area: Rect) {
    let hints = Line::from(vec![
        Span::styled(" ↑/↓ ", Style::default().fg(Color::Black).bg(Color::Cyan)),
        Span::raw(" select  "),
        Span::styled(" g/G ", Style::default().fg(Color::Black).bg(Color::Cyan)),
        Span::raw(" first/last  "),
        Span::styled(" q ", Style::default().fg(Color::Black).bg(Color::Cyan)),
        Span::raw(" quit "),
    ]);
    frame.render_widget(Paragraph::new(hints), area);
}

/// The display label + color for a lifecycle state.
fn state_style(state: LifecycleState) -> (&'static str, Color) {
    match state {
        LifecycleState::Seeded => ("seeded", Color::DarkGray),
        LifecycleState::UserSynthesized => ("user_synth", Color::Gray),
        LifecycleState::AssistantGenerated => ("generated", Color::Blue),
        LifecycleState::Verified => ("verified", Color::Blue),
        LifecycleState::Judged => ("judged", Color::Cyan),
        LifecycleState::Revising => ("revising", Color::Magenta),
        LifecycleState::NeedsReview => ("needs_review", Color::Yellow),
        LifecycleState::Admitted => ("admitted", Color::Green),
        LifecycleState::Rejected => ("rejected", Color::Red),
        LifecycleState::Formatted => ("formatted", Color::Green),
        LifecycleState::Exported => ("exported", Color::Green),
        LifecycleState::Error => ("error", Color::Red),
    }
}

/// Truncate `s` to at most `max` chars, appending `…` when clipped (keeps table columns aligned).
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let kept: String = s.chars().take(max.saturating_sub(1)).collect();
    format!("{kept}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_short_is_unchanged() {
        assert_eq!(truncate("abc", 10), "abc");
    }

    #[test]
    fn truncate_long_appends_ellipsis() {
        let out = truncate("abcdefghij", 5);
        assert_eq!(out.chars().count(), 5);
        assert!(out.ends_with('…'));
    }

    #[test]
    fn every_state_has_a_style() {
        // Touch each variant so a future state addition fails compilation here, not at runtime.
        for state in [
            LifecycleState::Seeded,
            LifecycleState::UserSynthesized,
            LifecycleState::AssistantGenerated,
            LifecycleState::Verified,
            LifecycleState::Judged,
            LifecycleState::Revising,
            LifecycleState::NeedsReview,
            LifecycleState::Admitted,
            LifecycleState::Rejected,
            LifecycleState::Formatted,
            LifecycleState::Exported,
            LifecycleState::Error,
        ] {
            let (label, _) = state_style(state);
            assert!(!label.is_empty());
        }
    }
}
