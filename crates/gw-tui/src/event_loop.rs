//! The async I/O shell: the single `tokio::select!` loop, terminal lifecycle, and the panic hook
//! (ARCHITECTURE §2.1–2.4, §6 core-loop sketch).
//!
//! This is the THIN, untested terminal-I/O layer. All business logic lives in the pure
//! [`App::update`](crate::App::update) / [`view()`](crate::view::view) layer; this module only:
//!
//! 1. enters raw mode + the alternate screen (restoring them on exit AND via a panic hook);
//! 2. multiplexes — in ONE `tokio::select!` — the crossterm [`EventStream`], a tick interval, a render
//!    interval, the engine [`EngineEvent`] receiver, and a [`CancellationToken`];
//! 3. maps each source onto an [`Action`], applies it via the pure model, and (only on the render arm)
//!    draws the frame — `terminal.draw` is the SOLE owner of the `Frame`.
//!
//! Cancellation has two triggers that BOTH end the loop cleanly: the user pressing `q`/Ctrl-C (which
//! also fires the shared [`CancellationToken`] so the caller's engine task tears down), and the caller
//! cancelling the token (engine finished / external shutdown). The terminal is restored on every exit
//! path.

use std::io::{Stdout, stdout};
use std::time::Duration;

use crossterm::event::{
    DisableMouseCapture, EnableMouseCapture, Event as CrosstermEvent, EventStream, KeyCode,
    KeyEvent, KeyEventKind, KeyModifiers,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use futures::StreamExt;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use tokio::sync::mpsc::Receiver;
use tokio::time::interval;
use tokio_util::sync::CancellationToken;

use gw_engine::EngineEvent;

use crate::action::Action;
use crate::error::Result;
use crate::model::App;
use crate::view::view;

/// The default UI tick rate (drives gauges/sparkline refresh and any animation): 4 ticks/second.
pub const DEFAULT_TICK_RATE: Duration = Duration::from_millis(250);

/// The default render rate (frame cap): ~30 fps. Decoupled from ticks so input stays responsive.
pub const DEFAULT_FRAME_RATE: Duration = Duration::from_millis(33);

/// Run the dashboard event loop until the user quits or `cancel` fires, consuming engine events from
/// `events`. Owns the terminal for its whole lifetime; restores it on every exit path.
///
/// The CALLER spawns the engine run and passes the [`Receiver`] half of
/// [`EventSink::subscribe`](gw_engine::EventSink::subscribe) plus a [`CancellationToken`] it shares with
/// that engine task. When the user quits, this fires `cancel` so the engine task also winds down; when
/// the engine finishes and the caller drops the sink (closing the channel) or cancels the token, this
/// loop exits.
///
/// # Errors
/// Returns [`TuiError::Io`](crate::TuiError::Io) if entering/leaving the terminal, reading the event
/// stream, or drawing a frame fails.
pub async fn run(
    events: Receiver<EngineEvent>,
    cancel: CancellationToken,
    tick_rate: Duration,
    frame_rate: Duration,
) -> Result<()> {
    install_panic_hook();
    let mut terminal = enter()?;
    let result = event_loop(&mut terminal, events, &cancel, tick_rate, frame_rate).await;
    // Restore the terminal regardless of how the loop ended (clean exit or error).
    let restored = exit(&mut terminal);
    result.and(restored)
}

/// The inner loop, factored out so [`run`] can ALWAYS restore the terminal even on an early error.
async fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    mut events: Receiver<EngineEvent>,
    cancel: &CancellationToken,
    tick_rate: Duration,
    frame_rate: Duration,
) -> Result<()> {
    let mut app = App::new();
    let mut reader = EventStream::new();
    let mut tick = interval(tick_rate);
    let mut render = interval(frame_rate);
    // Once the engine sink is dropped the channel CLOSES and `recv()` returns `None` immediately and
    // forever; without disabling the arm the `select!` would busy-spin at 100% CPU (LOOP-1). Track the
    // open state and guard the arm so a closed channel simply stops being polled.
    let mut events_open = true;

    loop {
        tokio::select! {
            // Caller-driven shutdown (engine finished / external cancel).
            () = cancel.cancelled() => break,

            // Engine events: map -> Action -> pure update. `None` => the sink was dropped (run over):
            // keep the final frame up but stop polling this arm (the caller will cancel to end us).
            maybe_event = events.recv(), if events_open => {
                match maybe_event {
                    Some(engine_event) => {
                        app.update(Action::from_engine_event(engine_event));
                    }
                    None => events_open = false,
                }
            }

            // Terminal input (keys/resize). A stream error or end-of-stream is treated as a quit.
            maybe_term = reader.next() => {
                match maybe_term {
                    Some(Ok(term_event)) => {
                        if let Some(action) = map_terminal_event(term_event) {
                            let quit = matches!(action, Action::Quit);
                            app.update(action);
                            if quit { cancel.cancel(); break; }
                        }
                    }
                    Some(Err(_)) | None => { cancel.cancel(); break; }
                }
            }

            _ = tick.tick() => { app.update(Action::Tick); }

            _ = render.tick() => {
                terminal.draw(|frame| view(frame, &mut app))?;
            }
        }

        if app.should_quit {
            cancel.cancel();
            break;
        }
    }
    Ok(())
}

/// Map a crossterm terminal event onto an [`Action`], or `None` to ignore it. Quit on `q`/`Esc`/Ctrl-C.
fn map_terminal_event(event: CrosstermEvent) -> Option<Action> {
    match event {
        CrosstermEvent::Key(key) => map_key(key),
        CrosstermEvent::Resize(width, height) => Some(Action::Resize { width, height }),
        _ => None,
    }
}

/// Map a key press onto an [`Action`]. Only `Press` events are honored (so a key is not double-counted
/// on terminals that also emit `Release`).
fn map_key(key: KeyEvent) -> Option<Action> {
    if key.kind != KeyEventKind::Press {
        return None;
    }
    // Ctrl-C is an unconditional quit.
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return Some(Action::Quit);
    }
    match key.code {
        KeyCode::Char('q') | KeyCode::Esc => Some(Action::Quit),
        KeyCode::Up | KeyCode::Char('k') => Some(Action::SelectUp),
        KeyCode::Down | KeyCode::Char('j') => Some(Action::SelectDown),
        KeyCode::Char('g') | KeyCode::Home => Some(Action::SelectFirst),
        KeyCode::Char('G') | KeyCode::End => Some(Action::SelectLast),
        _ => None,
    }
}

/// Enter raw mode + the alternate screen and build the ratatui terminal over stdout.
///
/// SELF-CLEANING (LOOP-2): once `enable_raw_mode` succeeds, any later failure (entering the alternate
/// screen, or building the terminal) UNDOES raw mode + the alt screen before returning the error — so a
/// partial-setup failure can never leave the caller's terminal in raw mode (the error propagates out of
/// `run` via `?` WITHOUT reaching `exit`).
fn enter() -> Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    enter_after_raw_mode().inspect_err(|_| {
        let _ = execute!(stdout(), LeaveAlternateScreen, DisableMouseCapture);
        let _ = disable_raw_mode();
    })
}

/// The post-raw-mode setup steps, split out so [`enter`] can undo raw mode on any failure here.
fn enter_after_raw_mode() -> Result<Terminal<CrosstermBackend<Stdout>>> {
    let mut out = stdout();
    execute!(out, EnterAlternateScreen, EnableMouseCapture)?;
    let terminal = Terminal::new(CrosstermBackend::new(out))?;
    Ok(terminal)
}

/// Leave the alternate screen and disable raw mode, then show the cursor. Best-effort: every step is
/// attempted even if an earlier one failed, so a partial failure still restores as much as possible.
fn exit(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
    let raw = disable_raw_mode();
    let screen = execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    );
    let cursor = terminal.show_cursor();
    raw?;
    screen?;
    cursor?;
    Ok(())
}

/// Install a panic hook that restores the terminal BEFORE the default hook prints the panic, so a panic
/// mid-draw never leaves the user in a broken raw-mode/alt-screen terminal. Chains the previous hook.
fn install_panic_hook() {
    let original = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // Best-effort restore; ignore errors (we are already panicking).
        let _ = disable_raw_mode();
        let _ = execute!(stdout(), LeaveAlternateScreen, DisableMouseCapture);
        original(info);
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn q_and_esc_quit() {
        assert_eq!(map_key(press(KeyCode::Char('q'))), Some(Action::Quit));
        assert_eq!(map_key(press(KeyCode::Esc)), Some(Action::Quit));
    }

    #[test]
    fn ctrl_c_quits() {
        let key = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert_eq!(map_key(key), Some(Action::Quit));
    }

    #[test]
    fn arrows_and_vim_keys_move_selection() {
        assert_eq!(map_key(press(KeyCode::Up)), Some(Action::SelectUp));
        assert_eq!(map_key(press(KeyCode::Char('k'))), Some(Action::SelectUp));
        assert_eq!(map_key(press(KeyCode::Down)), Some(Action::SelectDown));
        assert_eq!(map_key(press(KeyCode::Char('j'))), Some(Action::SelectDown));
        assert_eq!(
            map_key(press(KeyCode::Char('g'))),
            Some(Action::SelectFirst)
        );
        assert_eq!(map_key(press(KeyCode::Char('G'))), Some(Action::SelectLast));
    }

    #[test]
    fn non_press_kinds_are_ignored() {
        let mut key = press(KeyCode::Char('q'));
        key.kind = KeyEventKind::Release;
        assert_eq!(map_key(key), None);
    }

    #[test]
    fn resize_maps_through() {
        assert_eq!(
            map_terminal_event(CrosstermEvent::Resize(80, 24)),
            Some(Action::Resize {
                width: 80,
                height: 24
            })
        );
    }

    #[test]
    fn unmapped_key_is_none() {
        assert_eq!(map_key(press(KeyCode::Char('z'))), None);
    }
}
