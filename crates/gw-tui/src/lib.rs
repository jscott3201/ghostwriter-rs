//! `gw-tui` — the terminal UI.
//!
//! ratatui 0.30 Component + Action architecture: a single `tokio::select!` loop multiplexes a
//! crossterm `EventStream`, tick/render timers, and an `mpsc` action channel fed by streaming
//! generation/judging tasks. `terminal.draw` is the sole owner of the `Frame`. Depends on
//! `gw-schema` and `gw-engine` (through an Action/event interface only).
