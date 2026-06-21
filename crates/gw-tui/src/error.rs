//! The crate error type for the async I/O shell.
//!
//! The pure model/update/view layer never fails (it only mutates in-memory state), so errors arise
//! ONLY in the terminal-I/O shell ([`crate::event_loop`]): entering/leaving raw mode, reading the
//! crossterm event stream, and `terminal.draw`. Those are `std::io` failures, so [`TuiError`] wraps
//! [`std::io::Error`].

use thiserror::Error;

/// An error from the TUI's terminal-I/O shell. The model/update/view layer is infallible; only the
/// async loop ([`crate::run`]) — terminal setup, the crossterm event stream, and frame drawing — can
/// fail, and those are I/O failures.
#[derive(Debug, Error)]
pub enum TuiError {
    /// A terminal I/O failure: entering/leaving raw mode + the alternate screen, reading the event
    /// stream, or drawing a frame.
    #[error("terminal i/o error: {0}")]
    Io(#[from] std::io::Error),
}

/// The crate result alias.
pub type Result<T> = std::result::Result<T, TuiError>;
