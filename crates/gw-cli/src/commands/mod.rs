//! The command handlers — one module per leaf subcommand.
//!
//! Each handler takes its parsed clap args and returns a typed `anyhow::Result<_>` for the binary
//! boundary. The PURE handlers ([`export`], [`eval`]) touch only a [`Store`](gw_storage::Store) /
//! local files and are fully unit + integration tested. The LIVE handlers ([`run`], [`tui`],
//! [`replay`]) construct real
//! providers ([`crate::wire`]) and are NOT exercised against the network in tests (key-gated); their
//! non-network wiring is asserted in [`crate::wire`] + [`crate::config`].

pub mod eval;
pub mod export;
pub mod replay;
pub mod run;
pub mod tui;
