//! `gw` — the ghostwriter-rs command-line entrypoint.
//!
//! Subcommands (`gen run`, `gen tui`, `gen export`, `gen replay`, `eval ...`) are wired as the
//! engine and downstream crates fill out. This scaffold parses `--version`/`--help` and prints
//! a placeholder.

use clap::Parser;

/// ghostwriter-rs: generate graded chain-of-thought reasoning traces as fine-tuning data.
#[derive(Debug, Parser)]
#[command(name = "gw", version, about)]
struct Cli;

fn main() {
    let _cli = Cli::parse();
    println!(
        "gw {} — ghostwriter-rs (scaffold)",
        env!("CARGO_PKG_VERSION")
    );
}
