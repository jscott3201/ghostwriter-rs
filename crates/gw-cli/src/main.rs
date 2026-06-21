//! `gw` — the ghostwriter-rs command-line entrypoint.
//!
//! A thin shell over the [`gw_cli`] library: it owns only the tokio runtime and the process exit code.
//! The clap tree, the layered config, and every command handler live in the library so they are unit +
//! integration testable. See [`gw_cli`] for the full command surface and the security posture
//! (`OPENROUTER_API_KEY` is read from the environment ONLY).

/// Build the multi-threaded tokio runtime and drive [`gw_cli::run`] to completion, exiting non-zero on
/// an error (printing the full error chain to stderr — never the API key, which is never in scope).
fn main() -> std::process::ExitCode {
    let runtime = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("error: failed to start the async runtime: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    match runtime.block_on(gw_cli::run()) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            std::process::ExitCode::FAILURE
        }
    }
}
