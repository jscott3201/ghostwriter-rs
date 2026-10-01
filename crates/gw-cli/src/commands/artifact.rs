//! Provider-free verification for external consumers of one immutable byte snapshot.
use std::io::{Read, Write};

/// Read one Parquet snapshot from stdin, verify every row, and emit only a JSON success report.
///
/// # Errors
/// Fails on I/O, missing authoritative metadata, or any artifact integrity violation.
pub fn verify_stdin() -> anyhow::Result<()> {
    let mut snapshot = Vec::new();
    std::io::stdin().lock().read_to_end(&mut snapshot)?;
    let report = gw_storage::verify_artifact_snapshot(snapshot)?;
    let mut output = std::io::stdout().lock();
    serde_json::to_writer(&mut output, &report)?;
    writeln!(output)?;
    Ok(())
}
