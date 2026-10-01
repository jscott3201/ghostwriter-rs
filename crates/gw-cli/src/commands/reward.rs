//! Provider-free numeric corpus export and captured-byte reward batch evaluation.
use gw_schema::{NumericRewardArtifact, NumericTaskDocument};
use std::io::{Read, Write};
use std::path::PathBuf;

/// Maximum captured JSON input per file or stdin request (64 MiB).
pub const MAX_REWARD_INPUT_BYTES: u64 = 64 * 1024 * 1024;

fn capture(reader: impl Read) -> anyhow::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader
        .take(MAX_REWARD_INPUT_BYTES + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_REWARD_INPUT_BYTES {
        anyhow::bail!("numeric reward JSON exceeds the 64 MiB input limit");
    }
    Ok(bytes)
}

fn emit(value: &impl serde::Serialize) -> anyhow::Result<()> {
    // Complete serialization precedes stdout. Errors never expose a usable partial result batch.
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    std::io::stdout().lock().write_all(&bytes)?;
    Ok(())
}

/// Export a single ordered artifact from reviewed task documents, selecting declared Train tasks.
/// Output is strict JSON on stdout, suitable for capture as an immutable corpus file.
///
/// # Errors
/// Rejects any malformed document, duplicate/conflicting declaration, or unsupported training task.
pub fn export(paths: &[PathBuf]) -> anyhow::Result<()> {
    let mut documents = Vec::new();
    for path in paths {
        let bytes = capture(std::fs::File::open(path)?)?;
        let text = std::str::from_utf8(&bytes)?;
        documents.push(NumericTaskDocument::from_json(text).map_err(anyhow::Error::msg)?);
    }
    let artifact = NumericRewardArtifact::from_documents(documents).map_err(anyhow::Error::msg)?;
    emit(&artifact)
}

/// Verify exactly one raw numeric artifact snapshot read from stdin.
///
/// # Errors
/// Rejects malformed, oversized, stale, or unsupported artifacts without emitting a report.
pub fn verify_stdin() -> anyhow::Result<()> {
    let bytes = capture(std::io::stdin().lock())?;
    let report = NumericRewardArtifact::verify_snapshot(&bytes).map_err(anyhow::Error::msg)?;
    emit(&report)
}

/// Evaluate one complete strict stdin batch with the shared pure numeric evaluator.
///
/// # Errors
/// Rejects malformed, oversized, incomplete, or incorrectly bound requests without output.
/// Factual Unknown is a complete result with no numeric reward; the first TRL adapter aborts it.
pub fn evaluate_stdin() -> anyhow::Result<()> {
    let bytes = capture(std::io::stdin().lock())?;
    let report = gw_judge::evaluate_numeric_reward_json(&bytes).map_err(anyhow::Error::msg)?;
    emit(&report)
}
