//! Provider-free verification for external consumers of one immutable byte snapshot.
use std::io::{Read, Write};

/// Stream a bounded full inference checkpoint and emit measured byte/content evidence.
///
/// # Errors
/// Rejects unsafe model content, contradictory training declarations or malformed captured input.
pub fn verify_checkpoint_stdin() -> anyhow::Result<()> {
    let report = gw_storage::verify_training_checkpoint(std::io::stdin().lock())?;
    let mut output = std::io::stdout().lock();
    serde_json::to_writer(&mut output, &report)?;
    writeln!(output)?;
    Ok(())
}

/// Verify a bounded captured prepared input and its source, then emit a separate receipt.
///
/// # Errors
/// Rejects malformed framing/payload, unverified source, or contradictory source bindings.
pub fn verify_prepared_stdin() -> anyhow::Result<()> {
    let mut snapshot = Vec::new();
    std::io::stdin()
        .lock()
        .take(gw_schema::MAX_PREPARED_SFT_BYTES as u64 + 1)
        .read_to_end(&mut snapshot)?;
    let verified = gw_storage::verify_prepared_sft_snapshot(snapshot)?;
    let mut output = std::io::stdout().lock();
    serde_json::to_writer(&mut output, verified.report())?;
    writeln!(output)?;
    Ok(())
}

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

/// Stream and inspect the separate complete Gemma LoRA framing.
pub fn verify_lora_stdin() -> anyhow::Result<()> {
    let report = gw_storage::verify_lora_checkpoint(std::io::stdin().lock())?;
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}

/// Stream the distinct CUDA checkpoint domain and report measured source/tensor evidence.
///
/// # Errors
/// Rejects malformed framing, changed source bytes or unsupported CUDA state declarations.
pub fn verify_cuda_lora_stdin() -> anyhow::Result<()> {
    let report = gw_storage::verify_cuda_lora_checkpoint(std::io::stdin().lock())?;
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}
