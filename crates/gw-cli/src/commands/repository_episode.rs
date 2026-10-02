//! Bounded provider-free repository episode capture and internal verification.
use gw_schema::REPOSITORY_EPISODE_MAX_BYTES;
use std::io::{Read, Write};

fn input() -> anyhow::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    std::io::stdin()
        .lock()
        .take(REPOSITORY_EPISODE_MAX_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| anyhow::anyhow!("repository stdin read failed"))?;
    anyhow::ensure!(
        bytes.len() <= REPOSITORY_EPISODE_MAX_BYTES,
        "repository stdin exceeds 32 MiB bound"
    );
    Ok(bytes)
}
fn frame(value: &impl serde::Serialize) -> anyhow::Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(value)
        .map_err(|_| anyhow::anyhow!("repository output serialization failed"))?;
    bytes.push(b'\n');
    anyhow::ensure!(
        bytes.len() <= REPOSITORY_EPISODE_MAX_BYTES,
        "repository output exceeds 32 MiB bound"
    );
    Ok(bytes)
}
fn output(bytes: &[u8]) -> anyhow::Result<()> {
    let mut stdout = std::io::stdout().lock();
    stdout
        .write_all(bytes)
        .and_then(|()| stdout.flush())
        .map_err(|_| anyhow::anyhow!("repository stdout write failed"))
}
/// Capture and independently verify one complete artifact before writing any stdout bytes.
///
/// # Errors
/// Rejects invalid input, oversized output, failed internal verification or I/O failure.
pub fn capture_stdin() -> anyhow::Result<()> {
    let artifact = gw_judge::capture_repository_episode(&input()?)?;
    let bytes = frame(&artifact)?;
    gw_judge::verify_repository_episode(&bytes)?;
    output(&bytes)
}
/// Verify the complete saved artifact and emit one content-free JSON receipt.
///
/// # Errors
/// Rejects invalid/tampered input, oversized output or I/O failure before emitting a receipt.
pub fn verify_stdin() -> anyhow::Result<()> {
    let receipt = gw_judge::verify_repository_episode(&input()?)?;
    output(&frame(&receipt)?)
}
