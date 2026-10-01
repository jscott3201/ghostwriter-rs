//! Explicit local saved-function evaluation and fresh re-execution of captured declarations.
use crate::{
    CommandOutcome,
    coding::{CapturedCodingInput, CodingArtifact, observe_coding, replay_coding},
};
use anyhow::ensure;
use gw_schema::{CodingTaskDocument, ExecutionOutcome};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use tokio_util::sync::CancellationToken;

/// Offline coding evaluator inputs; the qualified image must already exist locally.
#[derive(Debug, clap::Args, PartialEq)]
pub struct CodingArgs {
    /// Strict reviewed coding task JSON, captured before any container starts.
    #[arg(long, value_name = "FILE")]
    pub tasks: PathBuf,
    /// Exactly one task label from the fully validated document.
    #[arg(long, value_name = "LABEL")]
    pub task: String,
    /// Complete saved UTF-8 Python module, captured exactly once.
    #[arg(long, value_name = "FILE")]
    pub candidate: PathBuf,
    /// New private evaluation artifact; existing files are never replaced.
    #[arg(long, value_name = "FILE")]
    pub output: PathBuf,
}
/// Saved replay explicitly starts fresh local containers; saved flags alone never establish success.
#[derive(Debug, clap::Args, PartialEq)]
pub struct CodingReplayArgs {
    /// Self-contained declaration to validate and re-execute from its captured inputs.
    #[arg(long, value_name = "FILE")]
    pub artifact: PathBuf,
    /// New private artifact describing the fresh observation.
    #[arg(long, value_name = "FILE")]
    pub output: PathBuf,
}

/// Evaluate one captured saved function, print a compact summary, and write its private artifact.
///
/// # Errors
/// Rejects invalid inputs, unsupported runtime, or failed private artifact publication.
pub async fn evaluate(args: CodingArgs) -> anyhow::Result<CommandOutcome> {
    ensure!(!args.output.exists(), "coding output already exists");
    let document = CodingTaskDocument::from_json(&capture(&args.tasks, 1024 * 1024)?)
        .map_err(anyhow::Error::msg)?;
    let code = String::from_utf8(capture(&args.candidate, 65_536)?)?;
    let input = CapturedCodingInput::new(&document, &args.task, code)?;
    let cancel = CancellationToken::new();
    let task = observe_coding(input, cancel.clone());
    tokio::pin!(task);
    let observed = tokio::select! {
        result = &mut task => result?,
        signal = tokio::signal::ctrl_c() => {signal?;cancel.cancel();task.await?}
    };
    publish(&args.output, &observed.consume())
}

/// Validate before Docker and consume only fresh matching local observations.
///
/// # Errors
/// Rejects stale/forged declarations, unsupported runtime, fresh mismatch, or publication failure.
pub async fn replay(args: CodingReplayArgs) -> anyhow::Result<CommandOutcome> {
    ensure!(!args.output.exists(), "coding output already exists");
    let saved = CodingArtifact::from_json(&capture(&args.artifact, 2 * 1024 * 1024)?)?;
    let cancel = CancellationToken::new();
    let task = replay_coding(saved, cancel.clone());
    tokio::pin!(task);
    let fresh = tokio::select! {
        result = &mut task => result?,
        signal = tokio::signal::ctrl_c() => {signal?;cancel.cancel();let _ = task.await;return Err(anyhow::anyhow!("coding replay cancelled after settlement attempts"));}
    };
    publish(&args.output, &fresh)
}
fn capture(path: &Path, limit: usize) -> anyhow::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take((limit + 1) as u64)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= limit,
        "coding input file exceeds its byte bound"
    );
    Ok(bytes)
}
fn publish(path: &Path, artifact: &CodingArtifact) -> anyhow::Result<CommandOutcome> {
    let bytes = serde_json::to_vec(artifact)?;
    CodingArtifact::from_json(&bytes)?;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut output = options.open(path)?;
    output.write_all(&bytes)?;
    output.sync_all()?;
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({"artifact_id":artifact.artifact_id,
        "input_id":artifact.input.input_id,"outcome":artifact.report.outcome,"cases":artifact.report.cases.len(),
        "runtime":artifact.runtime.recipe,"replayed_declaration_id":artifact.replayed_declaration_id}))?
    );
    Ok(if artifact.report.outcome == ExecutionOutcome::Passed {
        CommandOutcome::Success
    } else {
        CommandOutcome::GateRejected
    })
}
