//! Owned whole-population stdio bridge. Only public prompts and redacted observations cross it.
use crate::{
    CommandOutcome,
    coding::paired::{CodingPairedArtifact, observe_pair, replay_pair},
};
use anyhow::ensure;
use gw_schema::{CODING_PAIR_MAX_BYTES, CodingPairRequest, TaskSplitRole};
use gw_storage::Store;
use std::io::{Read, Write};
use std::path::PathBuf;
use tokio::io::AsyncReadExt;
use tokio_util::sync::CancellationToken;

/// Complete held-out split, never an arbitrary subset or training role.
#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq)]
pub enum CodingHeldoutSplit {
    /// All sixteen registered validation members.
    Validation,
    /// All thirty-two registered test members.
    Test,
}
impl From<CodingHeldoutSplit> for TaskSplitRole {
    fn from(value: CodingHeldoutSplit) -> Self {
        match value {
            CodingHeldoutSplit::Validation => Self::Validation,
            CodingHeldoutSplit::Test => Self::Test,
        }
    }
}
/// Capture one whole registered split, emit public prompts, then consume one complete request.
#[derive(Debug, clap::Args, PartialEq)]
pub struct CodingPairArgs {
    /// Existing local reference store; this command never registers or imports a population.
    #[arg(long)]
    pub db: PathBuf,
    /// Explicit operator registration with a complete committed native reference import.
    #[arg(long)]
    pub registration: String,
    /// Whole held-out partition.
    #[arg(long, value_enum)]
    pub split: CodingHeldoutSplit,
    /// Explicit bounded protocol: receive one JSON line, send one complete request, close stdin.
    #[arg(long, required = true)]
    pub stdio: bool,
}
/// Fresh native execution of saved generated modules; no model or training authority is recreated.
#[derive(Debug, clap::Args, PartialEq)]
pub struct CodingPairReplayArgs {
    /// Current registered reference store.
    #[arg(long)]
    pub db: PathBuf,
    /// One complete saved paired artifact on stdin; close the stream after writing it.
    #[arg(long, required = true)]
    pub stdin: bool,
}
fn frame(value: &impl serde::Serialize, limit: usize) -> anyhow::Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    ensure!(
        bytes.len() <= limit,
        "paired stdout frame exceeds byte bound including newline"
    );
    Ok(bytes)
}
fn print(value: &impl serde::Serialize) -> anyhow::Result<()> {
    let bytes = frame(value, CODING_PAIR_MAX_BYTES)?;
    let mut out = std::io::stdout().lock();
    out.write_all(&bytes)?;
    out.flush()?;
    Ok(())
}
async fn input() -> anyhow::Result<Vec<u8>> {
    let mut bytes = vec![];
    tokio::io::stdin()
        .take(CODING_PAIR_MAX_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .await?;
    ensure!(
        bytes.len() <= CODING_PAIR_MAX_BYTES,
        "paired stdin exceeds 32 MiB bound"
    );
    Ok(bytes)
}
fn request(bytes: &[u8]) -> anyhow::Result<CodingPairRequest> {
    let value = gw_schema::strict_coding_json(bytes).map_err(anyhow::Error::msg)?;
    let request = serde_json::from_value::<CodingPairRequest>(value.clone())?;
    ensure!(
        serde_json::to_value(&request)? == value,
        "paired request contains unknown or omitted fields"
    );
    Ok(request)
}
/// Run a single bounded whole-population bridge. Its owner closes stdin when cancelling.
///
/// # Errors
/// Rejects invalid registration or pair before execution; native uncertainty remains explicit.
pub async fn bridge(args: CodingPairArgs) -> anyhow::Result<CommandOutcome> {
    ensure!(
        args.db.is_file(),
        "paired evaluation requires an existing reference store"
    );
    let store = Store::open(&args.db).await?;
    let population = store
        .capture_coding_population(&args.registration, args.split.into())
        .await?;
    print(&serde_json::json!({"protocol_version":1,"population":population.public()}))?;
    let cancel = CancellationToken::new();
    let task = async {
        let request = request(&input().await?)?;
        observe_pair(population, request, cancel.clone()).await
    };
    tokio::pin!(task);
    let observed = tokio::select! {
        result = &mut task => result?,
        signal = tokio::signal::ctrl_c() => { signal?; cancel.cancel(); task.await? }
    };
    print(&observed.into_artifact())?;
    Ok(CommandOutcome::Success)
}
/// Re-execute a complete saved pair under the current registered private oracles.
///
/// # Errors
/// Rejects altered coverage or any fresh mismatch, without trusting saved pass declarations.
pub async fn replay(args: CodingPairReplayArgs) -> anyhow::Result<CommandOutcome> {
    ensure!(
        args.db.is_file(),
        "paired replay requires an existing reference store"
    );
    let saved = CodingPairedArtifact::from_json(&input().await?)?;
    let store = Store::open(&args.db).await?;
    let population = store
        .capture_coding_population(&saved.population.registration_id, saved.population.split)
        .await?;
    let cancel = CancellationToken::new();
    let task = replay_pair(population, saved, cancel.clone());
    tokio::pin!(task);
    let fresh = tokio::select! {
        result = &mut task => result?,
        signal = tokio::signal::ctrl_c() => { signal?; cancel.cancel(); task.await? }
    };
    print(&fresh.into_artifact())?;
    Ok(CommandOutcome::Success)
}
/// Verify saved bindings, complete coverage and arithmetic; this starts no model or container.
///
/// # Errors
/// Rejects inconsistent, incomplete, stale or forged saved declarations.
pub fn inspect() -> anyhow::Result<()> {
    let mut bytes = vec![];
    std::io::stdin()
        .take(CODING_PAIR_MAX_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    let artifact = CodingPairedArtifact::from_json(&bytes)?;
    print(
        &serde_json::json!({"report_version":1,"artifact_id":artifact.artifact_id,
        "population_id":artifact.population.population_id,"request_id":artifact.request.request_id(),
        "structural_validation":"passed","historical_training":"declared","historical_generation":"declared",
        "historical_execution":"declared","tokenizer_replay":"not_run","automatic_promotion":false,
        "comparable":artifact.comparable,"base":artifact.base,"candidate":artifact.candidate,
        "passed_difference":artifact.passed_difference}),
    )
}

#[cfg(test)]
mod frame_tests {
    #[test]
    fn serialized_frame_bound_counts_utf8_and_final_newline() {
        let value = serde_json::json!({"text":"é"});
        let json = serde_json::to_vec(&value).unwrap();
        let limit = json.len() + 1;
        let bytes = super::frame(&value, limit).unwrap();
        assert_eq!(bytes.len(), limit);
        assert_eq!(bytes.last(), Some(&b'\n'));
        assert_eq!(&bytes[..json.len()], json);
        assert!(super::frame(&value, limit - 1).is_err());
        assert!(
            super::frame(&value, String::from_utf8(json).unwrap().chars().count() + 1).is_err()
        );
    }
}
