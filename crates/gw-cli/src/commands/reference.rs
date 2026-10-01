//! Explicit operator registration and atomic native-verified reference imports.
use crate::CommandOutcome;
use anyhow::ensure;
use gw_schema::{ReferenceCapture, ReferenceCatalogue};
use std::{
    io::Read,
    path::{Path, PathBuf},
};
use tokio_util::sync::CancellationToken;

/// Exact local population sources and private application database.
#[derive(Debug, clap::Args, PartialEq)]
pub struct ReferenceArgs {
    /// Strict complete catalogue, with relative paths resolved from its parent directory.
    #[arg(long, value_name = "FILE")]
    pub catalogue: PathBuf,
    /// Private local application store; registration and import must use the same store.
    #[arg(long, value_name = "FILE")]
    pub db: PathBuf,
}
/// Explicit fresh revalidation policy for a registered reference import.
#[derive(Debug, clap::Args, PartialEq)]
pub struct ReferenceImportArgs {
    /// Catalogue and store.
    #[command(flatten)]
    pub source: ReferenceArgs,
    /// Execute all 112 members even if this batch is already committed; retain original facts.
    #[arg(long)]
    pub fresh_validation: bool,
}
fn read(path: &Path, limit: usize) -> anyhow::Result<String> {
    ensure!(
        std::fs::symlink_metadata(path)?.file_type().is_file(),
        "reference input must be a regular non-symlink file"
    );
    let file = std::fs::File::open(path)?;
    ensure!(
        file.metadata()?.is_file(),
        "reference input must be a regular file"
    );
    let mut bytes = Vec::new();
    file.take((limit + 1) as u64).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= limit,
        "reference input exceeds capture bound"
    );
    Ok(String::from_utf8(bytes)?)
}
fn relative(base: &Path, name: &str) -> anyhow::Result<PathBuf> {
    let mut path = base.to_owned();
    ensure!(
        !name.is_empty() && !name.contains('\\'),
        "invalid reference path"
    );
    for component in Path::new(name).components() {
        let std::path::Component::Normal(component) = component else {
            anyhow::bail!("reference path must be normal and relative");
        };
        path.push(component);
        ensure!(
            !std::fs::symlink_metadata(&path)?.file_type().is_symlink(),
            "reference paths cannot traverse symlinks"
        );
    }
    Ok(path)
}
fn capture(path: &Path) -> anyhow::Result<ReferenceCapture> {
    let catalogue = read(path, 1024 * 1024)?;
    let parsed: ReferenceCatalogue = serde_json::from_value(
        gw_schema::strict_coding_json(catalogue.as_bytes()).map_err(anyhow::Error::msg)?,
    )?;
    ensure!(
        parsed.version == 1 && parsed.members.len() == 112 && parsed.task_documents.len() <= 112,
        "unsupported or incomplete reference catalogue"
    );
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let task_documents = parsed
        .task_documents
        .iter()
        .map(|name| read(&relative(base, name)?, 1024 * 1024))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut modules = vec![];
    let mut reviews = vec![];
    for member in &parsed.members {
        modules.push(read(&relative(base, &member.module_path)?, 65_536)?);
        reviews.push(read(&relative(base, &member.review_path)?, 65_536)?);
    }
    let capture = ReferenceCapture {
        catalogue,
        task_documents,
        modules,
        reviews,
    };
    capture.validate().map_err(anyhow::Error::msg)?;
    Ok(capture)
}
/// Explicit local operator acceptance; this does not execute reference code.
///
/// # Errors
/// Rejects incomplete/invalid captures, input I/O errors or registration conflicts.
pub async fn register(args: ReferenceArgs) -> anyhow::Result<CommandOutcome> {
    let captured = capture(&args.catalogue)?;
    let store = gw_storage::Store::open(&args.db).await?;
    let registered = store.register_reference_catalogue(&captured).await?;
    println!(
        "{}",
        serde_json::json!({"status":"registered", "catalogue_id":registered.population().catalogue_id(),
        "registration_id":registered.registration_id(),"batch_id":registered.batch_id(),"members":112,
        "fresh_execution":false})
    );
    Ok(CommandOutcome::Success)
}
/// Match an existing registration and consume only fresh opaque native runtime observations.
///
/// # Errors
/// Rejects changed/unregistered inputs before execution; any member failure prevents batch commit.
pub async fn import(args: ReferenceImportArgs) -> anyhow::Result<CommandOutcome> {
    let captured = capture(&args.source.catalogue)?;
    let store = gw_storage::Store::open(&args.source.db).await?;
    let registered = store.registered_reference_catalogue(&captured).await?;
    let cancel = CancellationToken::new();
    let task = crate::coding::reference::import(
        &store,
        &registered,
        args.fresh_validation,
        cancel.clone(),
    );
    let (outcome, fresh) = settle_on_cancel(task, cancel, tokio::signal::ctrl_c()).await?;
    println!(
        "{}",
        serde_json::json!({"status":match outcome.status {
        gw_storage::RecordWriteStatus::Applied => "committed", gw_storage::RecordWriteStatus::AlreadyApplied => "already_committed" },
        "batch_id":outcome.batch_id,"training_records":outcome.records.len(),"private_held_out_members":outcome.held_out_count,
        "fresh_execution":fresh,"records":outcome.records.iter().map(|r| serde_json::json!({"record_id":r.record_id,"state":r.lifecycle.state})).collect::<Vec<_>>()})
    );
    Ok(CommandOutcome::Success)
}

async fn settle_on_cancel<T>(
    task: impl std::future::Future<Output = anyhow::Result<T>>,
    cancel: CancellationToken,
    signal: impl std::future::Future<Output = std::io::Result<()>>,
) -> anyhow::Result<T> {
    tokio::pin!(task);
    tokio::select! {
        result = &mut task => result,
        signal = signal => { signal?; cancel.cancel(); task.await }
    }
}

#[cfg(test)]
mod cancellation_tests {
    use super::*;

    #[tokio::test]
    async fn late_signal_retains_the_settled_commit_or_failure_outcome() {
        for committed in [true, false] {
            let cancel = CancellationToken::new();
            let observed = cancel.clone();
            let task = async move {
                observed.cancelled().await;
                if committed {
                    Ok(64)
                } else {
                    anyhow::bail!("reference import cancelled before commit")
                }
            };
            let result = settle_on_cancel(task, cancel, std::future::ready(Ok(()))).await;
            if committed {
                assert_eq!(result.unwrap(), 64);
            } else {
                assert!(result.unwrap_err().to_string().contains("before commit"));
            }
        }
    }
}

/// Publish one committed reference batch with reference-specific lifecycle acknowledgment.
#[derive(Debug, clap::Args, PartialEq)]
pub struct ReferenceExportArgs {
    /// Private source store containing the accepted registration and committed batch.
    #[arg(long)]
    pub db: PathBuf,
    /// Exact committed reference batch identity.
    #[arg(long)]
    pub batch_id: String,
    /// Local Parquet destination.
    #[arg(long)]
    pub out: PathBuf,
}
/// Export registered Train members and acknowledge only their reference lifecycle.
///
/// # Errors
/// Rejects a missing or non-reference batch, ineligible members or publication failure.
pub async fn export(args: ReferenceExportArgs) -> anyhow::Result<CommandOutcome> {
    let store = gw_storage::Store::open(&args.db).await?;
    ensure!(
        store.run_kind(&args.batch_id).await? == Some(gw_storage::RunKind::ReviewedReference),
        "export requires a committed reference batch"
    );
    let publication = store
        .publish_export(
            gw_schema::ExportOptions {
                target: gw_schema::TrlFormat::OpenAiMessages,
                cot_policy: gw_schema::CotPolicy::Stripped,
                dataset_version: None,
                scope: gw_schema::ExportScope::Run {
                    run_id: args.batch_id,
                },
            },
            &args.out,
            gw_storage::ExportPurpose::Reference,
        )
        .await?;
    println!("{}", serde_json::to_string(&publication.artifact)?);
    Ok(CommandOutcome::Success)
}
