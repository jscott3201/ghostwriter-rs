//! The `gen export` handler — PURE (no providers): scan a [`Store`] and write admitted records to a
//! Parquet shard.
//!
//! [`Store::publish_export`] requires both an Admit judging verdict and a selected lifecycle state
//! (Admitted, Formatted, or Exported), so this scans the store (optionally restricted to one
//! run) and hands the whole set to the exporter; the returned [`ExportManifest`](gw_schema::ExportManifest)
//! (`n_records` / `n_admitted` / `build_inputs_hash`) is printed as JSON so a build pipeline can
//! record exactly what was written.

use anyhow::Context;

use gw_schema::{ExportOptions, ExportScope};
use gw_storage::{ExportPurpose, Store};

use crate::cli::{ExportArgs, ExportCot, ExportFormat};

/// Export admitted records from the store at `args.db` to the Parquet file at `args.out`, printing the
/// resulting manifest as JSON.
///
/// # Errors
/// Propagates a store-open / scan failure, a Parquet encode failure, or a filesystem write failure.
pub async fn export(args: ExportArgs) -> anyhow::Result<()> {
    if args.resume_publication.is_some()
        && (args.out.is_some()
            || args.run_id.is_some()
            || args.format.is_some()
            || args.cot.is_some()
            || args.dataset_version.is_some())
    {
        anyhow::bail!(
            "--resume-publication uses the receipt's destination, scope and policies; export overrides conflict"
        );
    }
    let store = Store::open(&args.db)
        .await
        .with_context(|| format!("opening the store at {}", args.db.display()))?;

    let publication = if let Some(publication_id) = &args.resume_publication {
        store
            .resume_export(publication_id)
            .await
            .with_context(|| format!("recovering artifact publication {publication_id}"))?
    } else {
        let out = args
            .out
            .context("export requires --out or --resume-publication")?;
        store
            .publish_export(
                ExportOptions {
                    target: args.format.unwrap_or(ExportFormat::ChatMl).into(),
                    cot_policy: args.cot.unwrap_or(ExportCot::Supervised).into(),
                    dataset_version: args.dataset_version,
                    scope: args
                        .run_id
                        .map_or(ExportScope::Store, |run_id| ExportScope::Run { run_id }),
                },
                &out,
                ExportPurpose::Standalone,
            )
            .await
            .with_context(|| format!("exporting Parquet to {}", out.display()))?
    };
    tracing::info!(publication_id = %publication.publication_id, disposition = ?publication.disposition, artifact_id = %publication.artifact.artifact_id, "artifact publication acknowledged");

    let json = serde_json::to_string_pretty(&publication.artifact.manifest)
        .context("serializing the export manifest")?;
    println!("{json}");
    Ok(())
}
