//! The `gen export` handler — PURE (no providers): scan a [`Store`] and write admitted records to a
//! Parquet shard.
//!
//! [`export_parquet`] filters to admitted records
//! (`judging.verdict == Admit`) internally, so this scans the store (optionally restricted to one
//! run) and hands the whole set to the exporter; the returned [`ExportManifest`](gw_schema::ExportManifest)
//! (`n_records` / `n_admitted` / `build_inputs_hash`) is printed as JSON so a build pipeline can
//! record exactly what was written.

use anyhow::Context;

use gw_storage::{RecordFilter, Store, export_parquet};

use crate::cli::ExportArgs;

/// Export admitted records from the store at `args.db` to the Parquet file at `args.out`, printing the
/// resulting manifest as JSON.
///
/// # Errors
/// Propagates a store-open / scan failure, a Parquet encode failure, or a filesystem write failure.
pub async fn export(args: ExportArgs) -> anyhow::Result<()> {
    let store = Store::open(&args.db)
        .await
        .with_context(|| format!("opening the store at {}", args.db.display()))?;

    let mut filter = RecordFilter::new();
    if let Some(run_id) = &args.run_id {
        filter = filter.run_id(run_id);
    }
    let records = store
        .scan(&filter)
        .await
        .context("scanning records to export")?;

    let manifest = export_parquet(&records, args.format.into(), args.cot.into(), &args.out)
        .await
        .with_context(|| format!("exporting Parquet to {}", args.out.display()))?;

    let json =
        serde_json::to_string_pretty(&manifest).context("serializing the export manifest")?;
    println!("{json}");
    Ok(())
}
