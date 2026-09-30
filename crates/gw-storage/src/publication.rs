//! Same-directory staging, complete readback, one rename, then SQLite acknowledgment.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use gw_schema::{ExportArtifact, ExportOptions, ExportScope};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::artifact::{
    ARTIFACT_METADATA_KEY, ArtifactVerification, ExportPlan, integrity, verify_artifact,
};
use crate::export::write_parquet;
use crate::receipts::Receipt;
use crate::{ExportPurpose, RecordFilter, Result, StorageError, Store};

/// How this invocation obtained the verified destination artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublicationDisposition {
    /// A new prepared plan was staged, verified and renamed into place.
    Published,
    /// A pending plan was staged again because the destination contained different content.
    Republished,
    /// The expected artifact already existed and passed complete verification.
    AcknowledgedExisting,
}

/// A verified publication whose exact selected members have been acknowledged in SQLite.
#[derive(Debug, Clone)]
pub struct ExportPublication {
    /// Durable local receipt identity used by explicit recovery.
    pub publication_id: String,
    /// Self-contained footer metadata; its manifest is also the CLI's JSON result.
    pub artifact: ExportArtifact,
    /// Whether bytes were written, rewritten, or recovered from the existing destination.
    pub disposition: PublicationDisposition,
    /// Records with a newly committed engine export history event (empty for standalone export).
    pub advanced_record_ids: Vec<String>,
}

impl Store {
    /// Publish one self-contained artifact and acknowledge its exact frozen selection.
    ///
    /// A prepared receipt commits before creating a unique same-parent staging file. A fresh
    /// reader verifies all staged data before one rename replaces the destination. Only then does
    /// one SQLite transaction acknowledge the receipt and (for engine exports) its selected members.
    /// After a database failure, retry verifies the expected existing artifact and current selected
    /// members without provider calls. Different destination content is explicitly republished from
    /// the same frozen plan; content claiming the expected identity but failing verification errors.
    ///
    /// Standalone export never changes generation lifecycle or run status. Callers must serialize
    /// publications to a destination: this is not a cross-process lease or a filesystem/DB atomic
    /// transaction, and makes no power-loss guarantee. Once started, callers must await completion.
    ///
    /// # Errors
    /// Returns integrity, filesystem, encoding or database errors. Before rename, the prior
    /// destination is preserved; after rename, the valid artifact remains. A database error can
    /// leave the receipt prepared or acknowledged; explicit-ID recovery resolves the persisted state.
    /// Post-preparation errors include that ID. Cleanup removes only the owned staging file.
    pub async fn publish_export(
        &self,
        options: ExportOptions,
        dst: impl AsRef<Path>,
        purpose: ExportPurpose,
    ) -> Result<ExportPublication> {
        let dst = dst.as_ref().to_path_buf();
        let dst = tokio::task::spawn_blocking(move || destination_path(&dst)).await??;
        let destination = dst
            .to_str()
            .ok_or_else(|| integrity("export destination is not valid UTF-8"))?
            .to_string();
        let pending = self.pending_export(&destination, purpose, &options).await?;
        let recovering = pending.is_some();
        let (plan, receipt) = if let Some(receipt) = pending {
            let plan = self
                .restore_export_plan(&receipt)
                .await
                .map_err(|error| publication_error(&receipt.publication_id, error))?;
            (plan, receipt)
        } else {
            let filter = match &options.scope {
                ExportScope::Store => RecordFilter::new(),
                ExportScope::Run { run_id } => RecordFilter::new().run_id(run_id),
                ExportScope::Records => {
                    return Err(integrity("store publication needs a store or run scope"));
                }
            };
            let records = self.scan(&filter).await?;
            let plan = tokio::task::spawn_blocking(move || ExportPlan::prepare(&records, options))
                .await??;
            let receipt = self
                .prepare_export_receipt(&plan, &destination, purpose)
                .await?;
            (plan, receipt)
        };
        self.finish_publication(plan, receipt, recovering).await
    }

    /// Recover exactly one prepared or acknowledged publication, using its recorded destination.
    ///
    /// Verifies the original selected IDs and projection, then verifies the existing artifact or
    /// republishes the same frozen plan. The receipt's original acknowledgment mode is preserved:
    /// engine receipts may finish record export acknowledgment; standalone receipts do not change
    /// lifecycle. Run status is never changed and no generation/provider work occurs.
    ///
    /// # Errors
    /// Returns an error carrying the publication ID on missing/stale receipts, integrity failures,
    /// filesystem failures or database errors. A changed selected record is never acknowledged.
    pub async fn resume_export(&self, publication_id: &str) -> Result<ExportPublication> {
        let result = async {
            let receipt = self.load_export_receipt(publication_id).await?;
            let plan = self.restore_export_plan(&receipt).await?;
            self.finish_publication(plan, receipt, true).await
        }
        .await;
        result.map_err(|error| publication_error(publication_id, error))
    }

    async fn finish_publication(
        &self,
        plan: ExportPlan,
        receipt: Receipt,
        recovering: bool,
    ) -> Result<ExportPublication> {
        let publication_id = receipt.publication_id.clone();
        let result = async {
            let dst = PathBuf::from(&receipt.destination);
            // An acknowledged receipt needs no new preparation when its expected file still verifies.
            let existing_artifact = plan.artifact.clone();
            let existing_dst = dst.clone();
            let exists = tokio::task::spawn_blocking(move || {
                if !claims_expected_identity(&existing_dst, &existing_artifact.artifact_id) {
                    return Ok(false);
                }
                match verify_artifact(&existing_dst)? {
                    ArtifactVerification::Verified(artifact) if artifact == existing_artifact => {
                        Ok(true)
                    }
                    _ => Err(integrity(
                        "existing destination does not match prepared artifact",
                    )),
                }
            })
            .await??;
            let disposition = if exists {
                PublicationDisposition::AcknowledgedExisting
            } else {
                // A repeated publication starts with durable intent even when an earlier attempt was
                // acknowledged. This preserves the exact receipt identity, selected members and mode.
                if receipt.acknowledged {
                    self.prepare_export_receipt(&plan, &receipt.destination, receipt.purpose)
                        .await?;
                }
                tokio::task::spawn_blocking(move || -> Result<PublicationDisposition> {
                    publish_staged(&plan, &dst, Fault::default())?;
                    Ok(if recovering {
                        PublicationDisposition::Republished
                    } else {
                        PublicationDisposition::Published
                    })
                })
                .await??
            };
            let advanced_record_ids = self.acknowledge_export(&receipt, receipt.purpose).await?;
            Ok(ExportPublication {
                publication_id: receipt.publication_id,
                artifact: receipt.artifact,
                disposition,
                advanced_record_ids,
            })
        }
        .await;
        result.map_err(|error| publication_error(&publication_id, error))
    }
}

pub(crate) fn publication_error(id: &str, source: StorageError) -> StorageError {
    if matches!(source, StorageError::Publication { .. }) {
        return source;
    }
    StorageError::Publication {
        publication_id: id.into(),
        source: Box::new(source),
    }
}

fn destination_path(dst: &Path) -> Result<PathBuf> {
    let file_name = dst
        .file_name()
        .ok_or_else(|| integrity("export destination needs a filename"))?;
    let parent = dst
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    Ok(std::fs::canonicalize(parent)?.join(file_name))
}

/// Inspect claims only to choose recovery vs actual replacement; acceptance always uses full verify.
fn claims_expected_identity(dst: &Path, expected: &str) -> bool {
    // Avoid following a symlink or opening a FIFO/device while inspecting an old destination.
    if !std::fs::symlink_metadata(dst).is_ok_and(|metadata| metadata.is_file()) {
        return false;
    }
    let Ok(file) = File::open(dst) else {
        return false;
    };
    let Ok(reader) = ParquetRecordBatchReaderBuilder::try_new(file) else {
        return false;
    };
    reader
        .metadata()
        .file_metadata()
        .key_value_metadata()
        .into_iter()
        .flatten()
        .filter(|entry| entry.key == ARTIFACT_METADATA_KEY)
        .filter_map(|entry| entry.value.as_deref())
        .filter_map(|value| serde_json::from_str::<serde_json::Value>(value).ok())
        .any(|value| value.get("artifact_id").and_then(|id| id.as_str()) == Some(expected))
}

#[derive(Default, Clone, Copy)]
enum Fault {
    #[default]
    None,
    #[cfg(test)]
    Write,
    #[cfg(test)]
    Footer,
    #[cfg(test)]
    Validation,
    #[cfg(test)]
    Rename,
}

fn publish_staged(plan: &ExportPlan, dst: &Path, fault: Fault) -> Result<()> {
    let (stage, file) = create_stage(dst)?;
    let result = (|| {
        #[cfg(test)]
        match fault {
            Fault::Write | Fault::Footer => {
                let mut encoded = Vec::new();
                write_parquet(&plan.rows, &plan.artifact, &mut encoded)?;
                let remaining = if matches!(fault, Fault::Write) {
                    encoded.len() / 2
                } else {
                    encoded.len() - 1
                };
                write_parquet(
                    &plan.rows,
                    &plan.artifact,
                    FailingWriter {
                        file: &file,
                        remaining,
                    },
                )?;
                return Err(integrity("injected write unexpectedly succeeded"));
            }
            _ => {}
        }
        write_parquet(&plan.rows, &plan.artifact, &file)?;
        file.sync_all()?;
        drop(file);
        #[cfg(test)]
        if matches!(fault, Fault::Validation) {
            std::fs::write(&stage, b"corrupt staged artifact")?;
        }
        match verify_artifact(&stage)? {
            ArtifactVerification::Verified(actual) if actual == plan.artifact => {}
            _ => return Err(integrity("staged artifact does not match prepared plan")),
        }
        #[cfg(test)]
        if matches!(fault, Fault::Rename) {
            return Err(std::io::Error::other("injected rename failure").into());
        }
        let _ = fault;
        // A single same-parent rename is the visibility boundary. Never delete the old destination.
        std::fs::rename(&stage, dst)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&stage);
    }
    result
}

#[cfg(test)]
struct FailingWriter<'a> {
    file: &'a File,
    remaining: usize,
}

#[cfg(test)]
impl std::io::Write for FailingWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self.remaining == 0 {
            return Err(std::io::Error::other("injected write or footer failure"));
        }
        let count = bytes.len().min(self.remaining);
        let written = self.file.write(&bytes[..count])?;
        self.remaining -= written;
        Ok(written)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

fn create_stage(dst: &Path) -> Result<(PathBuf, File)> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let parent = dst
        .parent()
        .ok_or_else(|| integrity("destination parent missing"))?;
    loop {
        let path = parent.join(format!(
            ".gw-export-{}-{}.tmp",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gw_schema::{CotPolicy, TrlFormat};

    #[test]
    fn prepublication_failures_preserve_old_bytes_and_cleanup_owned_stage() {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "gw-stage-faults-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).unwrap();
        let dst = dir.join("out.parquet");
        let plan = ExportPlan::prepare(
            &[],
            ExportOptions {
                target: TrlFormat::ChatML,
                cot_policy: CotPolicy::Masked,
                dataset_version: None,
                scope: ExportScope::Store,
            },
        )
        .unwrap();
        for fault in [
            Fault::Write,
            Fault::Footer,
            Fault::Validation,
            Fault::Rename,
        ] {
            std::fs::write(&dst, b"previous good artifact").unwrap();
            assert!(publish_staged(&plan, &dst, fault).is_err());
            assert_eq!(std::fs::read(&dst).unwrap(), b"previous good artifact");
            assert_eq!(
                std::fs::read_dir(&dir).unwrap().count(),
                1,
                "only the destination remains"
            );
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}
