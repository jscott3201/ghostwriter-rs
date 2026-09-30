//! Immutable semantic comparison shared by CLI preflight and transactional launch registration.
use crate::{Result, StorageError, Store};
use gw_schema::RunManifest;

/// Whether an invocation may initialize an unknown run ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunMode {
    /// Start a new run, or resume an existing compatible run.
    CreateOrResume,
    /// Resume only an existing compatible run; unknown IDs fail closed.
    Replay,
}
pub(crate) type StoredManifest = (String, Option<i64>, Option<String>);

pub(crate) fn mismatch(run_id: &str, reason: impl Into<String>) -> StorageError {
    StorageError::RunManifest {
        run_id: run_id.into(),
        reason: reason.into(),
    }
}

pub(crate) fn check(
    run_id: &str,
    existing: Option<&StoredManifest>,
    manifest: &RunManifest,
    mode: RunMode,
) -> Result<()> {
    manifest
        .validate()
        .map_err(|reason| mismatch(run_id, reason))?;
    let Some((original, shards, digest)) = existing else {
        return if mode == RunMode::Replay {
            Err(mismatch(run_id, "unknown run ID for replay"))
        } else {
            Ok(())
        };
    };
    let stored: RunManifest = serde_json::from_str(original)
        .map_err(|_| mismatch(run_id, "legacy, unpinned, or malformed semantic manifest"))?;
    stored
        .validate()
        .map_err(|reason| mismatch(run_id, reason))?;
    // Compare original JSON values, not only a decoded struct: absent historical fields must not
    // be silently defaulted/adopted. Whitespace and object-key order do not change meaning.
    let original: serde_json::Value = serde_json::from_str(original)?;
    if original != serde_json::to_value(manifest)?
        || *shards != Some(manifest.input_plan.shard_items.len() as i64)
        || digest.as_deref() != Some(manifest.input_plan.content_hash.as_str())
    {
        return Err(mismatch(
            run_id,
            "incompatible generation, admission, client, or captured-input semantics",
        ));
    }
    Ok(())
}

impl Store {
    /// Read-only compatibility validation before CLI credential access. Execution must repeat this
    /// comparison inside [`Self::register_accounting_launch`] because another launch can race it.
    ///
    /// # Errors
    /// Rejects incompatible, unpinned, malformed or unsupported runs and unknown replay IDs.
    pub async fn validate_run_manifest(
        &self,
        run_id: &str,
        manifest: &RunManifest,
        mode: RunMode,
    ) -> Result<()> {
        let existing: Option<StoredManifest> = sqlx::query_as(
            "SELECT config_json, shard_count, prompts_hash FROM runs WHERE run_id=?",
        )
        .bind(run_id)
        .fetch_optional(self.pool())
        .await?;
        check(run_id, existing.as_ref(), manifest, mode)
    }
}
