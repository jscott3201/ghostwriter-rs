//! One atomic complete import; retries read current records instead of replaying historical envelopes.
use crate::{
    RecordWriteStatus, ReferenceMemberObservation, RegisteredReferenceCatalogue, Result, Store,
    artifact::integrity, now_rfc3339, record_data as data, reference_records as reference,
};
use gw_schema::{TaskSplitRole, TrainingRecord};
use sqlx::{Sqlite, Transaction};

/// A committed reference import acknowledgment; all held-out members remain private.
#[derive(Debug, Clone)]
pub struct ReferenceImportOutcome {
    /// New commit or recognition of an existing complete batch.
    pub status: RecordWriteStatus,
    /// Stable complete batch identity.
    pub batch_id: String,
    /// Current authoritative Train records, including later publication lifecycle.
    pub records: Vec<TrainingRecord>,
    /// Private Validation/Test member count, never represented as training records.
    pub held_out_count: usize,
}
async fn current(
    tx: &mut Transaction<'_, Sqlite>,
    registered: &RegisteredReferenceCatalogue,
) -> Result<Option<ReferenceImportOutcome>> {
    let exists: bool = sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM reference_batches WHERE batch_id=? AND registration_id=? AND member_count=112)")
        .bind(registered.batch_id()).bind(registered.registration_id()).fetch_one(&mut **tx).await?;
    if !exists {
        return Ok(None);
    }
    let rows: Vec<(i64, String, String, Option<String>, String, String)> = sqlx::query_as("SELECT ordinal, member_id, split, record_id, origin_json, evidence_json FROM reference_members WHERE batch_id=? ORDER BY ordinal")
        .bind(registered.batch_id()).fetch_all(&mut **tx).await?;
    if rows.len() != 112 {
        return Err(integrity(
            "committed reference batch has incomplete membership",
        ));
    }
    let mut records = vec![];
    for (index, ((ordinal, member_id, split, record_id, origin_json, evidence_json), member)) in
        rows.iter()
            .zip(registered.population().members())
            .enumerate()
    {
        let expected_split = match member.task.split.role {
            TaskSplitRole::Train => "train",
            TaskSplitRole::Validation => "validation",
            TaskSplitRole::Test => "test",
        };
        let origin: gw_schema::ReviewedReferenceOrigin = serde_json::from_str(origin_json)?;
        let evidence: serde_json::Value = serde_json::from_str(evidence_json)?;
        let verification =
            serde_json::from_value(evidence["report"]["native_verification"].clone())?;
        let observed = ReferenceMemberObservation::from_native_observation(
            member_id.clone(),
            origin.native_result_id.clone(),
            verification,
            evidence,
        )?;
        if *ordinal != index as i64
            || member_id != &member.member_id
            || split != expected_split
            || origin != reference::origin(registered, member, &observed)?
            || record_id.is_some() != (member.task.split.role == TaskSplitRole::Train)
        {
            return Err(integrity(
                "committed reference membership differs from its accepted capture",
            ));
        }
        if let Some(id) = record_id {
            let record = data::load(tx, id)
                .await?
                .ok_or_else(|| integrity("committed reference record missing"))?;
            data::check_history(tx, &record).await?;
            reference::eligible(tx, &record).await?;
            records.push(record);
        }
    }
    if records.len() != 64 {
        return Err(integrity("committed reference Train coverage differs"));
    }
    Ok(Some(ReferenceImportOutcome {
        status: RecordWriteStatus::AlreadyApplied,
        batch_id: registered.batch_id().into(),
        records,
        held_out_count: 48,
    }))
}
impl Store {
    /// Read a previously committed batch's current records. This reports historical completion,
    /// never a fresh verification, and preserves later Formatted/Exported states and history.
    ///
    /// # Errors
    /// Rejects stale registrations, incomplete membership or corrupted authoritative state.
    pub async fn committed_reference_import(
        &self,
        registered: &RegisteredReferenceCatalogue,
    ) -> Result<Option<ReferenceImportOutcome>> {
        self.registered_reference_catalogue(&registered.capture)
            .await?;
        let mut tx = self.pool().begin().await?;
        let result = current(&mut tx, registered).await?;
        tx.commit().await?;
        Ok(result)
    }
    /// Commit one complete batch from the trusted native execution adapter. The application must
    /// retain fresh opaque observations until it constructs this material; saved reports confer no
    /// fresh authority. Registration, all members, Train records and history commit together.
    /// Cancellation remains effective through writer-lock acquisition and preparation. The final
    /// selection between cancellation and completed preparation is the commit decision: after it,
    /// commit and acknowledgment settle without cancellation replacing their outcome.
    ///
    /// # Errors
    /// Rejects incomplete/non-Pass members, changed registration or any inconsistent binding.
    /// Returns [`crate::StorageError::ReferenceImportCancelled`] after rolling back preparation
    /// when cancellation wins. Once commit wins, errors can mean a lost acknowledgment.
    /// A lost commit acknowledgment must be retried through the current committed batch lookup.
    pub async fn commit_reference_import(
        &self,
        registered: &RegisteredReferenceCatalogue,
        observations: &[ReferenceMemberObservation],
        cancelled: impl std::future::Future<Output = ()>,
    ) -> Result<ReferenceImportOutcome> {
        tokio::pin!(cancelled);
        if observations.len() != 112 {
            return Err(integrity(
                "reference import requires all 112 fresh observations",
            ));
        }
        let at = now_rfc3339();
        let mut prepared = Vec::with_capacity(112);
        for (member, observed) in registered.population().members().iter().zip(observations) {
            let origin = reference::origin(registered, member, observed)?;
            let record = if member.task.split.role == TaskSplitRole::Train {
                Some(reference::make_record(registered, member, observed, &at)?)
            } else {
                None
            };
            prepared.push((origin, record));
        }
        #[cfg(test)]
        self.test_boundary("reference_import", "transaction_wait")
            .await?;
        let mut tx = tokio::select! {
            biased;
            () = &mut cancelled => return Err(crate::StorageError::ReferenceImportCancelled),
            begun = self.pool().begin_with("BEGIN IMMEDIATE") => begun?,
        };
        let admission = async {
            let capture: Option<String> = sqlx::query_scalar("SELECT capture_json FROM reference_registrations WHERE registration_id=? AND catalogue_id=?")
            .bind(registered.registration_id()).bind(registered.population().catalogue_id()).fetch_optional(&mut *tx).await?;
            if capture.as_deref() != Some(serde_json::to_string(&registered.capture)?.as_str()) {
                return Err(integrity(
                    "operator registration changed or is absent at import commit",
                ));
            }
            if let Some(outcome) = current(&mut tx, registered).await? {
                return Ok(outcome);
            }
            sqlx::query("INSERT INTO runs(run_id,config_json,status,created_at,run_kind) VALUES (?,'null','completed',?,'reviewed_reference')")
            .bind(registered.batch_id()).bind(&at).execute(&mut *tx).await?;
            sqlx::query("INSERT INTO reference_batches(batch_id,registration_id,committed_at,member_count) VALUES (?,?,?,112)")
            .bind(registered.batch_id()).bind(registered.registration_id()).bind(&at).execute(&mut *tx).await?;
            let mut records = vec![];
            for (ordinal, ((origin, record), observed)) in
                prepared.into_iter().zip(observations).enumerate()
            {
                let record_id = record.as_ref().map(|r| r.record_id.clone());
                if let Some(record) = record {
                    data::write(&mut tx, &record, &at).await?;
                    data::history(
                        &mut tx,
                        &record,
                        0,
                        None,
                        Some("registered reference native verification and admission"),
                    )
                    .await?;
                    records.push(record);
                }
                let split = match registered.population().members()[ordinal].task.split.role {
                    TaskSplitRole::Train => "train",
                    TaskSplitRole::Validation => "validation",
                    TaskSplitRole::Test => "test",
                };
                sqlx::query("INSERT INTO reference_members(batch_id,ordinal,member_id,split,record_id,origin_json,evidence_json) VALUES (?,?,?,?,?,?,?)")
                .bind(registered.batch_id()).bind(ordinal as i64).bind(&origin.member_id).bind(split).bind(record_id)
                .bind(serde_json::to_string(&origin)?).bind(serde_json::to_string(&observed.evidence)?).execute(&mut *tx).await?;
            }
            #[cfg(test)]
            self.test_boundary("reference_import", "precommit").await?;
            Ok(ReferenceImportOutcome {
                status: RecordWriteStatus::Applied,
                batch_id: registered.batch_id().into(),
                records,
                held_out_count: 48,
            })
        };
        let decision = tokio::select! {
            biased;
            () = &mut cancelled => None,
            prepared = admission => Some(prepared),
        };
        let outcome = match decision {
            None => {
                tx.rollback().await?;
                return Err(crate::StorageError::ReferenceImportCancelled);
            }
            Some(prepared) => prepared?,
        };
        // No cancellation selection after this decision: a successful commit is authoritative.
        tx.commit().await?;
        #[cfg(test)]
        if outcome.status == RecordWriteStatus::Applied {
            self.test_boundary("reference_import", "committed").await?;
        }
        Ok(outcome)
    }
}
