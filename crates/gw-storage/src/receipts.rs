//! The SQLite half of export publication: immutable intent and one atomic acknowledgment.

use gw_schema::{
    ExportArtifact, ExportOptions, ExportScope, LifecycleState, StateTransition, TrainingRecord,
};
use serde::{Deserialize, Serialize};

use crate::artifact::{ExportPlan, Member, integrity, projected_hash, validate_rows};
use crate::export::{Projected, is_sft_eligible, project};
use crate::publication::publication_error;
use crate::{Result, Store, now_rfc3339};

/// Whether acknowledgment records an engine lifecycle transition or only a standalone receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExportPurpose {
    /// Publication belongs to one generation run and acknowledges its exact selected records.
    Engine,
    /// Export bookkeeping only; generation lifecycle and run status remain untouched.
    Standalone,
}

impl ExportPurpose {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Engine => "engine",
            Self::Standalone => "standalone",
        }
    }
}

pub(crate) struct Receipt {
    pub publication_id: String,
    pub destination: String,
    pub purpose: ExportPurpose,
    pub acknowledged: bool,
    pub artifact: ExportArtifact,
    pub members: Vec<Member>,
}

impl Store {
    pub(crate) async fn pending_export(
        &self,
        destination: &str,
        purpose: ExportPurpose,
        options: &ExportOptions,
    ) -> Result<Option<Receipt>> {
        let rows: Vec<(String, String, String)> = sqlx::query_as(
            "SELECT publication_id, artifact_json, members_json FROM export_receipts \
             WHERE destination = ?1 AND purpose = ?2 AND state = 'prepared' ORDER BY rowid",
        )
        .bind(destination)
        .bind(purpose.as_str())
        .fetch_all(self.pool())
        .await?;
        if rows.len() > 1 {
            return Err(integrity(
                "multiple pending publications for this destination",
            ));
        }
        let Some((publication_id, artifact_json, members_json)) = rows.into_iter().next() else {
            return Ok(None);
        };
        let artifact: ExportArtifact = serde_json::from_str(&artifact_json)
            .map_err(|error| publication_error(&publication_id, error.into()))?;
        if artifact.scope != options.scope
            || artifact.manifest.target != options.target
            || artifact.manifest.cot_policy != options.cot_policy
            || artifact.manifest.dataset_version != options.dataset_version
        {
            return Err(publication_error(
                &publication_id,
                integrity("pending publication has different options or scope"),
            ));
        }
        let members = serde_json::from_str(&members_json)
            .map_err(|error| publication_error(&publication_id, error.into()))?;
        Ok(Some(Receipt {
            publication_id,
            destination: destination.to_string(),
            purpose,
            acknowledged: false,
            artifact,
            members,
        }))
    }

    pub(crate) async fn load_export_receipt(&self, publication_id: &str) -> Result<Receipt> {
        let row: Option<(String, String, String, String, String)> = sqlx::query_as(
            "SELECT destination, purpose, artifact_json, members_json, state FROM export_receipts WHERE publication_id = ?1",
        ).bind(publication_id).fetch_optional(self.pool()).await?;
        let Some((destination, purpose, artifact, members, state)) = row else {
            return Err(integrity("publication receipt does not exist"));
        };
        let purpose = match purpose.as_str() {
            "engine" => ExportPurpose::Engine,
            "standalone" => ExportPurpose::Standalone,
            _ => return Err(integrity("unsupported publication acknowledgment mode")),
        };
        let artifact: ExportArtifact = serde_json::from_str(&artifact)?;
        if publication_identity(&artifact.artifact_id, &destination, purpose) != publication_id {
            return Err(integrity("publication receipt identity mismatch"));
        }
        Ok(Receipt {
            publication_id: publication_id.into(),
            destination,
            purpose,
            acknowledged: state == "acknowledged",
            artifact,
            members: serde_json::from_str(&members)?,
        })
    }

    pub(crate) async fn restore_export_plan(&self, receipt: &Receipt) -> Result<ExportPlan> {
        let mut tx = self.pool().begin_with("BEGIN IMMEDIATE").await?;
        let records = selected_records(&mut tx, &receipt.artifact, &receipt.members).await?;
        let rows = checked_projection(&receipt.artifact, &receipt.members, &records)?;
        tx.commit().await?;
        Ok(ExportPlan {
            artifact: receipt.artifact.clone(),
            rows,
        })
    }

    pub(crate) async fn prepare_export_receipt(
        &self,
        plan: &ExportPlan,
        destination: &str,
        purpose: ExportPurpose,
    ) -> Result<Receipt> {
        if purpose == ExportPurpose::Engine
            && !matches!(plan.artifact.scope, ExportScope::Run { .. })
        {
            return Err(integrity("engine publication requires a run scope"));
        }
        let publication_id = publication_identity(&plan.artifact.artifact_id, destination, purpose);
        let members = plan.members()?;
        let mut tx = self.pool().begin_with("BEGIN IMMEDIATE").await?;
        let records = selected_records(&mut tx, &plan.artifact, &members).await?;
        checked_projection(&plan.artifact, &members, &records)?;
        sqlx::query("INSERT INTO export_receipts \
            (publication_id, artifact_id, destination, purpose, artifact_json, members_json, state, prepared_at) \
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'prepared', ?7) ON CONFLICT(publication_id) DO UPDATE SET \
            state = 'prepared', acknowledged_at = NULL")
            .bind(&publication_id).bind(&plan.artifact.artifact_id).bind(destination).bind(purpose.as_str())
            .bind(serde_json::to_string(&plan.artifact)?).bind(serde_json::to_string(&members)?)
            .bind(now_rfc3339()).execute(&mut *tx).await?;
        tx.commit()
            .await
            .map_err(|error| publication_error(&publication_id, error.into()))?;
        Ok(Receipt {
            publication_id,
            destination: destination.to_string(),
            purpose,
            acknowledged: false,
            artifact: plan.artifact.clone(),
            members,
        })
    }

    /// Atomically recheck the exact selected snapshot and acknowledge only those members.
    pub(crate) async fn acknowledge_export(
        &self,
        receipt: &Receipt,
        purpose: ExportPurpose,
    ) -> Result<Vec<String>> {
        let mut tx = self.pool().begin_with("BEGIN IMMEDIATE").await?;
        let stored: Option<(String, String, String, String)> = sqlx::query_as(
            "SELECT state, purpose, artifact_json, members_json FROM export_receipts WHERE publication_id = ?1",
        ).bind(&receipt.publication_id).fetch_optional(&mut *tx).await?;
        let Some((state, stored_purpose, artifact, members)) = stored else {
            return Err(integrity("prepared publication receipt is missing"));
        };
        if stored_purpose != purpose.as_str()
            || serde_json::from_str::<ExportArtifact>(&artifact)? != receipt.artifact
            || serde_json::from_str::<Vec<Member>>(&members)? != receipt.members
        {
            return Err(integrity("prepared publication receipt changed"));
        }
        let records = selected_records(&mut tx, &receipt.artifact, &receipt.members).await?;
        checked_projection(&receipt.artifact, &receipt.members, &records)?;
        if state == "acknowledged" {
            tx.commit().await?;
            return Ok(Vec::new());
        }
        let mut advanced = Vec::new();
        let at = now_rfc3339();
        if purpose == ExportPurpose::Engine {
            if !matches!(receipt.artifact.scope, ExportScope::Run { .. }) {
                return Err(integrity("engine receipt lost its run scope"));
            }
            let detail = format!("artifact:{}", receipt.artifact.artifact_id);
            for mut record in records {
                let already: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM lifecycle_history WHERE record_id = ?1 AND state = 'exported' AND detail = ?2")
                    .bind(&record.record_id).bind(&detail).fetch_one(&mut *tx).await?;
                if already.0 == 0 {
                    record.lifecycle.attempts = record.lifecycle.attempts.saturating_add(1);
                    record.lifecycle.history.push(StateTransition {
                        state: LifecycleState::Exported,
                        at: at.clone(),
                        attempt: record.lifecycle.attempts,
                    });
                    sqlx::query("INSERT INTO lifecycle_history (record_id, state, at, detail) VALUES (?1, 'exported', ?2, ?3)")
                        .bind(&record.record_id).bind(&at).bind(&detail).execute(&mut *tx).await?;
                }
                record.lifecycle.state = LifecycleState::Exported;
                record.lifecycle.error = None;
                sqlx::query("UPDATE records SET lifecycle_state = 'exported', record_json = ?1, updated_at = ?2 WHERE record_id = ?3")
                    .bind(serde_json::to_string(&record)?).bind(&at).bind(&record.record_id).execute(&mut *tx).await?;
                if already.0 == 0 {
                    advanced.push(record.record_id);
                }
            }
        }
        sqlx::query("UPDATE export_receipts SET state = 'acknowledged', acknowledged_at = ?1 WHERE publication_id = ?2")
            .bind(&at).bind(&receipt.publication_id).execute(&mut *tx).await?;
        tx.commit().await?;
        Ok(advanced)
    }
}

fn publication_identity(artifact_id: &str, destination: &str, purpose: ExportPurpose) -> String {
    let mut hash = blake3::Hasher::new_derive_key("ghostwriter.export.publication.v1");
    for part in [artifact_id, destination, purpose.as_str()] {
        hash.update(&(part.len() as u64).to_be_bytes());
        hash.update(part.as_bytes());
    }
    hash.finalize().to_hex().to_string()
}

async fn selected_records(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    artifact: &ExportArtifact,
    members: &[Member],
) -> Result<Vec<TrainingRecord>> {
    let mut records = Vec::with_capacity(members.len());
    for member in members {
        let json: Option<(String,)> =
            sqlx::query_as("SELECT record_json FROM records WHERE record_id = ?1")
                .bind(&member.record_id)
                .fetch_optional(&mut **tx)
                .await?;
        let Some((json,)) = json else {
            return Err(integrity("selected export record no longer exists"));
        };
        let record: TrainingRecord = serde_json::from_str(&json)?;
        if record.record_id != member.record_id || !is_sft_eligible(&record) {
            return Err(integrity("selected export record is no longer eligible"));
        }
        if let ExportScope::Run { run_id } = &artifact.scope
            && record.provenance.run_id != *run_id
        {
            return Err(integrity("selected export record changed run"));
        }
        records.push(record);
    }
    Ok(records)
}

fn checked_projection(
    artifact: &ExportArtifact,
    members: &[Member],
    records: &[TrainingRecord],
) -> Result<Vec<Projected>> {
    let version = artifact.manifest.column_schema_version;
    let rows: Vec<_> = records
        .iter()
        .map(|record| project(record, version))
        .collect::<Result<_>>()?;
    for (row, member) in rows.iter().zip(members) {
        if row.record_id != member.record_id
            || projected_hash(row, version)? != member.projected_hash
        {
            return Err(integrity(
                "selected export record changed after preparation",
            ));
        }
    }
    validate_rows(artifact, &rows)?;
    Ok(rows)
}
