//! Short durable transactions for model transmission evidence; no lock spans HTTP work.
use crate::{Result, StorageError, Store, now_rfc3339};
use gw_schema::{
    AccountingCapability, AttemptIntent, AttemptMetadata, AttemptObservation, AttemptReceipt,
    LaunchCoverage, OutputInterpretation, ReportedCost, TransportSettlement,
};
use sqlx::{Sqlite, Transaction};

impl Store {
    /// Record a fresh launch and the capabilities of its actual injected clients.
    /// Existing launch history is never upgraded by a later client replacement.
    pub async fn begin_model_launch(
        &self,
        run_id: &str,
        teacher: AccountingCapability,
        judge: AccountingCapability,
        embedding: AccountingCapability,
    ) -> Result<LaunchCoverage> {
        let mut tx = self.pool().begin_with("BEGIN IMMEDIATE").await?;
        sqlx::query(
            "UPDATE run_accounting SET history_complete=0, revision=revision+1 WHERE run_id = ?",
        )
        .bind(run_id)
        .execute(&mut *tx)
        .await?;
        let launch_id: String = sqlx::query_scalar("SELECT lower(hex(randomblob(16)))")
            .fetch_one(&mut *tx)
            .await?;
        let coverage = LaunchCoverage {
            version: 1,
            run_id: run_id.into(),
            launch_id,
            history: gw_schema::AccountingHistory::Unknown,
            policy: None,
            teacher,
            judge,
            embedding,
        };
        sqlx::query("INSERT INTO model_launches (launch_id, run_id, coverage_json, created_at) VALUES (?, ?, ?, ?)")
            .bind(&coverage.launch_id).bind(run_id).bind(serde_json::to_string(&coverage)?).bind(now_rfc3339()).execute(&mut *tx).await?;
        tx.commit().await?;
        Ok(coverage)
    }

    /// List every recorded launch. An empty result does not certify historical accounting.
    pub async fn model_launches(&self, run_id: &str) -> Result<Vec<LaunchCoverage>> {
        let rows: Vec<String> = sqlx::query_scalar("SELECT coverage_json FROM model_launches WHERE run_id = ? ORDER BY created_at, launch_id")
            .bind(run_id).fetch_all(self.pool()).await?;
        rows.into_iter().map(|row| decode_coverage(&row)).collect()
    }

    /// Commit one intent before the caller may transmit. A receipt needs no existing record.
    pub async fn begin_model_attempt(&self, intent: &AttemptIntent) -> Result<String> {
        validate_intent(intent)?;
        let mut tx = self.pool().begin_with("BEGIN IMMEDIATE").await?;
        if super::accounting::policy(&mut tx, &intent.context.run_id)
            .await?
            .is_some()
        {
            return Err(StorageError::Admission(
                gw_schema::AdmissionDenial::UnregisteredContext,
            ));
        }
        let launch: String = sqlx::query_scalar(
            "SELECT coverage_json FROM model_launches WHERE launch_id = ? AND run_id = ?",
        )
        .bind(&intent.context.launch_id)
        .bind(&intent.context.run_id)
        .fetch_one(&mut *tx)
        .await?;
        let _ = decode_coverage(&launch)?;
        let id: String = sqlx::query_scalar("SELECT lower(hex(randomblob(16)))")
            .fetch_one(&mut *tx)
            .await?;
        let receipt = AttemptReceipt {
            attempt_id: id.clone(),
            policy_epoch: None,
            intent: intent.clone(),
            metadata: AttemptMetadata::default(),
            observations: Vec::new(),
            conflicts: Vec::new(),
            transport: None,
            interpretation: None,
        };
        let now = now_rfc3339();
        sqlx::query("INSERT INTO model_attempts (attempt_id, run_id, launch_id, receipt_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)")
            .bind(&id).bind(&intent.context.run_id).bind(&intent.context.launch_id).bind(serde_json::to_string(&receipt)?).bind(&now).bind(&now).execute(&mut *tx).await?;
        tx.commit().await?;
        Ok(id)
    }

    /// Read receipts including unresolved intents, failed outputs, and conflicting observations.
    pub async fn model_attempts(&self, run_id: &str) -> Result<Vec<AttemptReceipt>> {
        let rows: Vec<String> = sqlx::query_scalar("SELECT receipt_json FROM model_attempts WHERE run_id = ? ORDER BY created_at, attempt_id")
            .bind(run_id).fetch_all(self.pool()).await?;
        rows.into_iter().map(|row| decode_receipt(&row)).collect()
    }

    /// Persist cumulative metadata in zero-based sequence order without summing repeats.
    /// Retrying an identical operation returns its original result. Reusing a sequence with
    /// different metadata, skipping a sequence, or decreasing cumulative values is a durable error.
    pub async fn observe_model_attempt(
        &self,
        id: &str,
        sequence: u64,
        metadata: &AttemptMetadata,
    ) -> Result<()> {
        let mut tx = self.pool().begin_with("BEGIN IMMEDIATE").await?;
        let mut receipt = load(&mut tx, id).await?;
        let mut metadata = metadata.clone();
        if matches!(metadata.cost_usd, ReportedCost::Known(value) if !value.is_finite() || value < 0.0)
        {
            metadata.cost_usd = ReportedCost::Invalid;
            if !metadata.invalid_fields.iter().any(|field| field == "cost") {
                metadata.invalid_fields.push("cost".into());
            }
        }
        if let Some(existing) = receipt.observations.iter().find(|observation| {
            observation.sequence == sequence && observation.metadata == metadata
        }) {
            return observation_result(id, existing);
        }
        let expected = receipt
            .observations
            .iter()
            .map(|observation| observation.sequence)
            .max()
            .map_or(Some(0), |previous| previous.checked_add(1));
        let conflicts = if Some(sequence) == expected {
            merge_metadata(&mut receipt, &metadata)
        } else {
            vec!["observation_sequence".into()]
        };
        for field in &conflicts {
            conflict(&mut receipt.conflicts, field);
        }
        let observation = AttemptObservation {
            sequence,
            metadata,
            conflicts,
        };
        receipt.observations.push(observation.clone());
        save(&mut tx, &receipt).await?;
        tx.commit().await?;
        observation_result(id, &observation)
    }

    /// Persist terminal transport evidence idempotently. A different settlement is a conflict.
    pub async fn settle_model_attempt(
        &self,
        id: &str,
        settlement: &TransportSettlement,
    ) -> Result<()> {
        let mut tx = self.pool().begin_with("BEGIN IMMEDIATE").await?;
        let mut receipt = load(&mut tx, id).await?;
        if receipt.transport.as_ref() == Some(settlement) {
            return Ok(());
        }
        let conflicting = receipt.transport.is_some();
        if conflicting {
            conflict(&mut receipt.conflicts, "transport");
        } else {
            receipt.transport = Some(settlement.clone());
        }
        save(&mut tx, &receipt).await?;
        tx.commit().await?;
        terminal_result(id, "transport", conflicting)
    }

    /// Record output interpretation independently from the transport settlement.
    pub async fn interpret_model_attempt(
        &self,
        id: &str,
        interpretation: OutputInterpretation,
    ) -> Result<()> {
        let mut tx = self.pool().begin_with("BEGIN IMMEDIATE").await?;
        let mut receipt = load(&mut tx, id).await?;
        if receipt.interpretation == Some(interpretation) {
            return Ok(());
        }
        let conflicting = receipt.interpretation.is_some();
        if conflicting {
            conflict(&mut receipt.conflicts, "interpretation");
        } else {
            receipt.interpretation = Some(interpretation);
        }
        save(&mut tx, &receipt).await?;
        tx.commit().await?;
        terminal_result(id, "interpretation", conflicting)
    }
}

pub(crate) fn validate_intent(intent: &AttemptIntent) -> Result<()> {
    if intent.version != 1
        || intent.context.run_id.is_empty()
        || intent.context.launch_id.is_empty()
        || intent.requested_model.is_empty()
        || intent.request_digest.len() != 64
        || !intent
            .request_digest
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(StorageError::Attempt("invalid model attempt intent".into()));
    }
    Ok(())
}
async fn load(tx: &mut Transaction<'_, Sqlite>, id: &str) -> Result<AttemptReceipt> {
    let json: String =
        sqlx::query_scalar("SELECT receipt_json FROM model_attempts WHERE attempt_id = ?")
            .bind(id)
            .fetch_one(&mut **tx)
            .await?;
    decode_receipt(&json)
}
async fn save(tx: &mut Transaction<'_, Sqlite>, receipt: &AttemptReceipt) -> Result<()> {
    super::accounting::bump(tx, &receipt.intent.context.run_id).await?;
    sqlx::query("UPDATE model_attempts SET receipt_json = ?, updated_at = ? WHERE attempt_id = ?")
        .bind(serde_json::to_string(receipt)?)
        .bind(now_rfc3339())
        .bind(&receipt.attempt_id)
        .execute(&mut **tx)
        .await?;
    Ok(())
}
fn terminal_result(id: &str, field: &str, conflicting: bool) -> Result<()> {
    if conflicting {
        Err(StorageError::Attempt(format!(
            "conflicting evidence for attempt {id}: {field}"
        )))
    } else {
        Ok(())
    }
}
fn conflict(conflicts: &mut Vec<String>, field: &str) {
    if !conflicts.iter().any(|f| f == field) {
        conflicts.push(field.into());
    }
}
fn observation_result(id: &str, observation: &AttemptObservation) -> Result<()> {
    if observation.conflicts.is_empty() {
        Ok(())
    } else {
        Err(StorageError::Attempt(format!(
            "conflicting observation {} for attempt {id}: {}",
            observation.sequence,
            observation.conflicts.join(", ")
        )))
    }
}
fn merge_metadata(receipt: &mut AttemptReceipt, patch: &AttemptMetadata) -> Vec<String> {
    let mut conflicts = Vec::new();
    let current = &mut receipt.metadata;
    for (name, old, new) in [
        (
            "prompt_tokens",
            &mut current.prompt_tokens,
            patch.prompt_tokens,
        ),
        (
            "completion_tokens",
            &mut current.completion_tokens,
            patch.completion_tokens,
        ),
        (
            "total_tokens",
            &mut current.total_tokens,
            patch.total_tokens,
        ),
        (
            "reasoning_tokens",
            &mut current.reasoning_tokens,
            patch.reasoning_tokens,
        ),
    ] {
        if let Some(value) = new {
            if old.is_some_and(|previous| value < previous) {
                conflict(&mut conflicts, name);
            }
            *old = Some(value);
        }
    }
    for (name, old, new) in [
        ("response_id", &mut current.response_id, &patch.response_id),
        ("model", &mut current.model, &patch.model),
        ("provider", &mut current.provider, &patch.provider),
    ] {
        if let Some(value) = new {
            if old.as_ref().is_some_and(|previous| previous != value) {
                conflict(&mut conflicts, name);
            }
            *old = Some(value.clone());
        }
    }
    match patch.cost_usd {
        ReportedCost::Missing => {}
        ReportedCost::Known(cost) if cost.is_finite() && cost >= 0.0 => {
            if matches!(current.cost_usd, ReportedCost::Known(previous) if cost < previous) {
                conflict(&mut conflicts, "cost_usd");
            }
            current.cost_usd = ReportedCost::Known(cost);
        }
        _ => {
            current.cost_usd = ReportedCost::Invalid;
        }
    }
    for field in &patch.invalid_fields {
        if !current.invalid_fields.contains(field) {
            current.invalid_fields.push(field.clone());
        }
    }
    conflicts
}

pub(crate) fn decode_coverage(json: &str) -> Result<LaunchCoverage> {
    let coverage: LaunchCoverage = serde_json::from_str(json)?;
    if coverage.version != 1 {
        return Err(StorageError::Attempt(
            "unsupported launch coverage version".into(),
        ));
    }
    Ok(coverage)
}
pub(crate) fn decode_receipt(json: &str) -> Result<AttemptReceipt> {
    let receipt: AttemptReceipt = serde_json::from_str(json)?;
    validate_intent(&receipt.intent)?;
    if matches!(receipt.metadata.cost_usd, ReportedCost::Known(value) if !value.is_finite() || value < 0.0)
    {
        return Err(StorageError::Attempt(
            "invalid known cost in persisted receipt".into(),
        ));
    }
    Ok(receipt)
}
