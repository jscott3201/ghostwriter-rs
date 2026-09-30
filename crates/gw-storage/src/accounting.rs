//! Consistent absolute ledger summaries; record projections are deliberately excluded.
use crate::{Result, StorageError, Store};
use gw_schema::{
    AccountingCapability, AccountingHistory, AccountingSnapshot, AttemptReceipt, PolicyState,
    ReportedCost, TokenEvidence,
};
use sqlx::{Sqlite, Transaction};

impl Store {
    /// Read authoritative run accounting in one database snapshot, independent of event delivery.
    pub async fn accounting_snapshot(&self, run_id: &str) -> Result<AccountingSnapshot> {
        let mut tx = self.pool().begin().await?;
        let result = snapshot(&mut tx, run_id).await?;
        tx.commit().await?;
        Ok(result)
    }
}
pub(crate) async fn bump(tx: &mut Transaction<'_, Sqlite>, run_id: &str) -> Result<()> {
    sqlx::query("UPDATE run_accounting SET revision = revision + 1 WHERE run_id = ?")
        .bind(run_id)
        .execute(&mut **tx)
        .await?;
    Ok(())
}
pub(crate) async fn policy(
    tx: &mut Transaction<'_, Sqlite>,
    run_id: &str,
) -> Result<Option<(PolicyState, bool, u64)>> {
    let row: Option<(String, bool, i64)> = sqlx::query_as(
        "SELECT policy_json, history_complete, revision FROM run_accounting WHERE run_id = ?",
    )
    .bind(run_id)
    .fetch_optional(&mut **tx)
    .await?;
    row.map(|(json, complete, revision)| {
        let state: PolicyState = serde_json::from_str(&json)?;
        if state.version != 1 || !state.policy.is_valid() || state.epoch == 0 || revision < 1 {
            return Err(StorageError::Attempt(
                "invalid persisted accounting policy".into(),
            ));
        }
        Ok((state, complete, revision as u64))
    })
    .transpose()
}
pub(crate) async fn receipts(
    tx: &mut Transaction<'_, Sqlite>,
    run_id: &str,
) -> Result<Vec<AttemptReceipt>> {
    let rows: Vec<String> = sqlx::query_scalar(
        "SELECT receipt_json FROM model_attempts WHERE run_id = ? ORDER BY created_at, attempt_id",
    )
    .bind(run_id)
    .fetch_all(&mut **tx)
    .await?;
    rows.iter()
        .map(|json| crate::attempts::decode_receipt(json))
        .collect()
}
pub(crate) async fn snapshot(
    tx: &mut Transaction<'_, Sqlite>,
    run_id: &str,
) -> Result<AccountingSnapshot> {
    let state = policy(tx, run_id).await?;
    let launches: Vec<String> =
        sqlx::query_scalar("SELECT coverage_json FROM model_launches WHERE run_id = ?")
            .bind(run_id)
            .fetch_all(&mut **tx)
            .await?;
    let mut unknown_coverage_lanes = 0;
    for json in launches {
        let launch = crate::attempts::decode_coverage(&json)?;
        unknown_coverage_lanes += [launch.teacher, launch.judge, launch.embedding]
            .iter()
            .filter(|cap| **cap == AccountingCapability::Unknown)
            .count() as u64;
    }
    let mut summary = AccountingSnapshot {
        revision: state.as_ref().map_or(0, |(_, _, revision)| *revision),
        configured: None,
        effective: state.as_ref().map(|(state, _, _)| state.clone()),
        history: if state.is_some_and(|(_, complete, _)| complete) {
            AccountingHistory::RecordedFromCreation
        } else {
            AccountingHistory::Unknown
        },
        unknown_coverage_lanes,
        attempts: 0,
        known_usd: Some(0.0),
        unknown_cost_attempts: 0,
        invalid_cost_attempts: 0,
        conflicting_attempts: 0,
        unresolved_attempts: 0,
        prompt_tokens: TokenEvidence::default(),
        completion_tokens: TokenEvidence::default(),
        total_tokens: TokenEvidence::default(),
        reasoning_tokens: TokenEvidence::default(),
        elapsed_ms: Some(0),
    };
    for receipt in receipts(tx, run_id).await? {
        summary.attempts += 1;
        if !receipt.conflicts.is_empty() {
            summary.conflicting_attempts += 1;
        }
        let metadata = &receipt.metadata;
        if metadata.invalid_fields.iter().any(|field| field == "cost")
            || matches!(metadata.cost_usd, ReportedCost::Invalid)
        {
            summary.invalid_cost_attempts += 1;
        } else {
            match metadata.cost_usd {
                ReportedCost::Known(cost)
                    if !receipt.conflicts.iter().any(|field| field == "cost_usd") =>
                {
                    summary.known_usd = summary.known_usd.and_then(|total| {
                        let sum = total + cost;
                        sum.is_finite().then_some(sum)
                    });
                }
                ReportedCost::Missing => summary.unknown_cost_attempts += 1,
                _ => {}
            }
        }
        for (field, value, output) in [
            (
                "prompt_tokens",
                metadata.prompt_tokens,
                &mut summary.prompt_tokens,
            ),
            (
                "completion_tokens",
                metadata.completion_tokens,
                &mut summary.completion_tokens,
            ),
            (
                "total_tokens",
                metadata.total_tokens,
                &mut summary.total_tokens,
            ),
            (
                "reasoning_tokens",
                metadata.reasoning_tokens,
                &mut summary.reasoning_tokens,
            ),
        ] {
            if metadata.invalid_fields.iter().any(|name| name == field)
                || receipt.conflicts.iter().any(|name| name == field)
            {
                output.invalid_attempts += 1;
            } else if let Some(value) = value {
                output.known = output.known.and_then(|sum| sum.checked_add(value));
            } else {
                output.missing_attempts += 1;
            }
        }
        if let Some(settlement) = receipt.transport {
            summary.elapsed_ms = summary
                .elapsed_ms
                .and_then(|sum| sum.checked_add(settlement.elapsed_ms));
        } else {
            summary.unresolved_attempts += 1;
        }
    }
    Ok(summary)
}
