//! Shared envelope normalization and indexed projection checks for every record writer.
use crate::{Result, StorageError};
use gw_schema::{LifecycleState, TrainingRecord, Verdict};
use sqlx::{Row, Sqlite, Transaction, sqlite::SqliteRow};

pub(crate) const COLUMNS: &str = "record_id, run_id, lifecycle_state, verdict, judge_aggregate, record_hash, prompt_hash, record_json";

pub(crate) fn state_str(state: LifecycleState) -> String {
    serde_json::to_value(state)
        .expect("enum serialization")
        .as_str()
        .expect("enum string")
        .into()
}

pub(crate) fn verdict_str(verdict: Verdict) -> String {
    serde_json::to_value(verdict)
        .expect("enum serialization")
        .as_str()
        .expect("enum string")
        .into()
}

pub(crate) fn integrity(id: &str, reason: &'static str) -> StorageError {
    StorageError::RecordIntegrity {
        record_id: id.into(),
        reason: reason.into(),
    }
}

pub(crate) fn snapshot(record: &TrainingRecord) -> Result<serde_json::Value> {
    let value = serde_json::to_value(record)?;
    // serde_json maps nonfinite floats to null. Reject any lossy envelope instead of acknowledging
    // a different value; custom schema serializers may reject earlier with their own serde error.
    let roundtrip: TrainingRecord = serde_json::from_value(value.clone())?;
    if roundtrip != *record {
        return Err(integrity(
            &record.record_id,
            "envelope cannot be serialized without changing data",
        ));
    }
    Ok(value)
}

pub(crate) fn normalize(record: &TrainingRecord) -> Result<TrainingRecord> {
    snapshot(record)?;
    let mut stored = record.clone();
    stored.hashes.record_hash = crate::record_hash(record)?;
    stored.hashes.prompt_hash = crate::prompt_hash(&record.messages)?;
    stored.hashes.completion_hash = crate::completion_hash(&record.messages)?;
    Ok(stored)
}

pub(crate) fn decode(row: &SqliteRow) -> Result<TrainingRecord> {
    let json: String = row.try_get("record_json")?;
    let record: TrainingRecord = serde_json::from_str(&json)?;
    let normalized = normalize(&record)?;
    if record != normalized
        || row.try_get::<String, _>("record_id")? != record.record_id
        || row.try_get::<String, _>("run_id")? != record.provenance.run_id
        || row.try_get::<String, _>("lifecycle_state")? != state_str(record.lifecycle.state)
        || row.try_get::<Option<String>, _>("verdict")? != record.judging.verdict.map(verdict_str)
        || row.try_get::<Option<f64>, _>("judge_aggregate")? != record.judging.aggregate
        || row.try_get::<String, _>("record_hash")? != record.hashes.record_hash
        || row.try_get::<String, _>("prompt_hash")? != record.hashes.prompt_hash
    {
        return Err(integrity(
            &record.record_id,
            "stored envelope, partition, hashes or projections disagree",
        ));
    }
    Ok(record)
}

pub(crate) async fn load(
    tx: &mut Transaction<'_, Sqlite>,
    id: &str,
) -> Result<Option<TrainingRecord>> {
    let row = sqlx::query(sqlx::AssertSqlSafe(format!(
        "SELECT {COLUMNS} FROM records WHERE record_id = ?"
    )))
    .bind(id)
    .fetch_optional(&mut **tx)
    .await?;
    row.as_ref().map(decode).transpose()
}

/// Caller has validated the snapshot or explicitly selected fixture/import replacement.
pub(crate) async fn write(
    tx: &mut Transaction<'_, Sqlite>,
    record: &TrainingRecord,
    at: &str,
) -> Result<()> {
    sqlx::query("INSERT INTO records (record_id, run_id, lifecycle_state, verdict, judge_aggregate, record_hash, prompt_hash, record_json, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(record_id) DO UPDATE SET run_id=excluded.run_id, lifecycle_state=excluded.lifecycle_state, verdict=excluded.verdict, judge_aggregate=excluded.judge_aggregate, record_hash=excluded.record_hash, prompt_hash=excluded.prompt_hash, record_json=excluded.record_json, updated_at=excluded.updated_at")
        .bind(&record.record_id).bind(&record.provenance.run_id).bind(state_str(record.lifecycle.state))
        .bind(record.judging.verdict.map(verdict_str)).bind(record.judging.aggregate)
        .bind(&record.hashes.record_hash).bind(&record.hashes.prompt_hash)
        .bind(serde_json::to_string(record)?).bind(at).execute(&mut **tx).await?;
    Ok(())
}

pub(crate) async fn history(
    tx: &mut Transaction<'_, Sqlite>,
    record: &TrainingRecord,
    start: usize,
    mutation_id: Option<&str>,
    detail: Option<&str>,
) -> Result<()> {
    for (ordinal, transition) in record.lifecycle.history.iter().enumerate().skip(start) {
        sqlx::query("INSERT INTO lifecycle_history (record_id, state, at, detail, mutation_id, history_ordinal, attempt) VALUES (?, ?, ?, ?, ?, ?, ?)")
            .bind(&record.record_id).bind(state_str(transition.state)).bind(&transition.at)
            .bind(if ordinal + 1 == record.lifecycle.history.len() { detail } else { None })
            .bind(mutation_id).bind(i64::try_from(ordinal).map_err(|_| integrity(&record.record_id, "history ordinal exhausted"))?)
            .bind(i64::from(transition.attempt)).execute(&mut **tx).await?;
    }
    Ok(())
}

pub(crate) async fn check_history(
    tx: &mut Transaction<'_, Sqlite>,
    record: &TrainingRecord,
) -> Result<()> {
    type HistoryRange = (i64, i64, i64, Option<i64>, Option<i64>);
    let ranges: Vec<HistoryRange> = sqlx::query_as("SELECT m.history_start, m.history_count, count(h.id), min(h.history_ordinal), max(h.history_ordinal) FROM record_mutations m LEFT JOIN lifecycle_history h ON h.mutation_id=m.mutation_id AND h.record_id=m.record_id WHERE m.record_id=? GROUP BY m.mutation_id")
        .bind(&record.record_id).fetch_all(&mut **tx).await?;
    for (start, expected, actual, first, last) in ranges {
        if expected != actual
            || (expected > 0 && (first != Some(start) || last != start.checked_add(expected - 1)))
        {
            return Err(integrity(
                &record.record_id,
                "committed mutation history is incomplete",
            ));
        }
    }
    let rows: Vec<(i64, String, String, Option<i64>)> = sqlx::query_as("SELECT history_ordinal, state, at, attempt FROM lifecycle_history WHERE record_id=? AND history_ordinal IS NOT NULL ORDER BY history_ordinal")
        .bind(&record.record_id).fetch_all(&mut **tx).await?;
    for (ordinal, state, at, attempt) in rows {
        let transition = usize::try_from(ordinal)
            .ok()
            .and_then(|index| record.lifecycle.history.get(index));
        if transition.is_none_or(|item| {
            state != state_str(item.state)
                || at != item.at
                || attempt != Some(i64::from(item.attempt))
        }) {
            return Err(integrity(
                &record.record_id,
                "relational history disagrees with the envelope",
            ));
        }
    }
    Ok(())
}

/// Capture every row of each declared run in one caller-owned transaction, before eligibility.
pub(crate) async fn screening_population(
    tx: &mut Transaction<'_, Sqlite>,
    runs: &[String],
) -> Result<Vec<TrainingRecord>> {
    let mut records = Vec::new();
    for run in runs {
        let rows = sqlx::query(sqlx::AssertSqlSafe(format!(
            "SELECT {COLUMNS} FROM records WHERE run_id = ? ORDER BY record_id"
        )))
        .bind(run)
        .fetch_all(&mut **tx)
        .await?;
        for row in rows {
            let record = decode(&row)?;
            check_history(tx, &record).await?;
            records.push(record);
        }
    }
    records.sort_by(|a, b| {
        (&a.provenance.run_id, &a.record_id).cmp(&(&b.provenance.run_id, &b.record_id))
    });
    Ok(records)
}
