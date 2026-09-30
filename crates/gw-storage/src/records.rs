//! Record queries and explicit fixture/import replacement. Production writes use guarded commands.

use futures::stream::{Stream, TryStreamExt};
use gw_schema::{LifecycleState, TrainingRecord, Verdict};
use sqlx::sqlite::SqliteRow;

use crate::error::{Result, StorageError};
use crate::record_data::{self as data, state_str, verdict_str};
use crate::store::{Store, now_rfc3339};

/// A predicate over the indexed projection columns, used by [`Store::scan`] /
/// [`Store::scan_stream`]. All set fields are AND-combined; an all-`None` filter matches every
/// record. Filtering happens in SQL (against the projection columns), not in Rust, so it pushes
/// down to the indexes.
#[derive(Debug, Clone, Default)]
pub struct RecordFilter {
    /// Restrict to one run.
    pub run_id: Option<String>,
    /// Restrict to one lifecycle state.
    pub lifecycle_state: Option<LifecycleState>,
    /// Restrict to one persisted envelope verdict.
    pub verdict: Option<Verdict>,
    /// Keep only records whose `judging.aggregate` is present and `>=` this floor.
    pub min_judge_aggregate: Option<f64>,
}

impl RecordFilter {
    /// An empty filter that matches every record.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Restrict to records in `run_id`.
    #[must_use]
    pub fn run_id(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = Some(run_id.into());
        self
    }

    /// Restrict to records in `state`.
    #[must_use]
    pub fn lifecycle_state(mut self, state: LifecycleState) -> Self {
        self.lifecycle_state = Some(state);
        self
    }

    /// Restrict to records with `verdict`.
    #[must_use]
    pub fn verdict(mut self, verdict: Verdict) -> Self {
        self.verdict = Some(verdict);
        self
    }

    /// Keep only records whose judge aggregate is `>= floor`.
    #[must_use]
    pub fn min_judge_aggregate(mut self, floor: f64) -> Self {
        self.min_judge_aggregate = Some(floor);
        self
    }
}

impl Store {
    /// Explicitly replace a fixture or imported record, including its relational history.
    /// This is an unguarded administrative replacement: it clears earlier mutation receipts and
    /// relational history, normalizes hashes/projections, and mirrors the supplied history atomically.
    /// Production generation and transition callers must use [`Self::insert_record`] or
    /// [`Self::transition_record`] so stale input cannot overwrite later progress.
    ///
    /// # Errors
    /// Returns a serialization/integrity or SQL error, including a missing run foreign key.
    pub async fn replace_record_for_import(&self, record: &TrainingRecord) -> Result<()> {
        let stored = data::normalize(record)?;
        let mut tx = self.pool().begin_with("BEGIN IMMEDIATE").await?;
        sqlx::query("DELETE FROM lifecycle_history WHERE record_id=?")
            .bind(&record.record_id)
            .execute(&mut *tx)
            .await?;
        sqlx::query("DELETE FROM record_mutations WHERE record_id=?")
            .bind(&record.record_id)
            .execute(&mut *tx)
            .await?;
        data::write(&mut tx, &stored, &now_rfc3339()).await?;
        data::history(&mut tx, &stored, 0, None, stored.lifecycle.error.as_deref()).await?;
        tx.commit().await?;
        Ok(())
    }

    /// Fetch one record by id, deserializing the stored envelope.
    ///
    /// # Errors
    /// Returns [`StorageError::NotFound`] if no row matches, or a
    /// SQL/serde error on failure.
    pub async fn get(&self, record_id: &str) -> Result<TrainingRecord> {
        let row = sqlx::query(sqlx::AssertSqlSafe(format!(
            "SELECT {} FROM records WHERE record_id = ?",
            data::COLUMNS
        )))
        .bind(record_id)
        .fetch_optional(self.pool())
        .await?;
        match row {
            Some(row) => data::decode(&row),
            None => Err(StorageError::NotFound(format!("record {record_id}"))),
        }
    }

    /// Scan all records matching `filter`, collected into a `Vec`.
    ///
    /// A convenience wrapper over [`scan_stream`](Self::scan_stream) for callers that want the
    /// whole result set in memory.
    ///
    /// # Errors
    /// Returns [`StorageError`] on a SQL fault or if a stored envelope fails
    /// to deserialize.
    pub async fn scan(&self, filter: &RecordFilter) -> Result<Vec<TrainingRecord>> {
        self.scan_stream(filter).try_collect().await
    }

    /// Stream records matching `filter` lazily from SQLite (one decode per yielded item).
    ///
    /// The predicate is AND-combined over the indexed projection columns and applied in SQL.
    /// The stream yields `Result<TrainingRecord>`; a decode failure surfaces as an `Err` item.
    pub fn scan_stream<'a>(
        &'a self,
        filter: &RecordFilter,
    ) -> impl Stream<Item = Result<TrainingRecord>> + 'a {
        // Build the WHERE clause with positional binds in a fixed order.
        let mut sql = format!("SELECT {} FROM records", data::COLUMNS);
        let mut clauses: Vec<&str> = Vec::new();
        if filter.run_id.is_some() {
            clauses.push("run_id = ?");
        }
        if filter.lifecycle_state.is_some() {
            clauses.push("lifecycle_state = ?");
        }
        if filter.verdict.is_some() {
            clauses.push("verdict = ?");
        }
        if filter.min_judge_aggregate.is_some() {
            // `>=` already excludes NULLs in SQLite (NULL comparisons are never true).
            clauses.push("judge_aggregate >= ?");
        }
        if !clauses.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&clauses.join(" AND "));
        }
        sql.push_str(" ORDER BY record_id");

        // `sql` is assembled solely from hardcoded clause literals above (all VALUES are bound,
        // never interpolated), so it is injection-safe — assert that for the sqlx 0.9 API.
        let mut q = sqlx::query(sqlx::AssertSqlSafe(sql));
        if let Some(run_id) = &filter.run_id {
            q = q.bind(run_id.clone());
        }
        if let Some(state) = filter.lifecycle_state {
            q = q.bind(state_str(state));
        }
        if let Some(verdict) = filter.verdict {
            q = q.bind(verdict_str(verdict));
        }
        if let Some(floor) = filter.min_judge_aggregate {
            q = q.bind(floor);
        }

        q.fetch(self.pool())
            .map_err(StorageError::from)
            .and_then(|row: SqliteRow| async move { data::decode(&row) })
    }

    /// The full ordered lifecycle history for a record: `(state, at, detail)` rows, oldest first.
    ///
    /// # Errors
    /// Returns [`StorageError`] on a SQL fault.
    pub async fn lifecycle_history(
        &self,
        record_id: &str,
    ) -> Result<Vec<(String, String, Option<String>)>> {
        let rows: Vec<(String, String, Option<String>)> = sqlx::query_as(
            "SELECT state, at, detail FROM lifecycle_history \
             WHERE record_id = ?1 ORDER BY id",
        )
        .bind(record_id)
        .fetch_all(self.pool())
        .await?;
        Ok(rows)
    }
}
