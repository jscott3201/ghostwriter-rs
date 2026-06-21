//! Record + lifecycle operations on the [`Store`]: idempotent `put`, `get`, filtered `scan`,
//! and the transactional `advance_lifecycle`.
//!
//! `put` is an UPSERT keyed by `record_id` ([INVARIANT d]): re-processing a seed must update the
//! row in place, never duplicate it. The indexed projection columns (`verdict`,
//! `judge_aggregate`, `record_hash`, `prompt_hash`, `lifecycle_state`) are derived from the
//! envelope on every write so the columnar filters in [`RecordFilter`] stay in sync with the
//! authoritative `record_json`.

use futures::stream::{Stream, TryStreamExt};
use gw_schema::{LifecycleState, StateTransition, TrainingRecord, Verdict};
use sqlx::Row;
use sqlx::sqlite::SqliteRow;

use crate::error::{Result, StorageError};
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

/// Serialize a [`LifecycleState`] to its snake_case wire spelling (the stored value).
fn state_str(state: LifecycleState) -> String {
    serde_json::to_value(state)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

/// Serialize a [`Verdict`] to its snake_case wire spelling (the stored value).
fn verdict_str(verdict: Verdict) -> String {
    serde_json::to_value(verdict)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

impl Store {
    /// Idempotently UPSERT a record by `record_id` ([INVARIANT d]).
    ///
    /// The full envelope is stored as JSON in `record_json`; `lifecycle_state`, `verdict`,
    /// `judge_aggregate`, `record_hash`, and `prompt_hash` are projected out for indexed
    /// filtering and dedup. Hashes (`record_hash` / `prompt_hash` / `completion_hash`) are read
    /// from the record's own [`gw_schema::Hashes`] when set, else computed from the envelope.
    /// The computed hashes are written BACK into the stored `record_json` so the persisted
    /// envelope and the indexed projection columns agree (a later `get`/`scan`/export sees a
    /// populated `hashes.record_hash`). Re-`put`-ting the same id overwrites the row — one row.
    ///
    /// # Errors
    /// Returns [`StorageError`] on a SQL fault, a serialization failure, or
    /// if the record's `provenance.run_id` does not reference an existing run (foreign key).
    pub async fn put(&self, rec: &TrainingRecord) -> Result<()> {
        // Compute-if-empty, then persist the hashes inside the stored envelope so the JSON and
        // the indexed columns never disagree.
        let mut stored = rec.clone();
        if stored.hashes.record_hash.is_empty() {
            stored.hashes.record_hash = crate::cache::record_hash(rec)?;
        }
        if stored.hashes.prompt_hash.is_empty() {
            stored.hashes.prompt_hash = crate::cache::prompt_hash(&rec.messages)?;
        }
        if stored.hashes.completion_hash.is_empty() {
            stored.hashes.completion_hash = crate::cache::completion_hash(&rec.messages)?;
        }
        let record_hash = stored.hashes.record_hash.clone();
        let prompt_hash = stored.hashes.prompt_hash.clone();
        let record_json = serde_json::to_string(&stored)?;
        let verdict = rec.judging.verdict.map(verdict_str);

        sqlx::query(
            "INSERT INTO records \
             (record_id, run_id, lifecycle_state, verdict, judge_aggregate, \
              record_hash, prompt_hash, record_json, updated_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9) \
             ON CONFLICT(record_id) DO UPDATE SET \
              run_id = excluded.run_id, \
              lifecycle_state = excluded.lifecycle_state, \
              verdict = excluded.verdict, \
              judge_aggregate = excluded.judge_aggregate, \
              record_hash = excluded.record_hash, \
              prompt_hash = excluded.prompt_hash, \
              record_json = excluded.record_json, \
              updated_at = excluded.updated_at",
        )
        .bind(&rec.record_id)
        .bind(&rec.provenance.run_id)
        .bind(state_str(rec.lifecycle.state))
        .bind(verdict)
        .bind(rec.judging.aggregate)
        .bind(record_hash)
        .bind(prompt_hash)
        .bind(record_json)
        .bind(now_rfc3339())
        .execute(self.pool())
        .await?;
        Ok(())
    }

    /// Fetch one record by id, deserializing the stored envelope.
    ///
    /// # Errors
    /// Returns [`StorageError::NotFound`] if no row matches, or a
    /// SQL/serde error on failure.
    pub async fn get(&self, record_id: &str) -> Result<TrainingRecord> {
        let row: Option<(String,)> =
            sqlx::query_as("SELECT record_json FROM records WHERE record_id = ?1")
                .bind(record_id)
                .fetch_optional(self.pool())
                .await?;
        match row {
            Some((json,)) => Ok(serde_json::from_str(&json)?),
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
        let mut sql = String::from("SELECT record_json FROM records");
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
            .and_then(|row: SqliteRow| async move {
                let json: String = row.try_get("record_json")?;
                Ok(serde_json::from_str(&json)?)
            })
    }

    /// Advance a record to `new_state`, keeping the stored envelope, the `lifecycle_state`
    /// column, and the `lifecycle_history` table all in sync in ONE transaction (event-sourced,
    /// DATA-SCHEMA §6.1).
    ///
    /// Within the transaction this: reads `record_json`, sets `lifecycle.state = new_state`,
    /// pushes a [`gw_schema::StateTransition`] onto `lifecycle.history` (so the envelope's own
    /// history grows alongside the `lifecycle_history` table), bumps `lifecycle.attempts`, sets
    /// `lifecycle.error` from `detail` when `new_state` is [`LifecycleState::Error`] (else clears
    /// it), re-serializes the envelope into `record_json`, updates the `lifecycle_state` column,
    /// and appends one `lifecycle_history` row. Either all writes land or none do.
    ///
    /// This is an UNCHECKED persistence primitive: it does NOT validate that the transition is
    /// legal. Enforcing the legal state-machine edges is the orchestrator's job; this method just
    /// records whatever transition it is told to.
    ///
    /// # Errors
    /// Returns [`StorageError::NotFound`] if `record_id` does not exist; a SQL or serde error
    /// rolls the transaction back.
    pub async fn advance_lifecycle(
        &self,
        record_id: &str,
        new_state: LifecycleState,
        detail: Option<&str>,
    ) -> Result<()> {
        let state = state_str(new_state);
        let at = now_rfc3339();
        let mut tx = self.pool().begin().await?;

        // Read the envelope inside the txn so the JSON we rewrite reflects the current row.
        let row: Option<(String,)> =
            sqlx::query_as("SELECT record_json FROM records WHERE record_id = ?1")
                .bind(record_id)
                .fetch_optional(&mut *tx)
                .await?;
        let Some((json,)) = row else {
            tx.rollback().await?;
            return Err(StorageError::NotFound(format!("record {record_id}")));
        };

        let mut rec: TrainingRecord = serde_json::from_str(&json)?;
        let attempt = rec.lifecycle.attempts.saturating_add(1);
        rec.lifecycle.state = new_state;
        rec.lifecycle.attempts = attempt;
        rec.lifecycle.error = if new_state == LifecycleState::Error {
            detail.map(str::to_owned)
        } else {
            None
        };
        rec.lifecycle.history.push(StateTransition {
            state: new_state,
            at: at.clone(),
            attempt,
        });
        let new_json = serde_json::to_string(&rec)?;

        sqlx::query(
            "UPDATE records SET lifecycle_state = ?1, record_json = ?2, updated_at = ?3 \
             WHERE record_id = ?4",
        )
        .bind(&state)
        .bind(&new_json)
        .bind(&at)
        .bind(record_id)
        .execute(&mut *tx)
        .await?;

        sqlx::query(
            "INSERT INTO lifecycle_history (record_id, state, at, detail) \
             VALUES (?1, ?2, ?3, ?4)",
        )
        .bind(record_id)
        .bind(&state)
        .bind(&at)
        .bind(detail)
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;
        Ok(())
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
