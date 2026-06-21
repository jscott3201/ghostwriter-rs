//! The run-ledger: run rows + per-shard checkpoints for crash recovery (DATA-SCHEMA §6.2).
//!
//! A *run* is the top-level unit of work; a *shard* is the unit of parallelism and of
//! checkpointing. After a shard commits a batch of records at some lifecycle state, the worker
//! persists a resume cursor here; on relaunch it reads the cursor back and resumes from the
//! furthest committed state instead of re-calling the (expensive) teacher.

use serde_json::Value;

use crate::error::{Result, StorageError};
use crate::store::{Store, now_rfc3339};

/// A run's coarse status in the ledger. Stored as its snake_case wire string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStatus {
    /// Accepting / processing work.
    Running,
    /// Finished cleanly (all shards drained).
    Completed,
    /// Halted on budget breach or operator stop.
    Halted,
    /// Terminated by an unrecoverable error.
    Failed,
}

impl RunStatus {
    /// The stored wire spelling.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            RunStatus::Running => "running",
            RunStatus::Completed => "completed",
            RunStatus::Halted => "halted",
            RunStatus::Failed => "failed",
        }
    }
}

/// A persisted resume point for one shard: the furthest committed lifecycle `state` (its wire
/// string) plus the opaque `cursor` the worker uses to resume (DATA-SCHEMA §6.2).
#[derive(Debug, Clone, PartialEq)]
pub struct ResumePoint {
    /// The furthest committed lifecycle state for the shard (snake_case wire string).
    pub state: String,
    /// The opaque resume cursor (e.g. `{"seed_offset": 4096}`), interpreted by the orchestrator.
    pub cursor: Value,
}

impl Store {
    /// Validate or record the run's seed partition manifest.
    ///
    /// The first launch of a run stamps the effective `shard_count` and ordered `prompts_hash`.
    /// Later launches of the same `run_id` must present the same values before any resume cursor is
    /// trusted; otherwise `(run_id, shard).next_offset` would refer to a different partition. Legacy
    /// rows with a missing manifest are rejected instead of guessed.
    ///
    /// This read/conditional-write runs under `BEGIN IMMEDIATE`, so two concurrent launches of the
    /// same run cannot race between checking and recording the manifest.
    ///
    /// # Errors
    /// Returns [`StorageError::RunPartitionMismatch`] when the existing manifest is missing or
    /// divergent, or [`StorageError::Sqlx`] on a SQL fault.
    pub async fn validate_or_record_run_partition(
        &self,
        run_id: &str,
        shard_count: usize,
        prompts_hash: &str,
    ) -> Result<()> {
        let shard_count = shard_count as i64;
        let mut tx = self.pool().begin_with("BEGIN IMMEDIATE").await?;
        let row: Option<(Option<i64>, Option<String>)> =
            sqlx::query_as("SELECT shard_count, prompts_hash FROM runs WHERE run_id = ?1")
                .bind(run_id)
                .fetch_optional(&mut *tx)
                .await?;

        match row {
            None => {
                sqlx::query(
                    "INSERT INTO runs \
                     (run_id, config_json, budget_usd, status, created_at, shard_count, prompts_hash) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                )
                .bind(run_id)
                .bind("{}")
                .bind(Option::<f64>::None)
                .bind(RunStatus::Running.as_str())
                .bind(now_rfc3339())
                .bind(shard_count)
                .bind(prompts_hash)
                .execute(&mut *tx)
                .await?;
            }
            Some((Some(existing_count), Some(existing_hash)))
                if existing_count == shard_count && existing_hash == prompts_hash => {}
            Some((existing_shard_count, existing_prompts_hash)) => {
                tx.rollback().await?;
                return Err(StorageError::RunPartitionMismatch {
                    run_id: run_id.to_string(),
                    existing_shard_count: existing_shard_count
                        .map(|n| n.to_string())
                        .unwrap_or_else(|| "<missing>".to_string()),
                    existing_prompts_hash: existing_prompts_hash
                        .unwrap_or_else(|| "<missing>".to_string()),
                    actual_shard_count: shard_count,
                    actual_prompts_hash: prompts_hash.to_string(),
                });
            }
        }

        tx.commit().await?;
        Ok(())
    }

    /// Create a run row. `config_json` is a snapshot of the [`gw_schema::Config`] in force (or any
    /// JSON the caller wants to pin); `budget_usd` mirrors the budget cap for audit.
    ///
    /// Idempotent by `run_id`: re-creating an existing run overwrites the snapshot and resets
    /// status to `running` (a relaunch of the same run id). It deliberately leaves the run partition
    /// manifest columns untouched; callers must validate them before calling this on a resume.
    ///
    /// # Errors
    /// Returns [`StorageError`] on a SQL fault.
    pub async fn create_run(
        &self,
        run_id: &str,
        config_json: &str,
        budget_usd: Option<f64>,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO runs (run_id, config_json, budget_usd, status, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5) \
             ON CONFLICT(run_id) DO UPDATE SET \
              config_json = excluded.config_json, \
              budget_usd = excluded.budget_usd, \
              status = excluded.status",
        )
        .bind(run_id)
        .bind(config_json)
        .bind(budget_usd)
        .bind(RunStatus::Running.as_str())
        .bind(now_rfc3339())
        .execute(self.pool())
        .await?;
        Ok(())
    }

    /// Update a run's status.
    ///
    /// # Errors
    /// Returns [`StorageError`] on a SQL fault. A missing `run_id` is a
    /// silent no-op (zero rows affected), matching SQLite `UPDATE` semantics.
    pub async fn set_run_status(&self, run_id: &str, status: RunStatus) -> Result<()> {
        sqlx::query("UPDATE runs SET status = ?1 WHERE run_id = ?2")
            .bind(status.as_str())
            .bind(run_id)
            .execute(self.pool())
            .await?;
        Ok(())
    }

    /// Read a run's status string, if the run exists.
    ///
    /// # Errors
    /// Returns [`StorageError`] on a SQL fault.
    pub async fn run_status(&self, run_id: &str) -> Result<Option<String>> {
        let row: Option<(String,)> = sqlx::query_as("SELECT status FROM runs WHERE run_id = ?1")
            .bind(run_id)
            .fetch_optional(self.pool())
            .await?;
        Ok(row.map(|(s,)| s))
    }

    /// Write (or replace) the resume checkpoint for `(run_id, shard)`: the furthest committed
    /// `state` and the opaque `cursor`. Called after a shard commits a state batch.
    ///
    /// # Errors
    /// Returns [`StorageError`] on a SQL fault or if `cursor` fails to
    /// serialize.
    pub async fn checkpoint(
        &self,
        run_id: &str,
        shard: i64,
        state: &str,
        cursor: &Value,
    ) -> Result<()> {
        let cursor_json = serde_json::to_string(cursor)?;
        sqlx::query(
            "INSERT INTO checkpoints (run_id, shard, state, cursor_json, updated_at) \
             VALUES (?1, ?2, ?3, ?4, ?5) \
             ON CONFLICT(run_id, shard) DO UPDATE SET \
              state = excluded.state, \
              cursor_json = excluded.cursor_json, \
              updated_at = excluded.updated_at",
        )
        .bind(run_id)
        .bind(shard)
        .bind(state)
        .bind(cursor_json)
        .bind(now_rfc3339())
        .execute(self.pool())
        .await?;
        Ok(())
    }

    /// Read back the resume point for `(run_id, shard)`, or `None` if the shard has never
    /// checkpointed (a fresh shard starts from the beginning).
    ///
    /// # Errors
    /// Returns [`StorageError`] on a SQL fault or if the stored cursor fails
    /// to parse as JSON.
    pub async fn resume_cursor(&self, run_id: &str, shard: i64) -> Result<Option<ResumePoint>> {
        let row: Option<(String, String)> = sqlx::query_as(
            "SELECT state, cursor_json FROM checkpoints WHERE run_id = ?1 AND shard = ?2",
        )
        .bind(run_id)
        .bind(shard)
        .fetch_optional(self.pool())
        .await?;
        match row {
            Some((state, cursor_json)) => Ok(Some(ResumePoint {
                state,
                cursor: serde_json::from_str(&cursor_json)?,
            })),
            None => Ok(None),
        }
    }
}
