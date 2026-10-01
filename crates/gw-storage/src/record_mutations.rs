//! Full-envelope compare-and-swap with deterministic committed-command recognition.
use crate::{Result, StorageError, Store, now_rfc3339, record_data as data};
use gw_schema::{LifecycleState, StateTransition, TrainingRecord};
use sqlx::{Sqlite, Transaction};

/// Whether this call committed a new command or recognized the same earlier committed command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordWriteStatus {
    /// All record, projection, mutation receipt and history writes committed together.
    Applied,
    /// This exact command committed earlier. Current later progress was retained.
    AlreadyApplied,
}

/// A record write acknowledgment, including the authoritative record at acknowledgment time.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordWriteOutcome {
    /// New commit versus recognition of a prior commit.
    pub status: RecordWriteStatus,
    /// Current envelope, possibly beyond the command's target after an idempotent retry.
    pub record: TrainingRecord,
}

fn conflict(id: &str, reason: &'static str) -> StorageError {
    StorageError::RecordConflict {
        record_id: id.into(),
        reason: reason.into(),
    }
}

fn command_id(
    kind: &str,
    expected: Option<&TrainingRecord>,
    updated: &TrainingRecord,
    to: Option<LifecycleState>,
    detail: Option<&str>,
) -> Result<String> {
    let expected = expected.map(data::snapshot).transpose()?;
    crate::canonical_json_hash(&serde_json::json!({
        "encoding":"record-mutation-v1", "kind":kind, "expected":expected,
        "record":data::snapshot(updated)?, "to":to, "detail":detail
    }))
}

pub(crate) async fn remember(
    tx: &mut Transaction<'_, Sqlite>,
    record_id: &str,
    id: &str,
    kind: &str,
    history: std::ops::Range<usize>,
    at: &str,
) -> Result<()> {
    let start = i64::try_from(history.start)
        .map_err(|_| data::integrity(record_id, "history ordinal exhausted"))?;
    let count = i64::try_from(history.len())
        .map_err(|_| data::integrity(record_id, "history count exhausted"))?;
    sqlx::query("INSERT INTO record_mutations (mutation_id, record_id, version, kind, history_start, history_count, committed_at) VALUES (?, ?, 1, ?, ?, ?, ?)")
        .bind(id).bind(record_id).bind(kind).bind(start).bind(count).bind(at).execute(&mut **tx).await?;
    Ok(())
}

pub(crate) fn append(
    record: &mut TrainingRecord,
    to: LifecycleState,
    detail: Option<&str>,
    at: &str,
) -> Result<()> {
    record.lifecycle.attempts =
        record.lifecycle.attempts.checked_add(1).ok_or_else(|| {
            data::integrity(&record.record_id, "lifecycle attempt counter exhausted")
        })?;
    record.lifecycle.state = to;
    record.lifecycle.error = if to == LifecycleState::Error {
        detail.map(str::to_owned)
    } else {
        None
    };
    record.lifecycle.history.push(StateTransition {
        state: to,
        at: at.into(),
        attempt: record.lifecycle.attempts,
    });
    Ok(())
}

impl Store {
    /// Insert a new generated record, or recognize this exact normalized initial command.
    /// A compatible retry returns the current record without overwriting any later progress.
    /// Supplied initial history is mirrored once, including multiple generation facts at attempt 0.
    /// Caller-supplied initial timestamps are part of this command; transaction timestamps are not.
    ///
    /// # Errors
    /// Returns [`StorageError::RecordConflict`] for incompatible reuse of an existing record ID,
    /// or a storage/serialization failure. A failed acknowledgment does not imply rollback.
    pub async fn insert_record(&self, record: &TrainingRecord) -> Result<RecordWriteOutcome> {
        self.insert_command(record, None, None).await
    }

    /// Atomically insert a missing record and append its transition, for pre-generation error stubs.
    /// Retrying the same command recognizes the earlier commit before testing whether the ID exists.
    ///
    /// # Errors
    /// Returns an identity conflict or storage/serialization failure. Transition legality belongs
    /// to the engine; this operation cannot replace an existing incompatible record.
    pub async fn insert_record_and_transition(
        &self,
        record: &TrainingRecord,
        to: LifecycleState,
        detail: Option<&str>,
    ) -> Result<RecordWriteOutcome> {
        self.insert_command(record, Some(to), detail).await
    }

    async fn insert_command(
        &self,
        record: &TrainingRecord,
        to: Option<LifecycleState>,
        detail: Option<&str>,
    ) -> Result<RecordWriteOutcome> {
        if record.origin.generated().is_none() {
            return Err(conflict(
                &record.record_id,
                "references require complete registered batch import",
            ));
        }
        let mut stored = data::normalize(record)?;
        let kind = if to.is_some() {
            "insert_transition"
        } else {
            "insert"
        };
        let id = command_id(kind, None, &stored, to, detail)?;
        let mut tx = self.pool().begin_with("BEGIN IMMEDIATE").await?;
        data::generated_partition(&mut tx, &stored).await?;
        if let Some(outcome) = already_applied(&mut tx, &stored.record_id, &id).await? {
            tx.commit().await?;
            return Ok(outcome);
        }
        if data::load(&mut tx, &stored.record_id).await?.is_some() {
            return Err(conflict(
                &stored.record_id,
                "record ID belongs to a different initial command",
            ));
        }
        let at = now_rfc3339();
        if let Some(to) = to {
            append(&mut stored, to, detail, &at)?;
        }
        data::write(&mut tx, &stored, &at).await?;
        #[cfg(test)]
        self.test_boundary("record_insert", "projected").await?;
        remember(
            &mut tx,
            &stored.record_id,
            &id,
            kind,
            0..stored.lifecycle.history.len(),
            &at,
        )
        .await?;
        data::history(&mut tx, &stored, 0, Some(&id), detail).await?;
        #[cfg(test)]
        self.test_boundary("record_insert", "history").await?;
        #[cfg(test)]
        self.test_boundary("record_insert", "precommit").await?;
        tx.commit().await?;
        #[cfg(test)]
        self.test_boundary("record_insert", "committed").await?;
        Ok(RecordWriteOutcome {
            status: RecordWriteStatus::Applied,
            record: stored,
        })
    }

    /// Compare the complete expected envelope (including provenance, facts, cost and history),
    /// replace non-lifecycle fields, normalize hashes/projections, and append exactly one transition.
    /// A previously committed deterministic command returns [`RecordWriteStatus::AlreadyApplied`]
    /// before comparing its now-stale snapshot, even after later publication or other progress.
    ///
    /// `updated` must retain the expected record/run identity and lifecycle. The store owns history
    /// append and timestamps; the engine owns legal state-machine edges. Legacy history rows remain
    /// unchanged, while each new transition has a unique envelope-history ordinal and mutation ID.
    ///
    /// # Errors
    /// Returns [`StorageError::RecordConflict`] for a stale snapshot or changed identity/history,
    /// [`StorageError::NotFound`] for a missing expected record, or a storage/serialization error.
    pub async fn transition_record(
        &self,
        expected: &TrainingRecord,
        updated: &TrainingRecord,
        to: LifecycleState,
        detail: Option<&str>,
    ) -> Result<RecordWriteOutcome> {
        if expected.record_id != updated.record_id
            || expected.run_id() != updated.run_id()
            || expected.lifecycle != updated.lifecycle
        {
            return Err(conflict(
                &expected.record_id,
                "command changed record identity or authoritative lifecycle",
            ));
        }
        if (expected.origin.generated().is_none() || updated.origin.generated().is_none())
            && expected != updated
        {
            return Err(conflict(
                &expected.record_id,
                "reference content and origin are immutable",
            ));
        }
        if expected.origin.generated().is_none()
            && !matches!(
                to,
                LifecycleState::Formatted | LifecycleState::Rejected | LifecycleState::Error
            )
        {
            return Err(conflict(
                &expected.record_id,
                "reference lifecycle requires truthful formatting, exclusion, or reference publication acknowledgment",
            ));
        }
        let mut stored = data::normalize(updated)?;
        let id = command_id("transition", Some(expected), &stored, Some(to), detail)?;
        let mut tx = self.pool().begin_with("BEGIN IMMEDIATE").await?;
        if let Some(outcome) = already_applied(&mut tx, &stored.record_id, &id).await? {
            tx.commit().await?;
            return Ok(outcome);
        }
        let current = data::load(&mut tx, &stored.record_id)
            .await?
            .ok_or_else(|| StorageError::NotFound(format!("record {}", stored.record_id)))?;
        data::check_history(&mut tx, &current).await?;
        crate::reference_records::eligible(&mut tx, &current).await?;
        if data::snapshot(&current)? != data::snapshot(expected)? {
            return Err(conflict(
                &stored.record_id,
                "complete expected snapshot is stale",
            ));
        }
        let start = current.lifecycle.history.len();
        stored.lifecycle = current.lifecycle;
        let at = now_rfc3339();
        append(&mut stored, to, detail, &at)?;
        data::write(&mut tx, &stored, &at).await?;
        #[cfg(test)]
        self.test_boundary("record_transition", "projected").await?;
        remember(
            &mut tx,
            &stored.record_id,
            &id,
            "transition",
            start..stored.lifecycle.history.len(),
            &at,
        )
        .await?;
        data::history(&mut tx, &stored, start, Some(&id), detail).await?;
        #[cfg(test)]
        self.test_boundary("record_transition", "history").await?;
        #[cfg(test)]
        self.test_boundary("record_transition", "precommit").await?;
        tx.commit().await?;
        #[cfg(test)]
        self.test_boundary("record_transition", "committed").await?;
        Ok(RecordWriteOutcome {
            status: RecordWriteStatus::Applied,
            record: stored,
        })
    }

    /// Append a transition without changing other fields, guarded by the complete expected record.
    ///
    /// # Errors
    /// Has the same conflicts and acknowledgment semantics as [`Self::transition_record`].
    pub async fn advance_lifecycle(
        &self,
        expected: &TrainingRecord,
        to: LifecycleState,
        detail: Option<&str>,
    ) -> Result<RecordWriteOutcome> {
        self.transition_record(expected, expected, to, detail).await
    }
}

async fn already_applied(
    tx: &mut Transaction<'_, Sqlite>,
    record_id: &str,
    id: &str,
) -> Result<Option<RecordWriteOutcome>> {
    let applied: bool = sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM record_mutations WHERE mutation_id=? AND record_id=? AND version=1)")
        .bind(id).bind(record_id).fetch_one(&mut **tx).await?;
    if !applied {
        return Ok(None);
    }
    let record = data::load(tx, record_id)
        .await?
        .ok_or_else(|| data::integrity(record_id, "committed mutation lost its record"))?;
    data::check_history(tx, &record).await?;
    Ok(Some(RecordWriteOutcome {
        status: RecordWriteStatus::AlreadyApplied,
        record,
    }))
}
