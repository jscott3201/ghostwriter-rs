//! Per-(run, shard) checkpointing + crash recovery (ARCHITECTURE §5, DATA-SCHEMA §6.2; INVARIANT 4).
//!
//! A shard is the unit of checkpointing. After a shard commits a seed item (its group finished
//! driving), the worker persists a RESUME CURSOR to the gw-storage run-ledger: the next un-committed
//! seed offset. On relaunch the shard reads the cursor back and SKIPS the already-committed items,
//! resuming from the first un-committed offset.
//!
//! ## Two layers of resume — and why both are needed
//!
//! 1. **Coarse (this module)** — the shard cursor (`next_offset`) lets a relaunch skip whole seed
//!    items that already fully committed, so it doesn't even re-query their (cached) producer calls.
//! 2. **Fine (the record lifecycle)** — for an item that was MID-FLIGHT when the crash hit (a record
//!    persisted at, say, `AssistantGenerated` but not yet `Judged`), the per-record `crate::step`
//!    machine re-enters at the record's LAST persisted state — NOT from scratch — because `step`
//!    reads `lifecycle.state` and the teacher call is content-hash cached. So even an item BELOW the
//!    cursor that crashed mid-drive resumes correctly: the cursor only advances AFTER an item's group
//!    fully terminates, so a mid-flight item is always re-driven (idempotently) on relaunch.
//!
//! The cursor is stored as the opaque `cursor` JSON of a [`ResumePoint`](gw_storage::ResumePoint); the
//! `state` field carries the shard's coarse furthest state for the run log.

use gw_storage::Store;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// The opaque resume cursor persisted per (run, shard): the next un-committed seed offset. A fresh
/// shard has no checkpoint → resumes at offset 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardCursor {
    /// The 0-based offset of the first seed item NOT yet committed for this shard. On relaunch the
    /// shard skips items `[0, next_offset)` and resumes at `next_offset`.
    pub next_offset: u64,
}

impl ShardCursor {
    /// A cursor at the start of the shard (nothing committed yet).
    #[must_use]
    pub fn start() -> Self {
        Self { next_offset: 0 }
    }
}

/// Read the resume cursor for `(run_id, shard)` — the offset to resume from. A shard that never
/// checkpointed resumes at offset 0 ([`ShardCursor::start`]).
///
/// # Errors
/// Returns [`EngineError::Storage`](crate::EngineError::Storage) on a SQL fault, or
/// [`EngineError::Serde`](crate::EngineError::Serde) if a stored cursor fails to parse.
pub async fn load_cursor(store: &Store, run_id: &str, shard: i64) -> Result<ShardCursor> {
    match store.resume_cursor(run_id, shard).await? {
        Some(point) => Ok(serde_json::from_value(point.cursor)?),
        None => Ok(ShardCursor::start()),
    }
}

/// Persist the resume cursor for `(run_id, shard)` after committing seed item at `committed_offset`:
/// the new cursor points at `committed_offset + 1` (the next un-committed item). The `state` string is
/// the shard's coarse furthest lifecycle state, recorded for the run log.
///
/// Called AFTER an item's best-of-k group fully terminates, so the cursor only advances past items
/// that are durably done — a mid-flight item is always re-driven on relaunch.
///
/// # Errors
/// Returns [`EngineError::Storage`](crate::EngineError::Storage) on a SQL fault, or
/// [`EngineError::Serde`](crate::EngineError::Serde) if the cursor fails to serialize.
pub async fn commit_cursor(
    store: &Store,
    run_id: &str,
    shard: i64,
    committed_offset: u64,
    state: &str,
) -> Result<()> {
    let cursor = ShardCursor {
        next_offset: committed_offset.saturating_add(1),
    };
    let value = serde_json::to_value(cursor)?;
    store.checkpoint(run_id, shard, state, &value).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fresh_shard_resumes_at_zero() {
        let store = Store::open_in_memory().await.unwrap();
        store.create_run("run-1", "{}", Some(25.0)).await.unwrap();
        let cursor = load_cursor(&store, "run-1", 0).await.unwrap();
        assert_eq!(cursor, ShardCursor::start());
        assert_eq!(cursor.next_offset, 0);
    }

    #[tokio::test]
    async fn commit_advances_the_cursor_past_the_committed_offset() {
        let store = Store::open_in_memory().await.unwrap();
        store.create_run("run-1", "{}", Some(25.0)).await.unwrap();

        // Commit offset 0 → cursor points at 1.
        commit_cursor(&store, "run-1", 0, 0, "exported")
            .await
            .unwrap();
        assert_eq!(
            load_cursor(&store, "run-1", 0).await.unwrap().next_offset,
            1
        );

        // Commit offset 3 → cursor points at 4 (last-write-wins on the shard checkpoint row).
        commit_cursor(&store, "run-1", 0, 3, "exported")
            .await
            .unwrap();
        assert_eq!(
            load_cursor(&store, "run-1", 0).await.unwrap().next_offset,
            4
        );
    }

    #[tokio::test]
    async fn cursors_are_per_shard() {
        let store = Store::open_in_memory().await.unwrap();
        store.create_run("run-1", "{}", Some(25.0)).await.unwrap();
        commit_cursor(&store, "run-1", 0, 5, "exported")
            .await
            .unwrap();
        commit_cursor(&store, "run-1", 1, 2, "exported")
            .await
            .unwrap();
        assert_eq!(
            load_cursor(&store, "run-1", 0).await.unwrap().next_offset,
            6
        );
        assert_eq!(
            load_cursor(&store, "run-1", 1).await.unwrap().next_offset,
            3
        );
        // A never-checkpointed shard is still at the start.
        assert_eq!(
            load_cursor(&store, "run-1", 2).await.unwrap().next_offset,
            0
        );
    }
}
