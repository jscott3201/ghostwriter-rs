//! Real files expose transaction, cancellation, and stale-writer recovery behavior.
use crate::{
    RecordWriteStatus, StorageError, Store,
    durability_support::*,
    test_hooks::{Action, Hook},
};
use gw_schema::LifecycleState;
use std::sync::Arc;

#[tokio::test]
async fn every_record_write_boundary_is_atomic_and_retry_recognizes_lost_ack() {
    for operation in ["record_insert", "record_transition"] {
        for stage in ["projected", "history", "precommit", "committed"] {
            let dir = Directory::new(&format!("{operation}-{stage}"));
            let store = initialized(&dir.db()).await;
            let expected = store.get("record").await.unwrap();
            let before = snapshot(&store).await;
            let mut updated = expected.clone();
            updated.cost.prompt_tokens = 9;
            updated.verification.all_passed = true;
            updated.judging.aggregate = Some(0.94);
            updated.judging.verdict = Some(gw_schema::Verdict::Admit);
            updated.tags.push("verified".into());
            if operation == "record_insert" {
                sqlx::query("DELETE FROM lifecycle_history")
                    .execute(store.pool())
                    .await
                    .unwrap();
                sqlx::query("DELETE FROM records")
                    .execute(store.pool())
                    .await
                    .unwrap();
            }
            store.set_test_hook(Some(Hook {
                operation,
                stage,
                action: Action::Fail,
            }));
            let result = if operation == "record_insert" {
                store.insert_record(&record()).await
            } else {
                store
                    .transition_record(&expected, &updated, LifecycleState::Verified, Some("done"))
                    .await
            };
            assert!(result.is_err());
            store.close().await;
            let reopened = Store::open(dir.db()).await.unwrap();
            integrity(&reopened).await;
            let committed = stage == "committed";
            if operation == "record_insert" {
                assert_eq!(reopened.get("record").await.is_ok(), committed);
                let outcome = reopened.insert_record(&record()).await.unwrap();
                assert_eq!(
                    outcome.status,
                    if committed {
                        RecordWriteStatus::AlreadyApplied
                    } else {
                        RecordWriteStatus::Applied
                    }
                );
                assert_eq!(reopened.lifecycle_history("record").await.unwrap().len(), 2);
            } else {
                if !committed {
                    assert_eq!(snapshot(&reopened).await, before);
                }
                let outcome = reopened
                    .transition_record(&expected, &updated, LifecycleState::Verified, Some("done"))
                    .await
                    .unwrap();
                assert_eq!(
                    outcome.status,
                    if committed {
                        RecordWriteStatus::AlreadyApplied
                    } else {
                        RecordWriteStatus::Applied
                    }
                );
                assert_eq!(outcome.record.cost.prompt_tokens, 9);
                assert_eq!(reopened.lifecycle_history("record").await.unwrap().len(), 3);
            }
            reopened.close().await;
        }
    }
}
#[tokio::test]
async fn cancellation_before_or_after_commit_preserves_atomicity_and_pool_reuse() {
    for stage in ["history", "committed"] {
        let dir = Directory::new("cancel");
        let store = initialized(&dir.db()).await;
        let before = snapshot(&store).await;
        let expected = store.get("record").await.unwrap();
        let entered = Arc::new(tokio::sync::Notify::new());
        store.set_test_hook(Some(Hook {
            operation: "record_transition",
            stage,
            action: Action::Pause {
                entered: entered.clone(),
                release: Arc::new(tokio::sync::Notify::new()),
            },
        }));
        let worker = store.clone();
        let saved = expected.clone();
        let task = tokio::spawn(async move {
            worker
                .advance_lifecycle(&saved, LifecycleState::Verified, None)
                .await
        });
        entered.notified().await;
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        store.set_test_hook(None);
        if stage == "history" {
            assert_eq!(snapshot(&store).await, before);
        }
        let retry = store
            .advance_lifecycle(&expected, LifecycleState::Verified, None)
            .await
            .unwrap();
        assert_eq!(
            retry.status,
            if stage == "committed" {
                RecordWriteStatus::AlreadyApplied
            } else {
                RecordWriteStatus::Applied
            }
        );
        store.close().await;
        let reopened = Store::open(dir.db()).await.unwrap();
        integrity(&reopened).await;
        assert_eq!(
            reopened.get("record").await.unwrap().lifecycle.state,
            LifecycleState::Verified
        );
        assert_eq!(reopened.lifecycle_history("record").await.unwrap().len(), 3);
        reopened.close().await;
    }
}
#[tokio::test]
async fn serialization_and_constraint_errors_leave_no_partial_record() {
    let dir = Directory::new("failed-write");
    let store = initialized(&dir.db()).await;
    let before = snapshot(&store).await;
    let expected = store.get("record").await.unwrap();
    let mut invalid = expected.clone();
    invalid.cost.usd = f64::NAN;
    assert!(
        store
            .transition_record(&expected, &invalid, LifecycleState::Verified, None)
            .await
            .is_err()
    );
    sqlx::query("CREATE TRIGGER refuse_history BEFORE INSERT ON lifecycle_history BEGIN SELECT RAISE(ABORT,'injected history constraint'); END").execute(store.pool()).await.unwrap();
    assert!(
        store
            .advance_lifecycle(&expected, LifecycleState::Verified, None)
            .await
            .is_err()
    );
    assert_eq!(snapshot(&store).await, before);
    sqlx::query("DROP TRIGGER refuse_history")
        .execute(store.pool())
        .await
        .unwrap();
    store.close().await;
    let reopened = Store::open(dir.db()).await.unwrap();
    assert_eq!(snapshot(&reopened).await, before);
    integrity(&reopened).await;
    reopened.close().await;
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn competing_store_handles_have_one_winner_and_a_typed_stale_conflict() {
    let dir = Directory::new("race");
    let first = initialized(&dir.db()).await;
    let second = Store::open(dir.db()).await.unwrap();
    let expected = first.get("record").await.unwrap();
    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    let mut tasks = vec![];
    for (store, tag) in [(first.clone(), "first"), (second.clone(), "second")] {
        let expected = expected.clone();
        let barrier = barrier.clone();
        tasks.push(tokio::spawn(async move {
            let mut updated = expected.clone();
            updated.tags.push(tag.into());
            barrier.wait().await;
            store
                .transition_record(&expected, &updated, LifecycleState::Verified, None)
                .await
        }));
    }
    let mut applied = 0;
    let mut conflicts = 0;
    for task in tasks {
        match task.await.unwrap() {
            Ok(value) => {
                assert_eq!(value.status, RecordWriteStatus::Applied);
                applied += 1;
            }
            Err(StorageError::RecordConflict { .. }) => conflicts += 1,
            other => panic!("{other:?}"),
        }
    }
    assert_eq!((applied, conflicts), (1, 1));
    assert_eq!(first.lifecycle_history("record").await.unwrap().len(), 3);
    integrity(&second).await;
    first.close().await;
    second.close().await;
}
#[tokio::test]
async fn corrupted_envelopes_projections_cache_and_cursors_are_explicit_errors() {
    use crate::RecordFilter;
    let dir = Directory::new("malformed");
    let store = initialized(&dir.db()).await;
    store
        .insert_historical_run("other", "{}", None)
        .await
        .unwrap();
    for statement in [
        "UPDATE records SET record_json='{'",
        "UPDATE records SET run_id='other'",
        "UPDATE records SET lifecycle_state='verified'",
        "UPDATE records SET record_hash='wrong'",
    ] {
        sqlx::query(statement).execute(store.pool()).await.unwrap();
        assert!(store.get("record").await.is_err());
        assert!(store.scan(&RecordFilter::new()).await.is_err());
        store.replace_record_for_import(&record()).await.unwrap();
    }
    store
        .cache_put(
            "key",
            "teacher",
            "model",
            None,
            &serde_json::json!({"content":"cached"}),
        )
        .await
        .unwrap();
    sqlx::query("UPDATE cache SET value_json='{'")
        .execute(store.pool())
        .await
        .unwrap();
    assert!(
        store
            .cache_get("key", "teacher", "model", None)
            .await
            .is_err()
    );
    store
        .checkpoint("run", 0, "verified", &serde_json::json!({"offset":1}))
        .await
        .unwrap();
    sqlx::query("UPDATE checkpoints SET cursor_json='{'")
        .execute(store.pool())
        .await
        .unwrap();
    assert!(store.resume_cursor("run", 0).await.is_err());
    integrity(&store).await;
    store.close().await;
}

#[tokio::test]
async fn noncontent_fields_and_history_are_part_of_the_expected_snapshot() {
    let dir = Directory::new("full-snapshot");
    let store = initialized(&dir.db()).await;
    let other = Store::open(dir.db()).await.unwrap();
    let document = gw_schema::NumericTaskDocument::from_json(include_str!(
        "../../../examples/reviewed-numeric-tasks.json"
    ))
    .unwrap();
    let mut original = record();
    original.task_provenance =
        Some(gw_schema::TaskProvenance::from_task(&document.tasks[0]).unwrap());
    for field in [
        "cost",
        "verification",
        "judging",
        "provenance",
        "history",
        "task_provenance",
    ] {
        store.replace_record_for_import(&original).await.unwrap();
        let stale = store.get("record").await.unwrap();
        let mut updated = stale.clone();
        match field {
            "cost" => updated.cost.usd = 1.0,
            "verification" => updated.verification.all_passed = true,
            "judging" => updated.judging.aggregate = Some(0.8),
            "provenance" => {
                updated
                    .origin
                    .generated_mut()
                    .expect("generated fixture")
                    .provenance
                    .teacher
                    .served_by = Some("new-route".into())
            }
            "history" => updated.lifecycle.history[0].at = "2026-02-01T00:00:00Z".into(),
            "task_provenance" => {
                updated.task_provenance.as_mut().unwrap().split.revision = "new-split".into()
            }
            _ => unreachable!(),
        }
        other.replace_record_for_import(&updated).await.unwrap();
        let newer = store.get("record").await.unwrap();
        assert_eq!(
            stale.hashes.record_hash, newer.hashes.record_hash,
            "{field}"
        );
        assert_eq!(stale.lifecycle.state, newer.lifecycle.state);
        assert!(
            matches!(
                store
                    .advance_lifecycle(&stale, LifecycleState::Verified, None)
                    .await,
                Err(StorageError::RecordConflict { .. })
            ),
            "{field}"
        );
        assert_eq!(store.get("record").await.unwrap(), newer);
    }
    store.close().await;
    other.close().await;
}
#[tokio::test]
async fn missing_or_substituted_committed_history_cannot_be_acknowledged() {
    for statement in [
        "DELETE FROM lifecycle_history WHERE history_ordinal=0",
        "UPDATE lifecycle_history SET at='changed' WHERE history_ordinal=0",
    ] {
        let dir = Directory::new("damaged-history");
        let store = initialized(&dir.db()).await;
        sqlx::query(statement).execute(store.pool()).await.unwrap();
        store.close().await;
        let reopened = Store::open(dir.db()).await.unwrap();
        assert!(matches!(
            reopened.insert_record(&record()).await,
            Err(StorageError::RecordIntegrity { .. })
        ));
        let current = reopened.get("record").await.unwrap();
        assert!(matches!(
            reopened
                .advance_lifecycle(&current, LifecycleState::Verified, None)
                .await,
            Err(StorageError::RecordIntegrity { .. })
        ));
        reopened.close().await;
    }
}

#[tokio::test]
async fn missing_error_stub_and_error_transition_have_one_commit_and_retry_identity() {
    for stage in ["projected", "history", "precommit", "committed"] {
        let dir = Directory::new("error-stub");
        let store = Store::open(dir.db()).await.unwrap();
        store
            .insert_historical_run("run", "{}", None)
            .await
            .unwrap();
        let mut stub = record();
        stub.messages.clear();
        stub.lifecycle.history.clear();
        stub.lifecycle.state = LifecycleState::Seeded;
        store.set_test_hook(Some(Hook {
            operation: "record_insert",
            stage,
            action: Action::Fail,
        }));
        assert!(
            store
                .insert_record_and_transition(
                    &stub,
                    LifecycleState::Error,
                    Some("generation failed")
                )
                .await
                .is_err()
        );
        store.close().await;
        let reopened = Store::open(dir.db()).await.unwrap();
        integrity(&reopened).await;
        assert_eq!(reopened.get("record").await.is_ok(), stage == "committed");
        let retried = reopened
            .insert_record_and_transition(&stub, LifecycleState::Error, Some("generation failed"))
            .await
            .unwrap();
        assert_eq!(
            retried.status,
            if stage == "committed" {
                RecordWriteStatus::AlreadyApplied
            } else {
                RecordWriteStatus::Applied
            }
        );
        assert_eq!(retried.record.lifecycle.state, LifecycleState::Error);
        assert_eq!(
            retried.record.lifecycle.error.as_deref(),
            Some("generation failed")
        );
        assert_eq!(retried.record.lifecycle.history.len(), 1);
        assert_eq!(reopened.lifecycle_history("record").await.unwrap().len(), 1);
        reopened.close().await;
    }
}
