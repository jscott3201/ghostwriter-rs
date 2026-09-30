//! Actual process death at acknowledged and ambiguous record/cache/checkpoint boundaries.
use crate::{
    RecordWriteStatus, Store,
    durability_support::*,
    test_hooks::{Action, Hook},
};
use gw_schema::LifecycleState;

#[test]
fn child_entry() {
    let Ok(path) = std::env::var("GW_DURABILITY_DB") else {
        return;
    };
    let scenario = std::env::var("GW_DURABILITY_SCENARIO").unwrap();
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let store = initialized(std::path::Path::new(&path)).await;
        store
            .set_run_status("run", crate::RunStatus::Running)
            .await
            .unwrap();
        store
            .cache_put(
                "request",
                "teacher",
                "fixture",
                None,
                &serde_json::json!({"content":"5","reasoning":"addition"}),
            )
            .await
            .unwrap();
        if scenario == "cache_ack" {
            pause(&store).await;
        }
        store
            .checkpoint(
                "run",
                0,
                "assistant_generated",
                &serde_json::json!({"offset":0}),
            )
            .await
            .unwrap();
        if scenario == "cursor_ack" {
            pause(&store).await;
        }
        store
            .set_run_status("run", crate::RunStatus::Halted)
            .await
            .unwrap();
        if scenario == "status_ack" {
            pause(&store).await;
        }
        let stage = match scenario.as_str() {
            "record_precommit" => Some("precommit"),
            "record_committed" => Some("committed"),
            _ => None,
        };
        if let Some(stage) = stage {
            store.set_test_hook(Some(Hook {
                operation: "record_transition",
                stage,
                action: Action::Process("kill"),
            }));
        }
        let expected = store.get("record").await.unwrap();
        let mut updated = expected.clone();
        updated.cost.prompt_tokens = 11;
        updated.tags.push("saved verification".into());
        store
            .transition_record(
                &expected,
                &updated,
                LifecycleState::Verified,
                Some("verified"),
            )
            .await
            .unwrap();
        // This acknowledgment is after COMMIT; deliberately do not close/checkpoint the pool.
        pause(&store).await;
    });
}

#[tokio::test]
async fn process_death_recovers_whole_records_and_independent_acknowledged_commits() {
    for scenario in [
        "record_precommit",
        "record_committed",
        "record_ack",
        "cache_ack",
        "cursor_ack",
        "status_ack",
    ] {
        let dir = Directory::new(scenario);
        kill_at(&dir, "process_recovery_tests::child_entry", scenario);
        let store = Store::open(dir.db()).await.unwrap();
        integrity(&store).await;
        let current = store.get("record").await.unwrap();
        let applied = matches!(scenario, "record_committed" | "record_ack");
        assert_eq!(
            current.lifecycle.state,
            if applied {
                LifecycleState::Verified
            } else {
                LifecycleState::AssistantGenerated
            }
        );
        assert_eq!(
            store.lifecycle_history("record").await.unwrap().len(),
            if applied { 3 } else { 2 }
        );
        assert_eq!(
            store
                .cache_get("request", "teacher", "fixture", None)
                .await
                .unwrap(),
            Some(serde_json::json!({"content":"5","reasoning":"addition"}))
        );
        let cursor = store.resume_cursor("run", 0).await.unwrap();
        assert_eq!(cursor.is_some(), scenario != "cache_ack");
        if let Some(cursor) = cursor {
            assert_eq!(cursor.cursor, serde_json::json!({"offset":0}));
        }
        assert_eq!(
            store.run_status("run").await.unwrap().as_deref(),
            Some(if matches!(scenario, "cache_ack" | "cursor_ack") {
                "running"
            } else {
                "halted"
            })
        );
        let expected = crate::record_data::normalize(&record()).unwrap();
        let mut updated = expected.clone();
        updated.cost.prompt_tokens = 11;
        updated.tags.push("saved verification".into());
        let retried = store
            .transition_record(
                &expected,
                &updated,
                LifecycleState::Verified,
                Some("verified"),
            )
            .await
            .unwrap();
        assert_eq!(
            retried.status,
            if applied {
                RecordWriteStatus::AlreadyApplied
            } else {
                RecordWriteStatus::Applied
            }
        );
        assert_eq!(store.lifecycle_history("record").await.unwrap().len(), 3);
        store.close().await;
    }
}
