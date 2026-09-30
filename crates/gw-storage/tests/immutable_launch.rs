//! The launch transaction checks immutable meaning before any operational mutation.
mod common;
use gw_schema::{AccountingCapability, AccountingPolicy, RunManifest};
use gw_storage::{LaunchRequest, RunMode, RunStatus, StorageError, Store};
fn request(manifest: RunManifest) -> LaunchRequest<'static> {
    LaunchRequest {
        run_id: "immutable",
        manifest,
        mode: RunMode::CreateOrResume,
        policy: &AccountingPolicy::ObservationOnly,
        teacher: AccountingCapability::NoModelRequests,
        judge: AccountingCapability::NoModelRequests,
        embedding: AccountingCapability::NoModelRequests,
    }
}
async fn row(store: &Store) -> (String, String, String, Option<i64>, Option<String>) {
    sqlx::query_as("SELECT config_json,created_at,status,shard_count,prompts_hash FROM runs WHERE run_id='immutable'").fetch_one(store.raw_pool()).await.unwrap()
}
#[tokio::test]
async fn conflicting_semantics_cannot_overwrite_configuration_or_advance_accounting() {
    let store = Store::open_in_memory().await.unwrap();
    let original = common::manifest();
    store
        .register_accounting_launch(request(original.clone()))
        .await
        .unwrap();
    store
        .set_run_status("immutable", RunStatus::Halted)
        .await
        .unwrap();
    store
        .checkpoint(
            "immutable",
            0,
            "admitted",
            &serde_json::json!({"seed_offset":1}),
        )
        .await
        .unwrap();
    let before = store.accounting_snapshot("immutable").await.unwrap();
    let launches = store.model_launches("immutable").await.unwrap();
    let metadata = row(&store).await;
    let cursor = store.resume_cursor("immutable", 0).await.unwrap();
    let mut changed = original;
    changed.clients.teacher.revision = "changed".into();
    let mut launch = request(changed);
    launch.policy = &AccountingPolicy::FiniteUsd { limit_usd: 5.0 };
    assert!(matches!(
        store.register_accounting_launch(launch).await,
        Err(StorageError::RunManifest { .. })
    ));
    assert_eq!(
        store.accounting_snapshot("immutable").await.unwrap(),
        before
    );
    assert_eq!(store.model_launches("immutable").await.unwrap(), launches);
    assert_eq!(row(&store).await, metadata);
    assert_eq!(store.resume_cursor("immutable", 0).await.unwrap(), cursor);
}
#[tokio::test]
async fn compatible_replay_preserves_exact_bytes_created_time_and_cursor() {
    let store = Store::open_in_memory().await.unwrap();
    let manifest = common::manifest();
    store
        .register_accounting_launch(request(manifest.clone()))
        .await
        .unwrap();
    let pretty = serde_json::to_string_pretty(&manifest).unwrap();
    sqlx::query("UPDATE runs SET config_json=?,created_at='original-time',status='halted' WHERE run_id='immutable'").bind(&pretty).execute(store.raw_pool()).await.unwrap();
    store
        .checkpoint(
            "immutable",
            0,
            "admitted",
            &serde_json::json!({"seed_offset":1}),
        )
        .await
        .unwrap();
    let cursor = store.resume_cursor("immutable", 0).await.unwrap();
    let mut launch = request(manifest);
    launch.mode = RunMode::Replay;
    launch.policy = &AccountingPolicy::FiniteUsd { limit_usd: 8.0 };
    store.register_accounting_launch(launch).await.unwrap();
    let saved = row(&store).await;
    assert_eq!(saved.0, pretty);
    assert_eq!(saved.1, "original-time");
    assert_eq!(store.resume_cursor("immutable", 0).await.unwrap(), cursor);
    assert_eq!(store.model_launches("immutable").await.unwrap().len(), 2);
}
#[tokio::test]
async fn unknown_legacy_malformed_unsupported_and_unpinned_runs_fail_closed() {
    let valid = common::manifest();
    let mut unsupported = serde_json::to_value(&valid).unwrap();
    unsupported["version"] = 999.into();
    let mut missing = serde_json::to_value(&valid).unwrap();
    missing["unattested_deployment"]
        .as_object_mut()
        .unwrap()
        .remove("weights");
    let examples = vec![
        "{}".into(),
        "not-json".into(),
        unsupported.to_string(),
        missing.to_string(),
        serde_json::to_string(&valid).unwrap(),
    ];
    let store = Store::open_in_memory().await.unwrap();
    let mut replay = request(valid.clone());
    replay.mode = RunMode::Replay;
    assert!(matches!(
        store.register_accounting_launch(replay).await,
        Err(StorageError::RunManifest { .. })
    ));
    assert!(store.run_status("immutable").await.unwrap().is_none());
    for original in examples {
        let store = Store::open_in_memory().await.unwrap();
        store
            .insert_historical_run("immutable", &original, None)
            .await
            .unwrap();
        let before = row(&store).await;
        assert!(
            store
                .validate_run_manifest("immutable", &valid, RunMode::Replay)
                .await
                .is_err()
        );
        assert!(matches!(
            store
                .register_accounting_launch(request(valid.clone()))
                .await,
            Err(StorageError::RunManifest { .. })
        ));
        assert_eq!(row(&store).await, before);
        assert!(store.model_launches("immutable").await.unwrap().is_empty());
    }
}
#[tokio::test]
async fn separate_connections_racing_incompatible_first_launches_have_one_winner() {
    let path = std::env::temp_dir().join(format!(
        "gw-manifest-race-{}-{}.sqlite",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let a = Store::open(&path).await.unwrap();
    let b = Store::open(&path).await.unwrap();
    let first = common::manifest();
    let mut second = first.clone();
    second.execution.revision = "other".into();
    // Both CLI-style read-only preflights pass before either initializer commits.
    a.validate_run_manifest("immutable", &first, RunMode::CreateOrResume)
        .await
        .unwrap();
    b.validate_run_manifest("immutable", &second, RunMode::CreateOrResume)
        .await
        .unwrap();
    let (one, two) = tokio::join!(
        a.register_accounting_launch(request(first.clone())),
        b.register_accounting_launch(request(second.clone()))
    );
    let loser = match (one, two) {
        (Ok(_), Err(error)) | (Err(error), Ok(_)) => error,
        result => panic!("exactly one incompatible initializer must succeed: {result:?}"),
    };
    assert!(matches!(loser, StorageError::RunManifest { .. }));
    let saved: RunManifest = serde_json::from_str(&row(&a).await.0).unwrap();
    assert!(saved == first || saved == second);
    assert_eq!(a.model_launches("immutable").await.unwrap().len(), 1);
    drop(a);
    drop(b);
    for suffix in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
    }
}
