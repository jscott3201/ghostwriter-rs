//! The real provider-free CLI reports publication errors and prints its embedded manifest.

mod common;

use common::{admit, cleanup_db, record, seed_store, unique_temp_path};
use gw_schema::{ExportManifest, LifecycleState, Verdict};
use gw_storage::{ArtifactVerification, RunStatus, Store, verify_artifact};
use std::process::Command;

fn command(db: &std::path::Path, out: &std::path::Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_gw"));
    command
        .args(["gen", "export", "--db"])
        .arg(db)
        .arg("--out")
        .arg(out)
        .args(["--run-id", "run", "--dataset-version", "2.4.6"])
        .env_remove("OPENROUTER_API_KEY");
    command
}

#[tokio::test]
async fn nonempty_and_empty_stdout_equal_embedded_manifest_without_generation_mutation() {
    for selected in [false, true] {
        let db = unique_temp_path("publication.sqlite");
        let out = unique_temp_path("publication.parquet");
        let rec = record(
            "r",
            "run",
            Some(if selected {
                Verdict::Admit
            } else {
                Verdict::Reject
            }),
            Some(0.9),
            true,
            "group",
        );
        let store = seed_store(&db, "run", &[rec]).await;
        if selected {
            admit(&store, "r").await;
        }
        store
            .set_run_status("run", RunStatus::Halted)
            .await
            .unwrap();
        let before = store.get("r").await.unwrap();
        let sidecar = out.with_extension("parquet.manifest.json");
        std::fs::write(&sidecar, b"user-owned historical manifest").unwrap();
        std::fs::write(&out, b"old output").unwrap();
        let result = command(&db, &out).output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let manifest: ExportManifest = serde_json::from_slice(&result.stdout).unwrap();
        let ArtifactVerification::Verified(artifact) = verify_artifact(&out).unwrap() else {
            panic!("missing metadata");
        };
        assert_eq!(manifest, artifact.manifest);
        assert_eq!(
            manifest.dataset_version,
            Some(semver::Version::new(2, 4, 6))
        );
        assert_eq!(
            (manifest.n_records, manifest.n_admitted),
            (1, u64::from(selected))
        );
        assert_eq!(store.get("r").await.unwrap(), before);
        assert_eq!(
            store.run_status("run").await.unwrap().as_deref(),
            Some("halted")
        );
        assert_eq!(
            std::fs::read(&sidecar).unwrap(),
            b"user-owned historical manifest"
        );
        store.close().await;
        std::fs::remove_file(out).unwrap();
        std::fs::remove_file(sidecar).unwrap();
        cleanup_db(&db);
    }
}

#[tokio::test]
async fn cli_postrename_failure_is_nonzero_and_reopened_retry_uses_prepared_receipt() {
    let db = unique_temp_path("retry-publication.sqlite");
    let out = unique_temp_path("retry-publication.parquet");
    let rec = record(
        "selected",
        "run",
        Some(Verdict::Admit),
        Some(0.9),
        true,
        "group",
    );
    let store = seed_store(&db, "run", &[rec]).await;
    admit(&store, "selected").await;
    store
        .set_run_status("run", RunStatus::Failed)
        .await
        .unwrap();
    sqlx::query("CREATE TRIGGER fail_cli_ack BEFORE UPDATE ON export_receipts WHEN NEW.state = 'acknowledged' BEGIN SELECT RAISE(FAIL, 'injected CLI acknowledgment failure'); END")
        .execute(store.raw_pool()).await.unwrap();
    let result = command(&db, &out).output().unwrap();
    assert!(!result.status.success());
    assert!(
        result.stdout.is_empty(),
        "failure cannot print a success manifest"
    );
    assert!(String::from_utf8_lossy(&result.stderr).contains("injected CLI acknowledgment"));
    let ArtifactVerification::Verified(expected) = verify_artifact(&out).unwrap() else {
        panic!("published file must be complete");
    };
    let later = record(
        "later",
        "run",
        Some(Verdict::Admit),
        Some(0.95),
        true,
        "later-group",
    );
    store.put(&later).await.unwrap();
    admit(&store, "later").await;
    sqlx::query("DROP TRIGGER fail_cli_ack")
        .execute(store.raw_pool())
        .await
        .unwrap();
    store.close().await;
    let result = command(&db, &out).output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let manifest: ExportManifest = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(manifest, expected.manifest);
    assert_eq!(
        manifest.n_records, 1,
        "retry retains the prepared population"
    );
    let reopened = Store::open(&db).await.unwrap();
    for id in ["selected", "later"] {
        assert_eq!(
            reopened.get(id).await.unwrap().lifecycle.state,
            LifecycleState::Admitted
        );
    }
    assert_eq!(
        reopened.run_status("run").await.unwrap().as_deref(),
        Some("failed")
    );
    let states: Vec<(String,)> = sqlx::query_as("SELECT state FROM export_receipts")
        .fetch_all(reopened.raw_pool())
        .await
        .unwrap();
    assert_eq!(states, [("acknowledged".into(),)]);
    reopened.close().await;
    std::fs::remove_file(out).unwrap();
    cleanup_db(&db);
}

#[tokio::test]
async fn explicit_cli_recovery_retains_engine_mode_without_completing_generation() {
    use gw_schema::{CotPolicy, ExportOptions, ExportScope, TrlFormat};
    use gw_storage::{ExportPurpose, StorageError};
    let db = unique_temp_path("engine-recovery.sqlite");
    let out = unique_temp_path("engine-recovery.parquet");
    let rec = record(
        "selected",
        "run",
        Some(Verdict::Admit),
        Some(0.9),
        true,
        "group",
    );
    let store = seed_store(&db, "run", &[rec]).await;
    admit(&store, "selected").await;
    sqlx::query("CREATE TRIGGER fail_engine_ack BEFORE UPDATE ON export_receipts WHEN NEW.state = 'acknowledged' BEGIN SELECT RAISE(FAIL, 'lost acknowledgment'); END")
        .execute(store.raw_pool()).await.unwrap();
    let error = store
        .publish_export(
            ExportOptions {
                target: TrlFormat::ChatML,
                cot_policy: CotPolicy::Masked,
                dataset_version: None,
                scope: ExportScope::Run {
                    run_id: "run".into(),
                },
            },
            &out,
            ExportPurpose::Engine,
        )
        .await
        .unwrap_err();
    let StorageError::Publication { publication_id, .. } = error else {
        panic!("recovery ID missing");
    };
    sqlx::query("DROP TRIGGER fail_engine_ack")
        .execute(store.raw_pool())
        .await
        .unwrap();
    store
        .set_run_status("run", RunStatus::Failed)
        .await
        .unwrap();
    let later = record(
        "later",
        "run",
        Some(Verdict::Admit),
        Some(0.95),
        true,
        "later",
    );
    store.put(&later).await.unwrap();
    admit(&store, "later").await;
    for _ in 0..2 {
        let result = Command::new(env!("CARGO_BIN_EXE_gw"))
            .args(["gen", "export", "--db"])
            .arg(&db)
            .args(["--resume-publication", &publication_id])
            .env_remove("OPENROUTER_API_KEY")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let manifest: ExportManifest = serde_json::from_slice(&result.stdout).unwrap();
        assert_eq!((manifest.n_records, manifest.n_admitted), (1, 1));
        assert_eq!(
            store.get("selected").await.unwrap().lifecycle.state,
            LifecycleState::Exported
        );
        assert_eq!(
            store.get("later").await.unwrap().lifecycle.state,
            LifecycleState::Admitted
        );
        assert_eq!(
            store.run_status("run").await.unwrap().as_deref(),
            Some("failed")
        );
    }
    let history = store.lifecycle_history("selected").await.unwrap();
    assert_eq!(
        history.iter().filter(|entry| entry.0 == "exported").count(),
        1
    );
    store.close().await;
    std::fs::remove_file(out).unwrap();
    cleanup_db(&db);
}

#[test]
fn explicit_recovery_rejects_every_export_override() {
    use clap::Parser;
    use gw_cli::cli::Cli;
    for override_args in [
        vec!["--out", "different.parquet"],
        vec!["--run-id", "different"],
        vec!["--format", "chat-ml"],
        vec!["--cot", "supervised"],
        vec!["--dataset-version", "1.0.0"],
    ] {
        let mut args = vec![
            "gw",
            "gen",
            "export",
            "--db",
            "source.sqlite",
            "--resume-publication",
            "receipt-id",
        ];
        args.extend(override_args);
        assert!(Cli::try_parse_from(args).is_err());
    }
    assert!(
        Cli::try_parse_from([
            "gw",
            "gen",
            "export",
            "--db",
            "source.sqlite",
            "--resume-publication",
            "receipt-id"
        ])
        .is_ok()
    );
}
