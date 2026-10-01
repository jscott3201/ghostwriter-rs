//! Real CLI startup checks replay identity before credentials.
mod common;
use std::process::Command;

#[test]
fn unknown_replay_is_rejected_before_missing_credentials() {
    let config = common::unique_temp_path("manifest.toml");
    let prompts = common::unique_temp_path("manifest.txt");
    let db = common::unique_temp_path("manifest.sqlite");
    std::fs::write(
        &config,
        "[area]\nadmission_intent='review_only'\n[[area.judges]]\nslug='judge'\nfamily='family'\n",
    )
    .unwrap();
    std::fs::write(&prompts, "What is 12*8?\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_gw"))
        .env_clear()
        .args(["gen", "replay", "--config"])
        .arg(&config)
        .args(["--run-id", "unknown", "--shards", "1", "--prompts"])
        .arg(&prompts)
        .arg("--db")
        .arg(&db)
        .output()
        .unwrap();
    std::fs::remove_file(config).unwrap();
    std::fs::remove_file(prompts).unwrap();
    common::cleanup_db(&db);
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        !error.contains("MODEL_API_KEY"),
        "replay identity must be checked first: {error}"
    );
    assert!(error.contains("unknown run"), "{error}");
}

const SETTINGS: &str =
    "[area]\nadmission_intent='review_only'\n[[area.judges]]\nslug='judge'\nfamily='family'\n";
fn configured() -> gw_cli::config::Config {
    serde_json::from_value(serde_json::json!({"area":{"admission_intent":"review_only","judges":[{"slug":"judge","family":"family"}]}})).unwrap()
}
#[tokio::test]
async fn run_replay_and_tui_check_existing_semantics_before_credentials_or_terminal() {
    use gw_schema::{AccountingCapability, AccountingPolicy};
    let config = common::unique_temp_path("manifest-shared.toml");
    let prompts = common::unique_temp_path("manifest-shared.txt");
    let db = common::unique_temp_path("manifest-shared.sqlite");
    std::fs::write(&config, SETTINGS).unwrap();
    std::fs::write(&prompts, "What is 12*8?\n").unwrap();
    let mut settings = configured();
    settings.db = db.clone();
    let source = gw_cli::seedsource::FileSeedSource::from_prompts_str("What is 12*8?", 1).unwrap();
    let prepared = gw_cli::wire::prepare_run(&settings, &source).unwrap();
    let store = gw_storage::Store::open(&db).await.unwrap();
    store
        .register_accounting_launch(gw_storage::LaunchRequest {
            run_id: "pinned",
            manifest: prepared.manifest().clone(),
            mode: gw_storage::RunMode::CreateOrResume,
            policy: &AccountingPolicy::ObservationOnly,
            teacher: AccountingCapability::NoModelRequests,
            judge: AccountingCapability::NoModelRequests,
            embedding: AccountingCapability::NoModelRequests,
        })
        .await
        .unwrap();
    store
        .set_run_status("pinned", gw_storage::RunStatus::Halted)
        .await
        .unwrap();
    let snapshot = store.accounting_snapshot("pinned").await.unwrap();
    let metadata: (String, String, String) =
        sqlx::query_as("SELECT config_json,created_at,status FROM runs WHERE run_id='pinned'")
            .fetch_one(store.raw_pool())
            .await
            .unwrap();
    for action in ["run", "replay", "tui"] {
        for (setting, value) in [
            ("GW_AREA__TEACHER_SLUG", "other-teacher"),
            ("GW_MODEL_API_BASE_URL", "http://localhost:9000/v1"),
        ] {
            let output = Command::new(env!("CARGO_BIN_EXE_gw"))
                .env_clear()
                .env(setting, value)
                .env("GW_MODEL_API_KEY_ENV", "UNREAD_MODEL_KEY")
                .args(["gen", action, "--config"])
                .arg(&config)
                .args(["--run-id", "pinned", "--shards", "1", "--prompts"])
                .arg(&prompts)
                .arg("--db")
                .arg(&db)
                .output()
                .unwrap();
            let error = String::from_utf8_lossy(&output.stderr);
            assert!(!output.status.success());
            assert!(error.contains("incompatible"), "{action}: {error}");
            assert!(error.contains("new run"), "{error}");
            assert!(!error.contains("UNREAD_MODEL_KEY"), "{action}: {error}");
            assert_eq!(store.accounting_snapshot("pinned").await.unwrap(), snapshot);
            let after: (String, String, String) = sqlx::query_as(
                "SELECT config_json,created_at,status FROM runs WHERE run_id='pinned'",
            )
            .fetch_one(store.raw_pool())
            .await
            .unwrap();
            assert_eq!(after, metadata);
        }
    }
    // Operational and key-reference changes preserve identity and reach the selected missing key.
    for action in ["run", "replay", "tui"] {
        let output = Command::new(env!("CARGO_BIN_EXE_gw"))
            .env_clear()
            .env("GW_MODEL_API_KEY_ENV", "ROTATED_MODEL_KEY")
            .env("GW_PROVIDER_RPM", "999")
            .env("GW_TICK_MS", "1")
            .args(["gen", action, "--config"])
            .arg(&config)
            .args([
                "--run-id",
                "pinned",
                "--shards",
                "1",
                "--max-in-flight",
                "7",
                "--accounting-policy",
                "observation-only",
                "--prompts",
            ])
            .arg(&prompts)
            .arg("--db")
            .arg(&db)
            .output()
            .unwrap();
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains("ROTATED_MODEL_KEY"), "{action}: {error}");
        assert!(!error.contains("incompatible"), "{action}: {error}");
        assert_eq!(store.accounting_snapshot("pinned").await.unwrap(), snapshot);
    }
    drop(store);
    std::fs::remove_file(config).unwrap();
    std::fs::remove_file(prompts).unwrap();
    common::cleanup_db(&db);
}
#[test]
fn pure_preparation_ignores_operational_config_and_resolves_effective_defaults() {
    let source = gw_cli::seedsource::FileSeedSource::from_prompts_str("q", 1).unwrap();
    let base = configured();
    let first = gw_cli::wire::prepare_run(&base, &source).unwrap();
    let mut other = base.clone();
    other.accounting_policy = Some(gw_schema::AccountingPolicy::ObservationOnly);
    other.model_api_key_env = "UNREAD_MODEL_KEY".into();
    other.provider_rpm = 999;
    other.tick_ms = 3;
    other.frame_ms = 2;
    other.db = "another.sqlite".into();
    other.export = Some(gw_cli::config::ExportSettings {
        out: "another.parquet".into(),
        format: gw_schema::TrlFormat::ChatML,
        cot: gw_schema::CotPolicy::Masked,
        dataset_version: None,
    });
    other.area.k = 0;
    other.area.teacher_max_tokens = Some(gw_engine::DEFAULT_MAX_TOKENS);
    other.area.judge_max_tokens = Some(gw_judge::DEFAULT_JUDGE_MAX_TOKENS);
    other.area.judge_reasoning_max_tokens = Some(gw_judge::DEFAULT_JUDGE_REASONING_MAX_TOKENS);
    assert_eq!(
        gw_cli::wire::prepare_run(&other, &source)
            .unwrap()
            .manifest(),
        first.manifest()
    );
    let mut embedding = base.clone();
    embedding.embedding = Some(gw_schema::EmbeddingConfig {
        api_key_env: Some("UNREAD_EMBEDDING_KEY".into()),
        ..Default::default()
    });
    assert!(gw_cli::wire::prepare_run(&embedding, &source).is_ok());
    for endpoint in [
        "https://SECRET@example.test/v1",
        "https://example.test/v1?key=SECRET",
        "https://example.test/v1#SECRET",
    ] {
        let mut invalid = base.clone();
        invalid.model_api_base_url = endpoint.into();
        let error = format!(
            "{:#}",
            gw_cli::wire::prepare_run(&invalid, &source).unwrap_err()
        );
        assert!(!error.contains("SECRET"));
        assert!(!error.contains("MODEL_API_KEY"));
    }
}
