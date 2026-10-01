//! Actual CLI startup rejects invalid panels before opening provider credentials or the store.

mod common;

use std::process::Command;

#[test]
fn duplicate_judge_aliases_fail_before_credentials_for_every_generation_entry() {
    for alias in ["", "family = 'alias'\n", "rubric_id = 'audit-alias'\n"] {
        for intent in ["automatic", "review_only"] {
            let config = common::unique_temp_path("duplicate-judges.toml");
            let prompts = common::unique_temp_path("duplicate-judges.txt");
            let db = common::unique_temp_path("duplicate-judges.sqlite");
            let second_family = if alias.starts_with("family") {
                ""
            } else {
                "family='family'\n"
            };
            std::fs::write(&config, format!("[area]\nadmission_intent='{intent}'\ncorrelation_rho=0.3\n[[area.judges]]\nslug='judge'\nfamily='family'\n[[area.judges]]\nslug='judge'\n{second_family}{alias}")).unwrap();
            std::fs::write(&prompts, "Explain the result\n").unwrap();
            for action in ["run", "replay", "tui"] {
                let output = Command::new(env!("CARGO_BIN_EXE_gw"))
                    .env_clear()
                    .args(["gen", action, "--config"])
                    .arg(&config)
                    .args(["--run-id", "duplicate", "--shards", "1", "--prompts"])
                    .arg(&prompts)
                    .arg("--db")
                    .arg(&db)
                    .output()
                    .unwrap();
                let stderr = String::from_utf8_lossy(&output.stderr);
                assert!(!output.status.success());
                assert!(stderr.contains("duplicate"), "{action}/{intent}: {stderr}");
                assert!(!stderr.contains("MODEL_API_KEY"), "{stderr}");
                assert!(!db.exists(), "invalid evidence must not create a run store");
            }
            std::fs::remove_file(config).unwrap();
            std::fs::remove_file(prompts).unwrap();
            common::cleanup_db(&db);
        }
    }
}

#[test]
fn generation_preflight_precedes_missing_provider_key() {
    for (label, panel, expected) in [
        ("empty", "", "empty"),
        (
            "impossible",
            "[[area.judges]]\nslug = 'judge'\nfamily = 'family'\n",
            "unattainable",
        ),
    ] {
        let config = common::unique_temp_path(&format!("preflight-{label}.toml"));
        let prompts = common::unique_temp_path(&format!("preflight-{label}.txt"));
        let db = common::unique_temp_path(&format!("preflight-{label}.sqlite"));
        std::fs::write(&config, panel).unwrap();
        std::fs::write(&prompts, "What is 12*8?\n").unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_gw"))
            .env_clear()
            .args(["gen", "run", "--config"])
            .arg(&config)
            .args(["--run-id", "preflight", "--prompts"])
            .arg(&prompts)
            .arg("--db")
            .arg(&db)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        let store_created = db.exists();
        std::fs::remove_file(config).unwrap();
        std::fs::remove_file(prompts).unwrap();
        common::cleanup_db(&db);
        assert!(!output.status.success());
        assert!(
            stderr.contains(expected),
            "expected panel preflight, got: {stderr}"
        );
        assert!(
            !stderr.contains("MODEL_API_KEY"),
            "credentials accessed before preflight"
        );
        assert!(
            !store_created,
            "invalid config should not create a run store"
        );
    }
}

#[test]
fn toml_environment_and_cli_intent_share_the_resolved_assessment() {
    let config = common::unique_temp_path("panel-layers.toml");
    let prompts = common::unique_temp_path("panel-layers.txt");
    std::fs::write(&config, "[area]\nadmission_intent = 'review_only'\n[[area.judges]]\nslug = 'judge'\nfamily = 'family'\n").unwrap();
    std::fs::write(&prompts, "What is 12*8?\n").unwrap();
    for (env_intent, cli_intent, min_eff, expected) in [
        (None, None, None, "MODEL_API_KEY"),
        (Some("automatic"), None, None, "unattainable"),
        (Some("review_only"), Some("automatic"), None, "unattainable"),
        (
            Some("automatic"),
            Some("review-only"),
            None,
            "MODEL_API_KEY",
        ),
        (Some("automatic"), None, Some("1.0"), "MODEL_API_KEY"),
        (Some("review_only"), None, Some("-1.0"), "min_n_eff"),
    ] {
        let db = common::unique_temp_path("panel-layers.sqlite");
        let mut command = Command::new(env!("CARGO_BIN_EXE_gw"));
        command
            .env_clear()
            .args(["gen", "run", "--config"])
            .arg(&config)
            .args(["--run-id", "layers", "--prompts"])
            .arg(&prompts)
            .arg("--db")
            .arg(&db);
        if let Some(intent) = env_intent {
            command.env("GW_AREA__ADMISSION_INTENT", intent);
        }
        if let Some(intent) = cli_intent {
            command.args(["--admission-intent", intent]);
        }
        if let Some(floor) = min_eff {
            command.env("GW_AREA__THRESHOLDS__MIN_N_EFF", floor);
        }
        let output = command.output().unwrap();
        common::cleanup_db(&db);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success());
        assert!(
            stderr.contains(expected),
            "expected {expected}, got: {stderr}"
        );
        if expected != "MODEL_API_KEY" {
            assert!(!stderr.contains("MODEL_API_KEY"));
        }
    }
    std::fs::remove_file(config).unwrap();
    std::fs::remove_file(prompts).unwrap();
}

#[test]
fn replay_applies_explicit_review_only_before_provider_construction() {
    let config = common::unique_temp_path("panel-replay.toml");
    let prompts = common::unique_temp_path("panel-replay.txt");
    let db = common::unique_temp_path("panel-replay.sqlite");
    std::fs::write(
        &config,
        "[[area.judges]]\nslug = 'judge'\nfamily = 'family'\n",
    )
    .unwrap();
    std::fs::write(&prompts, "What is 12*8?\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_gw"))
        .env_clear()
        .args(["gen", "replay", "--config"])
        .arg(&config)
        .args([
            "--run-id",
            "review",
            "--shards",
            "1",
            "--admission-intent",
            "review-only",
            "--prompts",
        ])
        .arg(&prompts)
        .arg("--db")
        .arg(&db)
        .output()
        .unwrap();
    common::cleanup_db(&db);
    std::fs::remove_file(config).unwrap();
    std::fs::remove_file(prompts).unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unknown run"),
        "explicit review-only passes feasibility before replay identity rejects an unknown run: {stderr}"
    );
    assert!(!stderr.contains("unattainable"));
}
