//! The admission repair changes execution identity, never historical votes or effective counts.

mod common;

use gw_schema::{AccountingCapability, AccountingPolicy, AdmissionIntent, LifecycleState, Verdict};
use std::process::Command;

#[tokio::test]
async fn prior_admission_runs_cannot_resume_but_remain_inspectable_exportable_and_recoverable() {
    for duplicated in [false, true] {
        let config = common::unique_temp_path("historical-judges.toml");
        let prompts = common::unique_temp_path("historical-judges.txt");
        let db = common::unique_temp_path("historical-judges.sqlite");
        let artifact = common::unique_temp_path("historical-judges.parquet");
        let mut settings: gw_cli::config::Config = serde_json::from_value(serde_json::json!({
            "area": { "correlation_rho": 0.3, "judges": [
                {"slug":"judge-a", "family":"family-a"},
                {"slug":"judge-b", "family":"family-b"}
            ] }
        }))
        .unwrap();
        settings.db = db.clone();
        let source =
            gw_cli::seedsource::FileSeedSource::from_prompts_str("Explain the result", 1).unwrap();
        let prepared = gw_cli::wire::prepare_run(&settings, &source).unwrap();
        assert_eq!(prepared.manifest().execution.revision, "4");
        assert_eq!(
            prepared.manifest().execution.configuration["judging"]["evidence"],
            "distinct-effective-request-and-interpretation-v1"
        );
        // This is the former behavior declaration, independently fixed to revision 3 and without
        // the new evidence requirement. The first case changes only that behavior declaration.
        let mut previous = prepared.manifest().clone();
        previous.execution.revision = "3".into();
        previous.execution.configuration["judging"]
            .as_object_mut()
            .unwrap()
            .remove("evidence");
        if duplicated {
            previous.execution.configuration["judging"]["request_contracts"][1] =
                previous.execution.configuration["judging"]["request_contracts"][0].clone();
        }
        let second = if duplicated { "judge-a" } else { "judge-b" };
        let family = if duplicated { "family-a" } else { "family-b" };
        std::fs::write(&config, format!("[area]\ncorrelation_rho=0.3\n[[area.judges]]\nslug='judge-a'\nfamily='family-a'\n[[area.judges]]\nslug='{second}'\nfamily='{family}'\n")).unwrap();
        std::fs::write(&prompts, "Explain the result\n").unwrap();
        let store = gw_storage::Store::open(&db).await.unwrap();
        store
            .register_accounting_launch(gw_storage::LaunchRequest {
                run_id: "historical",
                manifest: previous.clone(),
                mode: gw_storage::RunMode::CreateOrResume,
                policy: &AccountingPolicy::ObservationOnly,
                teacher: AccountingCapability::NoModelRequests,
                judge: AccountingCapability::NoModelRequests,
                embedding: AccountingCapability::NoModelRequests,
            })
            .await
            .unwrap();
        let mut record = common::record(
            "stored",
            "historical",
            Some(Verdict::Admit),
            Some(0.9),
            true,
            "question",
        );
        record.lifecycle.state = LifecycleState::Formatted;
        record.judging.admission_intent = AdmissionIntent::Automatic;
        record.judging.panel[0].judge_model = "judge-a".into();
        let mut second_vote = record.judging.panel[0].clone();
        second_vote.judge_model = second.into();
        record.judging.panel.push(second_vote);
        record.judging.decisive_count = Some(2);
        record.judging.n_eff = Some(20.0 / 13.0);
        record.judging.verdict_reason = Some("above_threshold".into());
        record.judging.threshold_at_decision = Some(0.8);
        store.replace_record_for_import(&record).await.unwrap();
        let before = store.get("stored").await.unwrap();
        assert!((before.judging.n_eff.unwrap() - 20.0 / 13.0).abs() < 1e-12);
        let bytes = serde_json::to_vec(&before).unwrap();
        let hash = gw_storage::record_hash(&before).unwrap();
        let snapshot = store.accounting_snapshot("historical").await.unwrap();
        let metadata: String =
            sqlx::query_scalar("SELECT config_json FROM runs WHERE run_id='historical'")
                .fetch_one(store.raw_pool())
                .await
                .unwrap();
        assert_eq!(
            serde_json::from_str::<gw_schema::RunManifest>(&metadata).unwrap(),
            previous
        );
        for action in ["run", "replay", "tui"] {
            let output = Command::new(env!("CARGO_BIN_EXE_gw"))
                .env_clear()
                .args(["gen", action, "--config"])
                .arg(&config)
                .args(["--run-id", "historical", "--shards", "1", "--prompts"])
                .arg(&prompts)
                .arg("--db")
                .arg(&db)
                .output()
                .unwrap();
            let error = String::from_utf8_lossy(&output.stderr);
            assert!(!output.status.success());
            assert!(
                error.contains(if duplicated {
                    "duplicate"
                } else {
                    "incompatible"
                }),
                "{error}"
            );
            assert!(!error.contains("MODEL_API_KEY"), "{error}");
            assert_eq!(
                store.accounting_snapshot("historical").await.unwrap(),
                snapshot
            );
            assert_eq!(
                serde_json::to_vec(&store.get("stored").await.unwrap()).unwrap(),
                bytes
            );
        }
        // Provider-free readback and rederivation preserve the recorded evidence. The historical
        // n_eff is not replaced with a guessed corrected value for duplicate positions.
        assert_eq!(
            gw_judge::rederive_verdict(&before.judging, gw_judge::AreaThresholds::default())
                .unwrap()
                .to_schema_verdict(),
            Some(Verdict::Admit)
        );
        assert!(store.run_status("historical").await.unwrap().is_some());
        let exported = Command::new(env!("CARGO_BIN_EXE_gw"))
            .env_clear()
            .args(["gen", "export", "--db"])
            .arg(&db)
            .arg("--out")
            .arg(&artifact)
            .args([
                "--run-id",
                "historical",
                "--format",
                "chat-ml",
                "--cot",
                "masked",
            ])
            .output()
            .unwrap();
        assert!(
            exported.status.success(),
            "{}",
            String::from_utf8_lossy(&exported.stderr)
        );
        let gw_storage::ArtifactVerification::Verified(verified) =
            gw_storage::verify_artifact(&artifact).unwrap()
        else {
            panic!("verified export")
        };
        assert_eq!(verified.manifest.n_admitted, 1);
        let publication_id: String =
            sqlx::query_scalar("SELECT publication_id FROM export_receipts")
                .fetch_one(store.raw_pool())
                .await
                .unwrap();
        let recovered = Command::new(env!("CARGO_BIN_EXE_gw"))
            .env_clear()
            .args(["gen", "export", "--db"])
            .arg(&db)
            .arg("--resume-publication")
            .arg(&publication_id)
            .output()
            .unwrap();
        assert!(
            recovered.status.success(),
            "{}",
            String::from_utf8_lossy(&recovered.stderr)
        );
        let after = store.get("stored").await.unwrap();
        assert_eq!(serde_json::to_vec(&after).unwrap(), bytes);
        assert_eq!(gw_storage::record_hash(&after).unwrap(), hash);
        assert_eq!(after.judging.n_eff, before.judging.n_eff);
        assert_eq!(after.judging.verdict, Some(Verdict::Admit));
        let after_metadata: String =
            sqlx::query_scalar("SELECT config_json FROM runs WHERE run_id='historical'")
                .fetch_one(store.raw_pool())
                .await
                .unwrap();
        assert_eq!(after_metadata, metadata);
        store.close().await;
        std::fs::remove_file(config).unwrap();
        std::fs::remove_file(prompts).unwrap();
        std::fs::remove_file(artifact).unwrap();
        common::cleanup_db(&db);
    }
}
