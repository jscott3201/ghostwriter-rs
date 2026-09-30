//! Current executable verification requires explicit policy and supported facts; historical export
//! remains provider-free and preserves the stored record.
mod common;
use gw_engine::SeedSource;
use gw_schema::{AccountingCapability, AccountingPolicy, LifecycleState, Verdict};
use std::process::Command;

#[tokio::test]
async fn legacy_facts_are_inspectable_exportable_and_denied_before_cli_credentials() {
    for variant in ["missing-policy", "missing-facts", "unsupported-facts"] {
        let config = common::unique_temp_path("verification.toml");
        let prompts = common::unique_temp_path("verification.txt");
        let db = common::unique_temp_path("verification.sqlite");
        let artifact = common::unique_temp_path("verification.parquet");
        std::fs::write(&config, "[area]\nadmission_intent='review_only'\n[[area.judges]]\nslug='judge'\nfamily='family'\n").unwrap();
        std::fs::write(&prompts, "Explain the result\n").unwrap();
        let mut settings: gw_cli::config::Config = serde_json::from_value(serde_json::json!({"area":{"admission_intent":"review_only","judges":[{"slug":"judge","family":"family"}]}})).unwrap();
        settings.db = db.clone();
        let source =
            gw_cli::seedsource::FileSeedSource::from_prompts_str("Explain the result", 1).unwrap();
        let contract = source.items_for_shard(0).remove(0).candidate.contract;
        assert_eq!(
            contract.answer_policy,
            Some(gw_schema::VerificationPolicy::Absent)
        );
        assert_eq!(
            contract.execution_policy,
            Some(gw_schema::VerificationPolicy::Absent)
        );
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
        let mut rec = common::record(
            "historical",
            "pinned",
            Some(Verdict::Admit),
            Some(0.99),
            true,
            "question",
        );
        rec.lifecycle.state = LifecycleState::Formatted;
        rec.verification_contract = Some(contract);
        if variant == "missing-policy" {
            rec.verification_contract.as_mut().unwrap().answer_policy = None;
        } else if variant == "unsupported-facts" {
            rec.verification = gw_judge::run_verifier(
                &gw_judge::VerifierInput {
                    messages: &rec.messages,
                    reasoning_tokens: 0,
                    cot_required: false,
                    contract: rec.verification_contract.as_ref(),
                    execution_evidence: None,
                    evidence_key: Default::default(),
                },
                &gw_judge::NullSandboxOracle,
            )
            .unwrap()
            .verification;
            rec.verification.interpretation.as_mut().unwrap().version += 1;
        }
        store.put(&rec).await.unwrap();
        let before = store.get("historical").await.unwrap();
        let bytes = serde_json::to_vec(&before).unwrap();
        let hash = gw_storage::record_hash(&before).unwrap();
        let accounting = store.accounting_snapshot("pinned").await.unwrap();
        for action in ["run", "replay", "tui"] {
            let output = Command::new(env!("CARGO_BIN_EXE_gw"))
                .env_clear()
                .args(["gen", action, "--config"])
                .arg(&config)
                .args(["--run-id", "pinned", "--shards", "1", "--prompts"])
                .arg(&prompts)
                .arg("--db")
                .arg(&db)
                .output()
                .unwrap();
            let error = String::from_utf8_lossy(&output.stderr);
            assert!(!output.status.success(), "{variant}/{action}");
            assert!(
                error.contains("verification"),
                "{variant}/{action}: {error}"
            );
            assert!(
                !error.contains("OPENROUTER_API_KEY"),
                "{variant}/{action}: {error}"
            );
            assert_eq!(
                store.accounting_snapshot("pinned").await.unwrap(),
                accounting
            );
            assert_eq!(
                serde_json::to_vec(&store.get("historical").await.unwrap()).unwrap(),
                bytes
            );
        }
        let exported = Command::new(env!("CARGO_BIN_EXE_gw"))
            .env_clear()
            .args(["gen", "export", "--db"])
            .arg(&db)
            .arg("--out")
            .arg(&artifact)
            .args([
                "--run-id", "pinned", "--format", "chat-ml", "--cot", "masked",
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
            panic!("export must carry verified artifact metadata")
        };
        assert_eq!(verified.manifest.n_admitted, 1);
        let after = store.get("historical").await.unwrap();
        assert_eq!(serde_json::to_vec(&after).unwrap(), bytes);
        assert_eq!(gw_storage::record_hash(&after).unwrap(), hash);
        drop(store);
        std::fs::remove_file(config).unwrap();
        std::fs::remove_file(prompts).unwrap();
        std::fs::remove_file(artifact).unwrap();
        common::cleanup_db(&db);
    }
}
