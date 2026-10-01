//! Explicit screened publication uses local protected inputs and transactional store membership.
use std::process::Command;

#[test]
fn screened_export_command_is_available_without_provider_configuration() {
    let output = Command::new(env!("CARGO_BIN_EXE_gw"))
        .args(["gen", "export-screened", "--help"])
        .env_remove("MODEL_API_KEY")
        .env("DATABASE_URL", "invalid://must-not-open")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let help = String::from_utf8(output.stdout).unwrap();
    for flag in ["--db", "--plan", "--protected", "--out"] {
        assert!(help.contains(flag), "missing {flag}");
    }
}

#[path = "../../gw-eval/tests/screened_publication_support/mod.rs"]
mod screened_publication_support;
#[path = "../../gw-eval/tests/screening_support/mod.rs"]
mod screening_support;
use gw_schema::*;
use screened_publication_support::*;
use screening_support::*;

fn dialogue(id: &str, first_reason: &str) -> TrainingRecord {
    let mut row = record(id, "start");
    row.messages[1].reasoning = Some(first_reason.into());
    row.messages[1].reasoning_details = Some(vec![ReasoningDetail::Text {
        text: first_reason.into(),
        index: 0,
        id: Some("detail-1".into()),
        signature: Some("synthetic-signature".into()),
        format: Some("synthetic-format".into()),
    }]);
    row.messages.extend([
        message(Role::User, "continue"),
        message(Role::Assistant, "final answer"),
    ]);
    row.messages[3].reasoning = Some("final reasoning".into());
    rebind_task(&mut row);
    row
}

#[tokio::test]
async fn actual_screened_cli_publication_generates_verified_consumer_golden_artifacts() {
    let fixtures =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../adapters/trl/tests/fixtures");
    for case in [
        "all",
        "final",
        "empty",
        "empty-final",
        "empty-gemma",
        "collision-a",
        "collision-b",
    ] {
        let mut a = dialogue("screened-a", "earlier left");
        let mut b = dialogue("screened-b", "earlier right");
        let collision = case.starts_with("collision");
        if !collision {
            b.provenance.run_id = a.provenance.run_id.clone();
            b.messages[1] = a.messages[1].clone();
            b.messages[3].content =
                Content::Text("longer final answer from the second candidate".into());
            a.generation.n_completions = Some(2);
            b.generation.n_completions = Some(2);
            b.generation.completion_index = Some(1);
        }
        let rows = vec![a, b];
        let mut declared = declaration(&rows);
        declared.policy.cot_policy = CotPolicy::Masked;
        if case.contains("final") || collision {
            declared.policy.multi_turn_loss = MultiTurnLoss::FinalTurnOnly;
        }
        if case.contains("gemma") {
            declared.policy.target = TrlFormat::Gemma4;
        }
        if case.starts_with("empty") {
            declared.output.record_ids.clear();
        }
        if case == "collision-b" {
            declared.output.run_id = rows[1].provenance.run_id.clone();
            declared.output.record_ids = vec![rows[1].record_id.clone()];
        }
        let sets = protected();
        let plan = gw_eval::screening::prepare_screening(&rows, &declared, &sets, None).unwrap();
        assert_eq!(
            plan.lexical_status,
            LexicalScreeningStatus::CompleteNoMatch,
            "{case}: {:?}",
            plan.incomplete
        );
        if collision {
            assert_eq!(
                plan.groups.len(),
                2,
                "source prompts differ in historical reasoning"
            );
        }
        let out = Temp::new();
        let store = setup(&rows, &out).await;
        let plan_path = out.0.join("plan.json");
        let protected_path = out.0.join("protected.json");
        std::fs::write(&plan_path, serde_json::to_vec(&plan).unwrap()).unwrap();
        std::fs::write(&protected_path, serde_json::to_vec(&sets).unwrap()).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_gw"))
            .args(["gen", "export-screened", "--db"])
            .arg(out.0.join("store.sqlite"))
            .arg("--plan")
            .arg(plan_path)
            .arg("--protected")
            .arg(&protected_path)
            .arg("--out")
            .arg(out.artifact())
            .env_remove("MODEL_API_KEY")
            .env("DATABASE_URL", "invalid://must-not-open")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{case}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            result["artifact"]["screening"]["population_check"],
            "transaction_checked"
        );
        let bytes = std::fs::read(out.artifact()).unwrap();
        let verified = gw_storage::verify_artifact_snapshot(bytes.clone()).unwrap();
        assert_eq!(
            serde_json::to_value(&verified.artifact).unwrap(),
            result["artifact"]
        );
        assert!(
            store
                .lifecycle_history("screened-a")
                .await
                .unwrap()
                .is_empty()
        );
        std::fs::remove_file(protected_path).unwrap();
        std::fs::remove_file(out.artifact()).unwrap();
        store
            .resume_export(result["publication_id"].as_str().unwrap())
            .await
            .unwrap();
        assert_eq!(std::fs::read(out.artifact()).unwrap(), bytes);
        let name = format!("screened-{case}.parquet");
        if let Some(destination) = std::env::var_os("GW_REGENERATE_SCREENED_TRL_FIXTURES") {
            std::fs::write(std::path::Path::new(&destination).join(&name), &bytes).unwrap();
        }
        let golden =
            gw_storage::verify_artifact_snapshot(std::fs::read(fixtures.join(name)).unwrap())
                .unwrap();
        assert_eq!(verified.artifact, golden.artifact);
    }
}
