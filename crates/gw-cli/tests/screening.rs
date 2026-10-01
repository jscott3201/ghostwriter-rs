//! Local supplied-corpus screening never requires provider or database configuration.
use std::process::Command;
mod common;
#[path = "../../gw-eval/tests/screening_support/mod.rs"]
mod screening_support;
use gw_schema::*;

struct Files {
    records: std::path::PathBuf,
    declaration: std::path::PathBuf,
    protected: std::path::PathBuf,
    plan: std::path::PathBuf,
}
impl Files {
    fn new(
        rows: &[TrainingRecord],
        declaration: &ScreeningDeclaration,
        protected: &[ProtectedScreeningSet],
    ) -> Self {
        let files = Self {
            records: common::unique_temp_path("screening-records.json"),
            declaration: common::unique_temp_path("screening-declaration.json"),
            protected: common::unique_temp_path("screening-protected.json"),
            plan: common::unique_temp_path("screening-plan.json"),
        };
        std::fs::write(&files.records, serde_json::to_vec(rows).unwrap()).unwrap();
        std::fs::write(&files.declaration, serde_json::to_vec(declaration).unwrap()).unwrap();
        std::fs::write(&files.protected, serde_json::to_vec(protected).unwrap()).unwrap();
        files
    }
    fn run(&self, check: bool) -> std::process::Output {
        let mut command = Command::new(env!("CARGO_BIN_EXE_gw"));
        command
            .args(["eval", "screen", "--records"])
            .arg(&self.records)
            .arg("--declaration")
            .arg(&self.declaration)
            .arg("--protected")
            .arg(&self.protected)
            .env_remove("OPENROUTER_API_KEY")
            .env("DATABASE_URL", "invalid://must-not-open");
        for (key, _) in std::env::vars_os() {
            if key.to_string_lossy().starts_with("GW_") {
                command.env_remove(key);
            }
        }
        if check {
            command.arg("--check-plan").arg(&self.plan);
        }
        command.output().unwrap()
    }
}
impl Drop for Files {
    fn drop(&mut self) {
        for path in [
            &self.records,
            &self.declaration,
            &self.protected,
            &self.plan,
        ] {
            let _ = std::fs::remove_file(path);
        }
    }
}

#[test]
fn frozen_screening_command_is_available_without_credentials() {
    let output = Command::new(env!("CARGO_BIN_EXE_gw"))
        .args(["eval", "screen", "--help"])
        .env_remove("OPENROUTER_API_KEY")
        .env("DATABASE_URL", "invalid://must-not-open")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let help = String::from_utf8(output.stdout).unwrap();
    for flag in ["--records", "--declaration", "--protected"] {
        assert!(help.contains(flag), "missing {flag}: {help}");
    }
}

#[test]
fn supplied_file_command_emits_and_revalidates_source_only_plan() {
    let rows = vec![screening_support::record("a", "independent synthetic task")];
    let mut declared = screening_support::declaration(&rows);
    let files = Files::new(&rows, &declared, &screening_support::protected());
    let first = files.run(false);
    assert_eq!(
        first.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let plan: FrozenScreeningPlan = serde_json::from_slice(&first.stdout).unwrap();
    assert_eq!(plan.lexical_status, LexicalScreeningStatus::CompleteNoMatch);
    assert_eq!(plan.semantic_status, SemanticScreeningStatus::NotRun);
    assert_eq!(
        plan.population_check,
        ScreeningPopulationCheck::SuppliedFilesOnly
    );
    assert_eq!(
        plan.effective_prompt_separation,
        EffectivePromptSeparation::Unknown
    );
    assert_eq!(plan.counts.population_records, 1);
    assert_eq!(plan.counts.eligible_output_records, 1);
    std::fs::write(&files.plan, &first.stdout).unwrap();
    let verified = files.run(true);
    assert_eq!(verified.status.code(), Some(0));
    assert_eq!(verified.stdout, first.stdout);
    declared.policy.cot_policy = CotPolicy::Stripped;
    std::fs::write(&files.declaration, serde_json::to_vec(&declared).unwrap()).unwrap();
    let stale = files.run(true);
    assert_eq!(stale.status.code(), Some(1));
    assert!(stale.stdout.is_empty());
    assert!(String::from_utf8_lossy(&stale.stderr).contains("stale or altered"));
}

#[test]
fn incomplete_and_matched_inputs_have_explicit_reports_and_nonzero_status() {
    let rows = vec![screening_support::record("a", "independent synthetic task")];
    let declared = screening_support::declaration(&rows);
    let mut sets = screening_support::protected();
    sets[0].rights = None;
    let files = Files::new(&rows, &declared, &sets);
    let incomplete = files.run(false);
    assert_eq!(incomplete.status.code(), Some(2));
    let plan: FrozenScreeningPlan = serde_json::from_slice(&incomplete.stdout).unwrap();
    assert_eq!(plan.lexical_status, LexicalScreeningStatus::Incomplete);
    sets = screening_support::protected();
    sets[0].items[0].prompt = vec![rows[0].messages[0].clone()];
    sets[0].content_digest =
        gw_eval::screening::protected_screening_content_digest(&sets[0].items).unwrap();
    std::fs::write(&files.protected, serde_json::to_vec(&sets).unwrap()).unwrap();
    let matched = files.run(false);
    assert_eq!(matched.status.code(), Some(2));
    let plan: FrozenScreeningPlan = serde_json::from_slice(&matched.stdout).unwrap();
    assert_eq!(
        plan.lexical_status,
        LexicalScreeningStatus::MatchQuarantined
    );
    assert_eq!(plan.counts.quarantined_groups, 1);
    assert!(plan.eligible_output.is_empty());
}

#[test]
fn ambiguous_threshold_json_rejects_before_a_report_is_emitted() {
    let rows = vec![screening_support::record("a", "independent synthetic task")];
    let declared = screening_support::declaration(&rows);
    let files = Files::new(&rows, &declared, &screening_support::protected());
    let text = serde_json::to_string(&declared).unwrap().replace(
        r#"{"binary64":"3fe999999999999a"}"#,
        r#"{"binary64":"7ff0000000000000","binary64":"3fe999999999999a"}"#,
    );
    std::fs::write(&files.declaration, text).unwrap();
    let output = files.run(false);
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
}
