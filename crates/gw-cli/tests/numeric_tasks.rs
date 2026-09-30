//! Real command paths reject invalid task input before credentials, SQLite, or provider requests.
mod common;
#[path = "../../gw-engine/tests/attempt_common/mod.rs"]
mod fixture;
use clap::Parser;
use gw_cli::cli::Cli;
use serde_json::{Value, json};
use std::process::Command;
const INPUT: &str = include_str!("../../../examples/reviewed-numeric-tasks.json");

struct Temp(std::path::PathBuf);
impl Temp {
    fn new() -> Self {
        let path = common::unique_temp_path("numeric-input");
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn run_replay_and_tui_require_exactly_one_explicit_source() {
    for action in ["run", "replay", "tui"] {
        let base = vec!["gw", "gen", action, "--run-id", "fixture", "--shards", "1"];
        assert!(Cli::try_parse_from(base.clone()).is_err());
        let mut both = base.clone();
        both.extend(["--prompts", "prompts.txt", "--tasks", "tasks.json"]);
        assert!(Cli::try_parse_from(both).is_err());
        for source in ["--prompts", "--tasks"] {
            let mut one = base.clone();
            one.extend([source, "source-file"]);
            assert!(Cli::try_parse_from(one).is_ok(), "{action}/{source}");
        }
    }
}

#[tokio::test]
async fn invalid_tasks_fail_before_credentials_store_creation_and_all_provider_calls() {
    let server = fixture::Server::new(|_, _, _| {
        let mut response = fixture::Response::ok("unexpected dispatch".into());
        response.status = 401;
        response
    })
    .await;
    let temp = Temp::new();
    let config = temp.0.join("config.toml");
    std::fs::write(&config, format!("provider_base_url = {:?}\n[area]\nadmission_intent='review_only'\n[[area.judges]]\nslug='fixture-judge'\nfamily='fixture'\n", server.url)).unwrap();
    let valid: Value = serde_json::from_str(INPUT).unwrap();
    let mut invalid = vec!["{bad-json".to_owned()];
    for (pointer, value) in [
        ("/version", json!(999)),
        ("/tasks/0/source/item", json!(" ")),
        ("/tasks/0/verification/oracle/expected", json!("1e309")),
        (
            "/tasks/0/verification/numeric/tolerance/absolute",
            json!(-1),
        ),
        (
            "/tasks/0/verification/oracle/oracle",
            json!("sandbox_execution"),
        ),
        (
            "/tasks/0/prompt/content",
            json!([{"type":"text","text":"not flattened"}]),
        ),
    ] {
        let value = {
            let mut changed = valid.clone();
            *changed.pointer_mut(pointer).unwrap() = value;
            changed
        };
        invalid.push(value.to_string());
    }
    let mut unknown = valid.clone();
    unknown["tasks"][0]["claimed_digest"] = json!("not-trusted");
    invalid.push(unknown.to_string());
    let mut duplicate = valid.clone();
    duplicate["tasks"][1] = duplicate["tasks"][0].clone();
    invalid.push(duplicate.to_string());
    let mut split = valid;
    split["tasks"][1]["group"] = split["tasks"][0]["group"].clone();
    split["tasks"][1]["split"]["role"] = json!("test");
    invalid.push(split.to_string());
    for (ordinal, input) in invalid.iter().enumerate() {
        let file = temp.0.join(format!("invalid-{ordinal}.json"));
        std::fs::write(&file, input).unwrap();
        for action in ["run", "replay", "tui"] {
            for has_key in [false, true] {
                let db = temp.0.join(format!("{ordinal}-{action}-{has_key}.sqlite"));
                let mut command = Command::new(env!("CARGO_BIN_EXE_gw"));
                command
                    .env_clear()
                    .env("RUST_LOG", "off")
                    .args(["gen", action, "--config"])
                    .arg(&config)
                    .args(["--run-id", "invalid", "--shards", "1", "--tasks"])
                    .arg(&file)
                    .arg("--db")
                    .arg(&db);
                if has_key {
                    command.env("OPENROUTER_API_KEY", "fixture");
                }
                let output = tokio::task::spawn_blocking(move || command.output().unwrap())
                    .await
                    .unwrap();
                let error = String::from_utf8_lossy(&output.stderr);
                assert!(!output.status.success(), "{ordinal}/{action}/{has_key}");
                assert!(
                    !error.contains("OPENROUTER_API_KEY"),
                    "input must precede credentials: {error}"
                );
                assert!(!db.exists(), "invalid task input created store: {error}");
                assert!(
                    server.requests.lock().unwrap().is_empty(),
                    "invalid task input dispatched a provider request"
                );
            }
        }
    }
}
