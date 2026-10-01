//! Real CLI terminal summaries read the durable ledger after normal and failed runs.
mod common;
#[path = "../../gw-engine/tests/attempt_common/mod.rs"]
mod fixture;
use common::{cleanup_db, unique_temp_path};
use fixture::{Response, Server};
use gw_storage::Store;
use std::process::Command;

#[tokio::test]
async fn completed_replay_reports_all_prior_attempts_without_a_second_request() {
    exercise(false, false).await;
}
#[tokio::test]
async fn failure_reports_spend_and_readback_failure_preserves_the_primary_error() {
    exercise(true, false).await;
    exercise(true, true).await;
}
async fn exercise(fail: bool, break_summary: bool) {
    let server = Server::new(move |_, req, _| {
        if req["model"] == "judge-fixture" {
            if fail {
                let mut response = Response::ok("unauthorized".into());
                response.status = 401;
                response
            } else {
                Response::ok(fixture::grade(
                    r#"{"score":0.95,"verdict":"accept","confidence":0.9}"#,
                    Some(0.2),
                ))
            }
        } else {
            Response::ok(fixture::teacher(
                "The answer follows from these steps.",
                "stop",
                Some(0.1),
            ))
        }
    })
    .await;
    let db = unique_temp_path("accounting.sqlite");
    let config = unique_temp_path("accounting.toml");
    let prompts = unique_temp_path("accounting-prompts.txt");
    std::fs::write(
        &prompts,
        "Explain how to compare two independent methods of solving a reasoning problem.\n",
    )
    .unwrap();
    std::fs::write(&config, format!("db = {:?}\nmodel_api_base_url = {:?}\naccounting_policy = {{ mode = \"finite_usd\", limit_usd = 5.0 }}\n[area]\nadmission_intent = \"review_only\"\nteacher_slug = \"teacher-fixture\"\n[[area.judges]]\nslug = \"judge-fixture\"\nfamily = \"fixture\"\n", db.to_str().unwrap(), server.url)).unwrap();
    let store = Store::open(&db).await.unwrap();
    if break_summary {
        sqlx::query("CREATE TRIGGER break_terminal_summary AFTER UPDATE OF status ON runs WHEN NEW.status = 'failed' BEGIN UPDATE run_accounting SET policy_json = 'invalid-json' WHERE run_id = NEW.run_id; END").execute(store.raw_pool()).await.unwrap();
    }
    let command = |mode: &str| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_gw"));
        command
            .env_clear()
            .env("MODEL_API_KEY", "fixture")
            .args(["gen", mode, "--config"])
            .arg(&config)
            .args(["--run-id", "report", "--prompts"])
            .arg(&prompts)
            .args(["--shards", "1"]);
        command
    };
    let mut child = command("run");
    let output = tokio::task::spawn_blocking(move || child.output().unwrap())
        .await
        .unwrap();
    let text = String::from_utf8(output.stdout).unwrap();
    let error = String::from_utf8(output.stderr).unwrap();
    assert_eq!(output.status.success(), !fail, "{text}\n{error}");
    assert!(text.contains("requested policy: finite_usd"), "{text}");
    if fail {
        assert!(error.contains("401"), "primary error lost: {error}");
    }
    if break_summary {
        assert!(error.contains("accounting unavailable"), "{error}");
    } else {
        assert!(text.contains("physical attempts 2"), "{text}");
        assert!(text.contains("client wall time:"), "{text}");
        assert!(text.contains("prompt/input tokens:"), "{text}");
        assert!(
            text.contains(if fail {
                "known USD $0.1000; unknown cost 1"
            } else {
                "known USD $0.3000; unknown cost 0"
            }),
            "{text}"
        );
    }
    if !fail {
        let mut child = command("replay");
        let replay = tokio::task::spawn_blocking(move || child.output().unwrap())
            .await
            .unwrap();
        assert!(
            replay.status.success(),
            "{}",
            String::from_utf8_lossy(&replay.stderr)
        );
        assert!(
            String::from_utf8_lossy(&replay.stdout)
                .contains("physical attempts 2; known USD $0.3000")
        );
    }
    assert_eq!(server.requests.lock().unwrap().len(), 2);
    store.close().await;
    cleanup_db(&db);
    std::fs::remove_file(config).unwrap();
    std::fs::remove_file(prompts).unwrap();
}
