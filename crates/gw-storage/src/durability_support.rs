//! Private file-backed fixtures and a kill/reap harness shared by durability tests.
use crate::Store;
use gw_schema::TrainingRecord;
use std::{
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};
static NEXT: AtomicU64 = AtomicU64::new(0);
pub(crate) struct Directory(pub PathBuf);
impl Directory {
    pub fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "gw-durability-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    pub fn db(&self) -> PathBuf {
        self.0.join("state.sqlite")
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        if std::thread::panicking() {
            eprintln!(
                "preserved recovery database and diagnostics: {}",
                self.0.display()
            );
        } else {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }
}
pub(crate) fn record() -> TrainingRecord {
    serde_json::from_value(serde_json::json!({
        "record_id":"record", "schema_version":"1.0.0", "training_area":"fixture",
        "messages":[{"role":"user","content":"2+3"},{"role":"assistant","content":"5","reasoning":"addition"}],
        "provenance":{"run_id":"run","teacher":{"provider":"fixture","slug":"fixture"},"harness_version":"fixture"},
        "generation":{}, "lifecycle":{"state":"assistant_generated","history":[
            {"state":"user_synthesized","at":"2026-01-01T00:00:00Z","attempt":0},
            {"state":"assistant_generated","at":"2026-01-01T00:00:00Z","attempt":0}
        ]}
    })).unwrap()
}
pub(crate) async fn initialized(path: &Path) -> Store {
    let store = Store::open(path).await.unwrap();
    store
        .insert_historical_run("run", "{}", None)
        .await
        .unwrap();
    store.insert_record(&record()).await.unwrap();
    store
}
pub(crate) async fn integrity(store: &Store) {
    let mode: String = sqlx::query_scalar("PRAGMA journal_mode")
        .fetch_one(store.pool())
        .await
        .unwrap();
    let sync: i64 = sqlx::query_scalar("PRAGMA synchronous")
        .fetch_one(store.pool())
        .await
        .unwrap();
    let fk: i64 = sqlx::query_scalar("PRAGMA foreign_keys")
        .fetch_one(store.pool())
        .await
        .unwrap();
    let check: String = sqlx::query_scalar("PRAGMA integrity_check")
        .fetch_one(store.pool())
        .await
        .unwrap();
    assert_eq!(
        (mode.as_str(), sync, fk, check.as_str()),
        ("wal", 1, 1, "ok")
    );
    assert!(
        sqlx::query("PRAGMA foreign_key_check")
            .fetch_all(store.pool())
            .await
            .unwrap()
            .is_empty()
    );
    let applied: Vec<(i64, Vec<u8>, bool)> =
        sqlx::query_as("SELECT version, checksum, success FROM _sqlx_migrations ORDER BY version")
            .fetch_all(store.pool())
            .await
            .unwrap();
    let migrations = sqlx::migrate!("./migrations");
    assert_eq!(applied.len(), migrations.iter().count());
    for ((version, checksum, success), migration) in applied.iter().zip(migrations.iter()) {
        assert_eq!(*version, migration.version);
        assert_eq!(checksum.as_slice(), migration.checksum.as_ref());
        assert!(*success);
    }
}
pub(crate) async fn snapshot(store: &Store) -> serde_json::Value {
    let envelope: (String,String,String,Option<String>,Option<f64>,String,String,String) = sqlx::query_as("SELECT record_id,run_id,lifecycle_state,verdict,judge_aggregate,record_hash,prompt_hash,record_json FROM records WHERE record_id='record'").fetch_one(store.pool()).await.unwrap();
    type HistoryRow = (
        String,
        String,
        Option<String>,
        Option<String>,
        Option<i64>,
        Option<i64>,
    );
    let history: Vec<HistoryRow> = sqlx::query_as("SELECT state,at,detail,mutation_id,history_ordinal,attempt FROM lifecycle_history WHERE record_id='record' ORDER BY id").fetch_all(store.pool()).await.unwrap();
    let commands: Vec<(String,String,String)> = sqlx::query_as("SELECT mutation_id,kind,committed_at FROM record_mutations WHERE record_id='record' ORDER BY mutation_id").fetch_all(store.pool()).await.unwrap();
    serde_json::json!([envelope, history, commands])
}
struct RunningChild(Child);
impl Drop for RunningChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
/// The parent kills only after a child emits an exact barrier. No timer guesses commit ordering.
pub(crate) fn kill_at(directory: &Directory, entry: &str, scenario: &str) {
    let diagnostics = std::fs::File::create(directory.0.join("child.stderr")).unwrap();
    let mut child = RunningChild(
        Command::new(std::env::current_exe().unwrap())
            .args(["--exact", entry, "--nocapture"])
            .env("GW_DURABILITY_DB", directory.db())
            .env("GW_DURABILITY_SCENARIO", scenario)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(diagnostics)
            .spawn()
            .unwrap(),
    );
    let stdout = child.0.stdout.take().unwrap();
    let (send, receive) = std::sync::mpsc::channel();
    let output = directory.0.join("child.stdout");
    let reader = std::thread::spawn(move || {
        use std::io::Write;
        let mut output = std::fs::File::create(output).unwrap();
        for line in BufReader::new(stdout).lines() {
            let line = line.unwrap();
            writeln!(output, "{line}").unwrap();
            if line.starts_with("GW_STAGE:") {
                let _ = send.send(line);
            }
        }
    });
    let observed = receive
        .recv_timeout(Duration::from_secs(20))
        .unwrap_or_else(|error| {
            panic!(
                "child did not reach {scenario}: {error}; diagnostics {}",
                directory.0.display()
            )
        });
    assert_eq!(observed, "GW_STAGE:kill");
    child.0.kill().unwrap();
    let status = child.0.wait().unwrap();
    assert!(!status.success());
    reader.join().unwrap();
    // Preserve the pre-recovery files as well as any subsequently recovered database on failure.
    for name in ["state.sqlite", "state.sqlite-wal", "state.sqlite-shm"] {
        let source = directory.0.join(name);
        if source.exists() {
            std::fs::copy(&source, directory.0.join(format!("killed-{name}"))).unwrap();
        }
    }
}
pub(crate) async fn pause(store: &Store) {
    store.set_test_hook(Some(crate::test_hooks::Hook {
        operation: "ack",
        stage: "ack",
        action: crate::test_hooks::Action::Process("kill"),
    }));
    store.test_boundary("ack", "ack").await.unwrap();
}
