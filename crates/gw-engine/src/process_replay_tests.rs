//! An interrupted real executor reuses committed output and cache on a new process.
use crate::{Clients, Engine, EventSink, ExportSpec, load_cursor};
use gw_schema::{CotPolicy, LifecycleState, TrlFormat};
use gw_storage::{RecordFilter, Store};
use std::{
    io::{BufRead, BufReader, Write},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};
use tokio_util::sync::CancellationToken;
#[path = "../tests/common/mod.rs"]
mod common;
use common::*;
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Directory(PathBuf);
impl Directory {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "gw-engine-recovery-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn db(&self) -> PathBuf {
        self.0.join("state.sqlite")
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        if std::thread::panicking() {
            eprintln!("preserved engine recovery files: {}", self.0.display());
        } else {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }
}
struct RunningChild(Child);
impl Drop for RunningChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
pub(crate) async fn boundary(clients: &Clients, stage: &str) {
    if clients.process_barrier != Some(stage) {
        return;
    }
    tokio::task::spawn_blocking(|| {
        println!("GW_ENGINE_STAGE:kill");
        std::io::stdout().flush().unwrap();
        let mut line = String::new();
        std::io::stdin().lock().read_line(&mut line).unwrap();
        assert_eq!(line.trim(), "RELEASE");
    })
    .await
    .unwrap();
}
fn kill_child(dir: &Directory, stage: &str) {
    let mut child = RunningChild(
        Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "process_replay_tests::child_entry",
                "--nocapture",
            ])
            .env("GW_ENGINE_RECOVERY_DB", dir.db())
            .env("GW_ENGINE_RECOVERY_STAGE", stage)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(std::fs::File::create(dir.0.join("child.stderr")).unwrap())
            .spawn()
            .unwrap(),
    );
    let stdout = child.0.stdout.take().unwrap();
    let (send, receive) = std::sync::mpsc::channel();
    let output = dir.0.join("child.stdout");
    let reader = std::thread::spawn(move || {
        let mut output = std::fs::File::create(output).unwrap();
        for line in BufReader::new(stdout).lines() {
            let line = line.unwrap();
            writeln!(output, "{line}").unwrap();
            if line.starts_with("GW_ENGINE_STAGE:") {
                let _ = send.send(line);
            }
        }
    });
    assert_eq!(
        receive
            .recv_timeout(Duration::from_secs(20))
            .unwrap_or_else(|error| panic!("{stage}: {error}; {}", dir.0.display())),
        "GW_ENGINE_STAGE:kill"
    );
    child.0.kill().unwrap();
    assert!(!child.0.wait().unwrap().success());
    reader.join().unwrap();
    for name in ["state.sqlite", "state.sqlite-wal", "state.sqlite-shm"] {
        let source = dir.0.join(name);
        if source.exists() {
            std::fs::copy(source, dir.0.join(format!("killed-{name}"))).unwrap();
        }
    }
}
fn engine(
    store: &Store,
    barrier: Option<&'static str>,
    replay: bool,
) -> (Engine, Arc<ScriptedTeacher>, Arc<ScriptedJudge>) {
    let teacher = Arc::new(ScriptedTeacher::new(
        if replay { vec![] } else { vec![good_cot(0.01)] },
        usize::from(!replay),
    ));
    let body = judge_body(0.95, "accept");
    let judge = Arc::new(ScriptedJudge::new(if replay {
        vec![]
    } else {
        vec![&body]
    }));
    let mut clients = clients(
        store.clone(),
        teacher.clone(),
        judge.clone(),
        EventSink::disconnected(),
    );
    clients.process_barrier = barrier;
    (
        Engine::new(clients, area_k1(one_judge(), lenient_thresholds()), 1),
        teacher,
        judge,
    )
}
#[test]
fn child_entry() {
    let Ok(path) = std::env::var("GW_ENGINE_RECOVERY_DB") else {
        return;
    };
    let stage = match std::env::var("GW_ENGINE_RECOVERY_STAGE").unwrap().as_str() {
        "before_cursor" => "before_cursor",
        "judge_cached" => "judge_cached",
        other => panic!("{other}"),
    };
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let store = Store::open(path).await.unwrap();
        let (engine, _, _) = engine(&store, Some(stage), false);
        engine
            .run("recovery-run", &one_item_source(), CancellationToken::new())
            .await
            .unwrap();
    });
}
fn export(dir: &Directory) -> ExportSpec {
    ExportSpec {
        dst: dir.0.join("data.parquet"),
        target: TrlFormat::ChatML,
        cot: CotPolicy::Masked,
        dataset_version: None,
    }
}
#[tokio::test]
async fn killed_executor_reuses_teacher_output_and_cached_judge_then_matches_clean_export() {
    let clean_dir = Directory::new("clean");
    let clean = Store::open(clean_dir.db()).await.unwrap();
    let (clean_engine, _, _) = engine(&clean, None, false);
    clean_engine
        .run("recovery-run", &one_item_source(), CancellationToken::new())
        .await
        .unwrap();
    let clean_manifest = clean_engine
        .export_shard("recovery-run", &export(&clean_dir))
        .await
        .unwrap();
    let clean_record = clean
        .scan(&RecordFilter::new().run_id("recovery-run"))
        .await
        .unwrap()
        .remove(0);
    for stage in ["before_cursor", "judge_cached"] {
        let dir = Directory::new(stage);
        kill_child(&dir, stage);
        let store = Store::open(dir.db()).await.unwrap();
        let before = store
            .scan(&RecordFilter::new().run_id("recovery-run"))
            .await
            .unwrap()
            .remove(0);
        assert_eq!(
            before.lifecycle.state,
            if stage == "before_cursor" {
                LifecycleState::Formatted
            } else {
                LifecycleState::Verified
            }
        );
        assert_eq!(
            load_cursor(&store, "recovery-run", 0)
                .await
                .unwrap()
                .next_offset,
            0
        );
        let (replay, teacher, judge) = engine(&store, None, true);
        let report = replay
            .run("recovery-run", &one_item_source(), CancellationToken::new())
            .await
            .unwrap();
        assert!(report.completed);
        assert_eq!(report.admitted, 1);
        assert_eq!(teacher.call_count(), 0);
        assert_eq!(judge.call_count(), 0);
        assert_eq!(
            load_cursor(&store, "recovery-run", 0)
                .await
                .unwrap()
                .next_offset,
            1
        );
        let manifest = replay
            .export_shard("recovery-run", &export(&dir))
            .await
            .unwrap();
        let after = store.get(&before.record_id).await.unwrap();
        assert_eq!(after.lifecycle.state, LifecycleState::Exported);
        assert_eq!(after.hashes.record_hash, clean_record.hashes.record_hash);
        assert_eq!(manifest.build_inputs_hash, clean_manifest.build_inputs_hash);
        assert_eq!(
            after.lifecycle.history.len(),
            clean_record.lifecycle.history.len()
        );
        store.close().await;
    }
    clean.close().await;
}
