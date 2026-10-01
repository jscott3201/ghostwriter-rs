//! Real owned Docker operations with delayed/lost client acknowledgments.
use super::containment_tests::probe_input;
use super::*;
use gw_schema::ExecutionOutcome;
use std::path::{Path, PathBuf};

struct Shim {
    directory: PathBuf,
    executable: PathBuf,
}
impl Shim {
    fn new(phase: &str, lost_ack: bool) -> Self {
        use std::os::unix::fs::PermissionsExt;
        let directory = std::env::temp_dir().join(format!("gw-coding-shim-{}", nonce().unwrap()));
        std::fs::create_dir(&directory).unwrap();
        let executable = directory.join("docker-control");
        // Paths are task-owned random hexadecimal names. This shim still executes the actual
        // local Docker client; no reported process/container result is synthesized as success.
        let body = if lost_ack {
            format!(
                "if [ ! -f '{0}/probe-settled' ]; then\ntouch '{0}/probe-settled'\nexec docker \"$@\"\nfi\ndocker \"$@\"\nexit 1",
                directory.display()
            )
        } else {
            format!(
                "touch '{}/dispatched'\nwhile [ ! -f '{}/release' ]; do sleep 0.02; done\nexec docker \"$@\"",
                directory.display(),
                directory.display()
            )
        };
        let script = format!(
            "#!/bin/sh\nfor argument in \"$@\"; do\nif [ \"$argument\" = '{phase}' ]; then\n{body}\nfi\ndone\nexec docker \"$@\"\n"
        );
        std::fs::write(&executable, script).unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
        Self {
            directory,
            executable,
        }
    }
    fn release(&self) {
        std::fs::write(self.directory.join("release"), "go").unwrap();
    }
}
impl Drop for Shim {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

async fn wait_path(path: &Path, present: bool) {
    tokio::time::timeout(Duration::from_secs(20), async {
        while path.exists() != present {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("supervisor did not settle within the controlled test bound");
}
fn containers() -> Vec<String> {
    let result = std::process::Command::new("docker")
        .args(["ps", "-aq", "--no-trunc"])
        .output()
        .unwrap();
    assert!(result.status.success());
    let mut ids = String::from_utf8(result.stdout)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    ids.sort();
    ids
}
fn owned_containers() -> Vec<String> {
    let output = std::process::Command::new("docker")
        .args([
            "ps",
            "-aq",
            "--no-trunc",
            "--filter",
            "label=io.ghostwriter.coding-owner",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect()
}

#[tokio::test]
#[ignore = "requires cached qualified local Docker; no pulls; serialized container inventory"]
async fn cancellation_settles_pending_create_and_start_before_absence() {
    for phase in ["create", "start"] {
        let before = containers();
        let shim = Shim::new(phase, false);
        let docker = Docker::connect_test(shim.executable.clone()).await.unwrap();
        let config = docker.config.clone();
        let cancel = CancellationToken::new();
        let task = tokio::spawn(observe(
            probe_input("def probe():\n    return True\n"),
            cancel.clone(),
            Some(docker),
        ));
        wait_path(&shim.directory.join("dispatched"), true).await;
        cancel.cancel();
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            !task.is_finished(),
            "cannot return from an early absence while {phase} can still execute"
        );
        shim.release();
        let artifact = task.await.unwrap().unwrap().consume();
        assert_eq!(artifact.report.outcome, ExecutionOutcome::Unknown);
        assert!(artifact.report.cases.iter().all(|case| case.settled));
        assert!(!config.exists());
        assert_eq!(containers(), before);
    }
}

#[tokio::test]
#[ignore = "requires cached qualified local Docker; no pulls; serialized container inventory"]
async fn dropping_the_caller_keeps_create_start_cleanup_supervised() {
    for phase in ["create", "start"] {
        let before = containers();
        let shim = Shim::new(phase, false);
        let docker = Docker::connect_test(shim.executable.clone()).await.unwrap();
        let config = docker.config.clone();
        let caller = tokio::spawn(observe(
            probe_input("def probe():\n    return True\n"),
            CancellationToken::new(),
            Some(docker),
        ));
        wait_path(&shim.directory.join("dispatched"), true).await;
        caller.abort();
        assert!(matches!(caller.await,Err(error) if error.is_cancelled()));
        assert!(
            config.exists(),
            "supervisor must retain ownership during the pending mutation"
        );
        shim.release();
        wait_path(&config, false).await;
        assert_eq!(containers(), before);
    }
}

#[tokio::test]
#[ignore = "requires cached qualified local Docker; no pulls; serialized container inventory"]
async fn lost_cleanup_acknowledgment_never_enters_positive_consumption() {
    let before = containers();
    let shim = Shim::new("rm", true);
    let docker = Docker::connect_test(shim.executable.clone()).await.unwrap();
    let artifact = observe(
        probe_input("def probe():\n    return True\n"),
        CancellationToken::new(),
        Some(docker),
    )
    .await
    .unwrap()
    .consume();
    assert_eq!(artifact.report.outcome, ExecutionOutcome::Unknown);
    assert_eq!(artifact.report.cases[0].reason, CodingCaseReason::Matched);
    assert_eq!(artifact.report.cases[0].status, TestStatus::Passed);
    assert!(!artifact.report.cases[0].settled);
    assert_eq!(
        artifact
            .report
            .native_verification
            .execution
            .observation
            .as_ref()
            .unwrap()
            .outcome,
        gw_schema::VerificationOutcome::Unknown
    );
    assert_eq!(
        containers(),
        before,
        "real rm happened despite its missing acknowledgment"
    );
}

#[tokio::test]
#[ignore = "requires cached qualified local Docker; no pulls; serialized container inventory"]
async fn cancellation_terminates_running_candidate_and_descendants() {
    let before = containers();
    let code = "def probe():\n    import subprocess,sys,time\n    subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n    while True:\n        time.sleep(0.01)\n";
    let cancel = CancellationToken::new();
    let task = tokio::spawn(observe_coding(probe_input(code), cancel.clone()));
    // The trusted runtime probe also executes first; wait for a candidate exec process using a
    // bounded Docker top read of only the newly owned containers.
    let started = tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            let mut found = false;
            for id in owned_containers()
                .into_iter()
                .filter(|id| !before.contains(id))
            {
                let output = std::process::Command::new("docker")
                    .args(["top", &id, "-eo", "pid,ppid,comm"])
                    .output()
                    .unwrap();
                // Trusted main + candidate wrapper + child. The interpreter-only probe has two.
                found |= output.status.success()
                    && String::from_utf8_lossy(&output.stdout).lines().count() >= 4;
            }
            if found {
                break;
            }
            tokio::time::sleep(Duration::from_millis(30)).await;
        }
    })
    .await;
    cancel.cancel();
    let artifact = task.await.unwrap().unwrap().consume();
    assert!(
        started.is_ok(),
        "candidate descendant was not observed before cancellation"
    );
    assert_eq!(artifact.report.outcome, ExecutionOutcome::Unknown);
    assert!(artifact.report.cases[0].settled);
    assert_eq!(containers(), before);
}
