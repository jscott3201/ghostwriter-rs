//! One owned container per invocation. Cleanup follows every dispatched mutation before return.
use super::process::{CONTROL_LIMIT, OUTPUT_LIMIT, Output, Running};
use super::runtime::{Docker, IMAGE, IMAGE_ID, LABEL, POLICY, PYTHON, WAIT_MAIN, nonce};
use anyhow::{Context, ensure};
use serde_json::Value;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

pub(super) struct Invocation {
    pub output: Option<Output>,
    pub stop: Option<Stop>,
    pub settled: bool,
    pub container_id: Option<String>,
    pub elapsed_ms: u64,
}
#[derive(Debug, Clone, Copy)]
pub(super) enum Stop {
    Cancelled,
    WallLimit,
    OutputLimit,
    Infrastructure,
}

struct Owned<'a> {
    docker: &'a Docker,
    name: String,
    owner: String,
    id: Option<String>,
    mutation_acknowledged: bool,
}
impl<'a> Owned<'a> {
    fn new(docker: &'a Docker) -> anyhow::Result<Self> {
        let owner = nonce()?;
        Ok(Self {
            docker,
            name: format!("gw-coding-{owner}"),
            owner,
            id: None,
            mutation_acknowledged: true,
        })
    }
    async fn create_start(&mut self, cancel: &CancellationToken) -> anyhow::Result<()> {
        ensure!(!cancel.is_cancelled(), "evaluation cancelled before create");
        let label = format!("{LABEL}={}", self.owner);
        let mut args = vec!["create", "--name", &self.name, "--label", &label];
        args.extend(POLICY);
        args.extend([IMAGE, "-I", "-S", "-B", "-u", "-c", WAIT_MAIN]);
        // Cancellation does not abandon create/start requests. The unique name and owner label
        // exist before dispatch, allowing cleanup after lost stdout or failed client completion.
        self.mutation_acknowledged = false;
        let created = self.docker.call(&args).await?;
        ensure!(
            created.status.success(),
            "container creation was not acknowledged"
        );
        self.mutation_acknowledged = true;
        let value = self
            .inspect()
            .await?
            .context("created container disappeared")?;
        self.id = Some(
            value["Id"]
                .as_str()
                .context("container identity missing")?
                .to_owned(),
        );
        validate(&value)?;
        ensure!(!cancel.is_cancelled(), "evaluation cancelled after create");
        self.mutation_acknowledged = false;
        let started = self
            .docker
            .call(&["start", self.id.as_deref().expect("inspected identity")])
            .await?;
        ensure!(
            started.status.success(),
            "container start was not acknowledged"
        );
        self.mutation_acknowledged = true;
        ensure!(!cancel.is_cancelled(), "evaluation cancelled after start");
        Ok(())
    }
    async fn inspect(&self) -> anyhow::Result<Option<Value>> {
        let name = self.id.as_deref().unwrap_or(&self.name);
        let output = self.docker.call(&["container", "inspect", name]).await?;
        if !output.status.success() {
            ensure!(
                self.absent().await?,
                "owned container could not be inspected"
            );
            return Ok(None);
        }
        let values: Value = serde_json::from_slice(&output.stdout)?;
        let value = &values[0];
        ensure!(
            value["Config"]["Labels"][LABEL] == self.owner,
            "container ownership label mismatch"
        );
        if let Some(id) = &self.id {
            ensure!(value["Id"] == *id, "owned container identity changed");
        }
        Ok(Some(value.clone()))
    }
    async fn absent(&self) -> anyhow::Result<bool> {
        let filter = if let Some(id) = &self.id {
            format!("id={id}")
        } else {
            format!("name=^/{}$", self.name)
        };
        let output = self
            .docker
            .call(&[
                "container",
                "ls",
                "--all",
                "--quiet",
                "--no-trunc",
                "--filter",
                &filter,
            ])
            .await?;
        ensure!(
            output.status.success(),
            "Docker absence query was not acknowledged"
        );
        Ok(output.stdout.iter().all(u8::is_ascii_whitespace))
    }
    async fn remove(&self) -> anyhow::Result<()> {
        if let Some(value) = self.inspect().await? {
            let id = value["Id"]
                .as_str()
                .context("owned container lacks identity")?;
            let output = self.docker.call(&["rm", "--force", id]).await?;
            ensure!(
                output.status.success(),
                "owned container removal was not acknowledged"
            );
        }
        ensure!(
            self.absent().await?,
            "owned container remains after cleanup"
        );
        ensure!(
            self.mutation_acknowledged,
            "an earlier daemon mutation remains uncertain"
        );
        Ok(())
    }
}

pub(super) async fn invoke(
    docker: &Docker,
    program: &str,
    input: Vec<u8>,
    cancel: &CancellationToken,
    wall: Duration,
) -> anyhow::Result<Invocation> {
    let began = Instant::now();
    let mut owned = Owned::new(docker)?;
    let created = owned.create_start(cancel).await;
    if created.is_err() {
        let settled = owned.remove().await.is_ok();
        return Ok(Invocation {
            output: None,
            stop: Some(if cancel.is_cancelled() {
                Stop::Cancelled
            } else {
                Stop::Infrastructure
            }),
            settled,
            container_id: owned.id,
            elapsed_ms: millis(began.elapsed()),
        });
    }
    let id = owned.id.as_deref().expect("started owned container");
    let mut command = docker.command(&[
        "exec",
        "--interactive",
        id,
        PYTHON,
        "-I",
        "-S",
        "-B",
        "-u",
        "-c",
        program,
    ]);
    let running = Running::spawn(&mut command, input, OUTPUT_LIMIT);
    let mut running = match running {
        Ok(value) => value,
        Err(_) => {
            let settled = owned.remove().await.is_ok();
            return Ok(Invocation {
                output: None,
                stop: Some(Stop::Infrastructure),
                settled,
                container_id: owned.id,
                elapsed_ms: millis(began.elapsed()),
            });
        }
    };
    let (status, stop) = tokio::select! {
        status = running.child.wait() => (status.ok(),None),
        _ = cancel.cancelled() => (None,Some(Stop::Cancelled)),
        _ = running.exceeded.cancelled() => (None,Some(Stop::OutputLimit)),
        _ = tokio::time::sleep(wall) => (None,Some(Stop::WallLimit)),
    };
    // The short start call already settled. Removing this immutable container kills its main,
    // candidate, and descendants, including any exec request still attaching to this identity.
    let mut settled = owned.remove().await.is_ok();
    let output = if let Some(status) = status {
        running.finish(status).await.ok()
    } else {
        match tokio::time::timeout(CONTROL_LIMIT, running.child.wait()).await {
            Ok(Ok(status)) => running.finish(status).await.ok(),
            _ => {
                settled = false;
                running.terminate().await.ok()
            }
        }
    };
    // Join the client/readers before the final absence proof, never certify from an early lookup.
    settled &= owned.absent().await.unwrap_or(false);
    let stop = if output.is_none() {
        Some(Stop::Infrastructure)
    } else {
        stop
    };
    Ok(Invocation {
        output,
        stop,
        settled,
        container_id: owned.id,
        elapsed_ms: millis(began.elapsed()),
    })
}
fn millis(duration: Duration) -> u64 {
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}

fn validate(value: &Value) -> anyhow::Result<()> {
    let config = &value["Config"];
    let host = &value["HostConfig"];
    ensure!(
        value["Image"] == IMAGE_ID
            && config["User"] == "65534:65534"
            && config["WorkingDir"] == "/tmp"
            && config["Entrypoint"] == serde_json::json!([PYTHON])
            && config["Cmd"] == serde_json::json!(["-I", "-S", "-B", "-u", "-c", WAIT_MAIN]),
        "container runtime configuration mismatch"
    );
    ensure!(
        host["NetworkMode"] == "none"
            && host["ReadonlyRootfs"] == true
            && host["Privileged"] == false
            && host["CapDrop"] == serde_json::json!(["ALL"])
            && host["SecurityOpt"] == serde_json::json!(["no-new-privileges=true"])
            && host["PidsLimit"] == 32
            && host["NanoCpus"] == 500_000_000
            && host["Memory"] == 134_217_728
            && host["MemorySwap"] == 134_217_728
            && host["IpcMode"] == "private"
            && host["CgroupnsMode"] == "private"
            && host["ShmSize"] == 1_048_576
            && host["RestartPolicy"]["Name"] == "no"
            && host["LogConfig"]["Type"] == "none",
        "container isolation/resource configuration mismatch"
    );
    ensure!(
        empty(&host["Binds"])
            && empty(&host["Mounts"])
            && empty(&value["Mounts"])
            && empty(&host["PortBindings"])
            && host["PublishAllPorts"] == false
            && empty(&host["Devices"])
            && empty(&host["DeviceRequests"])
            && empty(&host["CapAdd"])
            && empty(&host["VolumesFrom"])
            && host["PidMode"] == ""
            && host["UTSMode"] == "",
        "container has unexpected host access"
    );
    ensure!(
        host["Tmpfs"] == serde_json::json!({"/tmp":"rw,nosuid,nodev,noexec,size=16m,mode=1777"}),
        "container tmpfs configuration mismatch"
    );
    for (name, limit) in [
        ("cpu", 2),
        ("nofile", 64),
        ("core", 0),
        ("fsize", 1_048_576),
    ] {
        ensure!(
            host["Ulimits"].as_array().is_some_and(|values| values
                .iter()
                .any(|v| v["Name"] == name && v["Soft"] == limit && v["Hard"] == limit)),
            "container ulimit mismatch: {name}"
        );
    }
    ensure!(
        config["Env"]
            .as_array()
            .is_some_and(|values| values.iter().all(|v| v.as_str().is_some_and(|s| [
                "PATH=",
                "LANG=C.UTF-8",
                "GPG_KEY=",
                "PYTHON_VERSION=3.12.14",
                "PYTHON_SHA256="
            ]
            .iter()
            .any(|prefix| s.starts_with(prefix))))),
        "container environment contains an unexpected entry"
    );
    Ok(())
}
fn empty(value: &Value) -> bool {
    value.is_null()
        || value.as_array().is_some_and(Vec::is_empty)
        || value.as_object().is_some_and(serde_json::Map::is_empty)
}
