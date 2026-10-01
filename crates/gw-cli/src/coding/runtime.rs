//! Capture and qualify one local Unix Docker endpoint and immutable runtime recipe.
use super::process::{Output, control};
use anyhow::{Context, ensure};
use gw_schema::{CODING_RUNTIME_RECIPE, coding_digest};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;
use tokio::process::Command;

pub(super) const IMAGE: &str =
    "python@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f";
pub(super) const IMAGE_ID: &str =
    "sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f";
pub(super) const WRAPPER: &str = include_str!("wrapper.py");
pub(super) const WAIT_MAIN: &str = "import time; time.sleep(600)";
pub(super) const PYTHON: &str = "/usr/local/bin/python3";
pub(super) const PROBE: &str = "import json,sys,platform,os; print(json.dumps({'implementation':sys.implementation.name,'version':list(sys.version_info[:3]),'machine':platform.machine(),'uid':os.getuid(),'gid':os.getgid()}))";
pub(super) const LABEL: &str = "io.ghostwriter.coding-owner";

// All caller-independent resource and isolation options are committed into the runtime identity.
pub(super) const POLICY: &[&str] = &[
    "--pull",
    "never",
    "--platform",
    "linux/arm64",
    "--network",
    "none",
    "--user",
    "65534:65534",
    "--cap-drop",
    "ALL",
    "--security-opt",
    "no-new-privileges=true",
    "--read-only",
    "--tmpfs",
    "/tmp:rw,nosuid,nodev,noexec,size=16m,mode=1777",
    "--workdir",
    "/tmp",
    "--pids-limit",
    "32",
    "--cpus",
    "0.5",
    "--memory",
    "128m",
    "--memory-swap",
    "128m",
    "--ulimit",
    "cpu=2:2",
    "--ulimit",
    "nofile=64:64",
    "--ulimit",
    "core=0:0",
    "--ulimit",
    "fsize=1048576:1048576",
    "--log-driver",
    "none",
    "--restart",
    "no",
    "--ipc",
    "private",
    "--cgroupns",
    "private",
    "--shm-size",
    "1m",
    "--no-healthcheck",
    "--entrypoint",
    PYTHON,
];

/// Exact qualified local runtime declaration. Persisted copies never certify execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodingRuntimeIdentity {
    /// Fixed supported recipe version.
    pub recipe: String,
    /// Repository digest reference; pulling is forbidden.
    pub image: String,
    /// Concrete locally inspected image content identity.
    pub image_id: String,
    /// Required actual Linux ARM64 platform.
    pub platform: String,
    /// Required implementation/version, independently probed before positive consumption.
    pub interpreter: String,
    /// Qualified Docker daemon version; other versions require another recipe qualification.
    pub docker_version: String,
    /// Hash of actual trusted wrapper bytes.
    pub wrapper_id: String,
    /// Hash of exact isolation options, trusted main/probe, and controller limits.
    pub policy_id: String,
}
impl CodingRuntimeIdentity {
    pub(super) fn expected() -> Self {
        let policy = serde_json::to_vec(&(POLICY,WAIT_MAIN,PROBE,PYTHON,
            "control=15s;case=3s;evaluation=180s;output=32768;candidate=65536;fresh-container-per-case;controller-v1")).expect("policy");
        Self {
            recipe: CODING_RUNTIME_RECIPE.into(),
            image: IMAGE.into(),
            image_id: IMAGE_ID.into(),
            platform: "linux/arm64".into(),
            interpreter: "cpython-3.12.14".into(),
            docker_version: "29.8.1".into(),
            wrapper_id: coding_digest("ghostwriter.coding-wrapper.v1", WRAPPER.as_bytes()),
            policy_id: coding_digest("ghostwriter.coding-runtime-policy.v1", &policy),
        }
    }
}

pub(super) struct Docker {
    endpoint: String,
    pub config: PathBuf,
    executable: PathBuf,
}
impl Drop for Docker {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.config);
    }
}
impl Docker {
    pub async fn connect() -> anyhow::Result<Self> {
        Self::connect_with(PathBuf::from("docker")).await
    }
    async fn connect_with(executable: PathBuf) -> anyhow::Result<Self> {
        ensure!(
            cfg!(unix),
            "coding runtime requires a local Unix Docker endpoint"
        );
        for name in [
            "DOCKER_HOST",
            "DOCKER_CONTEXT",
            "DOCKER_TLS_VERIFY",
            "DOCKER_CERT_PATH",
        ] {
            ensure!(
                std::env::var_os(name).is_none(),
                "custom Docker environment selection is unsupported: {name}"
            );
        }
        let output = control(Command::new(&executable).args(["context", "show"])).await?;
        ensure!(
            output.status.success(),
            "cannot resolve active Docker context"
        );
        let context = String::from_utf8(output.stdout)?.trim().to_owned();
        ensure!(
            !context.is_empty() && context.len() < 256 && !context.starts_with('-'),
            "invalid Docker context"
        );
        let output =
            control(Command::new(&executable).args(["context", "inspect", &context])).await?;
        ensure!(
            output.status.success(),
            "cannot inspect active Docker context"
        );
        let contexts: Value = serde_json::from_slice(&output.stdout)?;
        let endpoint = contexts[0]["Endpoints"]["docker"]["Host"]
            .as_str()
            .context("Docker context has no endpoint")?
            .to_owned();
        let path = endpoint
            .strip_prefix("unix://")
            .context("remote/non-Unix Docker endpoints are unsupported")?;
        ensure!(
            path.starts_with('/') && !path.contains(['\n', '\r', '\0', '?', '#']),
            "unsupported Docker Unix endpoint"
        );
        let config = std::env::temp_dir().join(format!("gw-coding-config-{}", nonce()?));
        std::fs::create_dir(&config)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&config, std::fs::Permissions::from_mode(0o700))?;
        }
        let docker = Self {
            endpoint,
            config,
            executable,
        };
        std::fs::write(docker.config.join("config.json"), "{}")?;
        let info = docker.json(&["info", "--format", "{{json .}}"]).await?;
        ensure!(
            info["OSType"] == "linux"
                && info["Architecture"] == "aarch64"
                && info["CgroupVersion"] == "2"
                && info["ServerVersion"] == "29.8.1",
            "unsupported Docker platform, cgroup version, or daemon version"
        );
        ensure!(
            info["SecurityOptions"]
                .as_array()
                .is_some_and(|options| options
                    .iter()
                    .any(|item| item.as_str() == Some("name=seccomp,profile=builtin"))),
            "Docker requires built-in default seccomp"
        );
        let images = docker.json(&["image", "inspect", IMAGE]).await?;
        let image = &images[0];
        ensure!(
            image["Id"] == IMAGE_ID
                && image["Os"] == "linux"
                && image["Architecture"] == "arm64"
                && image["RepoDigests"]
                    .as_array()
                    .is_some_and(|refs| refs.iter().any(|v| v == IMAGE)),
            "cached pinned Linux ARM64 image is absent or mismatched; no pull was attempted"
        );
        ensure!(
            image["Config"]["Volumes"].is_null(),
            "runtime image must declare no volumes"
        );
        Ok(docker)
    }
    pub fn command(&self, args: &[&str]) -> Command {
        let mut command = Command::new(&self.executable);
        command.env_clear();
        if let Some(path) = std::env::var_os("PATH") {
            command.env("PATH", path);
        }
        command
            .env("HOME", &self.config)
            .env("LANG", "C.UTF-8")
            .arg("--config")
            .arg(&self.config)
            .args(["--host", &self.endpoint])
            .args(args);
        command
    }
    pub async fn call(&self, args: &[&str]) -> anyhow::Result<Output> {
        control(&mut self.command(args)).await
    }
    pub async fn json(&self, args: &[&str]) -> anyhow::Result<Value> {
        let output = self.call(args).await?;
        ensure!(
            output.status.success(),
            "Docker query failed: {}",
            args.first().unwrap_or(&"query")
        );
        Ok(serde_json::from_slice(&output.stdout)?)
    }
    #[cfg(test)]
    pub async fn connect_test(executable: PathBuf) -> anyhow::Result<Self> {
        Self::connect_with(executable).await
    }
}

pub(super) fn nonce() -> anyhow::Result<String> {
    use std::io::Read;
    let mut bytes = [0_u8; 16];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}
