//! Bounded client I/O. Mutating control calls are not abandoned on caller cancellation.
use std::process::ExitStatus;
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, Command};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

pub(super) const CONTROL_LIMIT: Duration = Duration::from_secs(15);
pub(super) const OUTPUT_LIMIT: usize = 32 * 1024;

pub(super) struct Output {
    pub status: ExitStatus,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub exceeded: bool,
}

pub(super) struct Running {
    pub child: Child,
    stdout: JoinHandle<std::io::Result<(Vec<u8>, bool)>>,
    stderr: JoinHandle<std::io::Result<(Vec<u8>, bool)>>,
    writer: JoinHandle<std::io::Result<()>>,
    pub exceeded: CancellationToken,
}
impl Running {
    pub fn spawn(command: &mut Command, input: Vec<u8>, limit: usize) -> anyhow::Result<Self> {
        use std::process::Stdio;
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut child = command.spawn()?;
        let stdout = child.stdout.take().expect("piped stdout");
        let stderr = child.stderr.take().expect("piped stderr");
        let mut stdin = child.stdin.take().expect("piped stdin");
        let exceeded = CancellationToken::new();
        let out_limit = exceeded.clone();
        let err_limit = exceeded.clone();
        Ok(Self {
            child,
            stdout: tokio::spawn(read_limited(stdout, limit, out_limit)),
            stderr: tokio::spawn(read_limited(stderr, limit, err_limit)),
            writer: tokio::spawn(async move {
                stdin.write_all(&input).await?;
                stdin.shutdown().await
            }),
            exceeded,
        })
    }
    pub async fn finish(self, status: ExitStatus) -> anyhow::Result<Output> {
        // Broken stdin can be a normal candidate failure. Child status and parsed result decide it.
        let _writer = self.writer.await?;
        let (stdout, out_exceeded) = self.stdout.await??;
        let (stderr, err_exceeded) = self.stderr.await??;
        Ok(Output {
            status,
            stdout,
            stderr,
            exceeded: out_exceeded || err_exceeded,
        })
    }
    pub async fn terminate(mut self) -> anyhow::Result<Output> {
        self.child.start_kill()?;
        let status = self.child.wait().await?;
        self.finish(status).await
    }
}

async fn read_limited(
    mut stream: impl AsyncRead + Unpin,
    limit: usize,
    signal: CancellationToken,
) -> std::io::Result<(Vec<u8>, bool)> {
    let mut output = Vec::new();
    let mut exceeded = false;
    let mut buffer = [0; 4096];
    loop {
        let count = stream.read(&mut buffer).await?;
        if count == 0 {
            break;
        }
        let take = count.min(limit.saturating_sub(output.len()));
        output.extend_from_slice(&buffer[..take]);
        if count > take {
            exceeded = true;
            signal.cancel();
        }
    }
    Ok((output, exceeded))
}

pub(super) async fn control(command: &mut Command) -> anyhow::Result<Output> {
    let mut running = Running::spawn(command, Vec::new(), 256 * 1024)?;
    match tokio::time::timeout(CONTROL_LIMIT, running.child.wait()).await {
        Ok(status) => {
            let result = running.finish(status?).await?;
            anyhow::ensure!(!result.exceeded, "Docker control output exceeded its bound");
            Ok(result)
        }
        Err(_) => {
            // Killing this client does NOT prove the daemon canceled the mutation. Callers must
            // retain Unknown even if their later best-effort cleanup finds no container.
            let _ = running.terminate().await;
            anyhow::bail!("Docker control acknowledgment timed out; daemon mutation is uncertain")
        }
    }
}
