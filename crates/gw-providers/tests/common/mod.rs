#![allow(dead_code)]
use gw_providers::{AttemptObserver, ObservationContext, ObservationError, ObservationFuture};
use gw_schema::{
    AccountingCapability as Cap, AttemptContext, AttemptIntent, AttemptMetadata, AttemptPurpose,
    AttemptRole, OutputInterpretation, TransportSettlement,
};
use gw_storage::Store;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    task::{JoinHandle, JoinSet},
};

pub struct Server {
    pub url: String,
    pub requests: Arc<Mutex<Vec<Vec<u8>>>>,
    pub posts: Arc<AtomicUsize>,
    task: JoinHandle<()>,
}
impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl Server {
    pub async fn responses(responses: Vec<(u16, String)>) -> Self {
        Self::handler(move |index, _| responses[index.min(responses.len() - 1)].clone()).await
    }
    pub async fn handler<F>(handler: F) -> Self
    where
        F: Fn(usize, &[u8]) -> (u16, String) + Send + Sync + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let posts = Arc::new(AtomicUsize::new(0));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let counter = posts.clone();
        let captured = requests.clone();
        let handler = Arc::new(handler);
        let task = tokio::spawn(async move {
            let mut connections = JoinSet::new();
            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                let counter = counter.clone();
                let captured = captured.clone();
                let handler = handler.clone();
                connections.spawn(async move {
                    let body = request_body(&mut socket).await;
                    let index = counter.fetch_add(1, Ordering::SeqCst);
                    captured.lock().unwrap().push(body.clone());
                    let (status, body) = handler(index, &body);
                    respond(&mut socket, status, &body).await;
                });
            }
        });
        Self {
            url,
            requests,
            posts,
            task,
        }
    }
}
pub async fn request_body(socket: &mut TcpStream) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut buf = [0; 4096];
    loop {
        let n = socket.read(&mut buf).await.unwrap();
        assert!(n > 0);
        bytes.extend_from_slice(&buf[..n]);
        if let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&bytes[..end]).to_ascii_lowercase();
            let len: usize = headers
                .lines()
                .find_map(|line| line.strip_prefix("content-length:"))
                .unwrap_or("0")
                .trim()
                .parse()
                .unwrap();
            if bytes.len() >= end + 4 + len {
                return bytes[end + 4..end + 4 + len].to_vec();
            }
        }
    }
}
pub async fn respond(socket: &mut TcpStream, status: u16, body: &str) {
    let message = format!(
        "HTTP/1.1 {status} Fixture\r\nContent-Length: {}\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    socket.write_all(message.as_bytes()).await.unwrap();
}
#[derive(Clone)]
pub struct StoreObserver(pub Store);
impl AttemptObserver for StoreObserver {
    fn begin(&self, intent: AttemptIntent) -> ObservationFuture<'_, String> {
        Box::pin(async move {
            self.0
                .begin_model_attempt(&intent)
                .await
                .map_err(|e| ObservationError(e.to_string()))
        })
    }
    fn metadata(&self, id: String, metadata: AttemptMetadata) -> ObservationFuture<'_, ()> {
        Box::pin(async move {
            self.0
                .observe_model_attempt(&id, &metadata)
                .await
                .map_err(|e| ObservationError(e.to_string()))
        })
    }
    fn settle(&self, id: String, settlement: TransportSettlement) -> ObservationFuture<'_, ()> {
        Box::pin(async move {
            self.0
                .settle_model_attempt(&id, &settlement)
                .await
                .map_err(|e| ObservationError(e.to_string()))
        })
    }
    fn interpret(&self, id: String, value: OutputInterpretation) -> ObservationFuture<'_, ()> {
        Box::pin(async move {
            self.0
                .interpret_model_attempt(&id, value)
                .await
                .map_err(|e| ObservationError(e.to_string()))
        })
    }
}
pub async fn context(
    store: &Store,
    role: AttemptRole,
    purpose: AttemptPurpose,
) -> ObservationContext {
    store.create_run("run", "{}", None).await.unwrap();
    let coverage = store
        .begin_model_launch(
            "run",
            Cap::PhysicalAttemptsV1,
            Cap::PhysicalAttemptsV1,
            Cap::PhysicalAttemptsV1,
        )
        .await
        .unwrap();
    ObservationContext {
        observer: Arc::new(StoreObserver(store.clone())),
        context: AttemptContext {
            run_id: "run".into(),
            launch_id: coverage.launch_id,
            record_id: Some("intended".into()),
            shard: Some(2),
            role,
            purpose,
        },
    }
}
pub fn sse(value: serde_json::Value, done: bool) -> String {
    format!(
        "data: {value}\n\n{}",
        if done { "data: [DONE]\n\n" } else { "" }
    )
}
