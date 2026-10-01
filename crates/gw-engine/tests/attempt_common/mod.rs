#![allow(dead_code)]
use gw_providers::{ChatCompletionsProvider, EmbeddingsClient, RetryPolicy};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    sync::Semaphore,
    task::{JoinHandle, JoinSet},
};

pub struct Response {
    pub status: u16,
    pub body: String,
    pub prefix: usize,
    pub started: Option<Arc<Semaphore>>,
    pub release: Option<Arc<Semaphore>>,
}
impl Response {
    pub fn ok(body: String) -> Self {
        Self {
            status: 200,
            body,
            prefix: 0,
            started: None,
            release: None,
        }
    }
    pub fn held(mut self, prefix: usize, started: Arc<Semaphore>, release: Arc<Semaphore>) -> Self {
        self.prefix = prefix;
        self.started = Some(started);
        self.release = Some(release);
        self
    }
}
pub struct Server {
    pub url: String,
    pub requests: Arc<Mutex<Vec<(String, Value)>>>,
    task: JoinHandle<()>,
}
impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl Server {
    pub async fn new(
        handler: impl Fn(&str, &Value, usize) -> Response + Send + Sync + 'static,
    ) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = requests.clone();
        let handler = Arc::new(handler);
        let task = tokio::spawn(async move {
            let mut connections = JoinSet::new();
            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                let captured = captured.clone();
                let handler = handler.clone();
                connections.spawn(async move {
                    let mut bytes = Vec::new();
                    let mut buffer = [0; 4096];
                    let (path, value) = loop {
                        let n = socket.read(&mut buffer).await.unwrap();
                        if n == 0 {
                            return;
                        }
                        bytes.extend_from_slice(&buffer[..n]);
                        if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                            let headers =
                                String::from_utf8_lossy(&bytes[..end]).to_ascii_lowercase();
                            let len: usize = headers
                                .lines()
                                .find_map(|l| l.strip_prefix("content-length:"))
                                .unwrap()
                                .trim()
                                .parse()
                                .unwrap();
                            if bytes.len() >= end + 4 + len {
                                let path = headers.split_whitespace().nth(1).unwrap().to_string();
                                let body: Value =
                                    serde_json::from_slice(&bytes[end + 4..end + 4 + len]).unwrap();
                                break (path, body);
                            }
                        }
                    };
                    let ordinal = {
                        let mut requests = captured.lock().unwrap();
                        let ordinal = requests.iter().filter(|(p, _)| p == &path).count();
                        requests.push((path.clone(), value.clone()));
                        ordinal
                    };
                    let response = handler(&path, &value, ordinal);
                    let headers = format!(
                        "HTTP/1.1 {} Fixture\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        response.status,
                        response.body.len()
                    );
                    socket.write_all(headers.as_bytes()).await.unwrap();
                    if response.prefix > 0 {
                        socket
                            .write_all(&response.body.as_bytes()[..response.prefix])
                            .await
                            .unwrap();
                    }
                    if let Some(started) = response.started {
                        started.add_permits(1);
                    }
                    if let Some(release) = response.release {
                        release.acquire().await.unwrap().forget();
                    }
                    let _ = socket
                        .write_all(&response.body.as_bytes()[response.prefix..])
                        .await;
                });
            }
        });
        Self {
            url,
            requests,
            task,
        }
    }
    pub fn provider(&self) -> ChatCompletionsProvider {
        ChatCompletionsProvider::builder()
            .base_url(&self.url)
            .rpm(60_000)
            .retry_policy(RetryPolicy {
                max_attempts: 1,
                ..Default::default()
            })
            .build_with_key("fixture")
            .unwrap()
    }
    pub fn embedder(&self) -> EmbeddingsClient {
        EmbeddingsClient::builder()
            .base_url(&self.url)
            .dim(2)
            .model("embedding-fixture")
            .build_with_key(None)
            .unwrap()
    }
}
pub fn embedding(cost: Option<f64>) -> String {
    let mut value = json!({"model":"embedding-actual","id":"embedding-response","usage":{"prompt_tokens":7,"total_tokens":7},"data":[{"index":0,"embedding":[1,0]}]});
    if let Some(cost) = cost {
        value["usage"]["cost"] = json!(cost);
    }
    value.to_string()
}
pub fn teacher(content: &str, finish: &str, cost: Option<f64>) -> String {
    let mut value = json!({"model":"teacher-actual","id":"teacher-response","provider":"local-fixture","choices":[{"delta":{"content":content,"reasoning":"Work out the answer carefully.","reasoning_details":[{"type":"reasoning.text","text":"Work out the answer carefully.","index":0}]},"finish_reason":finish}],"usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8,"completion_tokens_details":{"reasoning_tokens":4}}});
    if content.is_empty() && finish == "stop" {
        value["choices"][0]["delta"] = json!({"content":""});
    }
    if let Some(cost) = cost {
        value["usage"]["cost"] = json!(cost);
    }
    format!("data: {value}\n\ndata: [DONE]\n\n")
}
pub fn grade(body: &str, cost: Option<f64>) -> String {
    let mut value = json!({"id":"judge-response","model":"judge-actual","choices":[{"delta":{"content":body},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":5,"total_tokens":16}});
    if let Some(cost) = cost {
        value["usage"]["cost"] = json!(cost);
    }
    format!("data: {value}\n\ndata: [DONE]\n\n")
}
pub async fn signal(gate: &Semaphore) {
    tokio::time::timeout(std::time::Duration::from_secs(5), gate.acquire())
        .await
        .expect("fixture made no progress")
        .unwrap()
        .forget();
}
