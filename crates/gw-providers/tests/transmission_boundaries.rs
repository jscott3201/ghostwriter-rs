//! Physical transmission boundaries, exercised against local HTTP servers.
use gw_providers::{ChatRequest, EmbeddingsClient, OpenRouterProvider, Provider, RetryPolicy};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

async fn redirect_server() -> (String, Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let posts = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&posts);
    let task = tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0; 8192];
            let n = socket.read(&mut buf).await.unwrap();
            count.fetch_add(1, Ordering::SeqCst);
            let request = String::from_utf8_lossy(&buf[..n]);
            let response = if request.starts_with("POST /target") {
                "HTTP/1.1 200 OK\r\nContent-Length: 42\r\nConnection: close\r\n\r\n{\"data\":[{\"index\":0,\"embedding\":[1,2]}]}".to_string()
            } else {
                format!(
                    "HTTP/1.1 307 Temporary Redirect\r\nLocation: http://{address}/target\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                )
            };
            socket.write_all(response.as_bytes()).await.unwrap();
        }
    });
    (format!("http://{address}"), posts, task)
}

#[tokio::test]
async fn chat_and_embedding_do_not_follow_redirects() {
    let (base, posts, task) = redirect_server().await;
    let provider = OpenRouterProvider::builder()
        .base_url(&base)
        .retry_policy(RetryPolicy {
            max_attempts: 1,
            ..Default::default()
        })
        .build_with_key("fixture")
        .unwrap();
    let _ = provider
        .stream_chat(ChatRequest::new("fixture", vec![]))
        .await;
    let chat_posts = posts.load(Ordering::SeqCst);
    let client = EmbeddingsClient::builder()
        .base_url(&base)
        .dim(2)
        .build_with_key(None)
        .unwrap();
    let _ = client.embed_batch(&["fixture"]).await;
    let all_posts = posts.load(Ordering::SeqCst);
    task.abort();
    assert_eq!(chat_posts, 1, "chat must not transparently resend a POST");
    assert_eq!(
        all_posts, 2,
        "embeddings must not transparently resend a POST"
    );
}
