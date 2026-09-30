//! A real protocol NACK, including a default-client positive control for hidden resends.
use crate::{
    AttemptObserver, ChatRequest, EmbeddingsClient, ObservationContext, ObservationFuture,
    OpenRouterProvider, Provider, RetryPolicy,
};
use bytes::Bytes;
use gw_schema::{
    AttemptContext, AttemptIntent, AttemptMetadata, AttemptPurpose, AttemptRole,
    OutputInterpretation, TransportSettlement,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokio::{
    net::TcpListener,
    task::{JoinHandle, JoinSet},
};
#[derive(Default)]
struct Observer {
    begins: AtomicUsize,
    settlements: AtomicUsize,
}
impl AttemptObserver for Observer {
    fn begin(&self, _: AttemptIntent) -> ObservationFuture<'_, String> {
        Box::pin(async move { Ok(self.begins.fetch_add(1, Ordering::SeqCst).to_string()) })
    }
    fn metadata(&self, _: String, _: AttemptMetadata) -> ObservationFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }
    fn settle(&self, _: String, _: TransportSettlement) -> ObservationFuture<'_, ()> {
        Box::pin(async move {
            self.settlements.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
    }
    fn interpret(&self, _: String, _: OutputInterpretation) -> ObservationFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }
}
fn context(observer: Arc<Observer>) -> ObservationContext {
    ObservationContext {
        observer,
        context: AttemptContext {
            run_id: "run".into(),
            launch_id: "launch".into(),
            shard: Some(0),
            record_id: Some("record".into()),
            role: AttemptRole::Teacher,
            purpose: AttemptPurpose::Initial,
        },
    }
}

struct NackServer {
    url: String,
    posts: Arc<AtomicUsize>,
    task: JoinHandle<()>,
}
impl Drop for NackServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl NackServer {
    async fn new() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let posts = Arc::new(AtomicUsize::new(0));
        let counter = posts.clone();
        let task = tokio::spawn(async move {
            let mut connections = JoinSet::new();
            loop {
                let (socket, _) = listener.accept().await.unwrap();
                let counter = counter.clone();
                connections.spawn(async move {
                    let mut connection = h2::server::handshake(socket).await.unwrap();
                    while let Some(request) = connection.accept().await {
                        let (request, mut response) = request.unwrap();
                        assert_eq!(request.method(), http::Method::POST);
                        let number = counter.fetch_add(1, Ordering::SeqCst);
                        if number == 0 {
                            response.send_reset(h2::Reason::REFUSED_STREAM);
                        } else {
                            let response_head =
                                http::Response::builder().status(200).body(()).unwrap();
                            let mut stream = response.send_response(response_head, false).unwrap();
                            stream.send_data(Bytes::from_static(b"ok"), true).unwrap();
                        }
                    }
                });
            }
        });
        Self { url, posts, task }
    }
}

#[tokio::test]
async fn real_refused_stream_is_retried_by_default_but_never_hidden_by_model_clients() {
    let control = NackServer::new().await;
    let response = reqwest::Client::builder()
        .http2_prior_knowledge()
        .build()
        .unwrap()
        .post(&control.url)
        .body("control")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(
        control.posts.load(Ordering::SeqCst),
        2,
        "fixture must trigger the library's protocol-NACK retry"
    );

    let observer = Arc::new(Observer::default());
    let ctx = context(observer.clone());
    let chat = NackServer::new().await;
    let provider = OpenRouterProvider::builder()
        .base_url(&chat.url)
        .http2_for_test()
        .retry_policy(RetryPolicy {
            max_attempts: 1,
            ..Default::default()
        })
        .build_with_key("fixture")
        .unwrap();
    assert!(
        provider
            .stream_chat_observed(ChatRequest::new("fixture", vec![]), ctx.call())
            .await
            .is_err()
    );
    assert_eq!(chat.posts.load(Ordering::SeqCst), 1);

    let embedding = NackServer::new().await;
    let client = EmbeddingsClient::builder()
        .base_url(&embedding.url)
        .http2_for_test()
        .dim(2)
        .build_with_key(None)
        .unwrap();
    assert!(
        client
            .embed_batch_observed(&["fixture"], ctx.call())
            .await
            .is_err()
    );
    assert_eq!(embedding.posts.load(Ordering::SeqCst), 1);
    assert_eq!(observer.begins.load(Ordering::SeqCst), 2);
    assert_eq!(observer.settlements.load(Ordering::SeqCst), 2);
}
