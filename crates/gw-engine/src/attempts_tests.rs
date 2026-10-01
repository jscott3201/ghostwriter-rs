//! OnceCell waiters cannot restart a failed initializer while its owner awaits outer handling.
use super::*;
use crate::accounting_test_support::{clients, record};
use std::{
    future::poll_fn,
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};
use tokio::sync::{Semaphore, oneshot};

struct FailingEmbedder {
    config_fault: bool,
    calls: AtomicUsize,
    entered: Semaphore,
    release: Semaphore,
}
impl FailingEmbedder {
    fn new(config_fault: bool) -> Self {
        Self {
            config_fault,
            calls: AtomicUsize::new(0),
            entered: Semaphore::new(0),
            release: Semaphore::new(0),
        }
    }
}
impl Embedder for FailingEmbedder {
    fn embed<'a>(&'a self, _: &'a str) -> EmbeddingFuture<'a> {
        Box::pin(async move {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                self.entered.add_permits(1);
                self.release.acquire().await.unwrap().forget();
                Err(if self.config_fault {
                    ProviderError::Config("first initializer failed".into())
                } else {
                    ProviderError::Accounting {
                        stage: "custom embedding".into(),
                        detail: "first initializer failed".into(),
                        primary: None,
                    }
                })
            } else {
                Ok(vec![1.0, 0.0])
            }
        })
    }
}

/// Return only after this worker has polled the real OnceCell future. Keep the completed result
/// outside any executor/shard classifier until both callers have drained at the local boundary.
async fn start(
    clients: Clients,
    outer: Arc<Semaphore>,
) -> (
    oneshot::Receiver<crate::Result<()>>,
    tokio::task::JoinHandle<()>,
) {
    let (ready, polled) = oneshot::channel();
    let (done, result) = oneshot::channel();
    let worker = tokio::spawn(async move {
        let mut future = Box::pin(clients.prepare_generation_priors());
        let mut ready = Some(ready);
        let result = poll_fn(|cx| {
            let poll = future.as_mut().poll(cx);
            if let Some(ready) = ready.take() {
                let _ = ready.send(());
            }
            poll
        })
        .await;
        let _ = done.send(result);
        outer.acquire().await.unwrap().forget();
    });
    polled.await.unwrap();
    (result, worker)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failed_prior_scan_seals_already_queued_initializer_before_outer_classification() {
    let store = Store::open_in_memory().await.unwrap();
    let clients = clients(store.clone(), Arc::new(gw_generate::NullEmbedder)).await;
    store
        .replace_record_for_import(&record("r-s0-seed0-a0-c0"))
        .await
        .unwrap();
    sqlx::query("UPDATE records SET record_json = 'invalid-json'")
        .execute(store.raw_pool())
        .await
        .unwrap();
    // Holding the sole SQLite connection parks the owner inside the scan while a second worker
    // queues on the OnceCell. Releasing it produces a real store decode error outside the observer.
    let connection = store.raw_pool().acquire().await.unwrap();
    let outer = Arc::new(Semaphore::new(0));
    let (first, owner) = start(clients.clone(), outer.clone()).await;
    let (second, waiter) = start(clients.clone(), outer.clone()).await;
    drop(connection);
    let error = first.await.unwrap().unwrap_err();
    assert!(
        matches!(
            error,
            crate::EngineError::Storage(gw_storage::StorageError::Serde(_))
        ),
        "{error:?}"
    );
    let error = tokio::time::timeout(Duration::from_secs(5), second)
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    outer.add_permits(2);
    owner.await.unwrap();
    waiter.await.unwrap();
    assert!(
        error.is_halt(),
        "the waiter must see sealed launch cancellation, not a second scan: {error:?}"
    );
    assert!(
        clients
            .observation
            .as_ref()
            .unwrap()
            .observer
            .cancel
            .is_cancelled()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fatal_custom_prior_embedder_seals_already_queued_initializer_before_outer_classification()
{
    for config_fault in [false, true] {
        let store = Store::open_in_memory().await.unwrap();
        let embedder = Arc::new(FailingEmbedder::new(config_fault));
        let clients = clients(store.clone(), embedder.clone()).await;
        store
            .replace_record_for_import(&record("r-s0-seed0-a0-c0"))
            .await
            .unwrap();
        let outer = Arc::new(Semaphore::new(0));
        let (first, owner) = start(clients.clone(), outer.clone()).await;
        embedder.entered.acquire().await.unwrap().forget();
        let (second, waiter) = start(clients.clone(), outer.clone()).await;
        embedder.release.add_permits(1);
        let first = first.await.unwrap().unwrap_err();
        assert!(first.to_string().contains("first initializer failed"));
        assert!(!first.is_record_level());
        let second = tokio::time::timeout(Duration::from_secs(5), second)
            .await
            .unwrap()
            .unwrap();
        outer.add_permits(2);
        owner.await.unwrap();
        waiter.await.unwrap();
        assert!(
            second.is_err_and(|error| error.is_halt()),
            "queued initialization must stop before another embed"
        );
        assert_eq!(embedder.calls.load(Ordering::SeqCst), 1);
    }
}
