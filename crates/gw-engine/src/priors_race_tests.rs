//! Record ownership is distinct from seed-item exclusion, including concurrent resume/append.
use super::*;
use crate::accounting_test_support::{clients, record};
use gw_generate::EmbeddingFuture;
use gw_storage::Store;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

struct Blocked {
    entered: Semaphore,
    release: Semaphore,
}
impl Blocked {
    fn new() -> Self {
        Self {
            entered: Semaphore::new(0),
            release: Semaphore::new(0),
        }
    }
}
impl Embedder for Blocked {
    fn embed<'a>(&'a self, _: &'a str) -> EmbeddingFuture<'a> {
        Box::pin(async move {
            self.entered.add_permits(1);
            self.release.acquire().await.unwrap().forget();
            Ok(vec![1.0])
        })
    }
}
struct Ready;
impl Embedder for Ready {
    fn embed<'a>(&'a self, _: &'a str) -> EmbeddingFuture<'a> {
        Box::pin(async { Ok(vec![2.0]) })
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn same_record_resume_and_append_contribute_once_in_either_order_and_keep_new_admissions() {
    for resume_first in [true, false] {
        let store = Store::open_in_memory().await.unwrap();
        let resuming = Arc::new(Blocked::new());
        let clients = clients(store.clone(), resuming.clone()).await;
        let admitted = record("r-s0-seed0-a0-c0");
        store.replace_record_for_import(&admitted).await.unwrap();
        let appending = Arc::new(Blocked::new());
        let appender = appending.clone();
        let priors = clients.priors.clone();
        let append_task =
            tokio::spawn(async move { append_record(&priors, appender.as_ref(), &admitted).await });
        appending.entered.acquire().await.unwrap().forget();
        let seeding = clients.clone();
        let seed_task =
            tokio::spawn(async move { seed(&seeding, "r", &CancellationToken::new()).await });
        resuming.entered.acquire().await.unwrap().forget();
        // The historical scan has already completed; a different record is admitted meanwhile.
        let newer = record("r-s1-seed1-a0-c0");
        store.replace_record_for_import(&newer).await.unwrap();
        append_record(&clients.priors, &Ready, &newer)
            .await
            .unwrap();
        if resume_first {
            resuming.release.add_permits(1);
            seed_task.await.unwrap().unwrap();
            appending.release.add_permits(1);
            append_task.await.unwrap().unwrap();
        } else {
            appending.release.add_permits(1);
            append_task.await.unwrap().unwrap();
            resuming.release.add_permits(1);
            seed_task.await.unwrap().unwrap();
        }
        let all = snapshot(&clients.priors, "different-item");
        assert_eq!(
            all.len(),
            2,
            "one vector per record; resume_first={resume_first}: {all:?}"
        );
        assert_eq!(
            snapshot(&clients.priors, "r-s0-seed0").as_ref(),
            &[vec![2.0]]
        );
        assert_eq!(
            snapshot(&clients.priors, "r-s1-seed1").as_ref(),
            &[vec![1.0]]
        );
    }
}

#[tokio::test]
async fn distinct_records_of_the_same_item_both_contribute_and_are_both_excluded() {
    let store = Store::open_in_memory().await.unwrap();
    let clients = clients(store.clone(), Arc::new(Ready)).await;
    for id in ["r-s0-seed0-a0-c0", "r-s0-seed0-a1-c1"] {
        store.replace_record_for_import(&record(id)).await.unwrap();
    }
    seed(&clients, "r", &CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(snapshot(&clients.priors, "other-item").len(), 2);
    assert!(snapshot(&clients.priors, "r-s0-seed0").is_empty());
    append_record(&clients.priors, &Ready, &record("r-s0-seed0-a0-c0"))
        .await
        .unwrap();
    assert_eq!(
        snapshot(&clients.priors, "other-item").len(),
        2,
        "replaying one record cannot add another vector"
    );
}
