//! Gated request and cache ownership tests for the production panel scheduler.
use super::*;
use crate::{JudgeSampling, grade_one_cached};
use gw_providers::{DeltaStream, ProviderError, StreamChatFuture, StreamDelta};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex, atomic::AtomicUsize},
    time::Duration,
};
use tokio::sync::{Semaphore, mpsc};

#[derive(Clone, Copy)]
enum Reply {
    Good(f64),
    Parse,
    Fatal,
    Halt,
    Cancelled,
}
struct Plan {
    reply: Reply,
    release: Semaphore,
}
struct Controlled {
    plans: BTreeMap<(String, Option<i64>), Arc<Plan>>,
    seen: Mutex<Vec<ChatRequest>>,
    entered: mpsc::UnboundedSender<(String, Option<i64>)>,
    active: AtomicUsize,
    peak: AtomicUsize,
}
struct Active<'a>(&'a Controlled);
impl Drop for Active<'_> {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::SeqCst);
    }
}
impl Provider for Controlled {
    fn stream_chat(&self, req: ChatRequest) -> StreamChatFuture<'_> {
        Box::pin(async move {
            let key = (req.model.clone(), req.seed);
            self.seen.lock().unwrap().push(req);
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(active, Ordering::SeqCst);
            let _active = Active(self);
            self.entered.send(key.clone()).unwrap();
            let plan = &self.plans[&key];
            plan.release.acquire().await.unwrap().forget();
            let body = match plan.reply {
                Reply::Good(score) => format!("{{\"score\":{score},\"verdict\":\"accept\"}}"),
                Reply::Parse => "unparseable response".into(),
                Reply::Fatal => return Err(ProviderError::Config("systemic fixture".into())),
                Reply::Halt => {
                    return Err(ProviderError::Admission(
                        gw_schema::AdmissionDenial::UnknownCost,
                    ));
                }
                Reply::Cancelled => return Err(ProviderError::Cancelled),
            };
            let stream: DeltaStream = Box::pin(futures::stream::iter([Ok(StreamDelta {
                content: Some(body),
                finish_reason: Some("stop".into()),
                ..Default::default()
            })]));
            Ok(stream)
        })
    }
}
fn controlled(
    spec: &[(&str, Option<i64>, Reply)],
) -> (
    Arc<Controlled>,
    mpsc::UnboundedReceiver<(String, Option<i64>)>,
) {
    let (entered, receiver) = mpsc::unbounded_channel();
    let plans = spec
        .iter()
        .map(|(model, seed, reply)| {
            (
                (model.to_string(), *seed),
                Arc::new(Plan {
                    reply: *reply,
                    release: Semaphore::new(0),
                }),
            )
        })
        .collect();
    (
        Arc::new(Controlled {
            plans,
            seen: Mutex::new(vec![]),
            entered,
            active: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        }),
        receiver,
    )
}
fn release(provider: &Controlled, model: &str, seed: Option<i64>) {
    provider.plans[&(model.into(), seed)].release.add_permits(1);
}
async fn received<T>(receiver: &mut mpsc::UnboundedReceiver<T>) -> T {
    tokio::time::timeout(Duration::from_secs(5), receiver.recv())
        .await
        .unwrap()
        .unwrap()
}
fn class(error: &JudgeError) -> PanelFailure {
    match error {
        JudgeError::JudgeParse(_) => PanelFailure::Record,
        JudgeError::Provider(ProviderError::Cancelled) => PanelFailure::Cancelled,
        JudgeError::Provider(ProviderError::Admission(_)) => PanelFailure::Halt,
        _ => PanelFailure::Fatal,
    }
}
async fn cached(store: &Store, judge: &PanelJudge) -> bool {
    let request = build_judge_request(judge, "rubric", "sealed candidate");
    let key = request_fingerprint(&request, judge.rubric_id.as_deref()).unwrap();
    store
        .cache_get("hash", JUDGE_CACHE_KIND, &judge.slug, Some(&key))
        .await
        .unwrap()
        .is_some()
}
async fn wait_cached(store: &Store, judge: &PanelJudge) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while !cached(store, judge).await {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn mixed_and_full_cache_reuse_coalesce_effective_keys_and_preserve_every_ordered_position() {
    let store = Store::open_in_memory().await.unwrap();
    let a = PanelJudge::new("a", "family-a");
    let b = PanelJudge::new("b", "family-b");
    // Raw max_tokens differs but clamps to the same built request; family is not a request key.
    let duplicate = PanelJudge::new("a", "different-family").with_max_tokens(1);
    let sampled = a.clone().with_sampling(JudgeSampling {
        seed: Some(7),
        ..Default::default()
    });
    let judges = vec![a.clone(), b.clone(), duplicate, sampled.clone()];
    let (provider, mut entered) = controlled(&[
        ("a", None, Reply::Good(0.9)),
        ("b", None, Reply::Good(0.8)),
        ("a", Some(7), Reply::Good(0.7)),
    ]);
    release(&provider, "b", None);
    grade_one_cached(
        &store,
        provider.as_ref(),
        &b,
        "rubric",
        "sealed candidate",
        "hash",
    )
    .await
    .unwrap();
    assert_eq!(received(&mut entered).await, ("b".into(), None));
    let (task_store, task_provider, task_judges) =
        (store.clone(), provider.clone(), judges.clone());
    let running = tokio::spawn(async move {
        grade_panel_cached(
            &task_store,
            task_provider.as_ref(),
            &task_judges,
            "rubric",
            "sealed candidate",
            "hash",
            class,
        )
        .await
    });
    let first = received(&mut entered).await;
    let second = received(&mut entered).await;
    assert_ne!(first, second);
    assert_eq!(provider.active.load(Ordering::SeqCst), 2);
    release(&provider, "a", Some(7));
    wait_cached(&store, &sampled).await;
    assert!(
        !running.is_finished(),
        "the first panel position is still held"
    );
    release(&provider, "a", None);
    let grades = running.await.unwrap().unwrap();
    assert_eq!(
        grades.iter().map(|grade| grade.score).collect::<Vec<_>>(),
        [0.9, 0.8, 0.9, 0.7]
    );
    assert_eq!(
        grades[0], grades[2],
        "duplicate positions retain the same paid origin"
    );
    assert!(
        grades[0].raw["attempt_origin"].is_null(),
        "a fake provider cannot invent paid provenance"
    );
    assert_eq!(provider.peak.load(Ordering::SeqCst), 2);
    assert_eq!(provider.active.load(Ordering::SeqCst), 0);
    {
        let seen = provider.seen.lock().unwrap();
        assert_eq!(seen.len(), 3);
        for judge in [&a, &b, &sampled] {
            let expected =
                serde_json::to_value(build_judge_request(judge, "rubric", "sealed candidate"))
                    .unwrap();
            assert!(
                seen.iter()
                    .any(|request| serde_json::to_value(request).unwrap() == expected),
                "request construction and sealed prompts stay byte-equivalent"
            );
        }
    }
    let replay = grade_panel_cached(
        &store,
        provider.as_ref(),
        &judges,
        "rubric",
        "sealed candidate",
        "hash",
        class,
    )
    .await
    .unwrap();
    assert_eq!(replay, grades);
    assert_eq!(provider.seen.lock().unwrap().len(), 3);
    assert!(entered.try_recv().is_err());
}

#[tokio::test]
async fn errors_drain_started_siblings_preserve_successful_cache_and_apply_precedence() {
    for (first, second, expected) in [
        (Reply::Parse, Reply::Good(0.9), PanelFailure::Record),
        (Reply::Parse, Reply::Fatal, PanelFailure::Fatal),
        (Reply::Parse, Reply::Halt, PanelFailure::Halt),
        (Reply::Cancelled, Reply::Parse, PanelFailure::Record),
        (Reply::Fatal, Reply::Cancelled, PanelFailure::Fatal),
    ] {
        let store = Store::open_in_memory().await.unwrap();
        let judges = vec![PanelJudge::new("a", "fa"), PanelJudge::new("b", "fb")];
        let (provider, mut entered) = controlled(&[("a", None, first), ("b", None, second)]);
        let (errors, mut observed) = mpsc::unbounded_channel();
        let (task_store, task_provider, task_judges) =
            (store.clone(), provider.clone(), judges.clone());
        let running = tokio::spawn(async move {
            grade_panel_cached(
                &task_store,
                task_provider.as_ref(),
                &task_judges,
                "rubric",
                "sealed candidate",
                "hash",
                |error| {
                    let class = class(error);
                    errors.send(class).unwrap();
                    class
                },
            )
            .await
        });
        received(&mut entered).await;
        received(&mut entered).await;
        release(&provider, "a", None);
        let first_class = received(&mut observed).await;
        assert_eq!(
            first_class,
            match first {
                Reply::Parse => PanelFailure::Record,
                Reply::Cancelled => PanelFailure::Cancelled,
                _ => PanelFailure::Fatal,
            }
        );
        assert!(!running.is_finished());
        assert_eq!(
            provider.active.load(Ordering::SeqCst),
            1,
            "held sibling was not dropped"
        );
        release(&provider, "b", None);
        let error = running.await.unwrap().unwrap_err();
        assert_eq!(class(&error), expected);
        if expected == PanelFailure::Fatal {
            assert!(error.to_string().contains("systemic fixture"));
        }
        assert_eq!(
            cached(&store, &judges[1]).await,
            matches!(second, Reply::Good(_))
        );
        assert_eq!(provider.active.load(Ordering::SeqCst), 0);
        assert!(!cached(&store, &judges[0]).await);
    }
}

#[tokio::test]
async fn immediate_error_prevents_not_yet_started_logical_misses() {
    let store = Store::open_in_memory().await.unwrap();
    let judges = vec![PanelJudge::new("a", "fa"), PanelJudge::new("b", "fb")];
    let (provider, mut entered) =
        controlled(&[("a", None, Reply::Fatal), ("b", None, Reply::Good(0.9))]);
    release(&provider, "a", None);
    let error = grade_panel_cached(
        &store,
        provider.as_ref(),
        &judges,
        "rubric",
        "sealed candidate",
        "hash",
        class,
    )
    .await
    .unwrap_err();
    assert_eq!(class(&error), PanelFailure::Fatal);
    assert_eq!(received(&mut entered).await, ("a".into(), None));
    assert!(entered.try_recv().is_err());
    assert_eq!(provider.seen.lock().unwrap().len(), 1);
    assert_eq!(provider.active.load(Ordering::SeqCst), 0);
}
