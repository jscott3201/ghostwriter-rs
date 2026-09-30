//! Deterministic shard ownership and failure-drain regressions. No live providers.

mod common;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::*;
use gw_engine::{Engine, EngineEvent, EventSink, InMemorySeedSource, load_cursor, record_id};
use gw_providers::{ChatRequest, DeltaStream, Provider, ProviderError, StreamChatFuture};
use gw_schema::{Content, LifecycleState};
use gw_storage::Store;
use tokio_util::sync::CancellationToken;

#[derive(Clone, Copy)]
enum Reply {
    Good,
    Fatal(u16),
    Panic,
}

struct Gate {
    started: CancellationToken,
    release: CancellationToken,
    reply: Reply,
}

impl Gate {
    fn new(reply: Reply) -> Arc<Self> {
        Arc::new(Self {
            started: CancellationToken::new(),
            release: CancellationToken::new(),
            reply,
        })
    }
}

struct GatedTeacher {
    gates: BTreeMap<String, Arc<Gate>>,
    calls: Mutex<Vec<String>>,
}

impl GatedTeacher {
    fn new(gates: &[(&str, Arc<Gate>)]) -> Arc<Self> {
        Arc::new(Self {
            gates: gates
                .iter()
                .map(|(p, g)| ((*p).into(), g.clone()))
                .collect(),
            calls: Mutex::new(Vec::new()),
        })
    }

    fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }
}

impl Provider for GatedTeacher {
    fn stream_chat(&self, req: ChatRequest) -> StreamChatFuture<'_> {
        let Content::Text(prompt) = &req.messages[0].content else {
            panic!("expected a text prompt");
        };
        self.calls.lock().unwrap().push(prompt.clone());
        let gate = self
            .gates
            .get(prompt)
            .or_else(|| self.gates.get(&format!("{prompt}:{}", req.seed.unwrap())))
            .expect("unexpected dispatch")
            .clone();
        Box::pin(async move {
            gate.started.cancel();
            gate.release.cancelled().await;
            match gate.reply {
                Reply::Good => {
                    let stream: DeltaStream =
                        Box::pin(futures::stream::iter(good_cot(0.01).into_iter().map(Ok)));
                    Ok(stream)
                }
                Reply::Fatal(status) => Err(ProviderError::Status {
                    status,
                    retryable: false,
                    body: Some(format!("deliberate {status}")),
                }),
                Reply::Panic => panic!("deliberate shard panic"),
            }
        })
    }
}

fn source(prompts: &[&str], shards: usize) -> InMemorySeedSource {
    InMemorySeedSource::new(prompts.iter().map(|p| good_candidate(p)).collect(), shards)
}

async fn started(gate: &Gate) {
    tokio::time::timeout(Duration::from_secs(3), gate.started.cancelled())
        .await
        .expect("provider dispatch never started");
}

async fn failed_run_drains(slow_shard: usize, panic: bool) {
    let store = Store::open_in_memory().await.unwrap();
    let slow = Gate::new(Reply::Good);
    let failure = Gate::new(if panic {
        Reply::Panic
    } else {
        Reply::Fatal(401)
    });
    let teacher = GatedTeacher::new(&[("slow", slow.clone()), ("failure", failure.clone())]);
    let judge = Arc::new(FailingJudge::new(usize::MAX, &judge_body(0.95, "accept")));
    let (sink, mut events) = EventSink::subscribe();
    let engine = Engine::new(
        clients(store.clone(), teacher.clone(), judge.clone(), 25.0, sink),
        area_k1(one_judge(), lenient_thresholds()),
        2,
    );
    let prompts = if slow_shard == 0 {
        ["slow", "failure", "later-slow", "later-failure"]
    } else {
        ["failure", "slow", "later-failure", "later-slow"]
    };
    let seeds = source(&prompts, 2);
    let cancel = CancellationToken::new();
    let runner = engine.clone();
    let run_cancel = cancel.clone();
    let run_seeds = seeds.clone();
    let mut run = tokio::spawn(async move { runner.run("drain", &run_seeds, run_cancel).await });
    started(&slow).await;
    started(&failure).await;
    failure.release.cancel();

    // Completion before releasing the other shard proves ownership was lost. Waiting for the
    // cancellation signal proves a later shard's error is observed while shard zero is blocked.
    let cancelled_before_return = tokio::time::timeout(Duration::from_secs(3), async {
        tokio::select! {
            biased;
            result = &mut run => panic!("returned before owned shard drained: {result:?}"),
            () = cancel.cancelled() => {}
        }
        assert!(!run.is_finished());
        assert_eq!(
            store.run_status("drain").await.unwrap().as_deref(),
            Some("running")
        );
    })
    .await;
    slow.release.cancel();
    cancelled_before_return.expect("fatal shard did not promptly cancel the run");
    let error = run.await.unwrap().unwrap_err();
    assert!(error.to_string().contains(if panic {
        "deliberate shard panic"
    } else {
        "401"
    }));
    assert_eq!(teacher.calls().len(), 2, "no subsequent item may dispatch");
    assert_eq!(
        judge.call_count(),
        0,
        "no next transition after cancellation"
    );
    let rid = record_id("drain", slow_shard as i64, slow_shard as i64, 0, 0);
    let settled = store.get(&rid).await.unwrap();
    assert_eq!(settled.lifecycle.state, LifecycleState::AssistantGenerated);
    assert_eq!(settled.cost.usd, 0.01);
    for shard in 0..2 {
        assert_eq!(
            load_cursor(&store, "drain", shard)
                .await
                .unwrap()
                .next_offset,
            0
        );
    }
    assert_eq!(
        store.run_status("drain").await.unwrap().as_deref(),
        Some("failed")
    );

    // The finished event is last, and dropping the final engine closes the channel immediately:
    // no detached shard still owns an EventSink or can race the following resume.
    drop(engine);
    let mut observed = Vec::new();
    loop {
        match events.try_recv() {
            Ok(event) => observed.push(event),
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break,
            Err(other) => panic!("a shard retained its event owner after run returned: {other}"),
        }
    }
    assert!(matches!(
        observed.last(),
        Some(EngineEvent::RunFinished {
            completed: false,
            ..
        })
    ));
    assert_eq!(
        observed
            .iter()
            .filter(|event| matches!(event, EngineEvent::RunFinished { .. }))
            .count(),
        1,
        "failure is reported only once, after the final persisted transition"
    );
    let resumed_teacher = Arc::new(ScriptedTeacher::new(vec![], 3));
    let resumed = Engine::new(
        clients(
            store.clone(),
            resumed_teacher.clone(),
            judge,
            25.0,
            EventSink::disconnected(),
        ),
        area_k1(one_judge(), lenient_thresholds()),
        2,
    );
    let report = resumed
        .run("drain", &seeds, CancellationToken::new())
        .await
        .unwrap();
    assert!(report.completed);
    assert_eq!(report.exported, 4);
    assert_eq!(
        resumed_teacher.call_count(),
        3,
        "settled generation is reused"
    );
    assert_eq!(
        teacher.calls().len(),
        2,
        "the old owner cannot spend during resume"
    );
}

#[tokio::test]
async fn later_fatal_is_observed_while_shard_zero_is_blocked() {
    failed_run_drains(0, false).await;
}

#[tokio::test]
async fn early_fatal_does_not_detach_a_later_shard() {
    failed_run_drains(1, false).await;
}

#[tokio::test]
async fn shard_panic_cancels_and_drains_other_shards() {
    failed_run_drains(0, true).await;
}

#[tokio::test]
async fn first_observed_fatal_survives_later_fatal_during_drain() {
    let store = Store::open_in_memory().await.unwrap();
    let secondary = Gate::new(Reply::Fatal(403));
    let primary = Gate::new(Reply::Fatal(401));
    let teacher = GatedTeacher::new(&[
        ("secondary", secondary.clone()),
        ("primary", primary.clone()),
    ]);
    let engine = Engine::new(
        clients(
            store,
            teacher,
            Arc::new(ScriptedJudge::new(vec![])),
            25.0,
            EventSink::disconnected(),
        ),
        area_k1(one_judge(), lenient_thresholds()),
        2,
    );
    let cancel = CancellationToken::new();
    let run_cancel = cancel.clone();
    let run = tokio::spawn(async move {
        engine
            .run("primary", &source(&["secondary", "primary"], 2), run_cancel)
            .await
    });
    started(&secondary).await;
    started(&primary).await;
    primary.release.cancel();
    let signal = tokio::time::timeout(Duration::from_secs(3), cancel.cancelled()).await;
    secondary.release.cancel();
    signal.expect("first fatal was not observed");
    let error = run.await.unwrap().unwrap_err().to_string();
    assert!(error.contains("401"), "primary error replaced: {error}");
}

#[tokio::test]
async fn failed_status_storage_error_preserves_primary_and_terminal_event() {
    let store = Store::open_in_memory().await.unwrap();
    let failure = Gate::new(Reply::Fatal(401));
    let teacher = GatedTeacher::new(&[("failure", failure.clone())]);
    let (sink, mut events) = EventSink::subscribe();
    let engine = Engine::new(
        clients(
            store.clone(),
            teacher,
            Arc::new(ScriptedJudge::new(vec![])),
            25.0,
            sink,
        ),
        area_k1(one_judge(), lenient_thresholds()),
        1,
    );
    let run = tokio::spawn(async move {
        engine
            .run(
                "closed-store",
                &source(&["failure"], 1),
                CancellationToken::new(),
            )
            .await
    });
    started(&failure).await;
    store.close().await;
    failure.release.cancel();
    let error = run.await.unwrap().unwrap_err().to_string();
    assert!(
        error.contains("401"),
        "secondary persistence error masked primary: {error}"
    );
    let mut finished = 0;
    while let Some(event) = events.recv().await {
        if matches!(
            event,
            EngineEvent::RunFinished {
                completed: false,
                ..
            }
        ) {
            finished += 1;
        }
    }
    assert_eq!(finished, 1, "terminal failure event still attempted");
}

#[tokio::test]
async fn fatal_sibling_signals_before_its_own_group_finishes_draining() {
    let store = Store::open_in_memory().await.unwrap();
    let slow = Gate::new(Reply::Good);
    let failure = Gate::new(Reply::Fatal(401));
    let teacher = GatedTeacher::new(&[("group:0", slow.clone()), ("group:1", failure.clone())]);
    let judge = Arc::new(ScriptedJudge::new(vec![]));
    let engine = Engine::new(
        clients(
            store.clone(),
            teacher,
            judge.clone(),
            25.0,
            EventSink::disconnected(),
        ),
        area_k(one_judge(), lenient_thresholds(), 2),
        1,
    );
    let cancel = CancellationToken::new();
    let run_cancel = cancel.clone();
    let run = tokio::spawn(async move {
        engine
            .run("group", &source(&["group"], 1), run_cancel)
            .await
    });
    started(&slow).await;
    started(&failure).await;
    failure.release.cancel();
    let signal = tokio::time::timeout(Duration::from_secs(3), cancel.cancelled()).await;
    assert!(!run.is_finished(), "the slow sibling is still owned");
    slow.release.cancel();
    signal.expect("fatal notification waited for sibling drain");
    assert!(run.await.unwrap().is_err());
    assert_eq!(judge.call_count(), 0);
    let rec = store.get(&record_id("group", 0, 0, 0, 0)).await.unwrap();
    assert_eq!(rec.lifecycle.state, LifecycleState::AssistantGenerated);
}

#[tokio::test]
async fn permit_waiters_stop_before_the_active_item_releases_its_permit() {
    let store = Store::open_in_memory().await.unwrap();
    let gate = Gate::new(Reply::Good);
    let teacher = GatedTeacher::new(&[
        ("one", gate.clone()),
        ("two", gate.clone()),
        ("three", gate.clone()),
    ]);
    let (sink, mut events) = EventSink::subscribe();
    let engine = Engine::new(
        clients(
            store.clone(),
            teacher.clone(),
            Arc::new(ScriptedJudge::new(vec![])),
            25.0,
            sink,
        ),
        area_k1(one_judge(), lenient_thresholds()),
        1,
    );
    let cancel = CancellationToken::new();
    let run_cancel = cancel.clone();
    let run = tokio::spawn(async move {
        engine
            .run("permits", &source(&["one", "two", "three"], 3), run_cancel)
            .await
    });
    started(&gate).await;
    let mut shard_starts = 0;
    while shard_starts < 3 {
        if matches!(events.recv().await, Some(EngineEvent::ShardStarted { .. })) {
            shard_starts += 1;
        }
    }
    cancel.cancel();
    let waiters_stopped = tokio::time::timeout(Duration::from_secs(3), async {
        let mut finished = 0;
        while finished < 2 {
            if matches!(events.recv().await, Some(EngineEvent::ShardFinished { .. })) {
                finished += 1;
            }
        }
    })
    .await;
    assert!(
        !run.is_finished(),
        "the admitted teacher call must still drain"
    );
    gate.release.cancel();
    waiters_stopped.expect("cancelled permit waiters remained blocked");
    let report = run.await.unwrap().unwrap();
    assert!(!report.completed);
    assert_eq!(teacher.calls().len(), 1);
    for shard in 0..3 {
        assert_eq!(
            load_cursor(&store, "permits", shard)
                .await
                .unwrap()
                .next_offset,
            0
        );
    }
    assert_eq!(
        store.run_status("permits").await.unwrap().as_deref(),
        Some("halted")
    );
}
