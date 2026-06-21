//! Run-control regressions: cooperative cancellation plus Drain/Abort budget breach behavior.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use common::*;
use gw_engine::{Engine, EventSink, InMemorySeedSource, load_cursor};
use gw_providers::{
    ChatRequest, DeltaStream, Provider, ProviderError, StreamChatFuture, StreamDelta,
};
use gw_schema::{BudgetBreach, LifecycleState};
use gw_storage::{RecordFilter, RunStatus, Store};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

struct CancelingJudge {
    cancel: CancellationToken,
    calls: AtomicUsize,
}

impl CancelingJudge {
    fn new(cancel: CancellationToken) -> Self {
        Self {
            cancel,
            calls: AtomicUsize::new(0),
        }
    }
}

impl Provider for CancelingJudge {
    fn stream_chat(&self, _req: ChatRequest) -> StreamChatFuture<'_> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let cancel = self.cancel.clone();
        Box::pin(async move {
            cancel.cancel();
            Ok(judge_stream(&judge_body(0.95, "accept")))
        })
    }
}

struct SlowFirstJudge {
    calls: AtomicUsize,
    release_first: Arc<Notify>,
}

impl SlowFirstJudge {
    fn new() -> Self {
        Self {
            calls: AtomicUsize::new(0),
            release_first: Arc::new(Notify::new()),
        }
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl Provider for SlowFirstJudge {
    fn stream_chat(&self, _req: ChatRequest) -> StreamChatFuture<'_> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        let release_first = Arc::clone(&self.release_first);
        Box::pin(async move {
            if n == 1 {
                release_first.notified().await;
                tokio::time::sleep(Duration::from_millis(100)).await;
            } else if n == 2 {
                release_first.notify_waiters();
            }
            Ok(judge_stream(&judge_body(0.95, "accept")))
        })
    }
}

fn judge_stream(body: &str) -> DeltaStream {
    let delta = StreamDelta {
        content: Some(body.to_string()),
        finish_reason: Some("stop".into()),
        ..Default::default()
    };
    Box::pin(futures::stream::iter(vec![
        Ok::<StreamDelta, ProviderError>(delta),
    ]))
}

fn source(prompts: &[&str], shards: usize) -> InMemorySeedSource {
    InMemorySeedSource::new(
        prompts
            .iter()
            .map(|prompt| good_candidate(prompt))
            .collect(),
        shards,
    )
}

async fn states(store: &Store, run_id: &str) -> Vec<LifecycleState> {
    store
        .scan(&RecordFilter::new().run_id(run_id))
        .await
        .unwrap()
        .into_iter()
        .map(|record| record.lifecycle.state)
        .collect()
}

#[tokio::test]
async fn cancellation_stops_mid_item_at_transition_boundary_without_committing_cursor() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(ScriptedTeacher::new(vec![good_cot(0.01)], 1));
    let cancel = CancellationToken::new();
    let judge = Arc::new(CancelingJudge::new(cancel.clone()));
    let engine = Engine::new(
        clients(
            store.clone(),
            teacher.clone(),
            judge,
            25.0,
            EventSink::disconnected(),
        ),
        area_k1(one_judge(), lenient_thresholds()),
        1,
    );

    let report = engine
        .run("run-cancel-mid", &source(&["q0"], 1), cancel)
        .await
        .unwrap();

    assert!(!report.completed);
    assert_eq!(teacher.call_count(), 1);
    assert_eq!(
        states(&store, "run-cancel-mid").await,
        vec![LifecycleState::Judged],
        "cancel is observed after the Judged transition is fully persisted"
    );
    assert_eq!(
        load_cursor(&store, "run-cancel-mid", 0)
            .await
            .unwrap()
            .next_offset,
        0,
        "an interrupted item must not be checkpointed"
    );
    assert_eq!(
        store.run_status("run-cancel-mid").await.unwrap().as_deref(),
        Some(RunStatus::Halted.as_str())
    );
}

#[tokio::test]
async fn abort_budget_breach_interrupts_in_flight_item_and_resume_does_not_respend() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(BarrierTeacher::new(2, 0.01));
    let judge = Arc::new(SlowFirstJudge::new());
    let first_engine = Engine::new(
        clients(
            store.clone(),
            teacher.clone(),
            judge.clone(),
            0.015,
            EventSink::disconnected(),
        ),
        area_k1(one_judge(), lenient_thresholds()),
        2,
    )
    .with_on_breach(BudgetBreach::Abort);
    let seed_source = source(&["q0", "q1", "q2", "q3"], 2);

    let first = first_engine
        .run("run-abort-budget", &seed_source, CancellationToken::new())
        .await
        .unwrap();

    assert!(!first.completed);
    assert_eq!(
        teacher.call_count(),
        2,
        "only the first in-flight pair spent"
    );
    let first_states = states(&store, "run-abort-budget").await;
    assert!(
        first_states.contains(&LifecycleState::Judged),
        "Abort leaves one in-flight item at a persisted non-terminal boundary: {first_states:?}"
    );
    assert!(
        first_states.contains(&LifecycleState::Exported),
        "one sibling completed before the budget breach fired: {first_states:?}"
    );
    assert_eq!(
        store
            .run_status("run-abort-budget")
            .await
            .unwrap()
            .as_deref(),
        Some(RunStatus::Halted.as_str())
    );

    let resumed_engine = Engine::new(
        clients(
            store.clone(),
            teacher.clone(),
            judge.clone(),
            25.0,
            EventSink::disconnected(),
        ),
        area_k1(one_judge(), lenient_thresholds()),
        2,
    )
    .with_on_breach(BudgetBreach::Abort);
    let resumed = resumed_engine
        .run("run-abort-budget", &seed_source, CancellationToken::new())
        .await
        .unwrap();

    assert!(resumed.completed);
    assert_eq!(
        teacher.call_count(),
        4,
        "resume generates only the two never-started items; the interrupted item is not re-spent"
    );
    assert_eq!(judge.call_count(), 4);
    assert_eq!(
        states(&store, "run-abort-budget")
            .await
            .into_iter()
            .filter(|state| *state == LifecycleState::Exported)
            .count(),
        4
    );
}

#[tokio::test]
async fn drain_budget_breach_lets_in_flight_items_reach_terminal_states() {
    let store = Store::open_in_memory().await.unwrap();
    let teacher = Arc::new(BarrierTeacher::new(2, 0.01));
    let judge = Arc::new(SlowFirstJudge::new());
    let engine = Engine::new(
        clients(
            store.clone(),
            teacher.clone(),
            judge,
            0.015,
            EventSink::disconnected(),
        ),
        area_k1(one_judge(), lenient_thresholds()),
        2,
    )
    .with_on_breach(BudgetBreach::Drain);

    let report = engine
        .run(
            "run-drain-budget",
            &source(&["q0", "q1", "q2", "q3"], 2),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert!(!report.completed);
    assert_eq!(teacher.call_count(), 2);
    let run_states = states(&store, "run-drain-budget").await;
    assert_eq!(
        run_states
            .iter()
            .filter(|state| **state == LifecycleState::Exported)
            .count(),
        2,
        "Drain preserves the old behavior: already-started items finish"
    );
    assert!(
        !run_states.contains(&LifecycleState::Judged),
        "Drain must not interrupt an in-flight item at Judged: {run_states:?}"
    );
    assert_eq!(
        store
            .run_status("run-drain-budget")
            .await
            .unwrap()
            .as_deref(),
        Some(RunStatus::Halted.as_str())
    );
}
