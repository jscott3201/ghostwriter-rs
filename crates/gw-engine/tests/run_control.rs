//! Cooperative cancellation preserves unfinished checkpoints.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::*;
use gw_engine::{Engine, EventSink, InMemorySeedSource, load_cursor};
use gw_providers::{
    ChatRequest, DeltaStream, Provider, ProviderError, StreamChatFuture, StreamDelta,
};
use gw_schema::LifecycleState;
use gw_storage::{RecordFilter, RunStatus, Store};
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
