//! In-memory launch fixtures for private admission and prior-corpus boundaries.
use crate::{Clients, EventSink, attempts::LaunchObservation};
use gw_generate::Embedder;
use gw_providers::{ChatRequest, Provider, StreamChatFuture};
use gw_schema::*;
use gw_storage::{LaunchRequest, Store};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

struct NoChat;
impl Provider for NoChat {
    fn stream_chat(&self, _: ChatRequest) -> StreamChatFuture<'_> {
        Box::pin(async { panic!("these tests exercise embedding and storage only") })
    }
    fn accounting_capability(&self) -> AccountingCapability {
        AccountingCapability::NoModelRequests
    }
}
pub(crate) async fn clients(store: Store, embedder: Arc<dyn Embedder>) -> Clients {
    let mut clients = Clients::new(
        store.clone(),
        Arc::new(NoChat),
        Arc::new(NoChat),
        embedder,
        Arc::new(gw_judge::NullSandboxOracle),
        AccountingPolicy::ObservationOnly,
        EventSink::disconnected(),
        "test",
    );
    let coverage = store
        .register_accounting_launch(LaunchRequest {
            run_id: "r",
            manifest: manifest(),
            mode: gw_storage::RunMode::CreateOrResume,
            policy: &clients.policy,
            teacher: clients.teacher.accounting_capability(),
            judge: clients.judge.accounting_capability(),
            embedding: clients.embedder.accounting_capability(),
        })
        .await
        .unwrap();
    clients.observation = Some(LaunchObservation::new(
        coverage,
        store,
        CancellationToken::new(),
        EventSink::disconnected(),
    ));
    clients
}
pub(crate) fn record(id: &str) -> TrainingRecord {
    serde_json::from_value(serde_json::json!({
        "record_id": id, "schema_version": "1.0.0", "training_area": "fixture",
        "messages": [{"role":"user", "content":format!("question {id}")},
                     {"role":"assistant", "content":"answer", "reasoning":"why"}],
        "provenance": {"run_id":"r", "teacher":{"provider":"fixture", "slug":"fixture"}, "harness_version":"test"},
        "generation": {}, "lifecycle":{"state":"admitted"},
        "judging":{"panel":[], "verdict":"admit", "aggregate":0.9}
    })).unwrap()
}

pub(crate) fn manifest() -> RunManifest {
    let d = SemanticDeclaration::new(
        "test/transaction-fixture",
        "1",
        serde_json::json!({"behavior":"no-client-execution"}),
    );
    RunManifest {
        version: RUN_MANIFEST_VERSION,
        input_plan: InputPlanIdentity {
            content_hash: "a".repeat(64),
            shard_items: vec![1],
        },
        execution: d.clone(),
        clients: ClientSemantics {
            teacher: d.clone(),
            judge: d.clone(),
            embedding: d.clone(),
            sandbox: d.clone(),
            execution_evidence: d,
        },
        unattested_deployment: UnattestedDeployment::default(),
    }
}
