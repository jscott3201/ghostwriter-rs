//! Actual replaceable objects declare semantics independently from accounting and execution.
mod common;
use common::*;
use gw_engine::{Clients, Engine, EventSink};
use gw_generate::{Embedder, EmbeddingFuture};
use gw_judge::{ExecutionEvidenceSource, SandboxOracle};
use gw_providers::{ChatRequest, Provider, StreamChatFuture};
use gw_schema::{
    AccountingCapability, AccountingPolicy, EvidenceBinding, ExecutionEvidence, SemanticDeclaration,
};
use gw_storage::{RunMode, Store};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
struct Spy {
    revision: &'static str,
}
impl Spy {
    fn declaration(&self, lane: &str) -> Option<SemanticDeclaration> {
        Some(SemanticDeclaration::new(
            format!("test/pure-{lane}"),
            self.revision,
            serde_json::json!({"immutable_fixture_digest":"b".repeat(64)}),
        ))
    }
}
impl Provider for Spy {
    fn semantic_declaration(&self) -> Option<SemanticDeclaration> {
        self.declaration("chat")
    }
    fn stream_chat(&self, _: ChatRequest) -> StreamChatFuture<'_> {
        panic!("preparation must not dispatch")
    }
}
impl Embedder for Spy {
    fn semantic_declaration(&self) -> Option<SemanticDeclaration> {
        self.declaration("embedding")
    }
    fn embed<'a>(&'a self, _: &'a str) -> EmbeddingFuture<'a> {
        panic!("preparation must not embed")
    }
}
impl SandboxOracle for Spy {
    fn semantic_declaration(&self) -> Option<SemanticDeclaration> {
        self.declaration("sandbox")
    }
    fn execute(&self, _: &str) -> Result<String, String> {
        panic!("preparation must not execute")
    }
}
impl ExecutionEvidenceSource for Spy {
    fn semantic_declaration(&self) -> Option<SemanticDeclaration> {
        self.declaration("evidence")
    }
    fn evidence(&self, _: &EvidenceBinding) -> Option<ExecutionEvidence> {
        panic!("preparation must not look up evidence")
    }
}
struct Opaque;
impl Provider for Opaque {
    fn accounting_capability(&self) -> AccountingCapability {
        AccountingCapability::NoModelRequests
    }
    fn stream_chat(&self, _: ChatRequest) -> StreamChatFuture<'_> {
        panic!("opaque dispatch")
    }
}
impl Embedder for Opaque {
    fn embed<'a>(&'a self, _: &'a str) -> EmbeddingFuture<'a> {
        panic!("opaque embed")
    }
}
impl SandboxOracle for Opaque {
    fn execute(&self, _: &str) -> Result<String, String> {
        panic!("opaque execute")
    }
}
impl ExecutionEvidenceSource for Opaque {
    fn evidence(&self, _: &EvidenceBinding) -> Option<ExecutionEvidence> {
        panic!("opaque evidence")
    }
}
fn spies(store: &Store) -> Clients {
    let spy = Arc::new(Spy { revision: "1" });
    let mut clients = Clients::new(
        store.clone(),
        spy.clone(),
        spy.clone(),
        spy.clone(),
        spy.clone(),
        AccountingPolicy::ObservationOnly,
        EventSink::disconnected(),
        "test",
    );
    clients.execution_evidence = spy;
    clients
}
fn engine(clients: Clients) -> Engine {
    Engine::new(clients, area_k1(one_judge(), lenient_thresholds()), 1)
}
#[tokio::test]
async fn pure_declarations_do_not_execute_and_unknown_accounting_does_not_replace_semantics() {
    let store = Store::open_in_memory().await.unwrap();
    let engine = engine(spies(&store));
    let prepared = engine.prepare(&one_item_source()).unwrap();
    assert!(store.run_status("pure").await.unwrap().is_none());
    let cancel = CancellationToken::new();
    cancel.cancel();
    engine
        .run_prepared("pure", prepared, RunMode::CreateOrResume, cancel)
        .await
        .unwrap();
    let launches = store.model_launches("pure").await.unwrap();
    assert_eq!(launches.len(), 1);
    assert_eq!(launches[0].teacher, AccountingCapability::Unknown);
    assert_eq!(launches[0].embedding, AccountingCapability::Unknown);
}
#[tokio::test]
async fn each_opaque_seam_fails_before_registration_even_when_request_free() {
    for lane in ["teacher", "judge", "embedding", "sandbox", "evidence"] {
        let store = Store::open_in_memory().await.unwrap();
        let mut clients = spies(&store);
        let opaque = Arc::new(Opaque);
        match lane {
            "teacher" => clients.teacher = opaque,
            "judge" => clients.judge = opaque,
            "embedding" => clients.embedder = opaque,
            "sandbox" => clients.sandbox = opaque,
            "evidence" => clients.execution_evidence = opaque,
            _ => unreachable!(),
        }
        let error = engine(clients)
            .run("opaque", &one_item_source(), CancellationToken::new())
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("semantic declaration"),
            "{lane}: {error}"
        );
        assert!(store.run_status("opaque").await.unwrap().is_none());
        assert!(store.model_launches("opaque").await.unwrap().is_empty());
    }
}
#[tokio::test]
async fn replacements_on_every_public_seam_reject_both_replay_and_stale_preparation() {
    for lane in ["teacher", "judge", "embedding", "sandbox", "evidence"] {
        let store = Store::open_in_memory().await.unwrap();
        let original = engine(spies(&store));
        let prepared = original.prepare(&one_item_source()).unwrap();
        let cancel = CancellationToken::new();
        cancel.cancel();
        original
            .run_prepared(
                "pinned",
                prepared.clone(),
                RunMode::CreateOrResume,
                cancel.clone(),
            )
            .await
            .unwrap();
        let before = store.accounting_snapshot("pinned").await.unwrap();
        let mut clients = spies(&store);
        let replacement = Arc::new(Spy { revision: "2" });
        match lane {
            "teacher" => clients.teacher = replacement,
            "judge" => clients.judge = replacement,
            "embedding" => clients.embedder = replacement,
            "sandbox" => clients.sandbox = replacement,
            "evidence" => clients.execution_evidence = replacement,
            _ => unreachable!(),
        }
        let changed = engine(clients);
        let stale = changed
            .run_prepared("fresh", prepared, RunMode::CreateOrResume, cancel.clone())
            .await
            .unwrap_err();
        assert!(
            stale.to_string().contains("actual injected"),
            "{lane}: {stale}"
        );
        assert!(store.run_status("fresh").await.unwrap().is_none());
        let replay = changed
            .run("pinned", &one_item_source(), cancel)
            .await
            .unwrap_err();
        assert!(replay.to_string().contains("manifest"), "{lane}: {replay}");
        assert_eq!(store.accounting_snapshot("pinned").await.unwrap(), before);
        assert_eq!(store.model_launches("pinned").await.unwrap().len(), 1);
    }
}
