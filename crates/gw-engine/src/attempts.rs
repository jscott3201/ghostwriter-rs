//! Engine/storage integration for physical request evidence, without monetary serialization.
use crate::Clients;
use gw_generate::{Embedder, EmbeddingFuture};
use gw_providers::{
    AttemptObserver, CallObservation, ChatRequest, ObservationContext, ObservationError,
    ObservationFuture, Provider, StreamChatFuture,
};
use gw_schema::{
    AccountingCapability, AttemptContext, AttemptIntent, AttemptMetadata, AttemptPurpose,
    AttemptRole, LaunchCoverage, OutputInterpretation, TransportSettlement,
};
use gw_storage::Store;
use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct LaunchObservation {
    pub(crate) coverage: LaunchCoverage,
    observer: Arc<dyn AttemptObserver>,
}
impl LaunchObservation {
    pub(crate) fn new(coverage: LaunchCoverage, store: Store) -> Self {
        Self {
            coverage,
            observer: Arc::new(StoreObserver(store)),
        }
    }
}
struct StoreObserver(Store);
impl AttemptObserver for StoreObserver {
    fn begin(&self, intent: AttemptIntent) -> ObservationFuture<'_, String> {
        Box::pin(async move {
            self.0
                .begin_model_attempt(&intent)
                .await
                .map_err(|e| ObservationError(e.to_string()))
        })
    }
    fn metadata(
        &self,
        id: String,
        sequence: u64,
        metadata: AttemptMetadata,
    ) -> ObservationFuture<'_, ()> {
        Box::pin(async move {
            self.0
                .observe_model_attempt(&id, sequence, &metadata)
                .await
                .map_err(|e| ObservationError(e.to_string()))
        })
    }
    fn settle(&self, id: String, settlement: TransportSettlement) -> ObservationFuture<'_, ()> {
        Box::pin(async move {
            self.0
                .settle_model_attempt(&id, &settlement)
                .await
                .map_err(|e| ObservationError(e.to_string()))
        })
    }
    fn interpret(
        &self,
        id: String,
        interpretation: OutputInterpretation,
    ) -> ObservationFuture<'_, ()> {
        Box::pin(async move {
            self.0
                .interpret_model_attempt(&id, interpretation)
                .await
                .map_err(|e| ObservationError(e.to_string()))
        })
    }
}
impl Clients {
    fn observation(
        &self,
        record_id: &str,
        role: AttemptRole,
        purpose: AttemptPurpose,
    ) -> Option<ObservationContext> {
        self.observation.as_ref().map(|launch| {
            // Engine-owned record IDs carry a shard; arbitrary direct-call IDs remain unknown.
            let shard = record_id
                .strip_prefix(&format!("{}-s", launch.coverage.run_id))
                .and_then(|rest| rest.split_once("-seed"))
                .and_then(|(shard, _)| shard.parse().ok());
            ObservationContext {
                observer: launch.observer.clone(),
                context: AttemptContext {
                    run_id: launch.coverage.run_id.clone(),
                    launch_id: launch.coverage.launch_id.clone(),
                    shard,
                    record_id: Some(record_id.into()),
                    role,
                    purpose,
                },
            }
        })
    }
    pub(crate) fn teacher_for(
        &self,
        record_id: &str,
        purpose: AttemptPurpose,
    ) -> ContextProvider<'_> {
        ContextProvider {
            inner: self.teacher.as_ref(),
            observation: self.observation(record_id, AttemptRole::Teacher, purpose),
        }
    }
    pub(crate) fn judge_for(&self, record_id: &str) -> ContextProvider<'_> {
        ContextProvider {
            inner: self.judge.as_ref(),
            observation: self.observation(record_id, AttemptRole::Judge, AttemptPurpose::Grade),
        }
    }
    pub(crate) fn embedder_for(
        &self,
        record_id: &str,
        purpose: AttemptPurpose,
    ) -> ContextEmbedder<'_> {
        ContextEmbedder {
            inner: self.embedder.as_ref(),
            observation: self.observation(record_id, AttemptRole::Embedding, purpose),
        }
    }
}

pub(crate) struct ContextProvider<'a> {
    inner: &'a dyn Provider,
    observation: Option<ObservationContext>,
}
impl Provider for ContextProvider<'_> {
    fn stream_chat(&self, req: ChatRequest) -> StreamChatFuture<'_> {
        self.inner.stream_chat(req)
    }
    fn stream_chat_observed(
        &self,
        req: ChatRequest,
        observation: CallObservation,
    ) -> StreamChatFuture<'_> {
        self.inner.stream_chat_observed(req, observation)
    }
    fn accounting_capability(&self) -> AccountingCapability {
        self.inner.accounting_capability()
    }
    fn observation_context(&self) -> Option<ObservationContext> {
        self.observation.clone()
    }
}

pub(crate) struct ContextEmbedder<'a> {
    inner: &'a dyn Embedder,
    observation: Option<ObservationContext>,
}
impl Embedder for ContextEmbedder<'_> {
    fn embed<'a>(&'a self, text: &'a str) -> EmbeddingFuture<'a> {
        match &self.observation {
            Some(context) => self.inner.embed_observed(text, context.call()),
            None => self.inner.embed(text),
        }
    }
    fn accounting_capability(&self) -> AccountingCapability {
        self.inner.accounting_capability()
    }
}
