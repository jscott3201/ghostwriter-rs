//! Engine/storage integration for physical request evidence, with policy-aware admission at each physical send.
use crate::Clients;
use gw_generate::{Embedder, EmbeddingFuture};
use gw_providers::{
    CallObservation, ChatRequest, ObservationContext, Provider, ProviderError, StreamChatFuture,
};
use gw_schema::{
    AccountingCapability, AttemptContext, AttemptPurpose, AttemptRole, LaunchCoverage,
};
use gw_storage::Store;
use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct LaunchObservation {
    pub(crate) coverage: LaunchCoverage,
    pub(crate) observer: Arc<crate::admission::StoreObserver>,
    priors_seeded: Arc<tokio::sync::OnceCell<()>>,
}
impl LaunchObservation {
    pub(crate) fn new(
        coverage: LaunchCoverage,
        store: Store,
        cancel: tokio_util::sync::CancellationToken,
        events: crate::EventSink,
    ) -> Self {
        Self {
            priors_seeded: Arc::new(tokio::sync::OnceCell::new()),
            observer: Arc::new(crate::admission::StoreObserver::new(
                store,
                coverage.clone(),
                cancel,
                events,
            )),
            coverage,
        }
    }
}
impl Clients {
    pub(crate) async fn prepare_generation_priors(&self) -> crate::Result<()> {
        if let Some(launch) = &self.observation {
            launch
                .priors_seeded
                .get_or_try_init(|| {
                    crate::priors::seed(self, &launch.coverage.run_id, &launch.observer.cancel)
                })
                .await?;
        }
        Ok(())
    }
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
        if self.observation.is_none()
            && self.inner.accounting_capability() != AccountingCapability::NoModelRequests
        {
            return Box::pin(async {
                Err(ProviderError::Admission(
                    gw_schema::AdmissionDenial::UnregisteredContext,
                ))
            });
        }
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
            None if self.inner.accounting_capability() == AccountingCapability::NoModelRequests => {
                self.inner.embed(text)
            }
            None => Box::pin(async {
                Err(ProviderError::Admission(
                    gw_schema::AdmissionDenial::UnregisteredContext,
                ))
            }),
        }
    }
    fn accounting_capability(&self) -> AccountingCapability {
        self.inner.accounting_capability()
    }
}
