//! Storage-independent async observation at the physical transmission boundary.
use crate::ProviderError;
use gw_schema::{
    AttemptContext, AttemptIntent, AttemptMetadata, OutputInterpretation, TransportSettlement,
};
use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    time::Instant,
};

/// Observer failure. Storage implementations retain their own typed error until this boundary.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct ObservationError(pub String);

/// Object-safe asynchronous persistence operation.
pub type ObservationFuture<'a, T> =
    Pin<Box<dyn Future<Output = Result<T, ObservationError>> + Send + 'a>>;

/// Durable observer; successful `begin` authorizes exactly one client transmission.
/// Implementations must await persistence and must not hold a lock across network consumption.
pub trait AttemptObserver: Send + Sync {
    /// Persist intent before sending; failure prevents the POST.
    fn begin(&self, intent: AttemptIntent) -> ObservationFuture<'_, String>;
    /// Persist changed cumulative metadata, without adding repeated values.
    fn metadata(&self, id: String, metadata: AttemptMetadata) -> ObservationFuture<'_, ()>;
    /// Persist transport outcome; retrying this write must not duplicate spend.
    fn settle(&self, id: String, settlement: TransportSettlement) -> ObservationFuture<'_, ()>;
    /// Persist higher-layer output interpretation, separate from transport.
    fn interpret(
        &self,
        id: String,
        interpretation: OutputInterpretation,
    ) -> ObservationFuture<'_, ()>;
}

/// Engine context bound to a provider without changing its serialized request body.
#[derive(Clone)]
pub struct ObservationContext {
    /// Durable observer supplied by the engine integration.
    pub observer: Arc<dyn AttemptObserver>,
    /// Ownership and purpose of calls made through this context.
    pub context: AttemptContext,
}
impl ObservationContext {
    /// Start independent tracking for one logical call and its intentional transport retries.
    #[must_use]
    pub fn call(&self) -> CallObservation {
        CallObservation {
            context: self.clone(),
            latest: Arc::new(Mutex::new(None)),
        }
    }
}

/// Observation handle for one logical request. Dropping it performs no asynchronous writes.
#[derive(Clone)]
pub struct CallObservation {
    context: ObservationContext,
    latest: Arc<Mutex<Option<String>>>,
}
impl CallObservation {
    /// Record the final higher-layer interpretation of the latest physical attempt.
    /// A pre-send failure has no receipt to interpret.
    pub async fn interpret(
        &self,
        value: OutputInterpretation,
        primary: Option<String>,
    ) -> Result<(), ProviderError> {
        let id = self
            .latest
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        if let Some(id) = id {
            self.context
                .observer
                .interpret(id, value)
                .await
                .map_err(|e| accounting("interpretation", e, primary))?;
        }
        Ok(())
    }

    pub(crate) async fn begin(
        &self,
        body: &[u8],
        model: &str,
        endpoint: &str,
        retry_ordinal: u32,
    ) -> Result<ActiveAttempt, ProviderError> {
        let endpoint = redacted_endpoint(endpoint)?;
        let intent = AttemptIntent {
            version: 1,
            context: self.context.context.clone(),
            request_digest: blake3::hash(body).to_hex().to_string(),
            retry_ordinal,
            requested_model: model.into(),
            endpoint,
        };
        let id = self
            .context
            .observer
            .begin(intent)
            .await
            .map_err(|e| accounting("pre-send intent", e, None))?;
        *self
            .latest
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(id.clone());
        Ok(ActiveAttempt {
            id,
            observer: self.context.observer.clone(),
            started: Instant::now(),
            last_metadata: AttemptMetadata::default(),
        })
    }
}

pub(crate) struct ActiveAttempt {
    id: String,
    observer: Arc<dyn AttemptObserver>,
    started: Instant,
    last_metadata: AttemptMetadata,
}
impl ActiveAttempt {
    pub(crate) async fn metadata(&mut self, value: AttemptMetadata) -> Result<(), ProviderError> {
        let mut merged = self.last_metadata.clone();
        crate::metadata::merge_present(&mut merged, &value);
        if merged == self.last_metadata {
            return Ok(());
        }
        self.observer
            .metadata(self.id.clone(), value)
            .await
            .map_err(|e| accounting("metadata", e, None))?;
        self.last_metadata = merged;
        Ok(())
    }
    pub(crate) async fn settle(
        &self,
        outcome: gw_schema::TransportOutcome,
        http_status: Option<u16>,
        primary: Option<String>,
    ) -> Result<(), ProviderError> {
        let settlement = TransportSettlement {
            outcome,
            http_status,
            elapsed_ms: u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX),
        };
        self.observer
            .settle(self.id.clone(), settlement)
            .await
            .map_err(|e| accounting("transport settlement", e, primary))
    }
}

fn redacted_endpoint(endpoint: &str) -> Result<String, ProviderError> {
    let mut url = reqwest::Url::parse(endpoint)
        .map_err(|_| ProviderError::Config("invalid endpoint URL".into()))?;
    let _ = url.set_username("");
    let _ = url.set_password(None);
    url.set_query(None);
    url.set_fragment(None);
    Ok(url.to_string())
}
fn accounting(stage: &str, error: ObservationError, primary: Option<String>) -> ProviderError {
    ProviderError::Accounting {
        stage: stage.into(),
        detail: error.0,
        primary,
    }
}

/// Start streaming with a fresh per-call handle when the engine supplied observation context.
/// Custom providers retain their declared capability; this helper does not certify their internals.
pub async fn observed_chat<P: crate::Provider + ?Sized>(
    provider: &P,
    request: crate::ChatRequest,
) -> Result<(crate::DeltaStream, Option<CallObservation>), ProviderError> {
    let observation = provider.observation_context().map(|context| context.call());
    let result = match &observation {
        Some(call) => provider.stream_chat_observed(request, call.clone()).await,
        None => provider.stream_chat(request).await,
    };
    match result {
        Ok(stream) => Ok((stream, observation)),
        Err(error) => {
            if !error.is_accounting()
                && let Some(call) = observation
            {
                call.interpret(OutputInterpretation::Failed, Some(error.to_string()))
                    .await?;
            }
            Err(error)
        }
    }
}
