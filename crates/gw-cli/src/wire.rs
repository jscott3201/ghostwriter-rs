//! Live construction of the engine's injected [`Clients`] bundle + the [`Engine`].
//!
//! This is the FIRST real (non-fake) `Clients` in the workspace: the engine's own tests inject fakes,
//! and gw-cli wires the live providers. Each side-effecting dependency is constructed from the
//! [`Config`] plus the process environment:
//!
//! | field      | constructor                                                           |
//! |------------|-----------------------------------------------------------------------|
//! | `store`    | [`Store::open`] over `config.db`                                      |
//! | `teacher`  | [`OpenRouterProvider`] (base URL + rpm from config, KEY from env)      |
//! | `judge`    | the SAME provider `Arc` (one endpoint for both rails in v1)            |
//! | `embedder` | configured OpenAI-compatible client, or [`NullEmbedder`] when absent   |
//! | `sandbox`  | [`NullSandboxOracle`] (D-SANDBOX deferred per `gw-judge`)              |
//! | `policy`   | [`AccountingPolicy`] from `config.effective_policy()`                                 |
//! | `events`   | the caller's [`EventSink`] (disconnected for headless, subscribed for  |
//! |            | the TUI)                                                               |
//!
//! ## The API key flows from the environment ONLY
//!
//! [`build_provider`] calls [`OpenRouterProviderBuilder::build`](gw_providers::OpenRouterProviderBuilder::build),
//! which reads `OPENROUTER_API_KEY` from the process environment and moves it into a
//! `set_sensitive(true)` `Authorization` header. The key is never read from the config file, never a
//! CLI flag, and never logged: a missing key returns a clean
//! [`ProviderError::MissingApiKey`](gw_providers::ProviderError::MissingApiKey) that names only the
//! variable, never a value.
//!
//! ## Embeddings
//!
//! An absent embedding section preserves the hermetic [`NullEmbedder`] behavior. An
//! OpenAI-compatible section constructs the HTTP client; Candle-local remains unsupported in v1.

use std::sync::Arc;

use anyhow::Context;
use tokio_util::sync::CancellationToken;

use gw_engine::{
    AccountingPolicy, Clients, Engine, EventSink, ExportSpec, PreparedRun, SeedSource,
};
use gw_generate::{Embedder, NullEmbedder};
use gw_judge::{
    ExecutionEvidenceSource, NullExecutionEvidenceSource, NullSandboxOracle, SandboxOracle,
};
use gw_providers::{
    EmbeddingsClient, EmbeddingsClientBuilder, OpenRouterProvider, OpenRouterProviderBuilder,
    Provider,
};
use gw_schema::EmbeddingBackend;
use gw_storage::Store;

use crate::config::Config;

/// The harness version stamped into provenance (the crate version).
pub const HARNESS_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Build the live teacher/judge [`Provider`] from `config` + the `OPENROUTER_API_KEY` environment
/// variable.
///
/// The base URL and per-lane rpm come from the config; the API KEY comes from the environment and is
/// never logged. A v1 run routes BOTH the teacher and the judge rails through this one provider (the
/// returned `Arc` is cloned into both `Clients` fields by [`build_engine`]); a future split-endpoint
/// run would construct a second provider here.
///
/// # Errors
/// Returns the [`ProviderError`](gw_providers::ProviderError) (as `anyhow`) if `OPENROUTER_API_KEY`
/// is unset or the key/headers/HTTP client cannot be constructed.
pub fn build_provider(config: &Config) -> anyhow::Result<Arc<dyn Provider>> {
    let provider = provider_builder(config)
        .build()
        .context("constructing the OpenRouter provider (is OPENROUTER_API_KEY set?)")?;
    Ok(Arc::new(provider))
}

fn provider_builder(config: &Config) -> OpenRouterProviderBuilder {
    OpenRouterProvider::builder()
        .base_url(&config.provider_base_url)
        .rpm(config.provider_rpm)
        .title("ghostwriter-rs")
}
fn embedding_builder(
    config: &gw_schema::EmbeddingConfig,
) -> anyhow::Result<EmbeddingsClientBuilder> {
    match config.backend {
        EmbeddingBackend::CandleLocal => {
            anyhow::bail!("embedding backend candle_local is not constructible in v1")
        }
        EmbeddingBackend::OpenAiCompatible => Ok(EmbeddingsClient::builder()
            .base_url(
                config
                    .endpoint
                    .as_deref()
                    .unwrap_or(gw_schema::DEFAULT_EMBEDDING_ENDPOINT),
            )
            .model(&config.model)
            .dim(config.dim)
            .api_key_env(config.api_key_env.clone())
            .declared_revision(config.revision.clone())
            .declared_index(config.index)),
    }
}

/// Pure CLI preparation shared by run, replay and TUI. Uses the built-in clients' actual descriptor
/// builders without reading credentials, constructing clients or invoking any model/evidence seam.
///
/// # Errors
/// Rejects invalid effective settings, unsupported endpoint forms and invalid captured plans.
pub fn prepare_run(
    config: &Config,
    source: &(impl SeedSource + ?Sized),
) -> anyhow::Result<PreparedRun> {
    config.validate_accounting_policy()?;
    config.validate_generation_budgets()?;
    config.validate_judge_reasoning()?;
    let chat = provider_builder(config).semantic_declaration()?;
    let embedding = match &config.embedding {
        Some(config) => embedding_builder(config)?.semantic_declaration()?,
        None => NullEmbedder
            .semantic_declaration()
            .expect("built-in declaration"),
    };
    Ok(PreparedRun::capture(
        source,
        &config.area_config(),
        gw_schema::ClientSemantics {
            teacher: chat.clone(),
            judge: chat,
            embedding,
            sandbox: NullSandboxOracle
                .semantic_declaration()
                .expect("built-in declaration"),
            execution_evidence: NullExecutionEvidenceSource
                .semantic_declaration()
                .expect("built-in declaration"),
        },
    )?)
}

/// Open the SQLite [`Store`] at `config.db` (creating it + running migrations on first open).
///
/// # Errors
/// Returns the [`StorageError`](gw_storage::StorageError) (as `anyhow`) if the database cannot be
/// opened or migrated.
pub async fn open_store(config: &Config) -> anyhow::Result<Store> {
    Store::open(&config.db)
        .await
        .with_context(|| format!("opening the store at {}", config.db.display()))
}

/// Assemble the live [`Clients`] bundle from an already-opened [`Store`], a [`Provider`], the caller's
/// [`EventSink`], and `config.effective_policy()`.
///
/// The teacher and judge rails share one provider `Arc`. The embedder is configured when an
/// embedding section is present and otherwise uses [`NullEmbedder`]. The sandbox remains the
/// [`NullSandboxOracle`] default. The caller supplies the event sink.
///
/// # Errors
/// Returns a configuration error for an unsupported backend, missing configured key variable,
/// invalid key/header, or HTTP-client construction failure.
pub fn build_clients(
    store: Store,
    provider: Arc<dyn Provider>,
    events: EventSink,
    policy: AccountingPolicy,
    embedding: Option<&gw_schema::EmbeddingConfig>,
) -> anyhow::Result<Clients> {
    let embedder: Arc<dyn Embedder + Send + Sync> = match embedding {
        None => Arc::new(NullEmbedder),
        Some(config) => Arc::new(
            embedding_builder(config)?
                .build()
                .context("constructing the embeddings client")?,
        ),
    };
    Ok(Clients::new(
        store,
        Arc::clone(&provider), // teacher rail
        provider,              // judge rail (same endpoint in v1)
        embedder,
        Arc::new(NullSandboxOracle),
        policy,
        events,
        HARNESS_VERSION,
    ))
}

/// Prepare the captured inputs and effective contract, check stored compatibility, then construct
/// credential-bearing clients and the [`Engine`]. The engine repeats the comparison atomically.
///
/// Returns the engine, opened [`Store`], and immutable [`PreparedRun`] to execute. The `events` sink and `max_in_flight` cap are the caller's (a headless run passes
/// `EventSink::disconnected()`; the TUI passes a subscribed sink and shares its
/// [`CancellationToken`]).
///
/// # Errors
/// Propagates a store-open or provider-construction failure (the latter includes a missing
/// `OPENROUTER_API_KEY`).
pub async fn build_engine(
    config: &Config,
    events: EventSink,
    max_in_flight: u32,
    run_id: &str,
    source: &(impl SeedSource + ?Sized),
    mode: gw_storage::RunMode,
) -> anyhow::Result<(Engine, Store, PreparedRun)> {
    let prepared = prepare_run(config, source)?;
    let store = open_store(config).await?;
    store
        .validate_run_manifest(run_id, prepared.manifest(), mode)
        .await?;
    gw_engine::validate_run_verification(&store, run_id, &config.area_config()).await?;
    let provider = build_provider(config)?;
    let clients = build_clients(
        store.clone(),
        provider,
        events,
        config.effective_policy(),
        config.embedding.as_ref(),
    )?;
    let engine = configure_engine(
        Engine::new(clients, config.area_config(), max_in_flight),
        config,
    );
    Ok((engine, store, prepared))
}

fn configure_engine(mut engine: Engine, config: &Config) -> Engine {
    if let Some(spec) = export_spec(config) {
        engine = engine.with_export(spec);
    }
    engine
}

fn export_spec(config: &Config) -> Option<ExportSpec> {
    config.export.as_ref().map(|export| ExportSpec {
        dst: export.out.clone(),
        target: export.format,
        cot: export.cot,
        dataset_version: export.dataset_version.clone(),
    })
}

/// A fresh [`CancellationToken`] for a run. Factored out so `gen run` and `gen tui` mint it the same
/// way (the TUI clones it so the dashboard and the engine task share one shutdown signal).
#[must_use]
pub fn new_cancel_token() -> CancellationToken {
    CancellationToken::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ExportSettings;

    #[tokio::test]
    async fn build_clients_wires_shared_provider_into_both_rails() {
        // A provider built with an explicit key (no env, no network call) is enough to assert the
        // structural wiring: both rails point at the same Arc, the embedder/sandbox are the v1 seams,
        // and the clients carry the resolved policy.
        let provider: Arc<dyn Provider> = Arc::new(
            OpenRouterProvider::builder()
                .build_with_key("DUMMY-TEST-KEY-NOT-A-CREDENTIAL")
                .expect("provider builds with an explicit test key"),
        );
        let store = Store::open_in_memory().await.expect("in-memory store");
        let clients = build_clients(
            store,
            Arc::clone(&provider),
            EventSink::disconnected(),
            AccountingPolicy::FiniteUsd { limit_usd: 12.5 },
            None,
        )
        .expect("clients build");

        // Both rails are the SAME provider Arc in v1.
        assert!(Arc::ptr_eq(&clients.teacher, &clients.judge));
        // The finite policy flowed through.
        assert_eq!(
            clients.policy,
            AccountingPolicy::FiniteUsd { limit_usd: 12.5 }
        );
        // The harness version is the crate version.
        assert_eq!(clients.harness_version, HARNESS_VERSION);
    }

    #[test]
    fn provider_missing_key_is_a_clean_error_not_a_panic() {
        // Point the provider at an env var that is overwhelmingly unlikely to be set, so the failure
        // path is exercised without depending on the ambient environment.
        let mut cfg = Config::default();
        // build_provider always reads OPENROUTER_API_KEY; to test the missing-key path hermetically
        // we assert the error TYPE only when the key is genuinely absent. Skip if it happens to be set
        // (a developer machine with a real key) so the test never makes a network call or fails.
        if std::env::var("OPENROUTER_API_KEY").is_ok() {
            return;
        }
        cfg.provider_base_url = gw_providers::DEFAULT_BASE_URL.to_string();
        // `Arc<dyn Provider>` is not `Debug`, so match the Result rather than `expect_err`.
        match build_provider(&cfg) {
            Ok(_) => panic!("a missing key must error, not succeed"),
            Err(err) => {
                // The error chain names the key var, never a value.
                let msg = format!("{err:#}");
                assert!(msg.contains("OPENROUTER_API_KEY"), "got: {msg}");
            }
        }
    }

    #[test]
    fn export_config_maps_to_engine_spec() {
        let config = Config {
            export: Some(ExportSettings {
                out: "auto.parquet".into(),
                format: gw_schema::TrlFormat::ChatML,
                cot: gw_schema::CotPolicy::Masked,
                dataset_version: Some(semver::Version::new(1, 2, 3)),
            }),
            ..Config::default()
        };

        let spec = export_spec(&config).expect("export spec is built");
        assert_eq!(spec.dst, std::path::PathBuf::from("auto.parquet"));
        assert_eq!(spec.target, gw_schema::TrlFormat::ChatML);
        assert_eq!(spec.cot, gw_schema::CotPolicy::Masked);
        assert_eq!(spec.dataset_version, Some(semver::Version::new(1, 2, 3)));
    }

    fn test_provider() -> Arc<dyn Provider> {
        Arc::new(
            OpenRouterProvider::builder()
                .build_with_key("DUMMY-TEST-KEY-NOT-A-CREDENTIAL")
                .expect("provider builds"),
        )
    }

    #[tokio::test]
    async fn configured_key_env_missing_names_only_variable() {
        let store = Store::open_in_memory().await.expect("in-memory store");
        let embedding = gw_schema::EmbeddingConfig {
            api_key_env: Some("GW_TEST_EMBEDDING_KEY_DEFINITELY_UNSET_7C91".into()),
            ..gw_schema::EmbeddingConfig::default()
        };
        let result = build_clients(
            store,
            test_provider(),
            EventSink::disconnected(),
            AccountingPolicy::ObservationOnly,
            Some(&embedding),
        );
        let error = match result {
            Ok(_) => panic!("missing configured embedding key must fail"),
            Err(error) => format!("{error:#}"),
        };
        assert!(error.contains("GW_TEST_EMBEDDING_KEY_DEFINITELY_UNSET_7C91"));
        assert!(!error.contains("DUMMY-TEST-KEY"));
    }

    #[tokio::test]
    async fn candle_local_embedding_backend_is_clean_error() {
        let store = Store::open_in_memory().await.expect("in-memory store");
        let embedding = gw_schema::EmbeddingConfig {
            backend: EmbeddingBackend::CandleLocal,
            ..gw_schema::EmbeddingConfig::default()
        };
        let result = build_clients(
            store,
            test_provider(),
            EventSink::disconnected(),
            AccountingPolicy::ObservationOnly,
            Some(&embedding),
        );
        let error = match result {
            Ok(_) => panic!("Candle-local is unsupported"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("candle_local"));
    }

    #[tokio::test]
    async fn keyless_embedding_config_constructs_without_network() {
        let store = Store::open_in_memory().await.expect("in-memory store");
        let embedding = gw_schema::EmbeddingConfig::default();
        build_clients(
            store,
            test_provider(),
            EventSink::disconnected(),
            AccountingPolicy::ObservationOnly,
            Some(&embedding),
        )
        .expect("keyless client construction does not perform a request");
    }
}
