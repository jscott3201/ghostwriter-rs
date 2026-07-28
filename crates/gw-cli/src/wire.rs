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
//! | `budget`   | [`BudgetMeter`] at `config.budget_usd`                                 |
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

use gw_engine::{AreaConfig, BudgetMeter, Clients, Engine, EventSink, ExportSpec};
use gw_generate::{Embedder, NullEmbedder};
use gw_judge::NullSandboxOracle;
use gw_providers::{EmbeddingsClient, OpenRouterProvider, Provider};
use gw_schema::EmbeddingBackend;
use gw_storage::Store;

use crate::config::Config;
use crate::embedder::HttpEmbedder;

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
    let provider = OpenRouterProvider::builder()
        .base_url(&config.provider_base_url)
        .rpm(config.provider_rpm)
        .title("ghostwriter-rs")
        .build()
        .context("constructing the OpenRouter provider (is OPENROUTER_API_KEY set?)")?;
    Ok(Arc::new(provider))
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
/// [`EventSink`], and `config.budget_usd`.
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
    budget_usd: f64,
    embedding: Option<&gw_schema::EmbeddingConfig>,
) -> anyhow::Result<Clients> {
    let embedder: Arc<dyn Embedder + Send + Sync> = match embedding {
        None => Arc::new(NullEmbedder),
        Some(config) => match config.backend {
            EmbeddingBackend::OpenAiCompatible => {
                let client = EmbeddingsClient::builder()
                    .base_url(
                        config
                            .endpoint
                            .as_deref()
                            .unwrap_or(gw_schema::DEFAULT_EMBEDDING_ENDPOINT),
                    )
                    .model(&config.model)
                    .dim(config.dim)
                    .api_key_env(config.api_key_env.clone())
                    .build()
                    .context("constructing the embeddings client")?;
                Arc::new(HttpEmbedder::new(client))
            }
            EmbeddingBackend::CandleLocal => {
                anyhow::bail!("embedding backend candle_local is not constructible in v1")
            }
        },
    };
    Ok(Clients::new(
        store,
        Arc::clone(&provider), // teacher rail
        provider,              // judge rail (same endpoint in v1)
        embedder,
        Arc::new(NullSandboxOracle),
        BudgetMeter::new(budget_usd),
        events,
        HARNESS_VERSION,
    ))
}

/// Build the [`Engine`] end-to-end: open the store, build the provider (KEY from env), assemble the
/// clients, and map the config's area into an [`AreaConfig`].
///
/// Returns the engine plus the opened [`Store`] handle (the caller needs it for the post-run tally /
/// export). The `events` sink and `max_in_flight` cap are the caller's (a headless run passes
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
) -> anyhow::Result<(Engine, Store)> {
    let store = open_store(config).await?;
    let provider = build_provider(config)?;
    let clients = build_clients(
        store.clone(),
        provider,
        events,
        config.budget_usd,
        config.embedding.as_ref(),
    )?;
    let area: AreaConfig = config.area_config();
    let engine = configure_engine(Engine::new(clients, area, max_in_flight), config);
    Ok((engine, store))
}

fn configure_engine(mut engine: Engine, config: &Config) -> Engine {
    engine = engine.with_on_breach(config.on_breach);
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
        // and the budget meter carries the configured cap.
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
            12.5,
            None,
        )
        .expect("clients build");

        // Both rails are the SAME provider Arc in v1.
        assert!(Arc::ptr_eq(&clients.teacher, &clients.judge));
        // The budget cap flowed through.
        assert!((clients.budget.cap() - 12.5).abs() < 1e-12);
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

    #[tokio::test]
    async fn on_breach_config_maps_to_engine_policy() {
        let provider: Arc<dyn Provider> = Arc::new(
            OpenRouterProvider::builder()
                .build_with_key("DUMMY-TEST-KEY-NOT-A-CREDENTIAL")
                .expect("provider builds with an explicit test key"),
        );
        let store = Store::open_in_memory().await.expect("in-memory store");
        let clients = build_clients(store, provider, EventSink::disconnected(), 1.0, None)
            .expect("clients build");
        let config = Config {
            on_breach: gw_schema::BudgetBreach::Abort,
            ..Config::default()
        };

        let engine = configure_engine(Engine::new(clients, config.area_config(), 1), &config);

        assert_eq!(engine.on_breach(), gw_schema::BudgetBreach::Abort);
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
            1.0,
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
            1.0,
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
            1.0,
            Some(&embedding),
        )
        .expect("keyless client construction does not perform a request");
    }
}
