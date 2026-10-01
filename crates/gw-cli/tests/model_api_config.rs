//! Model API configuration names and credential selection, without network access.
#![allow(clippy::result_large_err)]

use std::path::Path;

use figment::Jail;
use gw_cli::{config::Config, wire::build_provider};
use gw_providers::{ChatCompletionsProvider, ProviderError};

#[test]
fn canonical_environment_overrides_file_without_capturing_credentials() {
    Jail::expect_with(|jail| {
        jail.clear_env();
        jail.create_file(
            "gw.toml",
            "model_api_base_url = 'http://localhost:8000/v1'\nmodel_api_key_env = 'FILE_MODEL_KEY'\n",
        )?;
        jail.set_env("GW_MODEL_API_BASE_URL", "http://localhost:9000/v1");
        jail.set_env("GW_MODEL_API_KEY_ENV", "CUSTOM_MODEL_KEY");
        jail.set_env("CUSTOM_MODEL_KEY", "credential-fixture-do-not-persist");
        let config = Config::load(Some(Path::new("gw.toml"))).unwrap();
        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["model_api_base_url"], "http://localhost:9000/v1");
        assert_eq!(json["model_api_key_env"], "CUSTOM_MODEL_KEY");
        let provider = build_provider(&config).unwrap();
        let declaration = provider.semantic_declaration().unwrap();
        assert_eq!(
            declaration.configuration["base_endpoint"],
            "http://localhost:9000/v1"
        );
        for output in [json.to_string(), format!("{config:?}")] {
            assert!(!output.contains("credential-fixture-do-not-persist"));
        }
        Ok(())
    });
}

#[test]
fn obsolete_endpoint_settings_diagnose_migration_before_deserialization() {
    Jail::expect_with(|jail| {
        for environment in [false, true] {
            for canonical in [false, true] {
                jail.clear_env();
                let mut toml = String::new();
                if environment {
                    jail.set_env("GW_PROVIDER_BASE_URL", "old-endpoint-secret-fixture");
                    if canonical {
                        jail.set_env("GW_MODEL_API_BASE_URL", "http://localhost:9000/v1");
                    }
                } else {
                    toml.push_str("provider_base_url = 'old-endpoint-secret-fixture'\n");
                    if canonical {
                        toml.push_str("model_api_base_url = 'http://localhost:9000/v1'\n");
                    }
                }
                jail.create_file("gw.toml", &toml)?;
                let error = Config::load(Some(Path::new("gw.toml")))
                    .expect_err("obsolete settings must require migration")
                    .to_string();
                assert!(error.contains("model_api_base_url"), "{error}");
                assert!(error.contains("GW_MODEL_API_BASE_URL"), "{error}");
                assert!(!error.contains("old-endpoint-secret-fixture"), "{error}");
            }
        }
        Ok(())
    });
}

#[test]
fn default_key_uses_model_api_key() {
    Jail::expect_with(|jail| {
        jail.clear_env();
        jail.set_env("MODEL_API_KEY", "model-key-fixture");
        assert!(ChatCompletionsProvider::from_env().is_ok());
        assert!(build_provider(&Config::load(None).unwrap()).is_ok());
        Ok(())
    });
}

#[test]
fn default_key_does_not_fall_back_to_openrouter() {
    Jail::expect_with(|jail| {
        jail.clear_env();
        jail.set_env("OPENROUTER_API_KEY", "old-key-fixture");
        assert!(matches!(
            ChatCompletionsProvider::from_env(),
            Err(ProviderError::MissingApiKey(name)) if name == "MODEL_API_KEY"
        ));
        Ok(())
    });
}

#[test]
fn explicit_key_references_select_custom_and_openrouter_keys() {
    Jail::expect_with(|jail| {
        for name in ["CUSTOM_MODEL_KEY", "OPENROUTER_API_KEY"] {
            jail.clear_env();
            jail.create_file("gw.toml", &format!("model_api_key_env = '{name}'\n"))?;
            jail.set_env(name, "explicit-key-fixture");
            let config = Config::load(Some(Path::new("gw.toml"))).unwrap();
            let provider = build_provider(&config).unwrap();
            assert!(
                !serde_json::to_string(&provider.semantic_declaration())
                    .unwrap()
                    .contains("explicit-key-fixture")
            );
        }
        Ok(())
    });
}

#[test]
fn invalid_key_references_fail_without_echo_or_credential_access() {
    Jail::expect_with(|jail| {
        jail.clear_env();
        for reference in [
            "",
            "invalid-key-secret-fixture",
            "BAD=secret-fixture",
            "9BAD",
        ] {
            let builder = ChatCompletionsProvider::builder().api_key_env(reference);
            let error = builder.semantic_declaration().unwrap_err().to_string();
            assert!(error.contains("API key environment variable"), "{error}");
            if !reference.is_empty() {
                assert!(!error.contains(reference), "{error}");
                assert!(!format!("{builder:?}").contains(reference));
                let config = Config {
                    model_api_key_env: reference.into(),
                    ..Default::default()
                };
                assert!(!format!("{config:?}").contains(reference));
            }
        }
        for value in [
            "'invalid-key-secret-fixture'",
            "['array-secret-fixture']",
            "12345",
        ] {
            jail.create_file("gw.toml", &format!("model_api_key_env = {value}\n"))?;
            let error = Config::load(Some(Path::new("gw.toml")))
                .unwrap_err()
                .to_string();
            assert!(error.contains("API key environment variable"), "{error}");
            assert!(!error.contains("secret-fixture"), "{error}");
            assert!(!error.contains("12345"), "{error}");
        }
        jail.set_env("GW_MODEL_API_KEY_ENV", "env-secret-fixture");
        let error = Config::load(None).unwrap_err().to_string();
        assert!(error.contains("API key environment variable"), "{error}");
        assert!(!error.contains("env-secret-fixture"), "{error}");
        Ok(())
    });
}

#[test]
fn missing_and_invalid_keys_are_redacted() {
    Jail::expect_with(|jail| {
        jail.clear_env();
        jail.set_env("GW_MODEL_API_KEY_ENV", "CUSTOM_MODEL_KEY");
        let config = Config::load(None).unwrap();
        let error = build_provider(&config)
            .err()
            .expect("invalid or absent key");
        assert!(format!("{error:#}").contains("CUSTOM_MODEL_KEY"));
        jail.set_env("CUSTOM_MODEL_KEY", "invalid\nkey-secret-fixture");
        let error = build_provider(&config)
            .err()
            .expect("invalid or absent key");
        let message = format!("{error:#}");
        assert!(message.contains("invalid API key"), "{message}");
        assert!(!message.contains("key-secret-fixture"));
        Ok(())
    });
}
