//! Integration tests for the figment-layered config: a TOML fixture + `GW_`-prefixed env vars + a
//! clap-style override merge correctly, and the API key is sourced from the ENVIRONMENT, never the
//! config file.
//!
//! Env mutation goes through [`figment::Jail`], which sandboxes the process env + cwd per closure and
//! restores them on exit — so these tests never touch the real environment (and avoid the forbidden
//! `unsafe std::env::set_var`).
//!
//! `Jail::expect_with`'s closure returns `figment::Result<()>`, whose error variant is large; the
//! `result_large_err` lint fires on that figment-dictated signature (not our code), so allow it here.
#![allow(clippy::result_large_err)]

use std::path::Path;

use figment::Jail;
use gw_cli::config::Config;

#[test]
fn toml_file_overrides_defaults() {
    Jail::expect_with(|jail| {
        jail.create_file(
            "gw.toml",
            r#"
                accounting_policy = { mode = "finite_usd", limit_usd = 42.0 }
                provider_rpm = 120

                [area]
                training_area = "rust-async"
                teacher_slug = "z-ai/glm-5.2"
                teacher_max_tokens = 20000
                teacher_reasoning_max_tokens = 12000
                k = 3
                judge_max_tokens = 3600
                judge_reasoning_max_tokens = 1800

                [export]
                out = "auto.parquet"
                format = "chatml"
                cot = "masked"
                dataset_version = "1.2.3"

                [[area.judges]]
                slug = "deepseek/deepseek-v4-pro"
                family = "deepseek"
                max_tokens = 3900
                reasoning_max_tokens = 1200
            "#,
        )?;
        let cfg = Config::load(Some(Path::new("gw.toml"))).expect("loads the TOML fixture");
        assert_eq!(
            cfg.effective_policy(),
            gw_schema::AccountingPolicy::FiniteUsd { limit_usd: 42.0 }
        );
        assert_eq!(cfg.provider_rpm, 120);
        assert_eq!(cfg.area.training_area, "rust-async");
        assert_eq!(cfg.area.k, 3);
        assert_eq!(cfg.area.teacher_max_tokens, Some(20_000));
        assert_eq!(cfg.area.teacher_reasoning_max_tokens, Some(12_000));
        assert_eq!(cfg.area.judge_max_tokens, Some(3600));
        assert_eq!(cfg.area.judge_reasoning_max_tokens, Some(1800));
        assert_eq!(cfg.area.judges.len(), 1);
        assert_eq!(cfg.area.judges[0].family, "deepseek");
        assert_eq!(cfg.area.judges[0].max_tokens, Some(3900));
        assert_eq!(cfg.area.judges[0].reasoning_max_tokens, Some(1200));
        let export = cfg.export.expect("export section parsed");
        assert_eq!(export.out, Path::new("auto.parquet"));
        assert_eq!(export.format, gw_schema::TrlFormat::ChatML);
        assert_eq!(export.cot, gw_schema::CotPolicy::Masked);
        assert_eq!(export.dataset_version, Some(semver::Version::new(1, 2, 3)));
        Ok(())
    });
}

#[test]
fn judge_reasoning_effort_conflicts_with_reasoning_max_tokens() {
    Jail::expect_with(|jail| {
        jail.create_file(
            "gw.toml",
            r#"
                [area]
                judge_reasoning_max_tokens = 1800
                judge_reasoning_effort = "low"

                [[area.judges]]
                slug = "deepseek/deepseek-v4-pro"
                family = "deepseek"
            "#,
        )?;
        let err = Config::load(Some(Path::new("gw.toml"))).expect_err("ambiguous area rejected");
        assert!(
            err.to_string()
                .contains("judge_reasoning_max_tokens and judge_reasoning_effort"),
            "got: {err}"
        );
        Ok(())
    });
}

#[test]
fn teacher_max_tokens_zero_is_rejected() {
    Jail::expect_with(|jail| {
        jail.create_file("gw.toml", "[area]\nteacher_max_tokens = 0\n")?;
        let err = Config::load(Some(Path::new("gw.toml"))).expect_err("zero cap rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("teacher_max_tokens must be greater than zero"),
            "got: {msg}"
        );
        Ok(())
    });
}

#[test]
fn export_cot_defaults_to_supervised() {
    Jail::expect_with(|jail| {
        jail.create_file(
            "gw.toml",
            r#"
                [export]
                out = "auto.parquet"
                format = "chatml"
            "#,
        )?;
        let cfg = Config::load(Some(Path::new("gw.toml"))).expect("loads the TOML fixture");
        let export = cfg.export.expect("export section parsed");
        assert_eq!(export.cot, gw_schema::CotPolicy::Supervised);
        assert!(export.dataset_version.is_none());
        Ok(())
    });
}

#[test]
fn full_embedding_section_parses_and_optional_fields_default() {
    Jail::expect_with(|jail| {
        jail.create_file(
            "gw.toml",
            r#"
                [embedding]
                backend = "open_ai_compatible"
                model = "test-model"
                dim = 3
                index = "usearch"
            "#,
        )?;
        jail.set_env("GW_EMBEDDING__DIM", "4");
        let cfg = Config::load(Some(Path::new("gw.toml"))).expect("full embedding config");
        let embedding = cfg.embedding.expect("embedding present");
        assert_eq!(embedding.dim, 4);
        assert!(embedding.endpoint.is_none());
        assert!(embedding.api_key_env.is_none());
        Ok(())
    });
}

#[test]
fn partial_embedding_section_reports_missing_required_field() {
    Jail::expect_with(|jail| {
        jail.create_file("gw.toml", "[embedding]\nmodel = \"test-model\"\n")?;
        let error = Config::load(Some(Path::new("gw.toml")))
            .expect_err("partial embedding config must fail");
        let message = error.to_string();
        assert!(
            message.contains("backend") || message.contains("dim"),
            "got: {message}"
        );
        Ok(())
    });
}

#[test]
fn unsupported_secret_fields_are_rejected_without_echoing_values() {
    Jail::expect_with(|jail| {
        jail.create_file(
            "gw.toml",
            "openrouter_api_key = \"fixture-not-secret-value\"\n",
        )?;
        let error = Config::load(Some(Path::new("gw.toml")))
            .unwrap_err()
            .to_string();
        assert!(error.contains("unknown field"));
        assert!(!error.contains("fixture-not-secret-value"));
        Ok(())
    });
}

#[test]
fn nested_env_key_sets_area_field() {
    Jail::expect_with(|jail| {
        jail.create_file(
            "gw.toml",
            "accounting_policy = { mode = \"finite_usd\", limit_usd = 5.0 }\n",
        )?;
        // `GW_AREA__K` (the `__` split) targets `area.k`.
        jail.set_env("GW_AREA__K", "6");
        let cfg = Config::load(Some(Path::new("gw.toml"))).expect("loads nested env");
        assert_eq!(cfg.area.k, 6, "GW_AREA__K must set area.k");
        Ok(())
    });
}

#[test]
fn openrouter_api_key_env_does_not_leak_into_config() {
    Jail::expect_with(|jail| {
        // The real key var is present in the environment but carries no `GW_` prefix, so the figment
        // Env layer (scoped to `GW_`) cannot reach it — it never enters the Config.
        jail.set_env("OPENROUTER_API_KEY", "LEAKED-KEY-FROM-ENV");
        jail.create_file(
            "gw.toml",
            "accounting_policy = { mode = \"finite_usd\", limit_usd = 2.0 }\n",
        )?;
        let cfg = Config::load(Some(Path::new("gw.toml"))).expect("loads");
        let json = serde_json::to_string(&cfg).expect("serialize");
        assert!(
            !json.contains("LEAKED-KEY-FROM-ENV"),
            "OPENROUTER_API_KEY must never reach the Config: {json}"
        );
        Ok(())
    });
}
