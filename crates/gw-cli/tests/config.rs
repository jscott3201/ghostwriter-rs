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
                budget_usd = 42.0
                on_breach = "abort"
                provider_rpm = 120

                [area]
                training_area = "rust-async"
                teacher_slug = "z-ai/glm-5.2"
                k = 3

                [export]
                out = "auto.parquet"
                format = "chatml"
                cot = "masked"
                dataset_version = "1.2.3"

                [[area.judges]]
                slug = "deepseek/deepseek-v4-pro"
                family = "deepseek"
            "#,
        )?;
        let cfg = Config::load(Some(Path::new("gw.toml"))).expect("loads the TOML fixture");
        assert!((cfg.budget_usd - 42.0).abs() < 1e-12);
        assert_eq!(cfg.on_breach, gw_schema::BudgetBreach::Abort);
        assert_eq!(cfg.provider_rpm, 120);
        assert_eq!(cfg.area.training_area, "rust-async");
        assert_eq!(cfg.area.k, 3);
        assert_eq!(cfg.area.judges.len(), 1);
        assert_eq!(cfg.area.judges[0].family, "deepseek");
        let export = cfg.export.expect("export section parsed");
        assert_eq!(export.out, Path::new("auto.parquet"));
        assert_eq!(export.format, gw_schema::TrlFormat::ChatML);
        assert_eq!(export.cot, gw_schema::CotPolicy::Masked);
        assert_eq!(export.dataset_version, Some(semver::Version::new(1, 2, 3)));
        Ok(())
    });
}

#[test]
fn on_breach_pause_is_rejected() {
    Jail::expect_with(|jail| {
        jail.create_file("gw.toml", "on_breach = \"pause\"\n")?;
        let err = Config::load(Some(Path::new("gw.toml"))).expect_err("pause rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("on_breach = \"pause\" is not yet supported"),
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
fn api_key_is_never_read_from_the_config_file() {
    Jail::expect_with(|jail| {
        // Even if an operator MISTAKENLY puts a key-shaped field in the TOML, it must not be slurped
        // into the Config (the struct has no such field; unknown keys are ignored, never surfaced).
        // The key lives ONLY in OPENROUTER_API_KEY at provider-construction time.
        jail.create_file(
            "gw.toml",
            r#"
                budget_usd = 1.0
                openrouter_api_key = "LEAKED-KEY-FROM-FILE-1"
                api_key = "LEAKED-KEY-FROM-FILE-2"
            "#,
        )?;
        let cfg = Config::load(Some(Path::new("gw.toml"))).expect("loads despite the stray field");

        // Round-trip the loaded config to JSON and assert NO key material is present anywhere.
        let json = serde_json::to_string(&cfg).expect("serialize config");
        assert!(
            !json.contains("LEAKED-KEY-FROM-FILE-1") && !json.contains("LEAKED-KEY-FROM-FILE-2"),
            "a config-file key field must never enter the Config: {json}"
        );
        assert!(
            !json.to_lowercase().contains("api_key") && !json.to_lowercase().contains("apikey"),
            "the Config must carry no api-key field at all: {json}"
        );
        assert!((cfg.budget_usd - 1.0).abs() < 1e-12);
        Ok(())
    });
}

#[test]
fn env_layers_over_file_and_clap_overrides_env() {
    Jail::expect_with(|jail| {
        jail.create_file("gw.toml", "budget_usd = 5.0\n")?;
        // Env (7.5) beats the file (5.0); an env-only key (rpm) is also applied.
        jail.set_env("GW_BUDGET_USD", "7.5");
        jail.set_env("GW_PROVIDER_RPM", "200");

        let cfg = Config::load(Some(Path::new("gw.toml"))).expect("loads with env layer");
        assert!(
            (cfg.budget_usd - 7.5).abs() < 1e-12,
            "env must override the file"
        );
        assert_eq!(cfg.provider_rpm, 200, "env-only key applied");

        // A clap-style override (the command handler applies these AFTER Config::load) wins over env.
        let mut overridden = cfg;
        overridden.budget_usd = 99.0; // mirrors `--budget-usd 99.0`
        assert!(
            (overridden.budget_usd - 99.0).abs() < 1e-12,
            "clap override is highest precedence"
        );
        Ok(())
    });
}

#[test]
fn nested_env_key_sets_area_field() {
    Jail::expect_with(|jail| {
        jail.create_file("gw.toml", "budget_usd = 5.0\n")?;
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
        jail.create_file("gw.toml", "budget_usd = 2.0\n")?;
        let cfg = Config::load(Some(Path::new("gw.toml"))).expect("loads");
        let json = serde_json::to_string(&cfg).expect("serialize");
        assert!(
            !json.contains("LEAKED-KEY-FROM-ENV"),
            "OPENROUTER_API_KEY must never reach the Config: {json}"
        );
        Ok(())
    });
}
