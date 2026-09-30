//! Shared configuration and explicit replacement semantics for every live command.
#![allow(clippy::result_large_err)]
use clap::Parser;
use figment::Jail;
use gw_cli::{
    cli::{AccountingArgs, AccountingMode, Cli, Command, GenCommand},
    config::Config,
};
use gw_schema::AccountingPolicy as Policy;
use std::path::Path;

#[test]
fn mode_replacement_never_inherits_a_stale_finite_limit() {
    Jail::expect_with(|jail| {
        jail.create_file(
            "gw.toml",
            "accounting_policy = { mode = \"finite_usd\", limit_usd = 5.0 }\n",
        )?;
        jail.set_env("GW_ACCOUNTING_POLICY__MODE", "observation_only");
        let mut config = Config::load(Some(Path::new("gw.toml"))).unwrap();
        assert_eq!(config.effective_policy(), Policy::ObservationOnly);
        AccountingArgs {
            accounting_policy: Some(AccountingMode::FiniteUsd),
            limit_usd: Some(9.0),
        }
        .apply(&mut config)
        .unwrap();
        assert_eq!(
            config.effective_policy(),
            Policy::FiniteUsd { limit_usd: 9.0 }
        );
        AccountingArgs {
            accounting_policy: Some(AccountingMode::ObservationOnly),
            limit_usd: None,
        }
        .apply(&mut config)
        .unwrap();
        assert_eq!(config.effective_policy(), Policy::ObservationOnly);
        Ok(())
    });
}

#[test]
fn absent_policy_defaults_after_layering_and_env_limit_can_override_file() {
    Jail::expect_with(|jail| {
        assert_eq!(
            Config::load(None).unwrap().effective_policy(),
            Policy::FiniteUsd { limit_usd: 5.0 }
        );
        jail.create_file(
            "gw.toml",
            "accounting_policy = { mode = \"finite_usd\", limit_usd = 2.0 }\n",
        )?;
        jail.set_env("GW_ACCOUNTING_POLICY__LIMIT_USD", "7.5");
        assert_eq!(
            Config::load(Some(Path::new("gw.toml")))
                .unwrap()
                .effective_policy(),
            Policy::FiniteUsd { limit_usd: 7.5 }
        );
        Ok(())
    });
}

#[test]
fn invalid_incomplete_and_removed_config_fail_closed() {
    Jail::expect_with(|jail| {
        for source in [
            "accounting_policy = { mode = \"observation_only\", limit_usd = 4.0 }",
            "accounting_policy = { mode = \"finite_usd\" }",
            "accounting_policy = { mode = \"finite_usd\", limit_usd = -1.0 }",
            "accounting_policy = { mode = \"finite_usd\", limit_usd = nan }",
            "accounting_policy = { mode = \"finite_usd\", limit_usd = inf }",
            "budget_usd = 5.0",
            "on_breach = \"drain\"",
            "[budget]\ncap_usd = 5.0",
        ] {
            jail.create_file("gw.toml", source)?;
            assert!(
                Config::load(Some(Path::new("gw.toml"))).is_err(),
                "accepted {source}"
            );
        }
        jail.set_env("GW_ON_BREACH", "abort");
        assert!(Config::load(None).is_err());
        Ok(())
    });
}

#[test]
fn all_live_commands_apply_the_same_policy_arguments_and_reject_old_flags() {
    for command in ["run", "replay", "tui"] {
        let base = [
            "gw",
            "gen",
            command,
            "--run-id",
            "r",
            "--prompts",
            "p",
            "--shards",
            "1",
        ];
        let mut arguments = base.to_vec();
        arguments.extend(["--accounting-policy", "observation-only"]);
        let cli = Cli::try_parse_from(arguments).unwrap();
        let args = match cli.command {
            Command::Gen(GenCommand::Run(args) | GenCommand::Tui(args)) => args.accounting,
            Command::Gen(GenCommand::Replay(args)) => args.accounting,
            _ => unreachable!(),
        };
        let mut config = Config::default();
        args.apply(&mut config).unwrap();
        assert_eq!(config.effective_policy(), Policy::ObservationOnly);
        for old in ["--budget-usd", "--on-breach"] {
            let mut invalid = base.to_vec();
            invalid.extend([old, "1"]);
            assert!(Cli::try_parse_from(invalid).is_err());
        }
        for (mode, limit) in [
            (AccountingMode::FiniteUsd, None),
            (AccountingMode::ObservationOnly, Some(1.0)),
            (AccountingMode::FiniteUsd, Some(f64::INFINITY)),
        ] {
            assert!(
                AccountingArgs {
                    accounting_policy: Some(mode),
                    limit_usd: limit
                }
                .apply(&mut config)
                .is_err()
            );
        }
    }
}

#[test]
fn documented_full_toml_configs_deserialize_with_real_format_spellings() {
    Jail::expect_with(|jail| {
        let readme = include_str!("../../../README.md");
        for (index, block) in readme.split("```toml\n").skip(1).enumerate() {
            let source = block.split("```").next().unwrap();
            jail.create_file("readme.toml", source)?;
            Config::load(Some(Path::new("readme.toml")))
                .unwrap_or_else(|e| panic!("README TOML block {index}: {e}"));
        }
        for format in [
            "gemma4",
            "chatml",
            "share_gpt",
            "open_ai_messages",
            "harmony",
            "trl_prompt_completion",
        ] {
            jail.create_file(
                "format.toml",
                &format!("[export]\nout = \"example.parquet\"\nformat = \"{format}\"\n"),
            )?;
            Config::load(Some(Path::new("format.toml"))).unwrap();
        }
        Ok(())
    });
}
