//! Integration tests for the clap command tree: every subcommand + its args parse correctly, and the
//! key error cases (missing required args, unknown subcommands) are rejected. No I/O, no network.

use std::path::PathBuf;

use clap::Parser;
use gw_cli::cli::{Cli, Command, EvalCommand, ExportCot, ExportFormat, GenCommand, OnBreach};

#[test]
fn gen_run_parses_required_and_default_args() {
    let cli = Cli::try_parse_from([
        "gw",
        "gen",
        "run",
        "--run-id",
        "r1",
        "--prompts",
        "seeds.txt",
    ])
    .expect("gen run parses");
    let Command::Gen(GenCommand::Run(args)) = cli.command else {
        panic!("expected gen run");
    };
    assert_eq!(args.run_id, "r1");
    assert_eq!(args.prompts, PathBuf::from("seeds.txt"));
    // Defaults.
    assert_eq!(args.shards, 1);
    assert_eq!(args.max_in_flight, 4);
    assert!(args.config.is_none());
    assert!(args.db.is_none());
    assert!(args.budget_usd.is_none());
    assert!(args.on_breach.is_none());
    assert!(args.k.is_none());
}

#[test]
fn gen_run_accepts_all_overrides() {
    let cli = Cli::try_parse_from([
        "gw",
        "gen",
        "run",
        "--config",
        "gw.toml",
        "--db",
        "store.sqlite",
        "--run-id",
        "r2",
        "--prompts",
        "p.txt",
        "--shards",
        "3",
        "--budget-usd",
        "12.5",
        "--on-breach",
        "abort",
        "--k",
        "4",
        "--max-in-flight",
        "8",
    ])
    .expect("gen run with overrides parses");
    let Command::Gen(GenCommand::Run(args)) = cli.command else {
        panic!("expected gen run");
    };
    assert_eq!(args.config, Some(PathBuf::from("gw.toml")));
    assert_eq!(args.db, Some(PathBuf::from("store.sqlite")));
    assert_eq!(args.shards, 3);
    assert_eq!(args.budget_usd, Some(12.5));
    assert_eq!(args.on_breach, Some(OnBreach::Abort));
    assert_eq!(args.k, Some(4));
    assert_eq!(args.max_in_flight, 8);
}

#[test]
fn gen_run_on_breach_rejects_pause() {
    let err = Cli::try_parse_from([
        "gw",
        "gen",
        "run",
        "--run-id",
        "r1",
        "--prompts",
        "p.txt",
        "--on-breach",
        "pause",
    ]);
    assert!(err.is_err(), "--on-breach exposes only drain|abort");
}

#[test]
fn gen_run_missing_run_id_is_an_error() {
    let err = Cli::try_parse_from(["gw", "gen", "run", "--prompts", "p.txt"]);
    assert!(err.is_err(), "missing --run-id must be rejected");
}

#[test]
fn gen_run_missing_prompts_is_an_error() {
    let err = Cli::try_parse_from(["gw", "gen", "run", "--run-id", "r1"]);
    assert!(err.is_err(), "missing --prompts must be rejected");
}

#[test]
fn gen_tui_parses_same_args_as_run() {
    let cli = Cli::try_parse_from(["gw", "gen", "tui", "--run-id", "t1", "--prompts", "p.txt"])
        .expect("gen tui parses");
    assert!(matches!(
        cli.command,
        Command::Gen(GenCommand::Tui(args)) if args.run_id == "t1"
    ));
}

#[test]
fn gen_export_parses_with_format_and_cot() {
    let cli = Cli::try_parse_from([
        "gw",
        "gen",
        "export",
        "--db",
        "store.sqlite",
        "--out",
        "out.parquet",
        "--run-id",
        "r1",
        "--format",
        "chat-ml",
        "--cot",
        "stripped",
    ])
    .expect("gen export parses");
    let Command::Gen(GenCommand::Export(args)) = cli.command else {
        panic!("expected gen export");
    };
    assert_eq!(args.db, PathBuf::from("store.sqlite"));
    assert_eq!(args.out, PathBuf::from("out.parquet"));
    assert_eq!(args.run_id, Some("r1".to_string()));
    assert_eq!(args.format, ExportFormat::ChatMl);
    assert_eq!(args.cot, ExportCot::Stripped);
}

#[test]
fn gen_export_format_and_cot_default() {
    let cli = Cli::try_parse_from([
        "gw",
        "gen",
        "export",
        "--db",
        "s.sqlite",
        "--out",
        "o.parquet",
    ])
    .expect("gen export defaults");
    let Command::Gen(GenCommand::Export(args)) = cli.command else {
        panic!("expected gen export");
    };
    assert_eq!(args.format, ExportFormat::ChatMl);
    assert_eq!(args.cot, ExportCot::Supervised);
    assert!(args.run_id.is_none());
}

#[test]
fn gen_export_rejects_unknown_format() {
    let err = Cli::try_parse_from([
        "gw",
        "gen",
        "export",
        "--db",
        "s",
        "--out",
        "o",
        "--format",
        "no-such-format",
    ]);
    assert!(err.is_err(), "an unknown --format value must be rejected");
}

#[test]
fn gen_replay_parses() {
    let cli = Cli::try_parse_from([
        "gw",
        "gen",
        "replay",
        "--run-id",
        "r1",
        "--prompts",
        "p.txt",
        "--shards",
        "2",
    ])
    .expect("gen replay parses");
    let Command::Gen(GenCommand::Replay(args)) = cli.command else {
        panic!("expected gen replay");
    };
    assert_eq!(args.run_id, "r1");
    assert_eq!(args.shards, 2);
}

#[test]
fn gen_replay_requires_explicit_shards() {
    // `--shards` has NO default on replay (unlike `gen run`): resume re-derives the seed→shard
    // partition as `index % shards`, so a silently-defaulted value would re-partition the space and
    // duplicate/orphan records against the persisted cursors. Omitting it must be rejected.
    let err = Cli::try_parse_from([
        "gw",
        "gen",
        "replay",
        "--run-id",
        "r1",
        "--prompts",
        "p.txt",
    ]);
    assert!(err.is_err(), "replay must require an explicit --shards");
}

#[test]
fn eval_audit_separation_parses() {
    let cli = Cli::try_parse_from([
        "gw",
        "eval",
        "audit-separation",
        "--db",
        "store.sqlite",
        "--run-id",
        "r1",
        "--min-decidable-groups",
        "5",
        "--min-decidable-fraction",
        "0.1",
        "--check",
    ])
    .expect("eval audit-separation parses");
    let Command::Eval(EvalCommand::AuditSeparation(args)) = cli.command else {
        panic!("expected eval audit-separation");
    };
    assert_eq!(args.db, PathBuf::from("store.sqlite"));
    assert_eq!(args.run_id, Some("r1".to_string()));
    assert_eq!(args.min_decidable_groups, Some(5));
    assert_eq!(args.min_decidable_fraction, Some(0.1));
    assert!(args.check);
}

#[test]
fn eval_audit_separation_check_defaults_to_false() {
    let cli = Cli::try_parse_from(["gw", "eval", "audit-separation", "--db", "store.sqlite"])
        .expect("eval audit-separation defaults");
    let Command::Eval(EvalCommand::AuditSeparation(args)) = cli.command else {
        panic!("expected eval audit-separation");
    };
    assert!(!args.check);
}

#[test]
fn eval_promote_parses_with_drift_exit() {
    let cli = Cli::try_parse_from([
        "gw",
        "eval",
        "promote",
        "--baseline",
        "base.json",
        "--candidate",
        "cand.json",
        "--drift-exit",
        "1",
        "--config",
        "promote.toml",
        "--check",
    ])
    .expect("eval promote parses");
    let Command::Eval(EvalCommand::Promote(args)) = cli.command else {
        panic!("expected eval promote");
    };
    assert_eq!(args.baseline, PathBuf::from("base.json"));
    assert_eq!(args.candidate, PathBuf::from("cand.json"));
    assert_eq!(args.drift_exit, 1);
    assert_eq!(args.config, Some(PathBuf::from("promote.toml")));
    assert!(args.check);
}

#[test]
fn eval_promote_drift_exit_defaults_to_zero() {
    let cli = Cli::try_parse_from([
        "gw",
        "eval",
        "promote",
        "--baseline",
        "b.json",
        "--candidate",
        "c.json",
    ])
    .expect("eval promote defaults");
    let Command::Eval(EvalCommand::Promote(args)) = cli.command else {
        panic!("expected eval promote");
    };
    assert_eq!(args.drift_exit, 0);
    assert!(!args.check);
}

#[test]
fn unknown_subcommand_is_rejected() {
    assert!(Cli::try_parse_from(["gw", "frobnicate"]).is_err());
    assert!(Cli::try_parse_from(["gw", "gen", "frobnicate"]).is_err());
    assert!(Cli::try_parse_from(["gw", "eval", "frobnicate"]).is_err());
}

#[test]
fn version_flag_is_accepted() {
    // `--version` short-circuits to a clap "DisplayVersion" error kind (not a real failure).
    let err = Cli::try_parse_from(["gw", "--version"]).unwrap_err();
    assert_eq!(err.kind(), clap::error::ErrorKind::DisplayVersion);
}
