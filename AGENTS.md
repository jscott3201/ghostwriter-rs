# AGENTS.md — ghostwriter-rs

Machine-facing guide to the **code**. The PR template and hooks point here for the
core invariants and validation commands.

## What this is

`ghostwriter-rs` is a Rust harness that **generates graded chain-of-thought
reasoning traces** as model fine-tuning data: teacher models produce full CoT, a
verifier + judge panel grades and admits each trace, and admitted records export
to a model-agnostic training set.

## Workspace & boundaries

A Cargo workspace (`crates/*`), acyclic by construction:

```
gw-schema → {gw-format, gw-providers, gw-storage} → {gw-generate, gw-judge} → gw-engine → {gw-tui, gw-cli}
{gw-schema, gw-storage} → gw-eval
```

- `gw-schema` depends on **nothing** internal and performs **no I/O** — it is the
  byte-reproducible serde data contract. Do not add I/O or internal deps to it.
- `gw-providers` owns the hand-rolled OpenRouter SSE client: a custom delta type
  captures `content` AND `reasoning`/`reasoning_details` (the typed `async-openai`
  delta silently drops the reasoning field).
- `gw-judge` owns the grading *algorithm* and the live trait-bearing types
  (`Decision`, the per-grade `Verdict`, consensus); `gw-schema` owns only the
  *persisted* envelope `Verdict { Admit, Reject, NeedsReview }`.
- `gw-engine` is headless-runnable; the TUI consumes its Action/event stream and
  never holds engine internals.

## Core invariants (every change must respect)

- **No `unsafe`** — the workspace `forbid`s it.
- **700-LOC cap per source file** (non-empty, non-comment lines). Split, don't grow.
- **Public APIs carry doc comments** (rustdoc runs with `-D warnings`).
- **Reasoning is a first-class sibling of `content`, never inlined** (data-contract
  INVARIANT-a; `Message.reasoning`, `PreferenceSide.reasoning`).
- **Verifier-first hard gate** — the deterministic verifier is authoritative where
  ground truth exists; the judge panel grades elsewhere.
- **Never plain-mean / majority admission** (INVARIANT-f) — consensus is the
  weighted design-effect compute, not an average.
- **Byte-reproducible records + content-hash caches** — re-runs never re-spend.
- **rustls-only TLS** — `native-tls`/`openssl` are banned (`deny.toml`).

## Validate

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo nextest run --workspace --locked        # tests (nextest)
cargo test --workspace --locked --doc         # doctests
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --workspace --document-private-items --locked
bash .github/scripts/check-file-size.sh
bash .github/scripts/check-no-secrets.sh
cargo deny check bans licenses sources        # on dependency changes
```

## Repo hygiene (this repo is open source)

Never commit internal planning, spec, handoff text, agent transcripts, secrets,
keys, local absolute paths, or personal data. Gitignored and never tracked: `_*/`
(e.g. `_research/`), `.aionforge/`, `.claude/`, `*-id.json`, `.env*`, `*.key`,
`*.pem`. PR bodies describe code, behavior, and validation only.
