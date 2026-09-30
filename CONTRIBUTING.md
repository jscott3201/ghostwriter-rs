# Contributing to ghostwriter-rs

Thanks for your interest! This guide is intentionally short; the authoritative
command list and project invariants live in [AGENTS.md](AGENTS.md).

## Setup

Development and pull requests are managed on Forgejo. Maintainers clone from
Forgejo and use it as their `origin`; GitHub is the public code mirror, with
GitHub Actions disabled. The public checkout remains available below.

```sh
git clone https://github.com/jscott3201/ghostwriter-rs
cd ghostwriter-rs
bash scripts/install-hooks.sh   # shared pre-commit (fmt + gates) and pre-push (clippy)
cargo build --workspace
```

The toolchain is pinned in `rust-toolchain.toml` (Rust 1.98.1, edition 2024).
This release toolchain fixes macOS optimized proc-macro loading; the workspace
minimum Rust version remains 1.95.

## Workflow

1. Branch off `development`: `feat/…`, `fix/…`, `docs/…`, `chore/…`, or `ci/…`.
2. Make focused changes. Keep each source file under the **700-LOC cap** and add
   doc comments to public APIs.
3. Run the local gates (see [AGENTS.md](AGENTS.md) for the canonical list):
   `cargo fmt --all --check`, `cargo clippy --workspace --all-targets --locked -- -D warnings`,
   `cargo nextest run --workspace --locked`, and the doctests.
4. Commit with **Conventional Commits**: `feat(scope): …`, `fix(scope): …`,
   `chore(scope): …`.
5. Open a Forgejo PR into `development` and fill out the template. Dev-PR CI runs
   formatting, file-size, secret, documentation, and dependency checks. The
   heavy build/clippy/test checks run on Linux at the `development → main`
   release gate. Each workflow reports a required `CI OK` status that succeeds
   only when every job passes.

The active workflows live in `.forgejo/workflows/`. The `.github/workflows/`
copies are retained for reference; GitHub receives code through the push mirror.

## macOS release validation

Before merging a release PR into `main`, run the gates in [AGENTS.md](AGENTS.md)
on macOS at the same commit tested by the Forgejo Linux release gate. Also run:

```sh
cargo build --workspace --release --locked
cargo deny check bans licenses sources
cargo audit --color always
```

Record the commit, macOS version, architecture, and results in the release PR.
The Forgejo `CI OK` status covers Linux; the maintainer checks the separate
macOS results before merging.

## Tests

We use [cargo-nextest](https://nexte.st/): `cargo nextest run --workspace`.
Doctests run separately: `cargo test --workspace --doc`.

## License

By contributing, you agree your contributions are dual-licensed under
[MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at the user's option.
