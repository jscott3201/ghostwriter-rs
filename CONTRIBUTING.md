# Contributing to ghostwriter-rs

Thanks for your interest! This guide is intentionally short; the authoritative
command list and project invariants live in [AGENTS.md](AGENTS.md).

## Setup

```sh
git clone https://github.com/jscott3201/ghostwriter-rs
cd ghostwriter-rs
bash scripts/install-hooks.sh   # shared pre-commit (fmt + gates) and pre-push (clippy)
cargo build --workspace
```

The toolchain is pinned in `rust-toolchain.toml` (Rust 1.95.0, edition 2024).

## Workflow

1. Branch off `development`: `feat/…`, `fix/…`, `docs/…`, `chore/…`, or `ci/…`.
2. Make focused changes. Keep each source file under the **700-LOC cap** and add
   doc comments to public APIs.
3. Run the local gates (see [AGENTS.md](AGENTS.md) for the canonical list):
   `cargo fmt --all --check`, `cargo clippy --workspace --all-targets --locked -- -D warnings`,
   `cargo nextest run --workspace --locked`, and the doctests.
4. Commit with **Conventional Commits**: `feat(scope): …`, `fix(scope): …`,
   `chore(scope): …`.
5. Open a PR into `development` and fill out the template. Dev-PR CI runs the fast
   gates; the heavy build/clippy/test matrix runs at the `development → main`
   release gate.

## Tests

We use [cargo-nextest](https://nexte.st/): `cargo nextest run --workspace`.
Doctests run separately: `cargo test --workspace --doc`.

## License

By contributing, you agree your contributions are dual-licensed under
[MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at the user's option.
