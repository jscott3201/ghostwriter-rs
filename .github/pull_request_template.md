<!--
  Keep this PR tight. Fill every section. Guidance lives in HTML comments so the
  rendered PR stays clean. Authoritative gate commands + core invariants live in
  AGENTS.md — this template points to them rather than restating them.
-->

## Summary

<!-- What changed and WHY, in a few bullets. -->

-

## Linked issue

<!-- "Closes #N" auto-closes the issue on merge. Use "Refs #N" if it only relates. -->

Refs #

## Type of change

- [ ] Bug fix
- [ ] Feature / enhancement
- [ ] Documentation
- [ ] Chore (deps, tooling, CI, refactor)

## Validation

<!-- Run these locally before opening the PR. Dev-PR CI runs only the FAST gates (fmt, file-size, no-secret, rustdoc, and deny on dependency changes); clippy + nextest run locally (the pre-push hook runs clippy) and in the dev->main release gate. Install hooks once: bash scripts/install-hooks.sh -->

- [ ] `cargo fmt --all --check`
- [ ] `cargo clippy --workspace --all-targets --locked -- -D warnings`
- [ ] `cargo nextest run --workspace --locked` (and `cargo test --workspace --locked --doc`)
- [ ] `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --workspace --document-private-items --locked`
- [ ] Repository fast gates: `bash .github/scripts/check-file-size.sh`, `bash .github/scripts/check-no-secrets.sh`
- [ ] Dependency change ran `cargo deny check bans licenses sources` — or N/A

<!-- Note: the heavy ubuntu + macOS build/clippy/test matrix runs at the development -> main release gate. -->

## Scope & invariants

<!-- Core invariants are defined in AGENTS.md. Confirm this change respects them. -->

- [ ] Change respects the acyclic crate boundaries (`gw-schema` depends on nothing), no `unsafe`, public APIs have doc comments, every source file under the 700-LOC cap, and the data-contract invariants (reasoning never inlined; verifier-first hard gate; never plain-mean admission). See AGENTS.md.

## Public-repo check

- [ ] This PR body describes code, behavior, and validation only. It does not include private planning notes, internal handoff text, agent transcripts, secrets, or personal data.
