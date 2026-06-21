# Security Policy

## Reporting a vulnerability

Please report security issues **privately** — do not open a public issue. Use
GitHub's private vulnerability reporting ("Report a vulnerability" under the
repository's Security tab) so the report stays confidential until a fix ships.

Include a description, reproduction steps, and the affected version/commit. We
aim to acknowledge within a few days.

## Scope & posture

- No secrets, API keys, private keys, or credentials are ever committed. Keys live
  in the runtime environment only. A baseline secret scan runs in CI and in the
  pre-commit hook.
- TLS is **rustls-only**; `native-tls`/`openssl` are banned via `deny.toml`, and
  `cargo audit` runs at the release gate.
- The workspace `forbid`s `unsafe` code.
