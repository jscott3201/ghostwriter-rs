# ghostwriter-rs

A Rust harness for generating **graded chain-of-thought reasoning traces** as model
fine-tuning data. Teacher models produce full chain-of-thought; a verifier + judge panel
grades and admits each trace; admitted records export to a model-agnostic training set.

> **Status:** early scaffold. Public usage guides land as the workspace fills out.

## Workspace

| Crate | Role |
|---|---|
| `gw-schema` | Canonical serde data contract (no I/O); everything depends on it. |
| `gw-format` | Chat-template rendering + SFT/preference export projection. |
| `gw-providers` | Streaming model client (chain-of-thought capture), rate limiting, retries. |
| `gw-storage` | Run/queue/provenance state + columnar dataset export. |
| `gw-generate` | Synthesizes user + assistant turns from teacher models. |
| `gw-judge` | Verifier + judge-panel grading and admission. |
| `gw-engine` | Headless orchestrator (generate → judge → admit → persist). |
| `gw-tui` | Terminal UI. |
| `gw-cli` | The `gw` command-line entrypoint. |
| `gw-eval` | Offline evaluation and dataset diagnostics. |

## Build

```sh
cargo build
```

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.
