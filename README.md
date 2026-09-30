# ghostwriter-rs

**A Rust harness for generating graded chain-of-thought reasoning traces as model fine-tuning data.**

![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)
![rust](https://img.shields.io/badge/rust-1.95%2B-orange)
![edition](https://img.shields.io/badge/edition-2024-555)
![unsafe](https://img.shields.io/badge/unsafe-forbidden-success)

ghostwriter-rs drives a **teacher** model to produce full chain-of-thought reasoning, runs each
trace through a deterministic **verifier** and a model **judge panel**, admits only the traces that
earn it, and exports the winners as a model-agnostic supervised-fine-tuning (SFT) dataset. The whole
run is event-sourced: it spends a budget you set, resumes after a crash without re-spending the
teacher, and records the full provenance of every decision.

It is built for one job — **minting high-quality reasoning data you can trust** — and it treats every
expensive call (teacher, judge) as something to be budgeted, graded, and accounted for.

> **Status:** active development toward `0.1.0`. The generation → grading → export pipeline runs
> end-to-end today; the surfaces and config are stabilizing. APIs may still shift before the first
> tagged release.

---

## Why

Distilling a strong reasoner into a smaller student is only as good as the data. Naively sampling a
teacher gives you fluent-but-wrong traces, silent truncations, and a corpus that quietly collapses
onto a few modes. ghostwriter-rs makes the quality bar **explicit and enforced**:

- a **verifier rail** rejects malformed or truncated traces before a judge ever sees them;
- a **judge panel** scores each trace against a rubric, and consensus is discounted when judges are
  correlated (so nine lookalike judges don't masquerade as nine independent votes);
- **best-of-k** generates several candidates per prompt and admits only the best;
- a hard **budget** governs spend, with a defined policy for what happens when it's reached.

The output is a self-contained Parquet dataset with an embedded manifest, ready to render into the
chat template of your target student model.

---

## Architecture

Ten crates in a strictly **acyclic** workspace — each tier depends only on the tiers above it.

<p align="center"><img src="assets/architecture.svg" alt="ghostwriter-rs crate architecture" width="760"></p>

| Crate | Role |
|---|---|
| `gw-schema` | Canonical serde data contract (no I/O). Everything depends on it. |
| `gw-providers` | Streaming OpenAI-compatible client with chain-of-thought capture, GCRA rate limiting, and retries. |
| `gw-storage` | Run / queue / provenance state (SQLite) and columnar Parquet export. |
| `gw-format` | Chat-template rendering and the SFT / preference export projection. |
| `gw-generate` | Synthesizes the user turn and the teacher's assistant turn. |
| `gw-judge` | The deterministic verifier and the model judge-panel grading + admission. |
| `gw-engine` | The headless orchestrator: the lifecycle state machine and sharded executor. |
| `gw-cli` | The `gw` command-line entrypoint. |
| `gw-tui` | A live [ratatui](https://ratatui.rs) dashboard over the engine event stream. |
| `gw-eval` | Off-path, model-free dataset diagnostics and promotion gating. |

The dependency spine: `schema → {format, providers, storage} → {generate, judge} → engine → {cli, tui}`,
with `gw-eval` consuming only `schema` + `storage`.

---

## How it works

Each prompt becomes a **record** that walks a 12-state lifecycle. Every transition is persisted, so a
run is fully resumable and every admission decision is auditable after the fact.

<p align="center"><img src="assets/pipeline.svg" alt="the record lifecycle from seed to exported training data" width="900"></p>

1. **Seeded** — a prompt is read from the seed source and a record id is minted.
2. **UserSynthesized** — the user turn is prepared and passes its gate.
3. **AssistantGenerated** — the teacher is called for **k** candidates, capturing content *and*
   reasoning.
4. **Verified** — a deterministic rail checks each candidate, including a *reasoning-present* hard
   gate when the area requires chain-of-thought.
5. **Judged** — the judge panel scores the surviving candidates; consensus is weighted by an
   **n_eff** design-effect that discounts correlated judges.
6. The verdict routes the record:
   - **Admitted** — the best of k (terminal-good for grading);
   - **Rejected** — retained for preference (DPO) pairs and judge auditing;
   - **Revising** — a single bounded re-generation, then re-judged;
   - **NeedsReview** — parked for human/verifier adjudication.
7. **Formatted → Exported** — admitted records are projected to the target template per the CoT
   policy and written into a versioned Parquet shard.

`Error` is terminal-until-requeue: a faulted record carries its last error and attempt count, and a
re-run picks it back up.

### Judge scoring and cache reuse

Judges return a JSON score and verdict. The implemented method is `json_score`; the audit records
that method and its interpretation version. There is no logprob scoring algorithm or automatic
scoring fallback. The former `GEvalLogprob` / `IntegerLikert` pins and scoring selector have been
removed from the pre-1.0 API; numeric score normalization is unchanged.

Judge cache entries use `judge-request-v2` fingerprints of the built request, optional rubric ID,
sampling bits, scoring method, and interpretation version. Changes to rubric text, candidate text,
prompt framing, or interpretation invalidate the grade. Raw token caps that produce the same
effective request can share a grade, as can identical requests from different runs. The old folded
key helper is removed; legacy cache entries and historical audit records remain untouched but are
not reused by the new cache. This identity does not cover provider-internal routing defaults,
endpoint changes, or resolved model revisions.

---

## Install

Requires **Rust 1.95+** (edition 2024). The repository pins **Rust 1.98.1**
for development and release builds, including a macOS optimized-build fix.

```sh
git clone https://github.com/jscott3201/ghostwriter-rs
cd ghostwriter-rs
cargo build --release
```

The binary is `gw` (`target/release/gw`). Run `cargo run -- <args>` during development, or install it:

```sh
cargo install --path crates/gw-cli
```

---

## Quickstart

**1. Provide your provider key via the environment.** It is read only from `OPENROUTER_API_KEY` and is
never a config field or a CLI flag, so it can't leak into shell history, a process listing, or a
`--help` dump.

```sh
export OPENROUTER_API_KEY=sk-or-...
```

**2. Write a prompts file** — one user turn per line (lines beginning with `#` are skipped):

```text
# prompts.txt
A train leaves Boston at 60 mph and another leaves NYC at 40 mph...
Prove that the square root of 2 is irrational.
Design a rate limiter for an API gateway and justify the algorithm.
```

**3. Write a review-only config** (`gw.toml`) — see the [reference](#configuration) below:

```toml
db         = "gw-run.sqlite"
budget_usd = 5.0

[area]
training_area = "reasoning"
admission_intent = "review_only"
teacher_slug  = "z-ai/glm-5.2"
cot_required  = true
k             = 3
rubric        = "Grade the reasoning for correctness, rigor, and explicit assumptions."

[area.thresholds]
accept_threshold = 0.80
reject_below     = 0.60
min_n_eff        = 1.5

[[area.judges]]
slug   = "deepseek/deepseek-v4-pro"
family = "deepseek"

```

**4. Run it** — generate, grade, and persist candidates for review; the report prints when it finishes:

```sh
gw gen run --config gw.toml --run-id demo-001 --prompts prompts.txt \
  --shards 8 --max-in-flight 8 --budget-usd 5.0
```

Prefer a live dashboard? Swap `run` for `tui`. Crashed or interrupted? Re-run the **same** `--run-id`
with the **same** `--prompts` and `--shards` to resume — already-committed work is skipped and the
teacher is never re-spent.

The quickstart keeps the conservative correlation prior and effective-count floor. Its single judge
cannot clear that floor, so collection explicitly uses `review_only`. Otherwise admitted candidates
stay at `NeedsReview`; no admitted dataset is produced. The `k = 3` setting generates three candidate
answers per prompt and does not add judges.

**5. Export an automatic-admission run** after configuring an attainable panel as described below.
Export is a provider-free step you can repeat:

```sh
gw gen export --db gw-run.sqlite --out out/dataset.parquet --run-id automatic-001 \
  --format chat-ml --cot supervised
```

You get one `out/dataset.parquet` file. Its footer records the target template, CoT policy, optional
`--dataset-version`, record counts, selection scope and artifact identity. The command prints the
same manifest as JSON. It records a local publication receipt without changing generation run status
or record lifecycle. Historical sidecars are ignored and left untouched.

### Artifact publication and recovery

Export freezes the selected IDs, all eight projected columns and the complete manifest before writing.
It records a prepared receipt in SQLite, writes a unique staging file beside the destination, closes
the writer, then opens a fresh reader to verify the footer, schema, every batch, counts, IDs, hashes
and decoded messages. One rename replaces the destination; SQLite acknowledgment follows. A failure
before rename preserves the old destination and removes only the owned staging file.

After rename, a database error leaves the verified artifact on disk. Retry checks the persisted
receipt and the exact selected records. Later unrelated admissions are not added to a prepared
publication. If the destination contains different content, retry republishes the same frozen plan
through staging before acknowledgment. A file claiming the expected identity but failing verification
is an integrity error. A database error does not prove rollback: the receipt may already be
acknowledged if the commit succeeded but its response was lost.

Errors after preparation report a publication ID. Recover that exact publication without provider
credentials or projection overrides:

```sh
gw gen export --db gw-run.sqlite --resume-publication PUBLICATION_ID
```

This uses the receipt's recorded destination and may replace that file with the original frozen
artifact. It accepts prepared or acknowledged receipts and leaves later admissions untouched.
The original acknowledgment mode applies: an engine receipt may finish marking its selected records
`Exported`; a standalone receipt leaves lifecycle unchanged. Recovery never changes generation run
status or claims that generation completed. `--out`, `--run-id`, `--format`, `--cot` and
`--dataset-version` conflict with explicit recovery. Ordinary export still selects a fresh population
when no prepared receipt requires recovery.

This protocol does not claim a filesystem/SQLite transaction, a cross-process
publication lease, or power-loss durability; callers must serialize publication to a destination.

The engine keeps selected records at `Formatted` until its configured artifact is published and
acknowledged. Without configured output, a successful run reports readiness and `exported = 0`.
Configured output is published even when no records were admitted, producing a valid empty artifact
that replaces stale output. Publication failure fails the run; success events and `Completed` follow
acknowledgment. Standalone `gen export` leaves generation states unchanged.

The authoritative footer key is `ghostwriter.export_artifact`. Metadata version 1 wraps the existing
manifest, scope and `artifact_id`; the eight-column `canonical_messages` schema remains unchanged.
`gw_storage::verify_artifact` reads every batch and returns an explicit `MissingLegacyMetadata` result
for historical files without this entry. Ordinary Parquet row readers can still read those files.
No adjacent file is used to infer metadata.

`build_inputs_hash` retains its original meaning: BLAKE3 of sorted admitted `record_hash` values,
each followed by a newline. The separate artifact identity covers the complete projection, manifest
and scope, including empty populations. It excludes the destination and its own ID field. Metadata
does not invent missing model identities or claim a complete generation provenance graph.

For independent version-1 identity implementations:

1. Sort rows by `record_id` in UTF-8 byte order. Reject duplicate IDs. A framed string is its UTF-8
   byte length as an unsigned 64-bit big-endian integer followed by those bytes.
2. Hash each row with BLAKE3 derive-key context `ghostwriter.export.projected-row.v1`: framed
   `record_id`, `training_area`, `record_hash`, `prompt_hash`; verdict presence byte (`0` absent,
   `1` present) and framed verdict when present; aggregate presence byte and its exact IEEE-754
   64-bit big-endian bits when present; unsigned 32-bit big-endian `reasoning_tokens`; framed
   `messages_json`. Encode the resulting digest as lowercase hexadecimal.
3. Hash the artifact with context `ghostwriter.export.artifact.v1`: unsigned 32-bit big-endian
   `metadata_version`; framed scope JSON; framed complete manifest JSON; unsigned 64-bit big-endian
   row count; each framed hexadecimal row digest in sorted order. Scope and manifest JSON use the
   version-1 typed fields, recursively sorted object keys, compact separators and UTF-8 strings.
   Optional absent manifest fields are omitted and defaulted fields use their serialized values.

---

## Configuration

Configuration is layered, lowest precedence first: **built-in defaults → TOML file (`--config`) →
`GW_`-prefixed environment variables → CLI flags**. Defaults deserialize without a file, but generation
requires a nonempty judge panel and a valid admission configuration before any provider is constructed.

```toml
# ─── run-wide ───────────────────────────────────────────────────────────────
db          = "gw-run.sqlite"   # SQLite store: run state, queue, provenance
budget_usd  = 5.0               # hard spend cap — the primary guard (default 5.0)
on_breach   = "drain"           # at the cap: "drain" (let in-flight finish) | "abort"
provider_base_url = "https://openrouter.ai/api/v1"   # OpenAI-compatible endpoint
provider_rpm      = 60          # per-lane requests/min for the rate limiter

# ─── the training area: what to generate and how to grade it ────────────────
[area]
training_area = "reasoning"     # stamped into provenance + the record-id prefix
admission_intent = "review_only" # explicit collection mode; default is "automatic"
teacher_slug  = "z-ai/glm-5.2"  # any OpenRouter chat model that emits reasoning
cot_required  = true            # require chain-of-thought (a hard Verify gate)
k             = 3               # best-of-k fan-out (default 1; clamped to >= 1)
correlation_rho = 0.7           # cold-start inter-judge correlation prior

# Optional teacher caps — keep a long reasoner from running away mid-trace:
teacher_max_tokens           = 20000   # combined completion cap (visible + reasoning)
teacher_reasoning_max_tokens = 12000   # reasoning-token sub-cap

# Optional area-wide judge headroom for dense traces (per-judge overrides below):
judge_max_tokens           = 8000
judge_reasoning_max_tokens = 6000      # mutually exclusive with judge_reasoning_effort

rubric = """
Grade the reasoning trace for correctness, rigor, and clarity.
Reward sound derivations and explicit assumptions; penalize unjustified leaps.
"""

# ─── admission thresholds + the correlation-guard floors ────────────────────
[area.thresholds]
accept_threshold = 0.80   # aggregate >= this (with trusted consensus) -> Admit
reject_below     = 0.60   # aggregate <  this -> Reject
min_n_eff_ratio  = 0.50   # floor on effective/nominal judge count
min_n_eff        = 1.5    # absolute effective-judge floor, else -> NeedsReview

# ─── the judge panel (one or more; same-family judges are de-weighted) ──────
[[area.judges]]
slug   = "deepseek/deepseek-v4-pro"
family = "deepseek"        # coarse family tag for same-family exclusion
# optional per-judge overrides: rubric_id, max_tokens, reasoning_max_tokens, reasoning_effort

# ─── optional end-of-run export ─────────────────────────────────────────────
[export]
out             = "out/dataset.parquet"
format          = "chat-ml"      # see Export targets below
cot             = "supervised"   # supervised | masked | stripped
dataset_version = "0.1.0"
```

Notes:

- **The API key is never in this file.** Only `OPENROUTER_API_KEY` (from the environment) authenticates.
- `judge_reasoning_max_tokens` and `judge_reasoning_effort` are **mutually exclusive** in the same
  table — set one or the other, not both. The same holds for per-judge `reasoning_max_tokens` /
  `reasoning_effort`.
- Any field can be overridden by an env var (`GW_BUDGET_USD=10.0`) or, for the common knobs, a CLI
  flag (`--budget-usd`, `--k`, `--shards`, `--on-breach`, `--admission-intent`). The intent environment
  setting is `GW_AREA__ADMISSION_INTENT=review_only`; its CLI spelling is `--admission-intent review-only`.

### Admission preflight

Automatic admission is the default intent. Before credentials or provider calls, the resolved panel
must be nonempty, score bands must satisfy finite `0 <= reject_below <= accept_threshold <= 1`, and
the effective-count floors must be finite and in range (`min_n_eff >= 0`, ratio in `[0,1]`). The
correlation prior is finite in `[0,1]` and must be positive and nonidentity for multiple judges.

Under equal cold-start weights, `d` decisive votes have
`n_eff = d / (1 + (d - 1) * correlation_rho)`. Preflight considers every `d` from one through the
number of judges, because uncertain votes are excluded. Some `d` must meet both the absolute and
relative floors. The default prior `0.7` and absolute floor `1.5` cannot do so, even with a larger
panel. `review_only` bypasses this attainability check while keeping all numeric safeguards. Its
intent is stored with each candidate; otherwise admitted candidates remain `NeedsReview` on replay
and rederivation. Verifier hard failures still reject.

**An attainable example under an assumed prior:** the following two-judge panel assumes `rho = 0.2`.
Two decisive judges then give `n_eff = 1.667` and `n_eff/d = 0.833`, clearing both stated floors.
This is a declared assumption, not measured calibration or evidence of judge independence. Use a
new run ID, such as `automatic-001`, when collecting under this configuration.

```toml
[area]
admission_intent = "automatic"
correlation_rho = 0.2
k = 3                         # candidate answers per prompt; there are two judges below

[area.thresholds]
accept_threshold = 0.80
reject_below = 0.60
min_n_eff = 1.5
min_n_eff_ratio = 0.7

[[area.judges]]
slug = "deepseek/deepseek-v4-pro"
family = "deepseek"

[[area.judges]]
slug = "z-ai/glm-5.2"
family = "glm"
```

---

## CLI reference

```
gw gen run      Headless run: generate → grade → admit → persist; prints the report.
gw gen tui      The same run with the live ratatui dashboard.
gw gen export   Export admitted records from a store to a Parquet shard (no providers).
gw gen replay   Resume a started run from its persisted checkpoints.
gw eval audit-separation   Selector-vs-random separation diagnostic over a store (JSON).
gw eval promote            Variance-aware promotion gate over two eval_results.json (JSON).
```

**`gw gen run` / `gw gen tui`**

| Flag | Meaning |
|---|---|
| `--config <FILE>` | TOML config (the figment base layer). |
| `--run-id <ID>` | Idempotency key. Re-running the same id over the same seeds **resumes**. |
| `--prompts <FILE>` | Newline-delimited prompts file (one user turn per line). |
| `--db <PATH>` | Override the SQLite store path. |
| `--shards <N>` | Partition the seed space into N shards (default `1`) — the primary concurrency axis. |
| `--max-in-flight <N>` | Cap on seed items in flight across all shards (default `4`). |
| `--k <K>` | Override the best-of-k fan-out. |
| `--budget-usd <USD>` | Override the run-wide budget cap. |
| `--on-breach <MODE>` | `drain` or `abort` at the cap. |

**`gw gen export`** — `--db`, `--out`, `--run-id` (optional), `--format`, `--cot`.
Standalone and automatic end-of-run exports use the same SFT eligibility rule: records need an
`admit` judging verdict and an `admitted`, `formatted`, or `exported` lifecycle state. Retained
best-of-k losers keep their individual grades in the store but do not enter the dataset. Unfinished,
rejected, review, and error states are excluded. The manifest counts all scanned records in
`n_records`; `n_admitted` and `build_inputs_hash` describe only the selected exported rows.

**`gw gen replay`** — resumes `--run-id` from a store; you must pass the **same** `--config`,
`--prompts`, and `--shards` the original run used (the seed→shard partition is `index % shards`, so a
different value would re-partition the space and duplicate or orphan records).

> **Concurrency, briefly.** The shard is the real unit of parallelism (one task per shard), and within
> a prompt the **k** best-of-k candidates fan out concurrently. `--shards` × `k` is your throughput
> dial; `--max-in-flight` bounds total concurrent items so you stay within provider limits.

A fatal shard error or panic stops new work and joins every shard before the run reports failure.
Surviving shards settle started transitions at a persisted boundary; grading may finish its panel and
internal retries. The first observed shard failure remains the returned error even if saving the
failed run status also fails. A provider that never returns can delay shutdown: the engine adds no
provider deadline. Dropping the run future aborts its shard tasks without guaranteeing that their
in-flight work is persisted.

---

## Export targets & CoT policy

`--format` records the target chat template in the export manifest. The Parquet shard itself is a
**columnar dump of the canonical conversation, not a pre-rendered template** — the target bytes are
produced downstream from the manifest plus the conversation column:

| `--format` | Target |
|---|---|
| `gemma4` | Gemma-4. Token bytes are transcribed from the research spec and golden-file tested; they are **pending a byte-for-byte diff-verify against the official pinned `chat_template.jinja`** before production SFT (nothing fetches the template at runtime). The system turn is folded into the following `user` turn and the upstream `<\|think\|>` marker is not emitted. |
| `chat-ml` | ChatML. |
| `sharegpt` | ShareGPT. |
| `openai-messages` | OpenAI `messages` conversational. |
| `harmony` | gpt-oss Harmony channels. |
| `trl-prompt-completion` | TRL prompt/completion. |

**Tool trajectories.** `openai-messages` and `trl-prompt-completion` carry `tool_calls` and each
tool result's `tool_call_id` verbatim. The other four targets (`gemma4`, `chat-ml`, `sharegpt`,
`harmony`) have nowhere to put them, so rendering a tool conversation for one **fails closed** with
a structured error naming the route, the dropped signal and the first offending message — it never
emits a training target in which a tool turn has been flattened into prose. The supported way to get
a tool-faithful target is to export the canonical `messages_json` conversation and apply the model's
official chat template in a consumer that owns that template. Text-only conversations are unaffected
on every target.

`--cot` controls how the captured reasoning is projected:

- **`supervised`** — reasoning is rendered into the loss region (train on the chain-of-thought).
- **`masked`** — reasoning is rendered but masked out of the loss.
- **`stripped`** — reasoning is dropped (answer-only).

The export is a columnar Parquet dump plus a manifest. One `messages_json` column holds the canonical
`Message[]` JSON in conversation order, losslessly: the `content` variant (including `null`),
`reasoning`, `reasoning_details`, `tool_calls` and the `tool_call_id` links all survive, so a
consumer decodes it straight back into `Message[]`. (The historical v1 `{role, content}` +
parallel `reasoning_json` pair was lossy — `reasoning_json` is gone, and `column_schema_version` in
the manifest records which contract a shard was written under.) The CoT policy and the target are
recorded as manifest metadata so a downstream trainer applies the matching loss mask and template.

---

## Evaluation

`gw-eval` is **off-path and model-free** — it reads persisted records and eval artifacts, never calls
a provider:

- **`gw eval audit-separation`** scores how well the selector separates admitted from random traces
  over a store (with a data-ceiling guard), printing a JSON report.
- **`gw eval promote`** is a variance-aware gate over two `eval_results.json` snapshots
  (baseline vs candidate) plus a drift exit code, printing a binary promotion decision.

Both accept `--check` to opt into decision-bearing process exits (`0` pass · `1` operational error ·
`2` gate rejects) for CI.

Promotion requires every benchmark supplied on either side, plus the configured headline metric,
to be present on both sides. Aggregate-only comparisons remain supported. Missing scores, reserved
aggregate names in benchmark maps, non-finite scores, and overflowing comparison arithmetic reject
with `evidence_valid: false` and structured `evidence_issues`. Retuning the report cannot override
that rejection. Valid threshold fields remain JSON numbers; invalid thresholds are `null` (the Rust
report fields `ab_min_delta` and `ab_sigma_k` are `Option<f64>`). Without `--check`, a gate rejection
still exits `0` and prints its report. Malformed JSON exits `1` in either mode.

`aggregate` and `eval_results.aggregate` are aliases for one headline, reported under the dotted
name. Either alias may name its sigma prior; if both priors are supplied, their normalized values
must agree. Negative or non-finite sigma priors still normalize to zero. Missing priors still use
zero, and `ab_avg_n` currently does not replace priors with measured variance.

These artifacts carry metric names and scores, without task-version, dataset, checkpoint, template,
or unit identity. The gate checks supplied evidence completeness; callers must establish those
compatibility conditions. It does not enforce an external required benchmark suite or a `[0, 1]`
score range.

---

## Security

- `OPENROUTER_API_KEY` is read from the environment by the provider constructor only — **never** from
  the config file, a CLI flag, serialization, or logs. The `GW_` env prefix can't slurp it.
- `unsafe` code is **forbidden** workspace-wide.
- See [SECURITY.md](SECURITY.md) for reporting.

---

## Contributing

Issues and PRs are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). The workspace builds with
`cargo build`, lints with `cargo clippy --workspace --all-targets`, and tests with
`cargo nextest run --workspace` (or `cargo test`).

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.
Unless you explicitly state otherwise, any contribution you submit shall be dual-licensed as above,
without additional terms or conditions.
