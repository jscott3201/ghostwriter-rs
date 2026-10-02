# ghostwriter-rs

**A Rust harness for generating graded chain-of-thought reasoning traces as model fine-tuning data.**

![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)
![rust](https://img.shields.io/badge/rust-1.95%2B-orange)
![edition](https://img.shields.io/badge/edition-2024-555)
![unsafe](https://img.shields.io/badge/unsafe-forbidden-success)

ghostwriter-rs drives a **teacher** model to produce full chain-of-thought reasoning, runs each
trace through a deterministic **verifier** and a model **judge panel**, admits only the traces that
earn it, and exports the winners as a model-agnostic supervised-fine-tuning (SFT) dataset. The whole
run persists lifecycle transitions, reuses completed teacher outputs on resume, and records the
model requests and grading evidence it observes.

Its goal is reproducible selection of reasoning traces for open-model training, with retained
verification, grading, request, and publication evidence. The current Rust core handles generation,
grading, persistence, artifact publication, and model-free diagnostics. External trainer adapters,
official tokenizer and loss-mask checks, immutable deployment identities, measured throughput,
and held-out student evaluation are further work; the pipeline alone does not establish student improvement.

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
- an explicit **accounting policy** either observes normal concurrent work or admits physical sends
  below a complete, known-spend dollar threshold.

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
| `gw-format` | Chat-template rendering, SFT projection and structural preference message projection. |
| `gw-generate` | Gates provided user turns and generates teacher assistant responses. |
| `gw-judge` | The deterministic verifier and the model judge-panel grading + admission. |
| `gw-engine` | The headless orchestrator: the lifecycle state machine and sharded executor. |
| `gw-cli` | The `gw` command-line entrypoint. |
| `gw-tui` | A live [ratatui](https://ratatui.rs) dashboard over the engine event stream. |
| `gw-eval` | Off-path, model-free dataset diagnostics, lexical screening plans and promotion gating. |

The dependency spine: `schema → {format, providers, storage} → {generate, judge} → engine → {cli, tui}`,
with `gw-eval` consuming `schema`, `storage` and the pure `format` projection layer.

---

## How it works

Each prompt becomes a **record** that walks a 12-state lifecycle. Persisted transitions and completed
grades support resume and admission audits. An unresolved request may have reached its provider, so
recovery cannot guarantee remote exactly-once execution.

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
   - **Rejected** — retained for audit and potential preference-pair construction; no DPO qualification is implied;
   - **Revising** — a single bounded re-generation, then re-judged;
   - **NeedsReview** — parked for human/verifier adjudication.
7. **Formatted → Exported** — `Formatted` records are ready for publication. `Exported` follows
   verified artifact publication and database acknowledgment. Parquet preserves canonical messages;
   the target template and CoT policy describe downstream training intent.

`Error` is terminal-until-requeue: a faulted record carries its last error and attempt count, and a
re-run picks it back up.

### Reviewed reference imports

`gw reference register`, `import`, and `export` provide an explicit local registration and
atomic native-verified path for a complete 112-member reference population. All 48 held-out
members remain in private storage. See [reviewed reference imports](docs/reviewed-references.md)
for the capture, review, execution and publication contracts.

### Local coding evaluation

`gw eval coding` evaluates a reviewed saved Python function using a cached, pinned
local Docker recipe and the native deterministic verifier. `gw eval coding-replay`
validates a self-contained saved declaration and executes its captured code again.
The controller keeps expected results outside candidate containers; training exports
retain redacted suite bindings. See the [owned coding controls](examples/coding/README.md)
for exact task/value contracts, supported runtime, limits, and qualification commands.

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
endpoint changes, or resolved model revisions. Cached grade audit data retains its originating
run and physical attempt when known; historical origins stay unknown. Consuming a cached grade
creates no new request receipt and imports no spend from the originating run.

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

**1. Provide your Model API key via the environment.** The default variable is `MODEL_API_KEY`.
The key value is never a config field or a CLI flag. A literal `export` command can still enter shell
history; use your shell or secret manager's protected input mechanism when entering a real key.

```sh
export MODEL_API_KEY=sk-or-...
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
db = "gw-run.sqlite"
accounting_policy = { mode = "finite_usd", limit_usd = 5.0 }

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
  --shards 8 --max-in-flight 8
```

Prefer a live dashboard? Swap `run` for `tui`. Crashed or interrupted? Re-run the **same** `--run-id`
with the **same effective generation/admission settings**, `--prompts`, and `--shards` to resume.
The immutable manifest is checked before credentials are read; committed work is skipped and
persisted teacher output is reused. Finite admission remains conservative about unresolved receipts.

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

### Verified SFT token labels

The [external TRL adapter](adapters/trl/README.md) prepares explicit loss labels for pinned
Qwen3-0.6B and Gemma 4 E2B text profiles in separately locked environments. It verifies one immutable Parquet byte snapshot through Rust, checks every
rendering against the official template, and supports supervised, masked, and stripped reasoning
with explicit assistant-prefix expansion. Its local CPU qualification inspects the real collator
and SFTTrainer dataloader without loading pretrained weights or running a training step.

The provider-free snapshot verifier is also available directly:

```sh
gw artifact verify --stdin < out/dataset.parquet
```

Its versioned JSON report contains the verified artifact, raw byte length, and raw BLAKE3 digest.
Missing legacy metadata and integrity failures exit with status 1 and emit no success report.
See the adapter guide for the pinned environment, rejection limits, and unresolved split/lineage
qualification.

### Fresh numeric rewards

The same [external TRL adapter](adapters/trl/README.md#fresh-numeric-rewards) can evaluate fresh
numeric completions against a frozen corpus of reviewed tasks. Export the corpus directly from
task documents, then verify the captured bytes:

```sh
gw reward export --tasks examples/reviewed-numeric-tasks.json > numeric-corpus.json
gw reward verify --stdin < numeric-corpus.json
```

Export validates every document and cross-document task/group declaration before selecting unique
declared Train tasks in source order. The self-contained artifact retains the prompt, literal
oracle, numeric semantics, source, rights, group, split, and review observations. The reward callback
passes only user messages and separate task references to TRL; teacher records are not inputs.

`gw reward evaluate --stdin` uses the shared pure numeric evaluator for a complete bound batch.
Factual Pass maps to `1.0`, decisive Fail to `0.0`, and Unknown to a null reward. The first Python
callback aborts the entire batch on Unknown, timeout, stale bindings, or incomplete output before
TRL receives numeric rewards. Its pinned CPU qualification exercises the real GRPO data loader
and reward dispatcher with synthetic completions, with model computation disabled. Real corpus
quality, split qualification, eligible student lineage, and a GRPO learning comparison remain open.

### Pure preference preparation

`gw_engine::prepare_preference_pair` validates two records against a versioned judge-ranking
assessment captured with `gw_storage::capture_preference_source`. It requires distinct record IDs,
matching run and task identities, equal complete original prefixes, a selected admitted chosen
record, supported verification authority, decisive grades, and a score gap strictly above the
configured margin. Supplied source snapshots must match the actual records. Preference evidence
encodes finite floating-point values as exact binary64 bit objects so JSON save-and-load preserves
score identity; ordinary training-record JSON stays unchanged. Missing content hashes
are recomputed; populated stale hashes are rejected. The score margin is a declared rule whose
quality still needs empirical validation.

The result contains explicit `prompt`, `chosen`, and `rejected` message arrays, source evidence,
and separate content and decision identities. Reasoning remains separate from answer content.
Supervised and stripped reasoning are supported; stripped removes reasoning from all three arrays
while retaining the original messages in the evidence. Ordered plaintext reasoning details are
supported only when their concatenation exactly equals the flat reasoning; training arrays emit
that reasoning once. Tools, multimodal content, summary/encrypted or mismatched reasoning details,
masked reasoning, and identical projected completions are rejected.

This API performs no provider or store I/O. It validates terminal assistant message shape; source
records do not capture stop/length metadata, so termination remains unknown. Receipt-to-output and
decision-execution provenance remain explicitly unbound. A protocol revision is a declaration,
not proof of which execution produced a grade. There is no DPO publication command or adapter
qualification yet. SFT artifacts exclude retained rejected siblings and cannot serve as pair proof;
real tokenizer loading and policy/reference likelihood masks still need separate qualification.

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
manifest, scope and `artifact_id`. New artifacts use the eleven-column v5 `tool_definitions`
schema: the ten v4 `record_origins` columns plus nullable `tools_json`. Frozen v2
`canonical_messages`, v3 `reviewed_tasks`, and v4 `record_origins` publications preserve their
original columns and identities during recovery.
`gw_storage::verify_artifact` reads every batch and returns an explicit `MissingLegacyMetadata` result
for historical files without this entry. Ordinary Parquet row readers can still read those files.
No adjacent file is used to infer metadata.

`build_inputs_hash` retains its original meaning: BLAKE3 of sorted admitted `record_hash` values,
each followed by a newline. The separate artifact identity covers the complete projection, manifest
and scope, including empty populations. It excludes the destination and its own ID field. Metadata
does not invent missing model identities or claim a complete generation provenance graph.

For independent implementations of metadata identity version 1:

1. Sort rows by `record_id` in UTF-8 byte order. Reject duplicate IDs. A framed string is its UTF-8
   byte length as an unsigned 64-bit big-endian integer followed by those bytes.
2. For column schema v2, hash each row with context `ghostwriter.export.projected-row.v1`: framed
   `record_id`, `training_area`, `record_hash`, `prompt_hash`; verdict presence byte (`0` absent,
   `1` present) and framed verdict when present; aggregate presence byte and its exact IEEE-754
   64-bit big-endian bits when present; unsigned 32-bit big-endian `reasoning_tokens`; framed
   `messages_json`. For column schema v3, use context
   `ghostwriter.export.projected-row.v2-reviewed-tasks`, encode those same fields, then append a
   task presence byte and framed canonical `task_json` when present. For v4, use context
   `ghostwriter.export.projected-row.v3-record-origins`, retain the v3 framing and append framed
   canonical `origin_json` (always present). For v5, use context
   `ghostwriter.export.projected-row.v4-tool-definitions`, retain the v4 framing and append a
   tools presence byte (`0` for SQL null, `1` for a present list), followed by framed canonical
   `tools_json` when present. An empty list is present: `1` followed by a frame containing `[]`.
   Encode each final row digest as lowercase hexadecimal. Task and origin JSON use strict typed
   fields and recursively sorted object keys. Tools JSON preserves definition order and nested
   JSON values, with recursively sorted object keys, compact separators and UTF-8 strings.
   Bind the actual exported tools payload even when the stored `record_hash` is unchanged.
3. Hash the artifact with context `ghostwriter.export.artifact.v1`: unsigned 32-bit big-endian
   `metadata_version`; framed scope JSON; framed complete manifest JSON; unsigned 64-bit big-endian
   row count; each framed hexadecimal row digest in sorted order. Scope and manifest JSON use the
   version-1 typed fields, recursively sorted object keys, compact separators and UTF-8 strings.
   Optional absent manifest fields are omitted and defaulted fields use their serialized values.

---

## Physical request receipts

`Engine::run` records physical chat and embedding attempts in SQLite. Each built-in HTTP client
awaits a durable intent immediately before its POST, after any rate-limit wait. An intent failure
prevents that request. Intentional chat retries each receive a new receipt; automatic redirects
and reqwest protocol retries are disabled for chat and embeddings. The HTTP/2 regression fixture
enables HTTP/2 only in test builds and exercises a real `REFUSED_STREAM` response through the
same client-builder retry policy.

A receipt binds the run and a fresh launch ID to an optional shard, intended record, role, purpose,
requested model, sanitized endpoint, exact request-body digest and retry ordinal. Intended records
need not exist yet. Context travels outside the serialized model request, so it does not alter
teacher sampling or judge-cache fingerprints. The captured response identifiers describe what the
backend reported; absent model revisions remain absent.

| Role | Purposes |
|---|---|
| Teacher | Initial generation, truncation retry, revision |
| Judge | One panel member's grade |
| Embedding | Candidate QC, newly admitted prior, prior rebuilt on resume |

`Store::model_attempts(run_id)` returns receipts. `Store::model_launches(run_id)` returns the actual
injected clients' capabilities assessed at each launch. Custom implementations default to
`Unknown`; `NullEmbedder` explicitly performs no model requests. A declaration is a cooperative
extension contract. Wrapping a logical call or accepting a no-op callback does not establish its
hidden transmission behavior. Engine helpers such as `step`, `drive`, `drive_to_judged`, `run_group`,
and `revise_once` reject model dispatch without registered launch context. Pure computation,
cache hits, and implementations declaring `NoModelRequests` remain usable. Raw provider APIs are
outside the engine's run accounting contract.

Usage fields are optional evidence. A reported zero differs from omitted cost, and explicit null,
negative, non-finite or malformed cost is invalid. Counts accept unsigned integers and integer
strings; explicit null, fractional or overflowing counts are marked invalid. A null usage container
means no measurement, and null optional response/model/provider identifiers add no new identity
evidence. Metadata extraction is independent of the teacher, judge and vector parsers, including
malformed output. An observed SSE response continues
to drain metadata after its first content-decoding error and returns that original error after
transport settles. Changed observations receive ordered identities within each physical attempt.
Exact persistence retries reuse that identity and return its stored result. A fresh return to older
cumulative values or reuse of an identity with different data remains a durable conflict. Cumulative
snapshots are never added together. Token deltas with unchanged metadata do not write the database.

Transport settlement and output interpretation are separate: a completed HTTP response can
contain an invalid grade or truncated reasoning. Elapsed milliseconds measure wall time from the
persisted intent to settlement; they do not measure GPU compute. Accounting failures are fatal
and cannot become an automatic resend or a skipped prior-building warning. A committed admission
remains recorded if its subsequent prior embedding fails.

### Accounting policy and replay

The shared policy is a tagged value:

- `observation_only` records available tokens, client wall time, and costs without requiring prices.
  It retains the engine's existing bounded concurrency, including concurrent unknown-priced calls.
- `finite_usd` requires a finite, nonnegative `limit_usd`. Every physical chat or embedding POST,
  including retries, checks its captured policy epoch, historical coverage, unsettled intents,
  invalid/conflicting evidence, and known spend in the transaction that writes its intent. At most
  one request is unresolved per run under this policy. It waits only for requests held by the live
  coordinator; an orphan or foreign unresolved receipt halts new dispatch.

A finite limit is a **dispatch threshold**, not an invoice ceiling. A last response may cross it;
local grading finalization, caching, persistence and publication can still complete. If another
request is needed, that send is denied and the item keeps its unfinished checkpoint. Zero is valid
and permits cache-only or local completion, while denying fresh physical requests.

Fresh engine runs atomically persist the immutable semantic manifest, operational policy and actual
client coverage before any startup embedding or model request. Older runs remain explicitly incomplete;
an empty legacy receipt table does not prove zero historical spend. A finite launch requires
complete history and known coverage of every lane. A later replacement with an unknown client is
reassessed and cannot inherit the previous client's certification.

The policy is operational authority with a durable epoch, separate from generation request and
cache identity. Reusing an unchanged policy preserves its epoch. Changing to a finite policy requires
complete, settled, valid price evidence; observation-only can supersede a finite policy despite an
unresolved receipt. Old authorized requests may settle, but the superseded coordinator cannot send
again. Replay uses the complete run receipt ledger, including failed or malformed outputs and
embeddings. It never reconstructs spend by summing record-level teacher projections.

CLI terminal reports refresh SQLite after run, replay, or TUI settlement, including best-effort
failure reporting without replacing the primary error. They show requested, configured and effective
policies, epochs, known dollars, unknown/invalid/conflicting/unresolved counts, coverage, optional token
totals and client wall-clock milliseconds. Token categories can overlap and are not added together.
The dashboard applies absolute snapshots by revision, so corrections may reduce a displayed subtotal;
dropped events never become the authoritative final accounting. Observation-only shows no monetary gauge.

Dropping a request performs no background persistence or drain. Process loss after intent leaves
an unresolved receipt: it may or may not have transmitted. Resuming does not establish remote
exactly-once execution. The current SQLite durability settings and receipts do not constitute
power-loss qualification or provider-invoice reconciliation.


### Persistence and process recovery

A successful storage write acknowledges a committed SQLite transaction. File stores keep WAL and
`synchronous=NORMAL`: acknowledged writes survive termination of the application process. OS crashes
and power loss can still discard acknowledged transactions. A canceled call or missing acknowledgment
can have committed already; it is not evidence of rollback.

Record updates commit their complete envelope, indexed projections, mutation receipt and lifecycle
history in one transaction. The expected snapshot includes verification, grading, cost, provenance and
history. A stale new command fails with a conflict. Retrying the same committed command recognizes its
receipt first and returns the current record, even after later publication, without another history
entry or state event. Initial insertion preserves both generation facts at attempt zero. Existing
historical rows are not rewritten by migration. The explicitly named `replace_record_for_import` API
replaces fixture/import data and its command history; engine writes use guarded insertion or transition.

The immutable manifest is compared before the launch transaction changes run status, accounting
policy or client coverage. Each attempt intent, metadata observation, transport settlement and output
interpretation commits independently. Cache entries, run status and shard cursors have separate write
acknowledgments. A settled accounting receipt does not prove that reusable teacher output or a judge
cache entry exists. There is no transaction spanning a provider call and these local writes.

After a process stops between saving a record and advancing its cursor, replay reuses persisted output
and committed caches. Publication keeps its prepared receipt and batch acknowledgment of the exact
selected records. Malformed envelopes, projections, cursors, cache values or accounting data produce
errors; they do not silently reset a run. Startup finishes and validates migrations before returning a
store, with SQLite's write lock covering migration discovery and application.

The file-backed recovery tests terminate child processes at explicit pre-commit, committed-before-ack
and acknowledged boundaries, then reopen and check integrity, foreign keys and migration checksums.
They also compare replayed record and export-input hashes with a clean fake-provider run. These are
application-process crash tests; they do not simulate physical power loss.

### Task verification policy

Each executable `VerificationContract` declares `answer_policy` and `execution_policy` independently:

| Policy | Pass | Fail | Unknown or missing evidence |
|---|---|---|---|
| `absent` | Axis is inactive | Axis is inactive | Axis is inactive |
| `advisory` | Quality panel proceeds | Failure is recorded; panel proceeds | Unknown is recorded; panel proceeds |
| `authoritative` | Quality panel still required | Reject before judge calls | NeedsReview before judge calls |

An authoritative failure takes precedence over an unknown result on another axis. Required plaintext
reasoning remains an authoritative gate. Plain prompt files declare both task axes `absent`.

Active answer policies require a supported comparator/oracle combination. Named-test authoritative
execution requires nonempty task-owned `required_tests`, with unique nonblank IDs matched verbatim.
The full captured plan is validated before teacher dispatch, including retry entry points. A valid
but unavailable sandbox or evidence source produces Unknown. Execution reports must match the exact
run, record, and completion hash; their own coverage list cannot reduce the task's requirements.

Verification persists typed Pass/Fail/Unknown observations, bounded reasons, applied policy, and an
interpretation version. Advisory failures are never stored as passing checks. `all_passed` now means
that no authoritative hard failure exists; it does not mean every factual observation passed.
Verified replay and reconciliation use supported persisted facts without rerunning an oracle.
Records missing explicit policy or a supported interpretation remain available for inspection and
standalone export, but executable replay/regrading rejects them before new model work. They are not
automatically rewritten or inferred from historical booleans.

### Reviewed numeric task files

Run, replay, and TUI accept exactly one of `--tasks FILE` or `--prompts FILE`. The
[reviewed arithmetic example](examples/reviewed-numeric-tasks.json) is a complete version 1 task
document. `NumericTaskSource::from_json` and `from_document` provide the same pure validation to
library callers. Invalid documents fail before credentials, store creation, or model dispatch.

Each ordered task declares a stable label; source namespace, item, immutable revision, and citation;
reviewed rights basis, evidence, reviewer, and permitted uses; a namespaced corpus group; split
manifest, revision, and role; one nonempty user-text prompt; literal numeric answer, extraction,
tolerances, and policies; and domain, difficulty, and QC observations. Unknown fields/versions,
other conversation structures, executable oracles, duplicate tasks, and conflicting assignments
for a group are rejected. Rights, difficulty, and QC fields record reviewed assertions. The harness
does not fetch their references or establish legal clearance, measured difficulty, or cross-corpus
split/decontamination qualification.

Numeric contracts persist these settings explicitly:

- `whole_content` requires the entire assistant content, except surrounding whitespace, to be a
  numeric token.
- `final_marker` requires exactly one literal marker, at the start of the final nonempty line
  after trimming that line. Only a numeric token may follow it. Missing, repeated, embedded, or
  ambiguous markers produce Unknown. Reasoning and arbitrary prose are never searched for numbers.
- Tokens follow `[+-]?(digits(.digits*)?|.digits+)([eE][+-]?digits+)?`, using ASCII digits.
  Currency, percent, separators, hexadecimal, NaN/infinity, overflow, and nonzero values that
  underflow to zero are rejected. Literal expected answers are strings so these failures remain
  detectable during intake.
- Values and arithmetic use IEEE-754 binary64, including its rounding of large integers; this
  does not provide arbitrary-precision decimal/integer equality. Tolerances are finite,
  nonnegative JSON numbers. A match uses the inclusive bound
  `abs(actual - expected) <= max(absolute, relative * abs(expected))`. Overflowing relative
  bounds are rejected; an overflowing distance exceeds every accepted finite bound. For example,
  `9007199254740993` and `9007199254740992` round to the same binary64 value.

The task prompt must ask the teacher for the chosen answer format. The harness does not inject the
reference answer or add format instructions. Authoritative wrong/unknown answers stop before the
quality panel; a correct answer still needs a passing quality decision. The run manifest pins the
numeric interpretation and the complete ordered materialized plan.

The versioned semantic task digest is derived from typed source identity/citation, exact prompt,
and numeric answer/extraction/tolerance semantics. Caller labels are never accepted as hashes.
Rights, group, split, policy, and QC declarations remain outside that semantic digest but are bound
by the full plan digest. Split changes therefore preserve the semantic identity while preventing
incompatible replay. Numeric seeds, shards, offsets, and record IDs keep their existing mapping;
the corpus group remains distinct from the prompt-hash `sibling_group_id`.

### Immutable run identity

Every new run stores a versioned generation/admission manifest in SQLite. It binds the full captured
seed plan, effective teacher and ordered judge requests, rubric, admission thresholds and intent,
verification rules, embedding behavior, and declared client endpoints. Every source shard is captured
once, including empty shards, and execution consumes those exact inputs. Message metadata, reasoning,
tools, oracle strings, task policies, exact required test IDs, QC flags, and whitespace all participate
in the input identity. The execution declaration also pins the verification interpretation revision.

`run`, `replay`, and `tui` check compatibility before reading credentials. The engine repeats the
check inside the launch transaction, so competing incompatible initializers cannot both succeed.
An incompatible launch leaves the original manifest, creation time, status, checkpoints, accounting
epoch, and launch history unchanged. Even a completed or fully cached run requires compatible meaning.
Use a new run ID to change generation or admission settings.

Accounting policy and amounts, rates, concurrency, UI timing, database/output paths, and export
projection remain operational choices. Compatible replay preserves the exact original manifest bytes
while registering the current accounting policy. Unknown replay IDs and legacy, unpinned, malformed,
or unsupported manifests fail execution with a new-run instruction. Historical runs remain available
for provider-free inspection and export; replay never adopts missing evidence.

Library extensions implement the pure `semantic_declaration()` getter on each actual `Provider`,
`Embedder`, `SandboxOracle`, and `ExecutionEvidenceSource`. Declare a stable implementation and behavior
revision plus immutable configuration or an evidence-collection digest. A type name, mutable path,
generic label, or accounting capability is insufficient. Preparation calls none of their execution
methods. These cooperative declarations identify requested models and configured routes; served
weights, tokenizers, templates, parsers, quantization, and deployment revisions remain unknown.
Configured embedding revisions are unenforced, and index labels do not select the engine's actual
in-memory cosine implementation. Source media URLs identify declared inputs without fetching media.
Chat and embedding base URLs reject userinfo, query, and fragment forms; use a credential-free route.

---

## Configuration

Configuration is layered, lowest precedence first: **built-in defaults → TOML file (`--config`) →
`GW_`-prefixed environment variables → CLI flags**. Defaults deserialize without a file, but generation
requires a nonempty judge panel and a valid admission configuration before any provider is constructed.

```toml
# ─── run-wide ───────────────────────────────────────────────────────────────
db          = "gw-run.sqlite"   # SQLite store: run state, queue, provenance
accounting_policy = { mode = "finite_usd", limit_usd = 5.0 }
# For local/unpriced backends: accounting_policy = { mode = "observation_only" }
model_api_base_url = "https://openrouter.ai/api/v1"   # Model API endpoint (base URL)
model_api_key_env  = "MODEL_API_KEY"                # API key environment variable name
provider_rpm      = 60          # shared teacher + judge chat requests/min

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

# ─── the judge panel (one or more distinct effective requests) ─────────────
[[area.judges]]
slug   = "deepseek/deepseek-v4-pro"
family = "deepseek"        # audit annotation; does not establish independence
# optional per-judge overrides: rubric_id, max_tokens, reasoning_max_tokens, reasoning_effort

# ─── optional end-of-run export ─────────────────────────────────────────────
[export]
out             = "out/dataset.parquet"
format          = "chatml"       # TOML enum spelling; CLI uses chat-ml
cot             = "supervised"   # supervised | masked | stripped
dataset_version = "0.1.0"
```

Notes:

- **The API key value is never in this file.** `model_api_key_env` names the environment variable
  that holds it; the default is `MODEL_API_KEY`. Names start with an ASCII letter or underscore and
  contain only ASCII letters, digits, or underscores.
- `GW_MODEL_API_BASE_URL` overrides the Model API endpoint (base URL); `GW_MODEL_API_KEY_ENV`
  overrides the API key environment variable **name**, never its value. For an existing OpenRouter
  setup, explicitly set `model_api_key_env = "OPENROUTER_API_KEY"`. There is no old-key fallback.
- Replace the removed `provider_base_url` / `GW_PROVIDER_BASE_URL` settings with
  `model_api_base_url` / `GW_MODEL_API_BASE_URL`. The default endpoint remains OpenRouter; changing
  its URL does not establish compatibility or qualify another backend. Embedding endpoint and key
  settings remain separate, including the local embedding default without authentication.
- `judge_reasoning_max_tokens` and `judge_reasoning_effort` are **mutually exclusive** in the same
  table — set one or the other, not both. The same holds for per-judge `reasoning_max_tokens` /
  `reasoning_effort`.
- Configuration keys can be overridden with `GW_` variables. For example,
  `GW_ACCOUNTING_POLICY__MODE=observation_only` selects a complete policy, replacing a finite
  policy from the file. `GW_ACCOUNTING_POLICY__MODE=finite_usd` requires a same-layer
  `GW_ACCOUNTING_POLICY__LIMIT_USD`. CLI `--accounting-policy finite-usd --limit-usd 10` or
  `--accounting-policy observation-only` replaces the complete file/env policy.
  The default is finite USD 5 only when no policy was supplied. Observation-only with a limit,
  nonfinite or negative limits, missing finite limits, removed `budget_usd` / `on_breach` keys,
  and removed flags are errors. Other common flags include `--k`, `--shards`, and `--admission-intent`. The intent environment
  setting is `GW_AREA__ADMISSION_INTENT=review_only`; its CLI spelling is `--admission-intent review-only`.

### Admission preflight

Automatic admission is the default intent. Before credentials or provider calls, the resolved panel
must be nonempty, score bands must satisfy finite `0 <= reject_below <= accept_threshold <= 1`, and
the effective-count floors must be finite and in range (`min_n_eff >= 0`, ratio in `[0,1]`). The
correlation prior is finite in `[0,1]` and must be positive and nonidentity for multiple judges.

Each panel position must have a distinct effective judge request and response interpretation.
Identical requests, family aliases, rubric audit IDs, equivalent token caps, and signed-zero
sampling variants cannot create additional evidence. Actual model, prompt, sampling, and reasoning
differences remain distinct contracts; they do not establish empirical independence. The library
grader requires the same evidence on live grades. The low-level cached collector still preserves
positions and coalesces matching cache keys for audit, but repeated positions cannot enter consensus.

Under equal cold-start weights, `d` decisive votes have
`n_eff = d / (1 + (d - 1) * correlation_rho)`. Preflight considers every `d` from one through the
number of judges, because uncertain votes are excluded. Some `d` must meet both the absolute and
relative floors. The default prior `0.7` and absolute floor `1.5` cannot do so, even with a larger
panel. `review_only` bypasses this attainability check while keeping evidence and numeric safeguards. Its
intent is stored with each candidate; otherwise admitted candidates remain `NeedsReview` on replay
and rederivation. Verifier hard failures still reject.

This evidence rule is pinned in the generation/admission manifest. Runs recorded under the previous
behavior need a new run ID for execution. Stored decisions and effective counts are preserved;
provider-free inspection, export, and publication recovery remain available.

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
gw eval fit-calibration    Offline group-equal judge fitting from supplied evidence (JSON).
gw eval screen             Frozen lexical groups/splits for a declared supplied corpus (JSON).
gw eval audit-separation   Score diagnostics and optional independent outcome evidence (JSON).
gw eval promote            Variance-aware promotion gate over two eval_results.json (JSON).
```

**`gw gen run` / `gw gen tui`**

| Flag | Meaning |
|---|---|
| `--config <FILE>` | TOML config (the figment base layer). |
| `--run-id <ID>` | Immutable run identity. Matching inputs and effective generation/admission settings **resume**. |
| `--prompts <FILE>` | Plain prompts with explicit judge-only policies; exclusive with `--tasks`. |
| `--tasks <FILE>` | Strict reviewed numeric task JSON; exclusive with `--prompts`. |
| `--db <PATH>` | Override the SQLite store path. |
| `--shards <N>` | Partition the seed space into N shards (default `1`) — the primary concurrency axis. |
| `--max-in-flight <N>` | Cap on seed items in flight across all shards (default `4`). |
| `--k <K>` | Override the best-of-k fan-out. |
| `--accounting-policy <MODE>` | `observation-only` or `finite-usd`; shared by run, replay, and TUI. |
| `--limit-usd <USD>` | Required with explicit `finite-usd`; forbidden with observation-only. |
| `--admission-intent <INTENT>` | `automatic` or `review-only`; also accepted by replay. |

**`gw gen export`** — `--db`, `--out`, optional `--run-id`, `--format`, `--cot`, and
`--dataset-version`. `--resume-publication <ID>` substitutes for `--out` and rejects projection overrides.
Standalone and automatic end-of-run exports use the same SFT eligibility rule: records need an
`admit` judging verdict and an `admitted`, `formatted`, or `exported` lifecycle state. Retained
best-of-k losers keep their individual grades in the store but do not enter the dataset. Unfinished,
rejected, review, and error states are excluded. The manifest counts all scanned records in
`n_records`; `n_admitted` and `build_inputs_hash` describe only the selected exported rows.

**`gw gen replay`** — resumes `--run-id` from a store using matching generation settings and the
**same** task or prompt input and `--shards` the original run used (the seed→shard partition is `index % shards`, so a
different value would re-partition the space and duplicate or orphan records). Replay accepts the
same accounting flags as run/TUI; an explicit policy change follows the epoch rules above. Unknown
run IDs fail. Incompatible generation/admission settings require a new run ID.

> **Concurrency.** `--max-in-flight` bounds seed groups across shard tasks; it does not count
> physical HTTP requests. Each group's `k` siblings may overlap under observation-only, while each
> sibling's distinct cached judge misses may also overlap, bounded by that panel's cardinality.
> Equal effective request keys within a panel share one result; cache hits make no model requests.
> Results retain panel order even when responses finish out of order. These are per-panel logical
> bounds; they do not set an endpoint-wide HTTP limit. Teacher and judge chat calls share the
> configured RPM limiter. Embedding calls have their own asynchronous path. Finite accounting
> serializes physical model requests regardless of these pipeline concurrency bounds.

A cached panel stops starting logical misses after an error and drains every operation it already
started, including provider futures waiting for RPM, retry backoff, or accounting admission.
Successful responses finish their cache writes, so replay requests only missing grades. Fatal
provider or cache errors seal engine dispatch before the panel finishes draining; a later fatal
error takes precedence over an earlier content error or cancellation. Existing provider waits can
still delay that drain.

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
| `gemma4` | Gemma-4. The renderer has golden-file tests; they are **pending a byte-for-byte diff-verify against the official pinned `chat_template.jinja`** before production SFT (nothing fetches the template at runtime). The system turn is folded into the following `user` turn and the upstream `<\|think\|>` marker is not emitted. |
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
a tool-faithful target is to export the canonical `messages_json` conversation with its `tools_json`
definitions and apply the model's official chat template in a consumer that owns that template.
Text-only conversations are unaffected on every target.

`--cot` records downstream training intent. Every policy preserves the same canonical messages in Parquet:

- **`supervised`** — train on captured reasoning.
- **`masked`** — include reasoning as context while excluding it from loss.
- **`stripped`** — construct an answer-only training example downstream.

The export is a columnar Parquet dump plus a manifest. One `messages_json` column holds the canonical
`Message[]` JSON in conversation order, losslessly: the `content` variant (including `null`),
`reasoning`, `reasoning_details`, `tool_calls` and the `tool_call_id` links all survive, so a
consumer decodes it straight back into `Message[]`. (The historical v1 `{role, content}` +
parallel `reasoning_json` pair was lossy — `reasoning_json` is gone, and `column_schema_version` in
the manifest records which contract a shard was written under.) The CoT policy and the target are
recorded as manifest metadata so a downstream trainer applies the matching loss mask and template.
Current exports use column schema v5 (`tool_definitions`): v3's nullable `task_json`, v4's required
`origin_json`, and nullable canonical `tools_json`. SQL null means absent tool definitions; `[]`
means an explicit empty list. Definition order and nested heterogeneous JSON values survive.
Generated origin is explicit; reviewed references carry stable registration, batch, member, code
and native-result bindings without fictitious teachers or judge scores. Reviewed-reference rows
require absent tools (SQL null); even an explicit empty list contradicts that source contract.
Task JSON contains typed task provenance plus the exact verification contract, validated against
the row's prompt. Plain prompts have null task provenance. Task declarations, origin and the actual
tools payload participate in row and artifact identity. Verification and receipt recovery continue
to honor v2/v3/v4's exact original columns and hash framing. A prepared or acknowledged publication
keeps its stored column version, identity, selected members, and acknowledgment history when
restored or republished; it is never upgraded during
recovery. A v2 receipt rejects selected records that acquired task provenance after preparation.
The exporter does not emit token IDs or loss labels and does not qualify official tokenizer,
truncation, or trainer behavior. TOML format values are `gemma4`, `chatml`, `share_gpt`,
`open_ai_messages`, `harmony`, and `trl_prompt_completion`; CLI spellings are shown in the table.

---

## Evaluation

`gw-eval` is **off-path and model-free** — it reads persisted records and eval artifacts, never calls
a provider:

- **`gw eval audit-separation`** reports verifier mixedness and judge-score spread. Optional
  independent outcomes evaluate the declared judge-score selection rule over a frozen corpus.
- **`gw eval promote`** is a variance-aware gate over two `eval_results.json` snapshots
  (baseline vs candidate) plus a drift exit code, printing a binary promotion decision.

Both accept `--check` to opt into decision-bearing process exits (`0` pass · `1` operational error ·
`2` gate rejects) for CI.

### Frozen source screening plans

```sh
gw eval screen --records records.json --declaration screening.json --protected protected.json > plan.json
gw eval screen --records records.json --declaration screening.json --protected protected.json --check-plan plan.json
```

This provider-free command reads ordinary `TrainingRecord[]`, a strict version 1
`ScreeningDeclaration`, and local `ProtectedScreeningSet[]` manifests. It opens no database,
loads no provider configuration, and acquires no benchmark contents. The pure Rust APIs are
`gw_eval::screening::prepare_screening`, `validate_screening_plan`, and
`protected_screening_content_digest`; the input/report types live in `gw-schema`.

Declare a nonempty `runs.run_ids` set and the complete expected source-item/revision and sibling
memberships. IDs are sorted canonically; duplicates reject. Every supplied record in those runs
enters the population, including Rejected records and relatives excluded from output. The separate
`output` selects candidate record IDs from one declared run. Required parents must resolve
unambiguously within the declared corpus. Missing task, source, parent or sibling evidence makes
the plan incomplete; observed completion counts alone do not assert that no relative is missing.
Records outside the declared runs are outside the claim. `supplied_files_only` explicitly means
that current database membership has not been checked.

Each protected manifest supplies a canonical set ID, immutable source revision, recomputed typed
content digest, item identities and actual local contents, reviewed permission for screening, and
field/language/text-media coverage. The ten canonical protected sets in `DecontamConfig` remain
required; `additional_protected_sets` adds to that union. Missing sets, unresolved rights, missing
coverage and unsupported media/encrypted reasoning produce incomplete coverage. These are supplied
assertions, not independent verification of rights, language or real benchmark completeness.
Unknown protected message/part/tool/reasoning fields reject. Protected payload text is omitted
from the emitted plan; match evidence carries only stable identifiers and integer counts.

The pinned `lexical-screen-v1` recipe lowercases ASCII A–Z and splits only on U+0009–U+000D,
U+0020, U+0085, U+00A0, U+1680, U+2000–U+200A, U+2028, U+2029, U+202F, U+205F and U+3000.
It discards empty tokens and preserves punctuation, signs, other case, accents and Unicode forms.
Turns, content parts, flat reasoning, reasoning details and supported tool fields remain separate
segments. Tool-argument and tool-definition object keys are independently screened text, while
retaining their exact structural meaning. Keys and values never share a shingle; evidence uses
deterministic member coordinates without copying protected key text. No shingle spans a segment
boundary. For each segment pair, a shared contiguous run of
`min_overlap_tokens` is required, then **any** n in the inclusive `ngram` range may qualify when
distinct-shingle intersection/union is at least `jaccard_threshold`. Empty intersections do not
match; repetitions add no set weight. The defaults `[8,13]`, minimum overlap 5, and threshold 0.8
are unqualified software parameters: the five-token gate adds no five-gram detector. Short
non-prompt segments have no shingle comparison. A whole nonempty structured prompt with user text
can match exactly regardless of length.

Grouping uses declared task/source/parent/sibling edges, any equal complete emitted training
example, and any pair of matching complete selected-target prompts. A lexical prompt edge requires
the same role/turn/part/tool-link structure and every corresponding text segment to match exactly
or by the shingle rule. A shared system instruction, reasoning fragment or short numeric answer
alone cannot create an edge. `all_assistant` contributes each full prefix before its selected
target; `final_turn_only` contributes only the final target's prefix. The shared format layer renders
complete examples under the pinned target/CoT/turn policies. Protected matching still examines all
captured supported source segments, including historical reasoning; it quarantines existing
components without merging otherwise unrelated records. Split conflicts quarantine the entire
component, including excluded relatives.

V1 ceilings are 64 MiB total text, 1 MiB and 65,536 tokens per segment, 100,000 segments,
2,000,000 stored distinct shingles, and 1,000,000 candidate comparisons. Structural unit checks
also consume the comparison budget. A separate `shingle_token_work` ceiling of 64,000,000 sums
`window_count * n` before processing each segment/length pair, including repeated windows and
the overlap gate. This bounds work even when repeated text creates few distinct shingles. Exact
indexes store start positions into token arrays, avoiding copied long tuples. Limits may be lowered
but not raised; exceeding a bound makes the result incomplete. Nothing is truncated or silently
dropped. These are operational limits, not empirical threshold qualification or a guarantee against
every allocation failure.

Thresholds use the strict finite binary64 codec: 0.8 is `{"binary64":"3fe999999999999a"}`.
Decimal alternatives, duplicate tags and nonfinite values reject. Signed zero and adjacent finite
values retain their bits through typed and JSON-Value replay. Counts stay integers; arbitrary
payload strings and ordinary record parsing retain their existing semantics.

The frozen report contains separate captured-input, policy, protected-input, grouping/split and
complete-plan identities, full population bindings, component memberships, protected coverage,
quarantine/exclusion counts and available task/domain/difficulty/teacher/length strata. Unknown
metadata remains explicit. Bindings include full typed messages and tools (including reasoning
detail IDs, indices, signatures and formats), task declarations, parent links, sibling fields,
persisted verdict and selected eligibility, while excluding publication-generated history/timestamps. `--check-plan`
recomputes the full report from the actual inputs: parsing and self-reported hashes confer no
authority. `--previous previous-plan.json` additionally revalidates the predecessor and retains its
group/split assignments when new members join. A bridge between previously separate groups is an
explicit persistent conflict. Predecessor chains are limited to 16 plans.

Exit status is `0` for complete lexical coverage with no quarantined component, `2` for incomplete
or quarantined results, and `1` for malformed/invalid input or failed exact plan verification.
`complete_no_match` describes the captured canonical source and pinned export policy only.
`semantic_status` is always `not_run`, and effective tokenizer/template prompt separation remains
`unknown`: a student template may omit historical reasoning retained by source prefixes. Unsupported
semantic/embedding policy fields reject. This command changes no production admission, export,
artifact or receipt behavior; existing unscreened v2/v3 artifacts retain their contracts. Actual
semantic recall, false-positive rates, real protected coverage and historical model exposure remain
unqualified. Screening cannot prove that a teacher or base model never saw a benchmark in pretraining.

### Transactional source-screened export

```sh
gw gen export-screened --db gw-run.sqlite --plan plan.json --protected protected.json \
  --out out/screened.parquet
```

The command captures the local protected contents once, queries every record in the plan's declared
runs, and reruns the complete pure planner against those captured inputs. It closes the initial read
transaction before the expensive rerun. Publication then rechecks the entire population and its
input bindings inside the SQLite write transaction. Added or removed members and changed inputs in
excluded records invalidate the plan. An unrelated run or a valid excluded aggregate-only update
leaves the screening identity unchanged; damaged stored projections or history always reject.

A complete plan may quarantine some components. The artifact emits exactly its eligible Train
output members; incomplete plans fail, including empty selections. The output-run record count,
requested candidate count, emitted row count and complete multi-run population count remain distinct.
Metadata version 3 uses the existing reviewed-task columns and binds the full text-free plan,
protected identities/coverage, exclusions, selected components and policy/layout. The publication
witness says `transaction_checked`; the nested planner report retains `supplied_files_only`.
The JSON result includes the publication ID and complete artifact qualification.

The strengthened frozen plan uses version 2, and its publication witness uses version 2 inside
artifact metadata version 3. Declarations, protected manifests and the lexical recipe retain their
v1 grammar. Each population binding includes a digest of all exported typed message fields
(including reasoning-detail metadata and tool turns), task declarations/contract, area and verdict.
The artifact verifier recomputes that digest from each row and checks its record hash and Train
split against the captured plan. Policy limits and complete protected-summary requirements share
the same pure validator used during planning. Earlier screened plans/artifacts lack this binding
and must be regenerated; raw metadata version 1 with v2/v3 columns remains supported unchanged.
Top-level tool definitions remain in the full raw input identity and are not added to the columns.
The frozen plan also retains the complete population's required field union, including excluded
records and tool definitions. Every protected summary must cover that union; independently decoded
rows must fit its declared fields and the same supported source shapes used by planning. Media,
encrypted reasoning, non-object arguments, invalid tool links, and unsupported training prefixes
cannot claim complete screening. The shared pure checks preserve the formatter's existing tool
and clean-field rules; calls without results retain their existing acceptance behavior.

Publication keeps the same staging/readback/rename/acknowledgment protocol as raw export. Explicit
recovery uses `gw gen export --db gw-run.sqlite --resume-publication <id>` and needs no protected files;
it rechecks the full current population even when the receipt was already acknowledged. A changed
population cannot be acknowledged or republished. Implicit retry requires the same raw/screened
flavor and frozen plan. Standalone export preserves record lifecycle; the shared
`Store::publish_screened_export` API also supports engine acknowledgment. Automatic engine exports
continue to use raw metadata version 1. Its historical v2/v3 column contracts and identities remain
unchanged. The filesystem rename and SQLite acknowledgment are separate boundaries.

The [TRL adapter](adapters/trl/README.md) accepts screened OpenAI-message inputs only when the
reasoning, turn and layout policies match exactly, including empty artifacts. Expanded examples
inherit connected component IDs and retain declared task groups separately. Semantic screening is
still `not_run`, and effective tokenizer/template separation is `unknown`. These source checks do
not establish empirical contamination recall or held-out training benefit.

### Offline judge calibration

```sh
gw eval fit-calibration --config panel.toml --records candidates.json --evidence calibration.json
```

This command resolves the configured area, rubric and ordered judges through the production request
builder. It reads a JSON array of complete `TrainingRecord` values and a strict version 1
`CalibrationEvidence` document, then prints a `CalibrationReport`. It makes no provider calls and
needs no credentials or database. Configuration layering remains defaults → TOML → `GW_` environment.
Request/control validation and duplicate effective-judge rejection apply; automatic-admission
feasibility is not a prerequisite for descriptive fitting. The records file is a lookup pool with
unique `(run_id, record_id)` pairs. The evidence's `fit` and `assessment` rows define the entire
measured population; unreferenced pool records are neither evaluated nor bound in the snapshot.

Prepare declarations with `gw_judge::ResolvedCalibrationPanel::new`, and bind already-collected raw
response text with its `observation` method. `gw_storage::capture_candidate_binding` recomputes full
candidate/run/area/prompt/content identities. These helpers establish request applicability only;
they do not attest historical collection, successful termination, served model weights, independent
tasks, blinding, or truthful labels. The evidence accepts only `supplied_unverified` runtime
provenance. The panel declaration is separate from each candidate-specific request contract; every
request is rebuilt from the complete ordered message sequence using the production renderer.

The evidence declares one versioned higher-is-better [0,1] reference target/protocol, an explicit
nonnegative `beta`, a versioned map to globally scoped prompt groups, and disjoint `fit` and
`assessment` populations. Equal prompt hashes must share a group across runs. Each declared
candidate needs a known label and exactly one usable observation from each ordered judge. Unknown
labels, uncertain verdicts and missing/failed collection make the population incomplete; rows and
judges are never silently dropped. Payload text is parsed using production JSON extraction,
normalization and verdict rules. A supplied normalized score must match that result exactly.

The `group_mean_squared_score_error_v1` recipe first averages squared quality-score errors within
each group, then averages groups equally. Weights are proportional to
`exp(-beta * (loss - minimum_loss))`; `beta = 0` is an explicit equal-weight control. The backend is
pinned to `libm = 0.2.16`, with sequential accumulation in canonical group/candidate order and
explicit panel order for weight normalization. Zero/underflowed or nonfinite weights reject. This
is score fitting, with no probability-calibration interpretation or reputation fallback.

Every declared floating-point field uses a finite binary64 bit object, such as
`{"binary64":"3fe0000000000000"}` for 0.5; decimal alternatives, duplicate tags, unknown keys,
malformed bit objects and nonfinite encodings reject. Optional unknown values remain explicit. Full production request JSON
and exact `raw.response` text remain opaque strings through persistence and hashing. Integer
versions, counts, seeds and caps remain integers. Ordinary record JSON is unchanged.

A complete snapshot seals the canonical evidence, exact losses/weights, group/candidate counts,
zero exclusions, frozen-weight descriptive assessment, the declared constant-rho assumption, and
`computed_unqualified` status. Fit identity excludes held-out inputs/results, whole-corpus
provenance containers and rho; the complete identity binds them. Held-out labels cannot influence
fitted weights. `gw_judge::verify_calibration_snapshot` refits and compares exact saved bits and
identities, rejecting mismatches without a tolerance or automatic rewrite.

Exit codes are `0` for `computed_unqualified`, `2` for `incomplete_evidence`, and `1` for invalid
input or an operational error. Semantic invalidity includes an `invalid_evidence` report;
malformed JSON/configuration fails before a report is emitted. Invalid/incomplete reports contain
no snapshot. There is no quality-qualified status or runtime snapshot adoption in this command.
Actual held-out quality, robustness, and empirical correlation remain unmeasured. Saved-bit replay
is distinct from target/toolchain/build-specific refitting checks: this recipe supports fixed Rust
exp dispatch on aarch64 and x86_64, while exact test results qualify only the configurations where
the fixtures actually run. Other architectures reject this recipe.

### Independent selector outcomes

Judge-score spread is descriptive: a group's maximum necessarily reaches or exceeds its own mean.
Without `--outcomes`, `outcome_evaluation.status` is `insufficient_evidence`, its `statistics` are
`null`, and `--check` exits `2`. The diagnostic `--min-decidable-groups` and
`--min-decidable-fraction` settings only control the mixedness warning.

```sh
gw eval audit-separation --db run.sqlite --run-id run-1 --outcomes outcomes.json --check
```

The version 1 outcome envelope uses one run, one area, one bounded metric, and one reference
protocol for an explicit frozen corpus. Its JSON structure is:

```json
{
  "version": 1,
  "run_id": "run-1",
  "training_area": "math",
  "metric": {
    "name": "task_success",
    "version": "v1",
    "direction": "higher_is_better"
  },
  "provenance": {
    "source": "deterministic_task_reference",
    "protocol_revision": "task-reference-v1",
    "reference_artifact_digest": {
      "algorithm": "sha256",
      "hex": "<full 64-character lowercase digest of the reference artifact>"
    }
  },
  "sampling_assumption": "independent_prompts",
  "corpus": [],
  "outcomes": []
}
```

Replace the digest placeholder and populate both lists before evaluating. `corpus` contains exact
bindings with `record_id`, `run_id`, `training_area`, `prompt_hash`, and `record_hash`. Use the full
recomputed hashes: the Rust helper `gw_eval::outcomes::CandidateBinding::from_record` constructs
them. `record_hash` follows the storage content-hash contract; run and area are also checked
explicitly. Each `outcomes` entry contains `candidate` (the same complete binding) and `outcome`,
either `{"status":"known","value":1.0}` or `{"status":"unknown","reason":"not measured"}`.
Known values must be finite and in `[0,1]`.

`adjudicated_reference` is the other accepted source; `blake3` is the other digest algorithm.
Every label uses the envelope's single metric and provenance contract. Per-label overrides,
unknown fields, duplicate labels, stale content hashes, conflicting identities, and labels outside
the declared corpus are invalid. Store filters must retain every declared member. Later records
outside the frozen member list do not enter its random-control population. A missing or unknown
outcome anywhere in the declared corpus blocks qualification, including when other prompts have
complete labels.

The evaluated population is all-pass prompt groups with at least two scored candidates. The policy
chooses the highest judge score and breaks ties by the lowest completion index; evaluated indices
must be present and distinct within each prompt. It compares that candidate's independent outcome
with the uniform-random expected outcome over the same scored candidates. Each distinct prompt hash
receives equal weight, so repeated prompts do not increase the independent sample count.

The report records the policy, metric, reference provenance, coverage, typed reasons, confidence
level, and the declared independent-prompt sampling assumption. Under that assumption, its
one-sided Hoeffding lower bound is `mean_gap - sqrt(2 * ln(1/alpha) / n)` for prompt gaps in `[-1,1]`,
where `alpha = 1 - confidence_level`. Qualification requires a positive lower bound and at least
30 evaluated prompts by default. `--confidence-level` (default `0.95`) and
`--min-evaluated-prompts` make those settings explicit.

Complete evidence with insufficient statistical support is `inconclusive`. Missing/unknown evidence
is `insufficient_evidence`; both exit `0` after successful analysis or `2` with `--check`. Semantic
invalidity prints an `invalid_evidence` report and exits `1` in either mode. Parse or I/O errors exit
`1` without a report. Absent numeric comparisons serialize as `null`.

Reference provenance is an auditable declaration, not proof of evaluator independence or blinding.
Qualification concerns this selection rule and declared corpus. It does not qualify the engine's
full admission policy, model quality outside that corpus, or downstream student learning benefit.
Synthetic fixtures establish software behavior only.

### Promotion

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

## Model identity vocabulary

`gw-schema` provides independent version 1 documents for pinned model artifacts, requested
execution semantics, model-policy document references, supplied deployment evidence, and observed
execution references. These are pure data contracts with structural validation and identity hashing.
They are not yet adopted by the engine, client requests, caches, records, or attempt receipts.
RunManifest v1 still requires every `UnattestedDeployment` value to be `None`; its canonical bytes,
record content hashes, and judge-request v2 identity are unchanged.

A pinned artifact declares its source and immutable revision, a nonempty file inventory with
purpose/algorithm/digest, lineage, and explicit tokenizer/template pins or unknowns. Lineage covers
base artifacts, derivatives, quantization, adapters, and checkpoints. Declaring a revision and file
hashes does not establish that files were obtained or loaded. An adapter's prior checkpoint and a
checkpoint's prior checkpoint or adapter each require an explicit declaration: unknown uses
`{"status":"unknown"}`, declared absence uses `{"status":"declared","value":null}`, and a
present link supplies an artifact identity as `value`. Omitting the link or a declared `value` is
invalid. These three states have distinct artifact hashes; declared absence remains a supplied
claim requiring independent review and does not establish model eligibility.

Chat-template declarations use the same three states: unknown, a pinned component, or explicit
`{"status":"declared","value":null}` absence. Existing present and unknown JSON and identities
are preserved. Chat preparation requires a present template; text completion and embedding may
declare absence. Supported text profiles still require a pinned tokenizer.

Model-policy references separately identify review, catalog, rights, lineage, serving-terms, and
output-terms documents for a declared
artifact, role, and intended use, including embeddings, teachers, judges, prompt synthesis, students, and
derivatives. They do not establish model eligibility. Task-source rights and candidate test-execution
evidence retain their existing meanings.

Requested execution binds an alias and operation to an explicit `ModelAdapterBehavior`, which uses
`SemanticDeclaration` for adapter behavior only. Serving endpoints and replica identities belong in
separate requested-execution or deployment-evidence fields. The existing full client/run declarations
remain endpoint-sensitive and cannot be used directly as adapter behavior. The pure
`gw_providers::builtin_adapter_behavior` helper reads the known built-in v1 descriptor shapes and
extracts their behavior parameters; unsupported implementations, versions, or fields are rejected.
For embeddings, the requested model stays in the execution alias, and unenforced model-revision/index
labels are excluded from adapter behavior. Existing client descriptors, fingerprints, constructors,
manifests, runtime behavior, and current caches are unchanged.

Execution also declares a serving-profile revision and any known artifact, tokenizer, template,
runtime, parser, and configuration behavior. Supplied deployment evidence records method/version, issuer and
claimed verifier, raw evidence reference/digest, claimed loaded artifacts and effective semantics,
endpoint/instance/incarnation, and claimed validity/revocation evidence. It requires independent
qualification before use; there is no deserialized approval or verification flag. Unknown expiration
is not perpetual validity. Requested and claimed effective identities may disagree and remain
separate evidence for a later consumer to assess.

Observed execution references the durable run, launch, physical attempt, and optional exact
observation sequence. A missing sequence means the attempt only, not its latest observation.
Supplementary native and normalized termination reasons preserve missing values. Resolved model,
provider, response ID, usage, and cost continue to belong to the existing attempt receipt.

All top-level documents reject unknown fields, unsupported versions, malformed digests, and unsafe
or duplicate file identifiers during deserialization. Their `from_json` and validation errors contain
no rejected input. File identifiers use nonempty `/`-separated ASCII alphanumeric, `.`, `_`, or `-`
segments; traversal, absolute paths, Windows reserved names, trailing dots, and case-insensitive
duplicates are rejected. Locators use a deliberately narrow HTTP(S) origin/path or `urn:namespace:id`
grammar; endpoints require HTTP(S). Userinfo, query strings, fragments, percent escapes, backslashes,
and whitespace are rejected. Paths cannot contain `.` or `..` segments. Locators are not normalized,
resolved, or fetched. All free text and semantic configuration must remain non-secret.

Identity hashes use distinct BLAKE3 derive-key domains for artifacts, policy documents, semantic
execution, deployment evidence, and observed execution. Canonical JSON recursively sorts object keys
and preserves configuration-array order. Artifact files sort by exact path; additional lineage parents
and declared artifact sets sort by digest, with duplicates rejected. An empty additional-artifact or
additional-parent set declares none and remains distinct from unknown. Policy-reference order has no
semantic effect. Artifact display labels are excluded; source locators, revisions, file identifiers,
purposes, algorithms, byte digests, components, and lineage contribute. Policy identities include
subject artifact, locator, revision, role, intended use, and document kind. Deployment identity includes all evidence,
locators, incarnation, and validity claims. Semantic execution identity excludes endpoint, policy,
incarnation, validity, and attempt references, while preserving unknowns and binding every semantic
declaration. Attempt/incarnation changes alter evidence binding. Equal declared semantics alone do
not establish deployment equivalence or authorize cache reuse.

## Offline artifact eligibility

`gw_providers::artifact_assessment` assesses an exact artifact, role, and intended use from an offline
bundle and a separately owner-supplied `TrustedArtifactCatalog`. Catalog construction belongs to a
trusted application or owner review channel. The catalog and its review inputs cannot be
deserialized from submitted model evidence. There are no built-in production catalog entries.

Each trusted review binds the recomputed artifact identity, allowed role/use, reviewed openness,
review reference/date/scope, and separate rights, serving-term, and output-term decisions. Reviewed
open weights may qualify without a complete open-training-artifact claim. Closed or unknown
openness and unreviewed applicable terms deny. Rights require reviewed evidence. Serving and output
terms each require either their own reviewed evidence or an explicit owner-reviewed inapplicability
scope. The assessor does not infer permissions from license names or interpret license text.

Both catalog construction and assessment check raw policy bytes against their declared SHA-256 or
BLAKE3 digest. The trusted review additionally pins the exact `PolicyDocumentIdentity` and an
independent BLAKE3 digest of the reviewed bytes. Submitted review/catalog documents, aliases,
claimed accepted digests, and approval flags cannot supply a trusted review. Changes to a policy's
subject, role, use, revision, bytes, or declaration require a new matching independent review.

Every referenced base, parent checkpoint, and adapter must resolve to its exact declaration and
its own reviewed obligations for the requested role/use. Unknown or missing lineage denies;
checkpoint and adapter links must have compatible kinds and the same declared base. Explicit
absence is useful only within the exact artifact claim reviewed by the owner. Duplicate or
conflicting catalog entries and duplicate submitted evidence are rejected. This first contract
supports one pinned document per artifact/kind/role/use. Additional submitted documents for that
same scope deny the assessment instead of expanding the trusted review. Real rights reviews may
require several source documents; those need an explicitly supported reviewed set in a later
contract. This limitation does not classify their text as legally contradictory. Collection ordering
is ignored where the contract defines a set; semantic configuration array ordering remains meaningful.

The structured report contains the selected artifact/role/use, exact catalog identity, matched
review and policy references, denial reasons, and an evidence digest. Catalog revisions and review
scope changes alter that binding. Reports are immutable results and cannot be deserialized into
trust. A positive report establishes only this offline artifact assessment. It does not prove that
model files were acquired or loaded, verify a deployment, qualify an actual client, or authorize
cache reuse or a model call. Existing request, manifest, cache, receipt, and concurrency behavior is
unchanged; this module performs no file, environment, credential, or network access.

## Offline serving profiles and gateway evidence

`gw_providers::serving_profile` provides versioned configured profiles, pure request preparation,
injected authentication, an offline chat-response normalizer, and supplied-snapshot consistency
checks. These APIs prepare and compare data. They are not connected to the live clients, engine,
CLI, manifests, caches, or request receipts and grant no execution authority.

`ServingProfile` holds the endpoint, authentication references, strict routing policy, and
operational defaults. Its separate `ProfileBehavior` declares the wire dialect, supported
operations, controls, and literal effort values. Only that behavior projection enters the serving
semantic declaration. Endpoints, credential-reference names, replica defaults, client in-flight
limits, rate limits, retry/deadline policy, and accounting policy are excluded. Backend batching,
client concurrency, and replica scaling have separate fields; declared capacity is not measured
throughput. `ServingProfile::modal` uses explicit `ObservationOnly` defaults.

The supported wire mappings are OpenRouter chat and vLLM-shaped chat, text-completion, and embedding
requests. Capabilities are supplied configuration claims, not approvals of any model or deployment.
Preparation requires complete pinned execution declarations and a matching profile/adapter behavior.
It uses canonical messages and sampling values, rejects unsupported required controls, and records
every omitted optional control. Effort names pass through exactly; `max` and `xhigh` are never
treated as equivalents. Template kwargs are explicit model/recipe declarations. OpenRouter
preparation requires a nonempty `provider.only` allowlist and emits `allow_fallbacks: false` and
`require_parameters: true`; no unapproved fallback is prepared. See
[OpenRouter provider routing](https://openrouter.ai/docs/guides/routing/provider-selection).

Authentication is explicit: no auth, one bearer-secret reference, or two distinct Modal proxy
references. Parsing and preparation do not resolve secrets. A separate injected resolver supplies
the values; missing, empty, whitespace-bearing, or invalid header values fail with static errors.
Modal token ID and token secret map to sensitive `Modal-Key` and `Modal-Secret` headers. Resolved
secrets and headers have redacted debug output and cannot be serialized. There is no SDK/CLI
credential discovery or no-auth fallback. See
[Modal proxy authentication](https://modal.com/docs/guide/webhook-proxy-auth).

The additive `normalize_chat_chunk` decoder reconciles `reasoning` and `reasoning_content` once
when both agree and rejects conflicting aliases. Content stays separate, structured reasoning
objects retain their fields, and missing token/cost values remain unknown rather than zero.
Native and normalized termination reasons remain distinct. This decoder handles supplied chat
JSON payloads; it does not replace the live stream decoder or measure tokenizer behavior. See
[vLLM reasoning outputs](https://docs.vllm.ai/en/latest/features/reasoning_outputs/).

`GatewayRequest` requires an immutable semantic target and its resolving declaration. An exact
instance/incarnation pin is optional, so equivalent replicas can satisfy the same fleet target.
`GatewayResponseEvidence` binds the actual replica evidence to endpoint, body digest, correlation,
and physical attempt. `check_gateway_consistency` compares these supplied documents against the
private immutable prepared request, supplied artifact declarations, and an explicit evaluation time
and revocation snapshot. Matching self-reported request/response body hashes alone are insufficient.
Every retry needs its own attempt and destination evidence.

The report separates semantic agreement, request/replica binding, and validity. A known mismatch
outranks unrelated unknown fields. Unknown values never establish completeness; unsupported
versions or measurement formats, installed-only claims, missing declarations, expiry, revocation,
and conflicting bindings cannot produce an aggregate `Consistent` result. Reports retain the
exact request/attempt coordinates and actual evidence identity that were checked.

The recognized loader-report format declares `ghostwriter/loaded-generation-report`, revision `1`,
with `{"scope":"loaded_generation"}`. Recognizing that format does not authenticate its claims.
Real qualification must establish that the serving replica measured the actual loaded artifacts
and effective configuration, that restart/reload changes its incarnation, and that the gateway
retains that same loaded generation atomically through inference. It must also authenticate the
response evidence and enforce freshness/revocation immediately before each physical send. Offline
fixtures establish none of those deployment lifecycle guarantees. Artifact policy eligibility and
actual-client/cache qualification remain separate requirements; no production model is approved here.

---

## Security

- The provider constructor reads the key from the configured environment variable (default
  `MODEL_API_KEY`) after run compatibility checks. Configuration stores only the variable name;
  never put the key value in TOML, `GW_` settings, CLI flags, or logs. Credentials are excluded from
  run manifests, receipts, cache identities, and provider debug output.
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
