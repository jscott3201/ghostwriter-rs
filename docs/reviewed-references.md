# Reviewed reference imports

The local reference path imports complete reviewed Python solutions without inventing a
teacher call, generation settings, quality-panel grades, or a judge admission event.
`TrainingRecord.origin` distinguishes generated records from reviewed references.
Generated records retain the exact historical generated-v1 JSON envelope. A reference
has a strict versioned `origin` object and common messages, task, verification and lifecycle.

## Local commands

```sh
gw reference register --catalogue corpus/catalogue.json --db private.sqlite
gw reference import --catalogue corpus/catalogue.json --db private.sqlite
gw reference export --db private.sqlite --batch-id <printed-batch-id> --out train.parquet
```

Registration is explicit local operator acceptance of captured bytes. It validates the
whole population and writes a private registration without running reference code.
Import requires those exact accepted bytes; changed whitespace also requires a new
registration. Neither an `approved` field nor a saved successful evaluation report can
register a population. There is no bypass flag.

The catalogue has `version: 1`, a public `training_area`, ordered `task_documents`
(relative file paths), and 112 ordered `members`. Each member declares `task_document`
(document index), `task_id`, `module_path`, `review_path`, and a namespace-qualified
`component`. Paths must be normal relative paths and cannot traverse symlinks.

Task documents use the existing reviewed coding-task contract and retain its 64-task
per-document limit. The full population must contain 28 families of four members:
16 Train families (64 members), four Validation families (16), and eight Test families
(32). Labels, source identities and semantic identities must be globally unique across
documents. Families and related components must have consistent complete split assignments.
Train tasks use private training cases; Validation/Test tasks use protected cases.

Each private review has `version: 1`, `task_digest`, `reference_code_id`, `author`,
`reviewer`, `independence`, `correctness`, `oracle`, `rights`, and `permitted_use`.
Actors declare `kind` (`human` or `agent`) and a nonempty private `label`. Evidence text
must be nonempty and cover the exact complete task and module. The required use is
`training` for Train and `evaluation` for held-out members, also present in task rights.
`reference_task_digest` binds canonical complete task semantics, including private cases
and declarations, under `ghostwriter.reference-task.v1`. Module identity binds exact UTF-8
bytes under `ghostwriter.coding-module.v1`. Both use BLAKE3 derive-key hashing.

These are operator declarations. Agent-reviewed work is labeled as agent-reviewed;
the software does not authenticate an external person or certify rights or review quality.
Private review labels and text stay in the registration store. Public task declarations
remain part of the existing task export contract.

## Execution and atomicity

Import captures catalogue, documents, modules and reviews once, validates the entire
population, and checks the existing registration before starting Docker. It executes only
those captured bytes under the existing pinned cached local runtime; it never pulls an
image. Each member must yield an opaque fresh observation consumed by the native verifier.
Saved `CodingArtifact` or Passed JSON is not accepted by the private CLI batch builder.
The storage API is an application-trusted adapter boundary, not a remote attestation service.

A Fail, Unknown, cancellation or uncertain cleanup prevents the complete new batch from
reaching its commit decision. Cancellation while waiting for the SQLite writer lock or
preparing admission rolls back every new batch fact. After preparation wins the final
cancellation check, the commit and its acknowledgment settle to a truthful outcome;
a late Ctrl-C cannot turn a successful commit into a cancellation report. A lost
acknowledgment is resolved from the authoritative batch state and idempotent reuse.
One immediate SQLite transaction commits the typed reference run, 112 private
member/evidence rows, 64 Train records and their truthful Verified/Admitted history.
The remaining 48 members never become `TrainingRecord`s or enter ordinary scans/exports.
The reference run has JSON null for its absent generation configuration and cannot enter
generation replay. SQLite durability follows the store's documented WAL/NORMAL policy.

The batch identity is deterministic. Repeating an already committed import returns current
records and reports historical completion, preserving later Formatted/Exported history.
`--fresh-validation` runs all members again, reports fresh execution, and preserves original
committed facts. A missing commit acknowledgment is resolved from current stored state.

## Publication and preparation

`gw reference export` uses the explicit reference publication purpose and acknowledges
only registered committed Train members. Standalone export validates the same eligibility
without advancing lifecycle. Engine publication accepts generated records. Serialized
origin fields alone cannot authorize publication, and references cannot be inserted through
ordinary record insertion or fixture replacement.

New Parquet uses v4 `record_origins`: v3 columns plus required strict canonical `origin_json`.
References expose stable catalogue, registration, batch, member, module, suite and native
result bindings, declared actor categories, component and use. Judge verdict/aggregate are
null and reasoning-token cost is zero. Reference down-projection to old column versions
is rejected. Stored v2/v3 publication recovery retains its original identities and columns.

Rust and Python validate reference row semantics. Every prepared SFT example retains the
exact origin column and native verification reconciles it with the embedded source bytes.
Unscreened reference grouping uses the declared reference component; explicit screening
uses the verified connected component. Hashes bind these declarations and contents; they
do not prove independent review, external authorship, or downstream model benefit.

Software tests use synthetic populations. They are not a real reviewed coding corpus.
