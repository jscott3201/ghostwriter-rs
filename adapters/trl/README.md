# Verified text-profile SFT labels and numeric rewards

This external Python adapter reads one canonical Ghostwriter Parquet snapshot,
asks `gw artifact verify --stdin` to verify **those exact bytes**, and prepares
`input_ids`, `attention_mask`, and explicit unshifted causal language-model
`labels`. It also retains the original `messages_json`, artifact/record/target
identities, character and token ownership, and a versioned immutable input build.
The installed command saves that build, verifies it through Rust, and replays the
pinned tokenizer before any optional trainer handoff.

Two named text profiles use the same source verifier, loss ownership, saved-input
format, replay, and trainer handoff:

| Profile | Exact publisher release | Official rendering entry point |
| --- | --- | --- |
| `qwen3_text_v1` (default) | [Qwen/Qwen3-0.6B, c1899de289a04d12100db370d81485cdf75e47ca](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca) | Tokenizer chat template |
| `gemma4_e2b_text_v1` | [google/gemma-4-E2B-it, 3e22461f65e89153144f8adb70e3b8c2cc9845a7](https://huggingface.co/google/gemma-4-E2B-it/tree/3e22461f65e89153144f8adb70e3b8c2cc9845a7) | Known `Gemma4Processor` with the captured template and tokenizer |

The pinned official rendering and tokenization are the independent oracle for
each profile. Both publisher cards declare Apache-2.0; parent revisions and full
execution lineage remain unresolved. Label and trainer-handoff qualification is separate from
the bounded full-SFT checkpoint path described below. Local software qualification
uses freshly initialized tiny models and establishes no learned quality benefit.

## Reproduce the qualified environment

Both checked environments were exercised on **CPython 3.12.14, macOS ARM64, CPU**.
Qwen retains its 55-package `requirements.lock`: Transformers 4.56.2, tokenizers
0.22.0, and safetensors 0.6.2. Gemma uses the separate 64-package
`requirements-gemma.lock`: Transformers 5.18.0, tokenizers 0.23.2, safetensors 0.8.0,
torchvision 0.23.0, and Pillow 12.3.0. Both use TRL 1.14.1, PyArrow 21.0.0,
PyTorch 2.8.0, and Accelerate 1.4.0. The locks include public-PyPI wheel hashes.
Keep the profiles in separate environments: their exact dependencies conflict.
The package extras declare the actual profile requirements and the adapter checks
every qualified version before preparation. Linux, CUDA, other Python releases,
and other dependency solutions remain unqualified.

From the repository root, with an installed CPython 3.12 interpreter:

```sh
uv venv --python python3.12 adapters/trl/.venv
uv pip sync --python adapters/trl/.venv/bin/python --only-binary :all: \
  --require-hashes adapters/trl/requirements.lock
uv pip install --python adapters/trl/.venv/bin/python --offline \
  --no-build-isolation -e 'adapters/trl[qwen]'
cargo build -p gw-cli --bin gw --locked
```

For Gemma, create a different environment and use its declared extra:

```sh
uv venv --python python3.12 adapters/trl/.venv-gemma
uv pip sync --python adapters/trl/.venv-gemma/bin/python --only-binary :all: \
  --require-hashes adapters/trl/requirements-gemma.lock
uv pip install --python adapters/trl/.venv-gemma/bin/python --offline \
  --no-build-isolation -e 'adapters/trl[gemma-e2b]'
```

Acquire only the files in the selected profile's manifest, at its exact revision:
six files for [Qwen](src/ghostwriter_trl/tokenizer_manifest.json), or seven files
totalling 32,226,083 bytes for [Gemma](src/ghostwriter_trl/profiles/gemma_manifest.json).
Gemma captures the model card, tokenizer, template, processor configuration, model
configuration, and generation configuration to identify the official text path.
No profile downloads weights. Use a new, empty output directory. The following
optional acquisition example is for Qwen; select the Gemma manifest, environment,
and a separate output directory for Gemma. Acquisition is separate from preparation.

```sh
adapters/trl/.venv/bin/python - <<'PY'
import hashlib, json, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
manifest = json.loads(Path("adapters/trl/src/ghostwriter_trl/tokenizer_manifest.json").read_text())
output = Path("qwen3-tokenizer")
output.mkdir(exist_ok=False)
for entry in manifest["files"]:
    cached = Path(hf_hub_download(manifest["repository"], entry["name"], revision=manifest["revision"]))
    data = cached.read_bytes()
    assert len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"]
    shutil.copyfile(cached, output / entry["name"])
PY
```

The adapter rejects extra files in this directory. It captures and hashes every pinned
files, then loads their verified snapshot with `local_files_only=True` and
`trust_remote_code=False`. It also checks the runtime tokenizer backend,
complete special-token map, wrapper configuration, and official template hash. The pinned
wrapper requires `split_special_tokens=False`; qualified encoding and offset calls
also pass that value explicitly. Internal pin data is immutable, and manifests
contain detached copies. Editing a returned manifest cannot change later loads or
builds. No remote code runs.

## Prepare and inspect

```sh
adapters/trl/.venv/bin/ghostwriter-trl \
  --artifact dataset.parquet --gw "$PWD/target/debug/gw" \
  --tokenizer "$PWD/qwen3-tokenizer" \
  --cot masked --turns all_assistant --max-length 2048 \
  --output prepared-sft
```

For Gemma, select the profile and thinking preamble explicitly:

```sh
adapters/trl/.venv-gemma/bin/ghostwriter-trl \
  --artifact dataset.parquet --gw "$PWD/target/debug/gw" \
  --tokenizer "$PWD/gemma-e2b-tokenizer" --profile gemma4_e2b_text_v1 \
  --thinking on --cot masked --turns all_assistant --max-length 2048 \
  --output prepared-gemma --qualify-handoff
```

`--thinking on` adds the official Gemma system thinking preamble; `off` omits it.
Gemma defaults to `off`, while Qwen retains its required `on` setting. Reasoning
loss policy is selected independently with `--cot`. Both choices, together with
`add_generation_prompt=False` and Gemma's `preserve_thinking=False`, are identity-bound.
When loading through Python, pass `profile="gemma4_e2b_text_v1"` to `load_tokenizer`.

The output directory must not already exist. Preparation writes:

- `prepared.gwsft`: the complete ordered examples, manifest, and exact original Parquet bytes.
- `verification.json`: a separate Rust receipt for the complete build and its reverified source.
- `replay.json`: a separate receipt for official rendering/tokenization replay by the Python loader.
- `handoff.json`: optional actual collator/trainer evidence, created with `--qualify-handoff`.

The input `build_id` exists before any trainer operation and stays unchanged when
handoff evidence is added. An empty or fully rejected input still carries complete
source verification, policies, counts, and rejection reasons. Inspect those counts
before consuming examples. Handoff qualification requires usable, unequal-length examples.

To verify a captured build with the provider-free Rust bridge:

```sh
gw artifact verify-prepared --stdin < prepared-sft/prepared.gwsft
```

To consume the saved build through the pinned Python loader:

```python
from pathlib import Path
from ghostwriter_trl.prepared import read_prepared
from ghostwriter_trl.handoff import qualify_prepared_handoff
from ghostwriter_trl.tokenizer import load_tokenizer

tokenizer = load_tokenizer(Path("qwen3-tokenizer"))
prepared = read_prepared(Path("prepared-sft/prepared.gwsft"), Path("target/debug/gw"), tokenizer)
examples = prepared.examples
manifest = prepared.manifest
report = qualify_prepared_handoff(prepared, tokenizer)
assert report["build_id"] == prepared.build_id
```

`VerifiedPrepared` can be created only through successful whole-build verification
and replay. Public examples, metadata, and receipts are defensive copies. Capture
happens once: replacing the bundle or original source path cannot change the bytes
that verification, replay, or handoff consumes. Publication uses an exclusive atomic
link and refuses existing files or symlinks. An optional handoff failure leaves the
saved input available for inspection; it does not produce a successful handoff receipt.

For a synthetic fixture, add `--qualify-handoff` to load the saved build and inspect both the actual
[TRL 1.14.1 collator and SFTTrainer](https://github.com/huggingface/trl/blob/fd74bbc7b5f852a70d4cc94377e0a8f94392fda1/trl/trainer/sft_trainer.py)
dataloader. This requires at least two unequal-length examples. It constructs a
small **random** CPU GPT-2 causal model covering the full tokenizer vocabulary,
then checks the real tensors. It performs no model forward pass, optimizer step,
pretrained-weight download, or cloud operation. The random architecture tests
the handoff only. Gemma weights, inference, optimization, and LoRA are outside this path.

The qualified recipe uses `skip_prepare_dataset=True`, `max_length=None`, `packing=False`,
`padding_free=False`, `assistant_only_loss=False`, `completion_only_loss=False`,
and the adapter's explicit labels. CPU, NLL loss, disabled mixed precision and
gradient checkpointing, no reporting/Hub push, and zero dataloader workers are
explicit. The package sets Hub/Datasets/Transformers offline and disables
telemetry before importing those libraries. The handoff checks all nonpadding
IDs/labels, right-padding IDs, attention masks, and `-100` padding labels.

## Loss policy and supported inputs

| Policy | Selected final assistant target |
| --- | --- |
| `supervised` | Reasoning, its thought wrapper, answer, and end-of-turn enter loss. |
| `masked` | Reasoning and its complete wrapper remain in input with `-100`; answer and end-of-turn enter loss. |
| `stripped` | Reasoning is removed; Qwen's official empty thought wrapper is masked; Gemma emits no thought wrapper; answer and end-of-turn enter loss. |

Headers, separators, system/user turns, and historical assistant turns are masked.
Every accepted example must retain a whole nonwhitespace answer token after the
causal shift. Stop or wrapper tokens alone do not satisfy this condition.

`final_turn_only` prepares the full conversation with only its final assistant
supervised. `all_assistant` explicitly selects **assistant_prefix_v1**: one
conversation prefix ending at each assistant, with only that prefix's final
assistant supervised. This is necessary because the official template omits
historical reasoning before the last user. Each target enters loss once;
repeated context remains masked. Expansion changes example-based weighting;
source-record and expanded-example counts remain distinct.

The supported shape is an optional initial system message followed by alternating
user/assistant string messages, ending with an assistant. Flat assistant reasoning
is supported, including absent reasoning. Ordered `reasoning.text` details are
accepted only when their strictly increasing unique indices and exact concatenated
text agree with present flat reasoning. Flat text is rendered once; original
source bytes are retained. Tools, media, null content, other roles/positions,
nonredundant structured reasoning, and every **pinned added-token literal** are
rejected in every clean source channel, including before stripping. This includes
all 26 Qwen tokens (including six non-special FIM/repository tokens) and all 24
Gemma control tokens (including modality delimiters). The source filter derives from immutable
pinned added-token data and cannot be weakened by editing a tokenizer's special-token
list.

Every accepted example must equal the official template rendering byte for byte.
The complete rendered text is tokenized once, without extra special tokens or
truncation. Explicit character ownership plus real fast-tokenizer offsets decides
labels; prefix lengths and substring matching do not decide boundaries. Overlength
examples and tokens crossing masked/supervised boundaries are rejected.

The Qwen tokenizer uses NFC normalization and original Python-codepoint offsets.
Some composed/reordered combining marks are omitted from raw offsets. The adapter
accounts for complete canonical combining sequences conservatively, requires one
owner and loss class per sequence, and rejects unexplained gaps. Leading combining
marks crossing role/reasoning/answer boundaries are rejected. Unsupported patterns,
including the observed Hangul Jamo composition gap, are explicit rejections; this
is not a claim of universal Unicode normalization support. Both original offsets
and expanded ownership offsets remain auditable.

Gemma preserves the qualified combining-mark and Hangul Jamo sequences in its
original offsets. The same conservative ownership checks apply. Its BOS, thinking
preamble, role headers, and trailing newline are masked; the selected model
turn's answer and `<turn|>` enter loss. Absent or empty reasoning produces no
thought wrapper. Whitespace-only reasoning follows the official template's
truthiness and is preserved in supervised/masked input. Historical reasoning is
omitted where the official template omits it.

## Integrity, grouping, and limits

The Rust bridge shares the existing complete v2/v3/v4/v5 artifact verifier: exact schema,
authoritative metadata, every batch, required values, message/task validation,
sorted unique IDs, counts, and logical identity. Missing legacy metadata is an
error for the bridge. Its versioned report carries the verified `ExportArtifact`,
raw byte length, and raw BLAKE3 digest. Python verifies that report and decodes the
same captured bytes through PyArrow `BufferReader`; replacing the source path
cannot change the consumed snapshot. The Python `VerifiedSnapshot` constructor is
disabled: only successful `verify_snapshot`/`read_snapshot` calls create one. Its
captured state is immutable, report access returns a defensive copy, and `build`
rejects snapshot lookalikes. Caller-supplied byte/report pairs cannot acquire
verified authority by recomputing a raw digest. Footer declarations do not reconstruct
lifecycle eligibility or independently establish rights.

### Complete prepared input contract

The version-one wire is `GWSFT001`, a 32-byte BLAKE3 digest, two big-endian unsigned
64-bit lengths, exact UTF-8 JSON payload bytes, and the exact source Parquet bytes.
The digest uses derive-key domain `ghostwriter.prepared-sft-input.v1` over both
lengths and all payload/source bytes. The maximum complete input is 256 MiB.
Trailing bytes, stale digests, malformed UTF-8, duplicate JSON fields, floating-point
JSON numbers, unsupported versions, and unknown payload fields are rejected.
Original decimal task/oracle values remain in captured Parquet and original task
JSON strings; this framing requires no second cross-language numeric hash convention.

The identity binds every ordered example and all IDs, attention, labels, rendered
text, ownership spans/offsets, source record/target/task/component references,
rejections, supervised/context counts, and shifted supervision/answer counts. It also
binds tokenizer files, official template, wrapper/backend policy, adapter source,
dependencies, producer runtime, layout, length, reasoning, and turn policies.
The `tokenizer_target` is the tokenizer repository/revision. Student weights and
execution/decision lineage remain explicitly `unbound`; semantic screening stays
`not_run`, and effective prompt separation stays `unknown`.

Current preparation emits recipe version 2, with an explicit closed profile name
and renderer controls. The frame, payload, and manifest remain version 1. Rust
checks exact profile commitments for tokenizer files, wrapper/backend policy,
dependencies, vocabulary, and controls, including Gemma's BOS, thinking preamble,
and end-of-turn structure. These structural checks do not execute tokenization.
Recursive profile source and policies contribute to the installed preparation
source identity.

Rust checks all framing, shape, ownership, accounting, source rows, target partitions,
and screened policy/component bindings before returning any imported payload. It
reverifies the actual embedded Parquet; an invented report plus a recomputed outer
digest cannot establish source verification. Rust reports `tokenizer_replay=not_run`:
it does not execute the official tokenizer or prove the producer's rendering choices.
The Python loader reconstructs the entire build from the captured verified source
under the pinned adapter/tokenizer/dependencies and compares every semantic field,
recipe/example identity, rejection, and count before exposing any examples.

Replay preserves the recorded producer runtime and original identities while recording
its own runtime separately in `replay.json`. A different OS or Python 3.12 patch alone
does not invalidate a build. Tokenizer, adapter source, dependency, and policy pins
must still match, and all rendered features must replay exactly. This compatibility
rule has regression coverage using altered producer-runtime declarations; execution
qualification remains macOS ARM64 CPU only. A later adapter implementation with a
different source identity requires preparation under that implementation.

Historical version-one `.gwsft` files keep their exact bytes and identities and
remain inspectable through `gw artifact verify-prepared`. The current installed
adapter explicitly rejects replay of an older recipe or preparation-source hash.
Prepare a new artifact from its captured canonical source with the current
profile. No historical artifact is relabelled or overwritten during migration.

Declared task groups/splits/rights are retained. Expanded prefixes keep their
source group. A declared validation or test role is a counted record rejection
from SFT preparation. A declared train role is still only a declaration. For
v2/plain records, grouping falls back to the source artifact/record and split
qualification stays unknown. Missing split evidence never becomes a train claim.
The manifest keeps provenance/rights and model execution lineage (#39/#45), grouped
split qualification (#41), contamination screening, and held-out training benefit
explicitly unknown where unproven. No bypass or evaluation adapter is included.

## Source-screened inputs

`gw gen export-screened` publishes metadata version 3 over the existing reviewed-task columns.
Its witness retains the complete text-free source-screening plan, declared run corpus, protected
input identities and coverage, population bindings, exclusions and emitted component members.
The actual command reruns the pure planner against captured database/protected inputs and checks
all declared database membership inside publication transactions. Its `transaction_checked`
witness is separate from the nested planner's `supplied_files_only` report.

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

The adapter verifies the same captured bytes through Rust and checks strict version-specific
metadata. Screened inputs currently require `open_ai_messages` with exactly matching CoT and turn
policies. `all_assistant` requires `assistant_prefix_v1`; `final_turn_only` requires
`full_conversation_final_v1`. These checks run before row decoding, including zero-row inputs.
The export-format target `gemma4` and mismatched policies fail before an output
directory is created. The `gemma4_e2b_text_v1` preparation profile consumes the
canonical `open_ai_messages` source through the same checks. Raw metadata version 1
retains its historical consumer behavior.

Each expanded example uses the frozen connected component as its group, retains the original
`declared_task.group`, and carries plan, policy, input, population and component identities.
Those identities enter the recipe, example and build identities. The manifest reports the supplied
source lexical result and transaction population check while retaining `semantic_screening=not_run`
and `effective_prompt_separation=unknown`. The pinned Qwen template can omit historical reasoning
and collapse source-distinct final prompts. The tested collision does not upgrade that claim.
Effective separation would require the entire held-out/excluded corpus, which this Train-only
artifact does not supply. Tool support and empirical contamination/benefit qualification remain
outside this adapter's current evidence.

## Tests and fixture provenance

```sh
GW_TRL_TOKENIZER="$PWD/qwen3-tokenizer" GW_TRL_GW="$PWD/target/debug/gw" \
  adapters/trl/.venv/bin/python -m pytest -q adapters/trl/tests \
    --ignore=adapters/trl/tests/gemma --ignore=adapters/trl/tests/gemma_lora \
    --ignore=adapters/trl/tests/gemma_comparison \
    --ignore=adapters/trl/tests/gemma_cuda_lora
GW_TRL_GEMMA_TOKENIZER="$PWD/gemma-e2b-tokenizer" GW_TRL_GW="$PWD/target/debug/gw" \
  adapters/trl/.venv-gemma/bin/python -m pytest -q adapters/trl/tests/gemma
```

These tests require the real pinned tokenizer and local Rust verifier; missing
qualification inputs fail rather than skip. They cover fixed independently audited
loss-token positions for repeated identical role/channel text, Unicode/NFC and BPE
boundaries, exact/overlength inputs, all policies and turn layouts, source/manifest
identities, real collator and trainer tensors, and source-path replacement.

`tests/fixtures/*.parquet` are small synthetic Rust-generated artifacts, with no
provider output or third-party training data. They cover empty/nonempty v2/v3,
raw JSON escape preservation, 1025 rows, reviewed task declarations, and held-out
roles, and conversations that produce unequal examples above 1024 tokens. Tests independently mutate the footer, schema, values, IDs, messages,
counts, hashes, and task JSON. The Rust verifier's existing corruption tests also
exercise both path and snapshot readers, including required nulls and later
batches. To intentionally regenerate the synthetic fixtures from their checked
Rust source:

```sh
GW_REGENERATE_TRL_FIXTURES="$PWD/adapters/trl/tests/fixtures" \
  cargo nextest run -p gw-storage -E 'test(snapshot_)' --locked --profile ci
```

Screened fixtures additionally pass through the actual Rust CLI, full trusted planner rerun and
SQLite publication/recovery. They cover prefix/final layouts, empty output, incompatible targets,
cross-task components, and source-distinct prompts that collide under the pinned template. Python
independently mutates witness versions, membership, components and qualification claims. The actual
installed CLI reaches the real collator and SFTTrainer dataloader with both supported layouts.
Regenerate only these synthetic fixtures with:

```sh
GW_REGENERATE_SCREENED_TRL_FIXTURES="$PWD/adapters/trl/tests/fixtures" \
  cargo nextest run -p gw-cli -E 'binary(screened_export)' --locked --profile ci
```

Historical prepared fixtures (`prepared-all.gwsft`, `prepared-empty.gwsft`, and
`prepared-long.gwsft`) remain unchanged for native compatibility and explicit
installed-replay rejection. `prepared-gemma.gwsft` is a separate current Gemma
output over the same synthetic source, inspected directly by Rust. Python tests
prepare private current fixtures through the selected installed profile; Rust
and Python independently consume the same `tests/prepared_cases.json` mutation
specification.
The corpus includes recomputed-hash attacks on later examples, source/component and
count mismatches, missing/duplicate/reordered examples, malformed framing/JSON, and
structurally valid token/recipe edits that only official replay can reject. Tests also
check all-rejected and empty inputs, unchanged IDs across producer/runtime differences,
and complete loaded features through the real collator and SFTTrainer dataloader.
Fail-if-called sentinels cover forward, generation, training, and optimizer steps.
To produce current synthetic fixtures after installing the adapter, choose a new
output directory. Existing historical files are never replaced:

```sh
cd adapters/trl
GW_TRL_GW="$PWD/../../target/debug/gw" .venv/bin/python -m tests.regenerate_prepared \
  --profile qwen3_text_v1 --tokenizer "$PWD/../../qwen3-tokenizer" \
  --output "$PWD/prepared-current"
```

## Completed full-SFT checkpoints

This checkpoint path retains its Qwen-only release scope and Qwen environment.
Gemma preparation does not qualify Gemma model loading or training.

`ghostwriter-trl-train` consumes a verified `.gwsft` input, trains all parameters on
one CPU process, and publishes one complete `.gwckpt` file. The original input
`build_id` remains unchanged; a separate `completion_id` covers the training
declarations, initial model, saved model, and exact original prepared input.

The public loader accepts one application-owned approved release start:

| Property | Fixed scope |
| --- | --- |
| Repository | `Qwen/Qwen3-0.6B` |
| Revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Role / purpose | Student / Training |
| Rights evidence | Pinned publisher Apache-2.0 license and model card |
| Declared parent | `Qwen/Qwen3-0.6B-Base`; exact parent revision remains unknown |
| Approval identifier | `qwen3_0_6b_c1899de_student_training_v1` |

This scoped release selection does not satisfy or change the separate generic
lineage assessor's resolved-parent requirements. Arbitrary local weights and
caller-supplied eligibility declarations cannot select this approval. The six
pinned tokenizer/license/card files plus `config.json`, `generation_config.json`,
and `model.safetensors` must already be present in a dedicated local directory.
All nine files are captured and checked against fixed sizes and SHA256 commitments
before a tokenizer or model loads. Extra files, pickle weights, remote-code
configuration, missing/extra parameters, conflicting tied weights, nonfinite
values, and unsupported conversions are rejected.

The safe loader constructs the known Qwen3 class directly and loads every
parameter strictly on CPU. BF16-to-F32 conversion is exact. Distinct trainable
parameters exclude the tied output-head alias; both names contribute to the
logical tensor content identity, and a serialized duplicate must contain equal
values. An initialized model object or a saved inspection report cannot replace
fresh approved loading.

```sh
ghostwriter-trl-train train \
  --prepared prepared-sft/prepared.gwsft \
  --release-directory qwen3-approved-release \
  --gw "$PWD/target/debug/gw" --output completed.gwckpt \
  --max-steps 3 --batch-size 1 --accumulation 2 \
  --learning-rate-millionths 100 --max-sequence-length 2048

gw artifact verify-checkpoint --stdin < completed.gwckpt

ghostwriter-trl-train reload --checkpoint completed.gwckpt \
  --tokenizer-directory qwen3-tokenizer --gw "$PWD/target/debug/gw"
```

Training uses FP32, AdamW, a constant learning rate, and a sequential sampler that
repeats complete epochs. Epoch tails retain partial batches and flush partial
gradient accumulation. Supported bounds are 1–32 optimizer updates, 1–8 examples
per microbatch, 1–8 accumulated microbatches, and complete sequences of at most
2048 tokens. There is no packing or truncation. Overflow is rejected before
training. Full pretrained-model optimization can require substantially more
memory than its weight file; the actual release has not been loaded or trained
as part of local software qualification.

Accounting is recorded after each successful real forward/backward call. It
counts actual nonpadding inputs, consumed example occurrences, and nonmasked
labels after the causal shift. Successful underlying optimizer calls are counted
separately from microbatches and Trainer progress. Dataset totals and planned steps
do not substitute for execution observations.

The completion contains exactly five files: initial config and safetensors,
final config and safetensors, and the original `.gwsft`. Strict framing carries
their exact lengths and hashes; model content identities normalize BF16 to exact
F32 bits. Native inspection streams large tensor payloads, verifies all finite
values and shapes, checks initial/final architecture agreement, verifies the
embedded prepared source, and independently reconstructs the declared batch
schedule. It accepts only final full inference weights. No pickled training
arguments, optimizer state, or resumable-training claim is saved.

Publication occurs only after native inspection, official tokenizer replay, a
fresh safe model reload, and trained-versus-reloaded logit agreement. The output
is linked atomically without replacing an existing file or symlink. Failed
optimization, save, verification, or reload produces no completed publication.
After the link is created, a directory open or synchronization failure retains the
complete checkpoint and raises `PublishedCheckpointError`. Its detached `report`
includes the completion and prepared input identities with
`status=published_durability_unknown`. A cleanup failure after successful directory
synchronization instead reports `status=published_cleanup_failed` and
`durability=confirmed`. Additional cleanup errors preserve the original outcome;
no such failure returns an `ObservedCompletion` receipt.

The training command returns exit code **3** for either post-publication outcome,
writes the publication report as one JSON object on stdout, and explains the failure
on stderr. Inspect the retained file with `verify-checkpoint` or `reload` and compare
its `completion_id` with the report before using it: another actor may have replaced
the pathname. The producer never deletes or replaces that destination during error
recovery. Retrying an existing destination is rejected before optimization; a
deliberate new training run needs an unused destination. Cleanup errors may leave
temporary files for inspection. Safe reloading confirms the captured file's current
contents, while historical training remains `declared`.

`ObservedCompletion` is a private-factory receipt for the current successful
producer call. `read_checkpoint` returns a separate `ReloadedCheckpoint`: its
current safe load and tokenizer replay are verified, while historical training
remains explicitly `declared`. Neither a content hash nor reloading authenticates
historical optimizer execution.

The training implementation has its own recursive source identity in a separate
subpackage and console entry point. Training source is excluded from the
preparation source identity, so optimizer-only changes do not require rebuilding
otherwise compatible prepared inputs.
Tiny random Qwen3 models live only in the test fixtures; no installed command
selects fixture authority. Qualification covers actual repeated/partial and long
training batches, independent supervision counts and reload logits, BF16/tied
aliases, rehashed corruption attempts, publication races, and injected failures.
The exercised Python platform remains CPython 3.12 on macOS ARM64 CPU. Pretrained
acquisition/training, Linux Python training, CUDA, and learning benefit are unqualified.

## Fresh numeric rewards

Numeric rewards retain the Qwen tokenizer and Qwen environment described below.

The `ghostwriter_trl.reward_artifact` and `ghostwriter_trl.rewards` modules support one
synchronous numeric reward callback for the pinned
[GRPOTrainer](https://github.com/huggingface/trl/blob/fd74bbc7b5f852a70d4cc94377e0a8f94392fda1/trl/trainer/grpo_trainer.py).
This is offline software and reward-dispatch qualification. A real RLVR experiment still needs
reviewed corpus/oracle quality, qualified splits, an eligible SFT-derived student, checkpoint/build
lineage, authorized compute, and a held-out comparison.

### Freeze task authority

```sh
gw reward export --tasks examples/reviewed-numeric-tasks.json > numeric-corpus.json
gw reward verify --stdin < numeric-corpus.json
```

Repeat `--tasks` for additional ordered task documents. All documents and cross-document duplicate
or conflicting declarations are checked before projection. Export selects each unique declared
Train task once and requires training rights assertions, passing reviewed QC assertions,
authoritative literal answers, and absent execution policy. The declarations retain their existing
evidentiary limits; export does not screen splits or prove corpus quality. Held-out tasks never enter
the artifact. Teacher Parquet siblings, assistant turns, reasoning, grades, and execution reports
are not accepted as task sources.

The version 1 self-contained artifact retains the complete reviewed task declaration, semantic
task identity, ordered corpus identity, and reward contract identity. The contract pins the existing
verification interpretation. The corpus identity includes source, rights, group/split, prompt,
oracle/extraction/tolerance, and review observations; metadata changes cannot reuse a prior corpus
receipt. The separate snapshot receipt binds every captured input byte, including whitespace.
Python reads the corpus once, sends those bytes to Rust, checks the receipt, and retains immutable
captured state. Public properties return detached copies. Both trainer-row projection and callback
construction require the exact factory-created corpus type and reject subclasses before tokenizer
validation or prompt projection.

Within this reward snapshot, `task.verification.numeric.tolerance.absolute` and `relative` use exact
finite binary64 objects, for example `{"binary64":"3ff89d89d89d89d9"}` for the selected binary64
representation of `20/13`. Each object contains exactly one key with 16 lowercase hexadecimal digits.
The codec preserves adjacent finite values and signed zero, rejects nonfinite values, and detects
duplicate keys before map conversion. Reviewed task-input documents retain their existing decimal
JSON number format; historical semantic task identities and other contracts retain their encoding.
The new snapshot preserves the exact values obtained from that intake rather than reparsing decimal
tolerances at each replay. Python forwards these bit objects unchanged and implements no comparator.

### Connect the callback

Use the same locally verified tokenizer and dependency pins as SFT preparation. In an application
that already supplies its authorized model and GRPO configuration:

```python
from pathlib import Path
from datasets import Dataset
from trl import GRPOTrainer
from ghostwriter_trl.tokenizer import load_tokenizer
from ghostwriter_trl.reward_artifact import read_numeric_corpus, reward_rows
from ghostwriter_trl.rewards import NumericRewardCallback

gw = Path("target/debug/gw").resolve()
tokenizer = load_tokenizer(Path("qwen3-tokenizer"))
corpus = read_numeric_corpus(Path("numeric-corpus.json"), gw)
reward = NumericRewardCallback(corpus, tokenizer, gw, timeout_seconds=30.0)
trainer = GRPOTrainer(
    model=model, args=grpo_config, processing_class=tokenizer,
    reward_funcs=[reward], train_dataset=Dataset.from_list(reward_rows(corpus, tokenizer)),
)
reward.bind_trainer(trainer)
```

The selected configuration requires one local process, `remove_unused_columns=False`, `beta=0`,
`use_vllm=False`, `use_transformers_continuous_batching=False`, no tools/environment/custom rollout,
and `chat_template_kwargs={"enable_thinking": False}`. Use valid GRPO batch/group sizes. Offline
qualification also disables reporting, Hub push, mixed precision, and gradient checkpointing.
Binding checks the actual trainer and repeats effective configuration checks on each callback.
Distributed execution needs a separately qualified coordinated-failure contract.

Trainer rows contain only one typed user message and the separate `gw_reward` identity references.
The callback keeps the oracle and full provenance in its verified corpus. The template is rendered
explicitly without thinking. This prevents exporter-induced oracle leakage; human corpus review
must still detect answers embedded in source prompt prose.

### Complete batch contract

TRL's real RepeatSampler can repeat task rows. Each callback instance creates a fresh run namespace.
Every callback entry allocates a monotonically increasing batch sequence and ordered positions;
failed calls consume a sequence too. Identical completions and
repeated tasks still have distinct reward attempts. These identities record reward evaluation and
do not establish model-generation or checkpoint lineage.

Every request binds corpus/task/reward-contract identity, fresh attempt, raw unpadded completion
IDs, exact decoded UTF-8 content, tokenizer identity, and decode policy. With Transformers 4.56.2 the
supported conversational completion is exactly one plain assistant string. Decoding uses
`skip_special_tokens=True` and disabled cleanup. The adapter checks raw IDs before decoding: all
26 pinned control tokens are rejected except one terminal pinned EOS. Invalid IDs, reasoning,
tools/media, extra turns, raw control literals, and mismatched token/text evidence are rejected.
Thus a role or media token hidden by skip-special decoding cannot become ordinary answer evidence.

The callback invokes `gw reward evaluate --stdin` once for the whole batch, with a finite timeout.
The strict version 1 request carries `artifact`, `completion_policy`, the effective
`mask_truncated_completions`, and ordered `items` containing `binding` and `completion`.
Rust validates every binding before evaluating any results. Its pure evaluator is shared with the
existing verifier: finite binary64 parsing and the declared extraction/tolerance settings determine
factual Pass, Fail, or Unknown from assistant content alone. Pass maps to `1.0`, Fail to `0.0`, and
Unknown carries `null`. Missing/malformed oracles are rejected at task intake; the pure evaluator
also preserves Unknown for unavailable oracle evidence.

The report binds the exact request bytes and policy, retains the mask setting, and returns one
ordered result per item. Python validates the complete report before returning any rewards. Unknown,
invalid/stale bindings, wrong order/cardinality, partial output, timeout, and evaluator failure all
abort the whole callback. No row is omitted, resampled, assigned a substitute zero, or returned as
`None`. `last_report` retains only a completely successful batch and is cleared at the next entry.

Termination is separate: a final pinned EOS gives `observed_eos`; every other supported sequence
retains `unknown`. No EOS or length equal to the configured cap does not identify a stop cause.
Numeric Pass/Fail can coexist with unknown termination. The adapter records and preserves the
effective truncation-mask setting without changing rewards based on an inferred stop cause.

Rust declares and validates tokenizer policy and completion hashes without loading a tokenizer.
The qualified Python adapter additionally checks the actual pinned tokenizer and token-to-text
decoding. A direct CLI caller's declared tokens/text are caller-supplied evidence. Hashes establish
binding and integrity, not authenticity. `gw-schema` stays pure and performs no I/O; CLI input/output
is local and provider-free, with a 64 MiB limit per captured input. Semantic errors emit no success
JSON; callers must check exit status and complete report parsing.

### Offline dispatcher evidence

`tests/test_rewards.py` constructs the real CPU GRPOTrainer with a tiny random GPT-2 model and the
pinned tokenizer. Its real data loader preserves repeated task metadata, and its unmodified
`_calculate_rewards` method consumes synthetic completions. Literal expected answers assert the
finite CPU float32 batch-by-one tensor and exact order. Fail-if-called sentinels cover model forward,
generation, preparation, prediction, evaluation, training, and optimizer construction.

The real dispatcher allocates an empty reward tensor before invoking the callback. Failure checks
prove that no numeric rewards reach its later tensor assignment or gather. Fixtures cover a valid
first row followed by an invalid row, Unknown, timeouts, partial/stale/reordered output, all reserved
IDs, reasoning-only/prompt-only markers, stale token/text, and distinct attempts for repeated rows.
Rust independently checks literal grammar, tolerance, overflow, and extraction vectors; agreement
between two runtime paths alone is not the numeric correctness oracle. No pretrained weights,
inference, optimizer step, or learned improvement is part of these checks.

## Gemma E2B LoRA completion

The separate `ghostwriter-trl-lora` entry point consumes a verified
`gemma4_e2b_text_v1` prepared build. Its exact environment is
`requirements-gemma-lora.lock`: the 64 Gemma preparation pins plus PEFT 0.21.2.
Install the `gemma-lora` extra in a separate CPython 3.12 environment using the
same hash-checked installation procedure above.

The supported recipe uses the official `Gemma4ForConditionalGeneration` text
path, rank 8, alpha 8, zero dropout, no bias, and only the exact text-attention
`q_proj` and `v_proj` projections. The selected full configuration has 35 q and
15 v projections because later layers share KV state. Embeddings, output head,
per-layer embeddings and projections, and all modality tensors remain frozen.
Every base parameter and persistent buffer must retain its captured bytes. The
only permitted infinite values are the directed scalar clipping bounds in the
exact official vision/audio clippable-linear inventory. Weights, adapters,
losses and adapter gradients must be finite.

The CPU float32 recipe permits 2–32 AdamW updates, batch size and accumulation
1–8, complete sequences up to 2,048 tokens, and a learning rate of 1–10,000
millionths. Every adapter tensor must change. Defaults are two updates, batch
size one, accumulation one, and a learning rate of 100 millionths. It uses the
complete sequential batch order, explicit causal-shift labels, no packing, and
no truncation. The observed optimizer population must equal all and only the
selected adapters. Seed zero is applied before PEFT initializes the adapter. The
execution scope restores caller Python, NumPy and CPU PyTorch random state after
both success and failure.

Supply a local directory containing exactly the seven pinned Gemma tokenizer
files plus the approved `model.safetensors` (10,246,621,918 bytes; SHA256
`2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550`).
The overlapping `config.json` is captured once. The loader owns regular file
descriptors, bounds and streams capture, and rejects altered bytes, unexpected
files, pickle, remote code, and automatic Hub base resolution.

```sh
ghostwriter-trl-lora train \
  --prepared prepared-gemma/prepared.gwsft \
  --release-directory gemma-e2b-release \
  --gw "$PWD/target/debug/gw" --output completed.gwlora \
  --max-steps 2 --batch-size 1 --accumulation 2

gw artifact verify-lora --stdin < completed.gwlora

ghostwriter-trl-lora reload --checkpoint completed.gwlora \
  --tokenizer-directory gemma-e2b-tokenizer --gw "$PWD/target/debug/gw"
```

`GWLORA01` is a separate versioned framing and hash domain. It contains the base
once, initial and final adapter safetensors, strict inert adapter configuration,
and the original complete prepared input. Native inspection streams the captured
bytes and verifies shapes, counts, configuration, source and dependency bindings,
and the full declared batch order. It reports historical training as `declared`
and model reload as `not_run`. Qwen full-SFT framing remains separate and rejects
LoRA completions.

The producer additionally reloads a fresh base from the captured safe tensors,
attaches only the captured local adapter, compares actual tensor identities, and
checks finite matching logits before exclusive publication. Safe PEFT extraction
explicitly uses `save_embedding_layers=False`. Only a successful actual producer
returns a private observation receipt owning its fresh reload and binding the
base, adapter, prepared build, recipe and completion identities. Reading saved
JSON or successfully reloading an arbitrary completion does not grant observed
training authority. A failure after publication retains the complete target and
reports its exact identity without returning a successful receipt; cleanup never
deletes a replacement destination.

Local qualification uses an owned seeded random text-only Gemma fixture with the
real 262,144-token vocabulary, both attention kinds, different local/global head
widths, KV sharing, nonzero per-layer embeddings and tied embeddings. It has
8,402,844 distinct base parameters and 1,728 trainable adapter parameters across
six targets and 12 tensors. This checks actual CPU forward/backward, changed
adapters, unchanged frozen state, native inspection and fresh reload. Full-release
shape checks use the official architecture on the meta device. Neither result
establishes actual pretrained loading, CUDA/BF16 execution, accelerator memory
fit, learned benefit, or an immutable publisher-parent revision.


### Reference origins

Version-four `record_origins` and version-five `tool_definitions` rows carry exact `origin_json` into every prepared example.
The native verifier checks the reference module and suite bindings, declared Train use,
and absent judge fields. Preparation binds origin to the captured Parquet and uses the
reference component for grouping unless an explicit screened component is present.
Historical v2/v3 artifacts retain their exact columns and identities. These exported
declarations do not authenticate an external reviewer or reconstruct local registration.

### Paired Gemma coding comparison

`ghostwriter-trl-compare train-and-compare` runs the bounded CPU LoRA producer and
passes its live completion directly to the paired controller. A saved checkpoint,
inspection report, supplied model object or copied receipt cannot authorize this
fresh path. The controller captures the original checkpoint once, independently
loads an untouched base and a second base with its measured final adapter, and
rechecks both complete tensor inventories after generation.

```sh
ghostwriter-trl-compare train-and-compare \
  --prepared prepared-gemma/prepared.gwsft --release-directory gemma-e2b-release \
  --checkpoint-output completed.gwlora --output paired-test.json \
  --gw "$PWD/target/debug/gw" --db reference.sqlite --registration REGISTRATION_ID \
  --split test --max-steps 2 --max-new-tokens 128 --max-prompt-tokens 1024

ghostwriter-trl-compare inspect --artifact paired-test.json \
  --tokenizer-directory gemma-e2b-tokenizer --gw "$PWD/target/debug/gw"

ghostwriter-trl-compare replay --artifact paired-test.json \
  --tokenizer-directory gemma-e2b-tokenizer --gw "$PWD/target/debug/gw" \
  --db reference.sqlite --output paired-test-replay.json
```

The selected registration must already contain the complete committed reviewed
112-member reference import. Rust captures **all 32 Test members or all 16
Validation members**, in accepted order, with private cases held inside the native
process. Completeness applies to that exact split. Generation receives public
problem text and redacted task, family, component, split and suite identities;
private inputs, expected values, reviews and held-out reference solutions are absent.
The capture includes compact native-derived content hashes of all 64 accepted Train
records. Python recomputes each hash from the verified prepared messages, task,
origin and stable record identity; a caller-supplied digest cannot replace this check.
Every complete bridge message is bounded to 32 MiB of serialized UTF-8, including
the newline on native output. Both sides enforce the opening population limit.
An actual release comparison requires the prepared build to cover all 64 accepted
Train members from that same registration/import. The owned random fixture can
use explicit synthetic software training. Both paths check task/family/component
separation and exact prompt IDs rendered under the generation recipe. Source
screening remains declared and semantic screening remains `not_run`. A held-out
prompt rejected by the text protocol retains both Unknown rows and marks effective
prompt separation incomplete. All renderable prompts still undergo collision checks.

Version one uses the actual pinned Gemma processor with
`add_generation_prompt=True`, `enable_thinking=False` and
`preserve_thinking=False`. Preparation retains its separate existing rendering.
Each model receives one complete unpadded prompt per member, greedy decoding,
one beam and sequence, the same fixed suffix bound and system text, float32 CPU
parameters, eager attention, one thread, deterministic algorithms and a fresh
ordinary dynamic cache. Compilation is disabled. The full effective generation
configuration is saved, including inactive defaults. There is no truncation.

The report retains prompt IDs, attention masks, full returned sequences, exact
suffix/body IDs and unfiltered decoded text. Stops are IDs 1, 106 and 50. Only one
actually observed final 1 or 106 is removed; tool handoff 50 stays in the body and
fails the representation gate. A terminal exactly at the token bound records both
facts. A complete suffix at the bound remains exact module bytes even without a
terminal. Unexpected short returns or prefix mismatches remain Unknown. Added
control tokens and control spellings, empty bodies and oversized bodies fail the
representation gate. Whitespace, Unicode, syntax errors and ordinary invalid code
are preserved for native execution; the controller does not extract or repair code.

Rust validates both ordered answer sets before running their modules through the
existing external-oracle coding observer and verifier. Every item remains in the
report, with native case coverage or an explicit failure/Unknown reason. Counts
retain Unknown separately; a pass-count difference is present only when both
whole populations are comparable. No result automatically promotes a candidate.
Cancellation waits for the owned native process and its container cleanup.
Publication validates first, then atomically links synced complete bytes without
overwriting. Errors after linking report the retained artifact identity and
uncertain durability or cleanup; they never remove a replacement target.

Saved inspection checks complete bindings, representation, arithmetic and actual
tokenizer decoding. Historical training, generation and execution stay `declared`.
Saved replay freshly executes the exact modules against current registered native
oracles and compares stable case results; it never reloads or regenerates a model.
The native `gw eval coding-pair --stdio` bridge likewise grants only fresh native
execution, not authenticity to caller-supplied model history.

The local qualification uses the reduced random CPU Gemma fixture and cached
native ARM64 Docker runtime. It covers complete paired generation/execution,
positive/wrong/syntax/Unknown controls, replay, copied receipt and altered artifact
rejection, cancellation and publication settlement. It does not establish learned
benefit, an actual pretrained comparison, GPU execution, accelerator memory fit,
semantic decontamination or an immutable publisher-parent revision.

The complete paired qualification additionally needs an explicit synthetic store fixture:

```sh
PAIR_FIXTURE=$(mktemp -d)
PAIR_REJECTED_FIXTURE=$(mktemp -d)
PAIR_LARGE_FIXTURES=$(mktemp -d)
GW_PAIR_TEST_DIRECTORY="$PAIR_FIXTURE" cargo nextest run -p gw-cli --locked \
  --test coding_pair --retries 0
GW_PAIR_TEST_DIRECTORY="$PAIR_REJECTED_FIXTURE" GW_PAIR_TEST_REJECTED_PROMPT=1 \
  cargo nextest run -p gw-cli --locked --test coding_pair --retries 0
for metadata in train heldout; do
  GW_PAIR_TEST_DIRECTORY="$PAIR_LARGE_FIXTURES/$metadata" GW_PAIR_TEST_LARGE_METADATA="$metadata" \
    cargo nextest run -p gw-cli --locked --test coding_pair --retries 0
done
GW_PAIR_TEST_DIRECTORY="$PAIR_FIXTURE" GW_PAIR_REJECTED_TEST_DIRECTORY="$PAIR_REJECTED_FIXTURE" \
  GW_PAIR_LARGE_TEST_DIRECTORY="$PAIR_LARGE_FIXTURES" \
  GW_TRL_GW="$PWD/target/debug/gw" \
  GW_TRL_GEMMA_TOKENIZER="$PWD/gemma-e2b-tokenizer" \
  adapters/trl/.venv-gemma/bin/python -m pytest -q adapters/trl/tests/gemma_comparison
```

This store models the prior native reference-import boundary with explicitly synthetic
observations. Its paired tests perform fresh actual Docker execution. The separate
ignored `reference_full_population_cached_runtime_roundtrip` test qualifies actual
112-member reference import. Preserve that distinction when reporting evidence.

### Separate CUDA checkpoint policy

`ghostwriter-trl-cuda-lora` consumes the same complete verified Gemma prepared
input and exact local approved release bytes. It uses the distinct `GWCUDA01`
frame and `ghostwriter.completed-gemma-cuda-lora.v1` digest domain. Existing CPU
LoRA and comparison commands retain their v1 contracts and source identities.

The CUDA policy requires one BF16-capable Linux x86_64 CUDA device and a standalone
process. Frozen parameters use BF16; the exact rank-8 q/v adapters, their
gradients and AdamW moments use FP32. Floating buffers stay FP32, including
nonpersistent RoPE buffers. The loader preserves tied aliases and frozen towers.
Training uses eager attention, BF16 autocast, explicit complete labels and
sequential batches. TF32, quantization, checkpointing, compilation, offload and
automatic batch reduction are disabled. Actual operator observations must show
the FP32 normalization, RoPE, softmax and loss operations.

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 ghostwriter-trl-cuda-lora train \
  --prepared prepared-gemma/prepared.gwsft \
  --release-directory gemma-e2b-release \
  --gw "$PWD/target/debug/gw" --output completed-cuda.gwckpt \
  --max-steps 2 --batch-size 1 --accumulation 1 --max-sequence-length 2048

gw artifact verify-cuda-lora --stdin < completed-cuda.gwckpt

ghostwriter-trl-cuda-lora reload --checkpoint completed-cuda.gwckpt \
  --tokenizer-directory gemma-e2b-tokenizer --gw "$PWD/target/debug/gw"
```

No acquisition occurs. The installed named dependency versions must exactly match
`requirements-gemma-lora.lock`; CUDA wheel local-version suffixes are not
normalized. The actual Linux CUDA wheel/image closure and device must be
qualified separately. The separate CUDA policy checks the 15 NVIDIA/Triton versions declared by Torch
2.8.0 for CUDA 12.8. It records runtime build facts and commitments to installed
distribution METADATA/RECORD files; those are installed-wheel declarations, not
independent image attestation. Driver process-residency measurement requires `nvidia-smi`.
The commands above describe the implemented interface, not completed hardware
qualification or an acquired pretrained release.

The native reader checks original source bytes, derives BF16 frozen-parameter
hashes, and verifies FP32 persistent-buffer and adapter hashes. Nonpersistent
buffer values remain declared until independent model allocation reproduces the
complete typed state. Saved training observations always remain `declared`.
The producer requires an exact independent reload probe under its separate
`cuda_bf16_exact_probe_v1` policy; it does not inherit the CPU floating tolerance.

A successful Python `cuda_lora.producer.train(...)` returns a model-free
`ObservedCompletion`. It owns a private complete capture, survives replacement
of the public output path, and supports a context manager or explicit `close()`.
`cuda_lora.ownership.consume(...)` transfers the capture once and revokes further
live use. Saved paths, reports and reloaded models cannot create this capability.
The CLI prints the receipt and closes its capture. Training and verification
models are released sequentially before issuance; allocator peaks, host RSS and
this process's driver-reported GPU residency are recorded. Python, NumPy, CPU
Torch and selected CUDA RNG states and changed backend flags are restored.

The explicit `--owned-fixture-tokenizer` alternative generates the reduced
random official architecture locally. Real-device qualification controls are
opt-in:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 GW_TRL_CUDA_QUALIFICATION=1 \
  GW_TRL_GW="$PWD/target/debug/gw" \
  GW_TRL_GEMMA_TOKENIZER="$PWD/gemma-e2b-tokenizer" \
  python -m pytest adapters/trl/tests/gemma_cuda_lora -q
```

Host controls cover typed state, aliases, private-capture lifetime, native
accept/reject cases and unsupported-host rejection. Opt-in CUDA controls add
real updates, padding/partial accumulation, independent reload and RNG recovery.
Host controls do not establish CUDA execution, memory fit, throughput or model
quality. Sequential CUDA comparison and an owned remote controller are separate
interfaces and are not provided by this checkpoint command.


### Complete tool artifacts (column schema v5)

New Parquet publications use `tool_definitions`: the v4 columns plus nullable UTF-8
`tools_json`. SQL null preserves absent definitions; `[]` preserves an explicit empty list.
The column contains canonical JSON with sorted object keys, ordered definitions, and
unchanged nested JSON values. `messages_json` retains null assistant content, parallel calls,
explicit result IDs, reasoning, and raw argument evidence. The new row identity binds
actual definition bytes even when a producer supplies an unchanged `record_hash`.
Historical v2/v3/v4 receipts replay their original columns and identity domains.

Prepared examples from v5 copy the exact `tools_json` field, including explicit null.
Historical examples omit the field. Rust source verification rejects a missing or changed
copy. The current text adapter accepts v5 text sources with absent or empty definitions
and rejects nonempty tool definitions and tool trajectories before tokenization.

`gw_format::validate_tool_training_source` is a separate complete-source check. It requires
unique function definitions, supported explicit object parameter schemas, object arguments,
assistant calls with unique IDs, and exactly one later result linked by ID per call.
Parallel same-name calls and reversed result order are valid sources. Supported schema
keywords are `type`, `description`, `properties`, `required`, `items`, `enum`, and boolean
`additionalProperties`; references, compositions, and content parts are unsupported.
It checks schema structure and linkage, not argument conformance to JSON Schema.

`validate_tool_projection_delimiters` separately checks a consumer-supplied set of control
tokens recursively in messages, reasoning, parsed arguments, and definition keys/values.
Raw argument strings are evidence, never training targets. Canonical export applies neither
restriction and preserves source evidence. This artifact support does not qualify official
31B template bytes, serial/parallel consumer behavior, trainer labels, or student quality.
