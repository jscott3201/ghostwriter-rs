# Verified Qwen3 SFT label preparation

This external Python adapter reads one canonical Ghostwriter Parquet snapshot,
asks `gw artifact verify --stdin` to verify **those exact bytes**, and prepares
`input_ids`, `attention_mask`, and explicit unshifted causal language-model
`labels`. It also retains the original `messages_json`, artifact/record/target
identities, character and token ownership, and a versioned build manifest.

The supported candidate is
[Qwen/Qwen3-0.6B at c1899de289a04d12100db370d81485cdf75e47ca](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca).
The pinned official tokenizer and template are the rendering/token oracle. The
repository declares Apache-2.0; its parent model revision and full execution
lineage remain unresolved. This is label and trainer-handoff qualification, not
evidence of model quality or training eligibility.

## Reproduce the qualified environment

The checked lock was resolved and exercised on **CPython 3.12.14, macOS ARM64,
CPU**, with TRL 1.14.1, Transformers 4.56.2, tokenizers 0.22.0, PyArrow 21.0.0,
PyTorch 2.8.0, and Accelerate 1.4.0. It pins all 55 development dependencies and
public-PyPI wheel hashes. It does not qualify Linux, CUDA, other Python releases,
or another dependency solution. The package requires Python 3.12 and checks the
installed dependency versions before preparation.

From the repository root, with an installed CPython 3.12 interpreter:

```sh
uv venv --python python3.12 adapters/trl/.venv
uv pip sync --python adapters/trl/.venv/bin/python --only-binary :all: \
  --require-hashes adapters/trl/requirements.lock
uv pip install --python adapters/trl/.venv/bin/python --offline \
  --no-deps --no-build-isolation -e adapters/trl
cargo build -p gw-cli --bin gw --locked
```

Acquire only the six tokenizer/license/readme files listed in
[`tokenizer_manifest.json`](src/ghostwriter_trl/tokenizer_manifest.json), at that
exact revision. The following optional acquisition step uses the public Hub and
downloads no weights, model config, or generation config. Use a new, empty output
directory. Acquisition is separate from offline preparation.

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

The adapter rejects extra files in this directory. It captures and hashes all six
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

The output directory must not already exist. `examples.jsonl` contains the token
features and audit information; `manifest.json` contains policies, installed
versions, actual runtime platform, source report, identities, counts, and explicit
rejections. An empty or fully rejected input produces zero examples and explicit
counts. Inspect those counts before using any output. The adapter uses its own
versioned SHA256 JSON identities; it never reimplements Rust's framed logical
BLAKE3 artifact identity.

For a synthetic fixture, add `--qualify-handoff` to inspect both the actual
[TRL 1.14.1 collator and SFTTrainer](https://github.com/huggingface/trl/blob/fd74bbc7b5f852a70d4cc94377e0a8f94392fda1/trl/trainer/sft_trainer.py)
dataloader. This requires at least two unequal-length examples. It constructs a
small **random** CPU GPT-2 causal model covering the full tokenizer vocabulary,
then checks the real tensors. It performs no model forward pass, optimizer step,
pretrained-weight download, or cloud operation. The random architecture tests
the handoff only; it is not a Qwen training run.

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
| `stripped` | Reasoning is removed; the official empty thought wrapper is masked; answer and end-of-turn enter loss. |

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
nonredundant structured reasoning, and all **26 pinned added-token literals** are
rejected in every clean source channel, including before stripping. This includes
the six non-special FIM/repository tokens. The source filter derives from immutable
pinned added-token data and cannot be weakened by editing a tokenizer's special-token
list.

Every accepted example must equal the official template rendering byte for byte.
The complete rendered text is tokenized once, without extra special tokens or
truncation. Explicit character ownership plus real fast-tokenizer offsets decides
labels; prefix lengths and substring matching do not decide boundaries. Overlength
examples and tokens crossing masked/supervised boundaries are rejected.

The pinned tokenizer uses NFC normalization and original Python-codepoint offsets.
Some composed/reordered combining marks are omitted from raw offsets. The adapter
accounts for complete canonical combining sequences conservatively, requires one
owner and loss class per sequence, and rejects unexplained gaps. Leading combining
marks crossing role/reasoning/answer boundaries are rejected. Unsupported patterns,
including the observed Hangul Jamo composition gap, are explicit rejections; this
is not a claim of universal Unicode normalization support. Both original offsets
and expanded ownership offsets remain auditable.

## Integrity, grouping, and limits

The Rust bridge shares the existing complete v2/v3 artifact verifier: exact schema,
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
Gemma4 and mismatched policies fail before an output directory is created. Raw metadata version 1
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
  adapters/trl/.venv/bin/python -m pytest -q adapters/trl/tests
```

These tests require the real pinned tokenizer and local Rust verifier; missing
qualification inputs fail rather than skip. They cover fixed independently audited
loss-token positions for repeated identical role/channel text, Unicode/NFC and BPE
boundaries, exact/overlength inputs, all policies and turn layouts, source/manifest
identities, real collator and trainer tensors, and source-path replacement.

`tests/fixtures/*.parquet` are small synthetic Rust-generated artifacts, with no
provider output or third-party training data. They cover empty/nonempty v2/v3,
raw JSON escape preservation, 1025 rows, reviewed task declarations, and held-out
roles. Tests independently mutate the footer, schema, values, IDs, messages,
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
