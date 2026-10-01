# Verified Qwen3 SFT labels and numeric rewards

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

## Fresh numeric rewards

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
