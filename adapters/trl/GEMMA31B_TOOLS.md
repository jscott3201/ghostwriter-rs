# Official Gemma4 31B serial tool labels

`ghostwriter-trl-gemma31b` prepares complete canonical v5 text/tool records for
`gemma4_31b_tools_v1`. It uses the official
[google/gemma-4-31B-it release at 842da3794eaa0b77d5f08bae87a17459d91ff475](https://huggingface.co/google/gemma-4-31B-it/tree/842da3794eaa0b77d5f08bae87a17459d91ff475).
The package pins all seven tokenizer/processor files, the template, runtime
wrapper, backend, added tokens, and dependency versions. These pins identify the
tokenizer contract; student weights and training benefit remain unqualified.

## Supported source and loss

Validation runs on each **complete source record before assistant-prefix
expansion**. A supported conversation has an optional initial system message,
a user message, serial assistant/tool exchanges, and an ordinary assistant
answer. Later user/assistant episodes are supported. Every call has one explicit,
unique ID, exactly one defined function and object arguments. Its immediate
text-only tool reply must link that ID, followed by an assistant continuation,
further call, or final answer. Canonical null and empty-string call content both
work. Nonempty content beside a call, parallel calls, missing definitions,
incomplete links, multimodal content, and unsupported message order cause a
record-level rejection. The consumer never executes tools.

Definitions support function name/description and object parameters. Parameter
schemas support explicit primitive types, object properties/required fields,
array items, string enums, and descriptions. Constraints that the official
formatter would omit, such as `additionalProperties` or function `strict`, are
rejected. This checks schema shape and linkage, not argument conformance to a
JSON Schema. Unescaped names and object keys must use letters, digits or
underscores and begin with a letter or underscore; keys that collide after
lowercasing are rejected. All pinned control-token literals are rejected in
projected keys and values. Retained `raw_arguments` remains source evidence;
only parsed structured arguments are rendered.

Argument values may include nested objects/lists, strings, booleans, null,
integers from −2^63 through 2^64−1, and finite decimal floats. Floats must be zero
or have magnitude at least 0.0001 and less than 10^16. Scientific JSON spellings
within that value range, such as `1e3`, render according to the official template
as `1000.0`. Scientific output outside that qualified domain, nonfinite values,
overflow and duplicate JSON keys are rejected without rounding or coercion.

| Ownership | Labels |
| --- | --- |
| Current function name, structured arguments, call wrappers | Supervised |
| Current call handoff `<\|tool_response>` (token 50) | Supervised |
| Current ordinary answer and `<turn\|>` (token 106) | Supervised |
| Definitions, system/user text, earlier assistants | Masked |
| External observations and both response markers | Masked |
| Trailing newline and batch padding | Masked |
| Current reasoning and channel wrappers | Selected by `--cot` |

`--thinking` controls the official thinking preamble. `--preserve-thinking`
controls historical tool-call reasoning according to the official template.
Both decisions are explicit and bound into the recipe. Earlier reasoning stays
masked even when retained. `--cot stripped` removes reasoning; `masked` retains
it without loss; `supervised` includes reasoning belonging to the selected
assistant target. Reasoning remains separate from ordinary content in source.

Call targets record `target_kind: tool_call` and shifted call-token indices.
They require a whole function/argument token to survive the causal shift.
Ordinary answers keep the whole-answer-token requirement. Each full rendering
must match the pinned official processor output exactly before tokenization.
Unicode ownership checks reject ambiguous masked/supervised token boundaries.
Overlength examples become explicit target rejections with complete accounting;
truncation is disabled.

## Reproduce

Use the exact `requirements-gemma.lock` environment with CPython 3.12 and the
`gemma-31b-tools` extra. Acquire only the seven files named in
[`gemma31b/manifest.json`](src/ghostwriter_trl/gemma31b/manifest.json) at the pinned
revision into one local directory. No model weight download is needed.

```sh
python -m pip install --no-deps '.[gemma-31b-tools]'
ghostwriter-trl-gemma31b \
  --artifact canonical-v5.parquet --gw /path/to/gw \
  --tokenizer /path/to/exact-local-31b-files \
  --thinking on --preserve-thinking on --cot masked \
  --turns all_assistant --max-length 4096 \
  --output prepared-31b --qualify-handoff
```

The command captures and verifies the source, saves `prepared.gwsft`, performs
native verification and Python replay, and writes separate verification and
replay receipts. `--qualify-handoff` also checks the actual TRL collator and a
real `SFTTrainer` dataloader using a tiny random CPU model. A forward hook rejects
model execution; no forward pass or optimization is performed. Existing output
paths are refused. Python's `read_prepared` API supports later verified replay
of saved bytes and `qualify_prepared_handoff` consumes that opaque verified input.

Rust independently reconstructs rendered text and every ownership span from the
captured complete source. Rehashed edits to a call, argument, observation owner,
or handoff fail native verification. Python independently reruns the exact
processor/tokenizer and compares the complete build. Native verification does
not implement BPE; the two receipts state that distinction. The independent
consumer package binds both its own files and the shared preparation closure.
The Qwen and E2B preparation source identities remain unchanged by this addition.

Run the actual local qualification with:

```sh
GW_TRL_GW=/path/to/gw \
GW_TRL_GEMMA31B_TOKENIZER=/path/to/exact-local-31b-files \
python -m pytest tests/gemma31b
```

The literal numeric corpus is consumed independently by Rust and Python, including
raw duplicate-key cases. The serial fixture covers Unicode and combining marks,
heterogeneous nested arguments, call-only prefixes, external observations, and a
later text episode. Qualification covers all reasoning/thinking/preservation
combinations, malformed complete sources, overlength accounting, rehashed
ownership tampering, captured-file replacement, and actual trainer batch labels.
