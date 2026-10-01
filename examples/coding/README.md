# Owned Python coding controls

These small original functions qualify the saved-function evaluator. Their literal
expected results were reviewed independently of the implementations. They are not a
benchmark population or evidence of model learning.

| Family | Task role | Correct behavior | Deliberate error |
| --- | --- | --- | --- |
| `merge_closed` | Train | Sort and merge overlapping or touching closed intervals | Fails to merge touching intervals |
| `runs` | Train | Count consecutive identical Unicode codepoints | Counts characters across separate runs |
| `common_prefix` | Test | Find the prefix shared by every string, preserving exact codepoints | Examines only the first two strings |

The complete third family is held out. Empty values, negative endpoints, point
intervals, nonadjacent repeats, astral codepoints, and distinct Unicode normalization
forms provide explicit boundary cases. `syntax_error.py` is the parse-failure control.

The reviewed document is [`reviewed-coding-tasks.json`](../reviewed-coding-tasks.json).
It records source, owned-use assertions, review evidence, family groups, split
assignment, signature, visible examples, private cases, comparison, and runtime.
Visible examples enter the prompt. Training-private and protected cases contribute
only suite/case identities and their partition category to exported task metadata.
The exported partition category must agree with the task's declared split, so a
protected suite cannot be relabeled for training without changing its task semantics.
Each isolated invocation
receives its current arguments; expected results and the rest of the private suite
remain in the external controller.

## Run a saved function

The first recipe requires Docker 29.8.1 over the active context's local Unix socket,
Linux ARM64, cgroup v2, built-in default seccomp, and this already cached image:

```text
python@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f
```

The controller verifies the concrete image ID and platform, then probes CPython
3.12.14 as UID/GID 65534. Unsupported configurations fail explicitly. It never
pulls an image. Custom `DOCKER_HOST`, `DOCKER_CONTEXT`, and TLS environment overrides
are unsupported; select a supported local Docker context before invoking it.

```sh
cargo run -p gw-cli --locked -- eval coding \
  --tasks examples/reviewed-coding-tasks.json --task merge-closed \
  --candidate examples/coding/merge_closed.correct.py \
  --output /tmp/merge-closed-evaluation.json

cargo run -p gw-cli --locked -- eval coding-replay \
  --artifact /tmp/merge-closed-evaluation.json \
  --output /tmp/merge-closed-reobserved.json
```

Output files must be new and are created with private permissions. The evaluation
artifact contains exact captured code and the complete reviewed task, including
private oracles. Keep it separate from training exports. The command prints a
compact identity/outcome summary. Exit 0 means all cases passed; exit 2 means a
completed Fail or Unknown result; invalid input, unsupported runtime, and replay
mismatch exit 1. A Ctrl-C requests cancellation, waits for settlement attempts, and
saves Unknown when a report can be produced.

Replay validates schema, capture identities, runtime recipe, complete case coverage,
and internally consistent declarations before contacting Docker. It then executes
the captured input again and compares stable results. Every replay has fresh run
and container IDs. Matching checksums or saved success flags do not authenticate a
past run; matching fresh observations establish the new run only. A mismatch cannot
return the saved success.

## Value and execution contract

Candidates are complete UTF-8 modules of at most 64 KiB. There is one synchronous
module-level entry point with the exact ordered positional signature. Defaults,
variadics, positional-only and keyword-only arguments are unsupported. Functions
return `None`, exact `bool`, signed 64-bit `int`, `str`, `list`, or `dict` with string
keys. Floats, nonfinite values, subclasses, tuples, sets, and arbitrary objects are
unsupported. Booleans differ from integers; list order and length matter; object
key order does not. Strings retain exact codepoints without normalization.

Typed values are bounded to depth 16, 2048 nodes, and 16 KiB of UTF-8 text. Documents
and complete selected captures are bounded to 1 MiB; documents contain at most 64
tasks, and suites at most 64 cases. A task has at most eight arguments. Each
invocation input is bounded to 1 MiB and each stdout/stderr stream to 32 KiB.

Each case gets a fresh container with no network, ports, host/repository/socket
mounts, or inherited credentials. It uses UID/GID 65534, drops all capabilities,
enables no-new-privileges and built-in seccomp, and keeps the root filesystem read
only. Scratch space is a 16 MiB `noexec,nosuid,nodev` tmpfs. Limits are 32 PIDs,
0.5 CPU, 128 MiB memory with no additional swap, two seconds of per-process CPU,
64 open files, a 1 MiB file size, and no core dumps. The external case wall limit is
three seconds. The whole evaluation has a 180-second dispatch deadline; bounded
Docker control and cleanup calls can extend final settlement beyond that deadline.
Each such call has a 15-second acknowledgment bound.

The trusted container main is separate from the candidate exec stream. Cancellation
retains already dispatched create/start clients, removes the whole owned container,
joins exec/input/output work, and checks absence. Dropping the caller signals the
supervisor to continue cleanup while the Tokio runtime remains alive. An unavailable
daemon or unacknowledged mutation/cleanup yields Unknown with settlement unconfirmed.
No positive result follows from a killed client or an early absence lookup.

The Python wrapper transports candidate results. Its stdout carries no pass/fail
authority. The Rust controller compares results against externally held literals;
only its opaque observed result enters this command's native verifier consumer.
Existing numeric identities and ordinary generation/run replay remain unchanged.
These owned controls qualify the tested workload and recipe. Other platforms,
broader containment claims, model generation, and student training need their own
evidence.

## Qualification

Real Docker tests are explicitly ignored in normal workspace runs, so absence of
the supported runtime cannot be mistaken for a pass. With the cached recipe ready:

```sh
cargo nextest run -p gw-cli --locked --profile ci --run-ignored only \
  --test-threads 1 -E 'test(coding::) | binary(coding_eval)'
```

The suite covers correct/wrong/syntax functions, stale and forged declarations,
fresh replay, capture-once behavior, private payloads, actual containment limits,
pending create/start cancellation, caller-drop cleanup, descendants, and lost
cleanup acknowledgment. Separate Rust/Python artifact tests verify redacted
Parquet import, held-out exclusion, complete-module SFT preparation, and length
rejection without truncation. Host Rust checks and Linux ARM64 candidate execution
are separate qualifications.

Runtime controls follow the primary references for
[Docker run options](https://docs.docker.com/reference/cli/docker/container/run/),
[resource constraints](https://docs.docker.com/engine/containers/resource_constraints/),
[default seccomp](https://docs.docker.com/engine/security/seccomp/), and
[Python isolated mode](https://docs.python.org/3.12/using/cmdline.html).
