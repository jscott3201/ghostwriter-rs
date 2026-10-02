# Capturing repository episodes

Capture an externally supplied repository attempt as one portable JSON artifact:

```sh
gw artifact capture-repository --stdin < request.json > episode.json
gw artifact verify-repository --stdin < episode.json > receipt.json
```

Both commands consume one complete JSON document, with a 32 MiB input limit. Capture
sorts changed paths, computes three separate identities, and verifies the complete
artifact before writing it to stdout. Verification recomputes every identity and the
report diagnostic from the saved material. Output is also limited to 32 MiB, including
its final newline. The Rust capture API reserves that newline in the complete artifact limit
before returning an artifact. Invalid input produces no stdout. Diagnostics contain fixed messages
without supplied field names, enum values, paths, or report contents. An output-device
failure can still interrupt a write; consumers must check the exit status and verify
saved bytes before using them.

The commands use no provider credentials, configuration, database, repository checkout,
container, tool, or model. Source and report references are opaque strings and are never
opened. The saved artifact is the persistence boundary.

## Request

[The authored request fixture](../crates/gw-cli/tests/fixtures/repository-episode-request.json)
is a complete runnable example. All public types are exported by `gw-schema` under the
`Repository` prefix. The top-level request has these fields:

| Field | Contents |
|---|---|
| `version` | `1` |
| `task` | Source, rights, group, split, public problem, repository and immutable task declarations |
| `candidate` | Attempt, ordered canonical messages, full tool definitions and declared file delta |
| `generation` | Supplied model/serving claims, settings and known-or-unknown usage |
| `report` | `null` or the original publisher report and optional normalized execution declaration |

Unknown fields in protected structures and duplicate object keys anywhere are rejected.
Arguments, tool definitions, generation settings, model/serving values, and the original
publisher report remain opaque JSON. Their complete values participate in the relevant
identities. Canonical message fields retain reasoning separately from content, preserve
null versus empty content, and retain raw argument strings alongside structured arguments.
Optional fields with unknown values serialize as `null` in new protocol structures; the existing
canonical message representation omits absent optional fields.

### Task declarations

The task uses the existing `TaskSource`, `ReviewedTaskRights`, `NamespacedTaskId`, and
`TaskSplit` declarations. `repository` accepts a credential-free HTTP(S) or URN reference;
credentials, query strings, fragments and path traversal are rejected.

`upstream_base` and `actor_baseline` are distinct `{algorithm, hex}` declarations.
Algorithms are `git_sha1` (40 lowercase hex characters) and `git_sha256` (64). The upstream
base identifies the original source. The actor baseline identifies the prepared workspace
seen by the model; setup may have created a different commit before the attempt starts.
These declarations do not establish that the objects exist or contain the stated files.

`environment` carries a `{algorithm, hex}` digest (`sha256` or `blake3`), platform, and
opaque recipe identity. `private_contract` carries only its digest and exact
`required_test_ids`. Private tests, reference fixes, and test patches have no structured
fields in this public contract. Arbitrary supplied text and raw reports are not scanned
for hidden private material; the producer remains responsible for appropriate redaction.

Missing bases, environment, private contract, required IDs, or source revision remain
inspectable. The artifact records pending reason codes and has no complete task or
candidate identity until those declarations are present.

### Trajectory and delta

Every call has a unique nonempty ID, object arguments, and a declared function name. Every
tool result explicitly identifies a preceding call. Complete trajectories require exactly
one result for each call. Parallel same-name calls may return in either order because IDs
carry their relation. Missing links, duplicate links and contradictory names are rejected.
Raw argument strings are preserved as evidence; they are not reparsed to replace the
structured arguments. Tool definitions are captured without imposing a training model's
schema or template restrictions. Unsupported tool kinds and multimodal content leave the
supported candidate identity pending.

Set `trajectory_complete` to the producer's declaration. Set `delta.enumeration` to
`complete` only when the producer declares the entire workspace delta was enumerated,
including untracked additions. Ghostwriter checks internal consistency; it does not
observe the workspace or certify that declaration.

Each `delta.changes` member contains a path plus `before` and `after` states:

```json
{
  "path": "src/example.py",
  "before": {"kind": "text", "text": "old\r\n", "mode": "100644"},
  "after": {"kind": "text", "text": "new", "mode": "100755"}
}
```

Use `{"kind":"absent"}` for the missing side of an addition or deletion. Mode-only changes
are supported. A rename is deletion plus addition. UTF-8 contents preserve exact bytes,
including line endings and a missing final newline; embedded NUL requires an unsupported
binary declaration. Only regular file modes `100644` and `100755` are accepted.

Paths use relative slash-separated NFC Unicode names. Absolute paths, empty/dot/traversal
segments, backslashes, control characters, Windows reserved names (including COM/LPT with
superscript ¹, ², or ³) or punctuation, trailing
dots/spaces, `.git` components, duplicate paths and case aliases are rejected. Alias checks
use lowercase then uppercase Unicode folding and NFC; text contents are never normalized.
A regular file cannot also be the ancestor of another regular file in either represented
snapshot. An old file may be deleted and replaced by a directory containing a new file.

Declare unsupported categories through `delta.unsupported`: `binary`, `symlink`,
`submodule`, or `non_utf8_path`. These declarations and incomplete enumeration retain the
capture but prevent a complete supported candidate identity. Null `tools` means definitions
are unavailable and also leaves the candidate pending; an empty array declares no tools.

### Numbers and usage

This protocol has its own strict JSON decoder, shared by capture and verification.
Integers must fit signed 64-bit or unsigned 64-bit ranges. Fractional/exponent numbers
use finite, correctly rounded binary64 values. Nonfinite numbers and nonzero values that
underflow to zero are rejected. Bare `-0`, `-0.0`, and `-0e0` canonicalize to floating
`-0.0`; integer `0` and floating `0.0` keep distinct canonical encodings. Other numeric
spellings are not retained. Use `raw_arguments` to retain the original argument text.

Usage has optional `input_tokens`, `output_tokens`, `reasoning_tokens`, and `cost_usd`.
Null means unknown. A reported zero stays known zero. Cost must be finite and nonnegative.
Model and serving fields use `{"status":"unknown"}` or
`{"status":"declared","value":...}`. These are supplied claims, without physical-attempt
receipts or proof that a model was loaded.

## Identity and reports

Identities use canonical JSON with sorted object keys, exact strings, and ordered arrays.
Only changed-path entries are sorted. Each identity uses its own BLAKE3 derive-key domain:

| Identity | Domain and bound material |
|---|---|
| Task | `ghostwriter.repository-task.v1`: source, public problem, repository, both bases, environment and private contract |
| Candidate | `ghostwriter.repository-candidate.v1`: task identity, attempt, ordered messages/tools and complete supported delta |
| Capture | `ghostwriter.repository-capture.v1`: every canonical request declaration, including rights, group, split, usage, model/settings and report |

Task labels, rights, group and split are excluded from task and candidate identities. They
remain bound by capture identity. Changing a result, tool definition, raw argument, file
state or attempt changes candidate identity even if assistant prose stays the same.
Changing the source, either base, environment or private contract changes task identity.

An attached report contains `publisher`, optional `source_ref`, original `raw` JSON and
optional `execution` using the existing `ExecutionEvidence` shape. Original publisher
statuses are retained in `raw`; the normalized projection is a supplied declaration too.
Its binding uses the following fields:

- `task`: derived repository task identity;
- `attempt`: the candidate attempt string;
- `patch_hash`: the complete repository candidate identity, including trajectory and delta.

The candidate identity excludes the report, so a producer can derive it before attaching
its evaluation declaration. `RepositoryEpisodeRequest::identities()` provides that pure
operation. A task/attempt mismatch is `foreign`; a candidate mismatch is `stale`. Those
reports remain in the capture, and their declared outcome is `unknown` for this candidate.

For matching reports, the existing factual interpretation uses the task's required IDs,
not the report's claimed coverage. A declared pass requires zero exit status, unique
readable cases, no reported errors or failures, and all required cases passing. Missing or
skipped required cases fail when the report is otherwise readable. Missing exit status,
empty/duplicate cases, or a reported unknown remain unknown. The original report is never
rewritten to match the diagnostic.

Every artifact and verification receipt has `observed_execution: "unknown"` and
`training_eligible: false`, including a fully corroborated declared pass. Verification
rejects altered authority fields even if the request hash was recomputed. The API verifies
internal consistency and does not authenticate publishers or qualify execution. Admission,
qualified observation, database intake, and training export are separate operations.
