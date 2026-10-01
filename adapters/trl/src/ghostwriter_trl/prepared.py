"""Whole immutable SFT inputs: captured Parquet, Rust import, and pinned tokenizer replay."""
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

import blake3

from .artifact import ContractError, VerifiedSnapshot, strict_json, verify_snapshot
from .build import build, identity, source_identity
from .profiles import QWEN, controls

MAGIC = b"GWSFT001"
HASH_DOMAIN = "ghostwriter.prepared-sft-input.v1"
MAX_BYTES = 256 * 1024 * 1024
HEADER_BYTES = 56


def _json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _payload_json(data: bytes):
    try:
        value = strict_json(data.decode("utf-8"))
    except UnicodeError as error:
        raise ContractError("invalid prepared input UTF-8") from error

    def validate(item):
        if type(item) in {str, int, bool, type(None)}:
            return
        if type(item) is list:
            for child in item:
                validate(child)
        elif type(item) is dict:
            for child in item.values():
                validate(child)
        else:
            raise ContractError("prepared input JSON requires integer numbers")
    validate(value)
    return value


def _frame(payload: bytes, source: bytes) -> bytes:
    framed = struct.pack(">QQ", len(payload), len(source)) + payload + source
    if len(framed) + 40 > MAX_BYTES:
        raise ContractError("prepared input exceeds size limit")
    digest = blake3.blake3(framed, derive_key_context=HASH_DOMAIN).digest()
    return MAGIC + digest + framed


def _unframe(data: bytes) -> tuple[str, bytes, bytes]:
    if type(data) is not bytes or not HEADER_BYTES <= len(data) <= MAX_BYTES or data[:8] != MAGIC:
        raise ContractError("unsupported or malformed prepared input framing")
    payload_length, source_length = struct.unpack(">QQ", data[40:HEADER_BYTES])
    boundary = HEADER_BYTES + payload_length
    if boundary + source_length != len(data):
        raise ContractError("prepared input length mismatch or trailing bytes")
    digest = blake3.blake3(data[40:], derive_key_context=HASH_DOMAIN).digest()
    if data[8:40] != digest:
        raise ContractError("prepared input complete payload/source digest mismatch")
    return digest.hex(), data[HEADER_BYTES:boundary], data[boundary:]


def prepare(snapshot: VerifiedSnapshot, tokenizer, *, cot: str, turns: str, max_length: int,
            profile: str = QWEN, enable_thinking: bool | None = None) -> bytes:
    """Produce a complete input identity before any trainer or optimization operation.

    The original source is embedded rather than trusting a portable report supplied by a caller.
    This returns bytes to save; consumption requires ``verify_prepared`` or ``read_prepared``.
    """
    examples, manifest = build(snapshot, tokenizer, cot=cot, turns=turns, max_length=max_length,
                               profile=profile, enable_thinking=enable_thinking)
    report = snapshot.report
    source = {key: report[key] for key in ("byte_length", "snapshot_blake3")}
    source["artifact_id"] = report["artifact"]["artifact_id"]
    payload = {"version": 1, "source": source, "manifest": manifest, "examples": examples}
    return _frame(_json_bytes(payload), snapshot.data)


class VerifiedPrepared:
    """Opaque immutable whole build, created only after Rust verification and official replay.

    Public metadata and examples are defensive copies. The original bytes and receipts remain
    fixed, even if a source path, bundle path, or returned nested dictionary is changed later.
    """
    __slots__ = ("__state",)

    def __new__(cls, *args, **kwargs):
        raise TypeError("VerifiedPrepared must be created by verify_prepared or read_prepared")

    def __setattr__(self, name, value):
        raise AttributeError("verified prepared input state is immutable")

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    @property
    def data(self) -> bytes:
        """Complete immutable captured build bytes."""
        return self.__state[0]

    @property
    def build_id(self) -> str:
        """Identity of the complete payload and source, fixed before any trainer handoff."""
        return self.data[8:40].hex()

    @property
    def payload(self) -> dict:
        """Defensive copy of the complete validated and replayed payload."""
        return _payload_json(self.__state[1])

    @property
    def examples(self) -> list[dict]:
        """All replayed examples in their original order; no partial import."""
        return self.payload["examples"]

    @property
    def manifest(self) -> dict:
        """Defensive copy of preparation recipe, counts, rejections, and qualification limits."""
        return self.payload["manifest"]

    @property
    def report(self) -> dict:
        """Separate actual Rust source/structure verification evidence."""
        return strict_json(self.__state[2].decode("utf-8"))

    @property
    def replay_report(self) -> dict:
        """Separate pinned Python recipe, renderer and tokenization replay evidence."""
        return strict_json(self.__state[3].decode("utf-8"))


def verify_prepared(data: bytes, gw: Path, tokenizer) -> VerifiedPrepared:
    """Verify all captured bytes with Rust, then replay every example from verified source.

    Rust validates framing, strict schema, labels/ownership/counts and original Parquet bindings.
    Python proves this installed pinned tokenizer/adapter actually produces the complete payload.
    Neither check reconstructs lifecycle eligibility, weights, execution/decision lineage, or benefit.
    """
    build_id, raw_payload, source = _unframe(data)
    payload = _payload_json(raw_payload)
    result = subprocess.run(
        [str(gw.resolve(strict=True)), "artifact", "verify-prepared", "--stdin"],
        input=data, capture_output=True, check=False,
    )
    if result.returncode != 0:
        raise ContractError("Rust prepared input verification failed")
    try:
        report = strict_json(result.stdout.decode("utf-8"))
    except UnicodeError as error:
        raise ContractError("invalid prepared verifier report encoding") from error
    if not isinstance(report, dict) or set(report) != {
        "report_version", "build_id", "byte_length", "snapshot_blake3", "source_verification",
        "example_count", "structural_source_validation", "tokenizer_replay",
    }:
        raise ContractError("unexpected prepared verifier report fields")
    if (type(report["report_version"]) is not int or report["report_version"] != 1
            or type(report["byte_length"]) is not int or report["byte_length"] != len(data)
            or report["build_id"] != build_id or report["snapshot_blake3"] != blake3.blake3(data).hexdigest()
            or report["structural_source_validation"] != "passed" or report["tokenizer_replay"] != "not_run"):
        raise ContractError("prepared verifier receipt does not match captured build")
    # Create actual verified-source authority through its guarded factory, not an embedded report.
    snapshot = verify_snapshot(source, gw)
    if report["source_verification"] != snapshot.report:
        raise ContractError("prepared verifier source receipt mismatch")
    try:
        recipe = payload["manifest"]["recipe"]
        if recipe["version"] != 2 or recipe["adapter_source_sha256"] != source_identity():
            raise ContractError("incompatible installed preparation recipe/source; native historical inspection remains available")
        profile = recipe["preparation_profile"]
        settings = controls(profile["name"], profile["controls"]["enable_thinking"])
        if profile != {"name": profile["name"], "controls": settings}:
            raise ContractError("unsupported prepared rendering controls")
        examples, manifest = build(snapshot, tokenizer, cot=recipe["cot_policy"],
                                   turns=recipe["multi_turn_loss"], max_length=recipe["max_length"],
                                   profile=profile["name"], enable_thinking=settings["enable_thinking"])
        replay_runtime = manifest["recipe"]["runtime"]
        # Runtime is recorded producer provenance. Replay uses the installed pins but must not
        # replace historical IDs merely because the consumer OS or Python patch differs.
        manifest["recipe"]["runtime"] = recipe["runtime"]
        manifest["recipe_id"] = identity(manifest["recipe"])
        for example in examples:
            example["example_id"] = identity(["ghostwriter.sft-example.v1", manifest["recipe_id"],
                example["source"], example["target_index"], example["input_ids"], example["labels"]])
        manifest["example_ids"] = [example["example_id"] for example in examples]
        expected_source = {key: snapshot.report[key] for key in ("byte_length", "snapshot_blake3")}
        expected_source["artifact_id"] = snapshot.report["artifact"]["artifact_id"]
        expected = {"version": 1, "source": expected_source, "manifest": manifest, "examples": examples}
        # Serialization normalizes tuples and rejects Boolean-vs-integer equality shortcuts.
        if _json_bytes(payload) != _json_bytes(expected):
            raise ContractError("prepared input differs from pinned recipe/rendering/tokenization replay")
    except (KeyError, TypeError, UnicodeError) as error:
        raise ContractError("invalid prepared input payload") from error
    if type(report["example_count"]) is not int or report["example_count"] != len(examples):
        raise ContractError("prepared verifier example count mismatch")
    replay = {"report_version": 1, "build_id": build_id, "recipe_id": manifest["recipe_id"],
              "example_count": len(examples), "official_tokenizer_replay": "passed",
              "runtime": replay_runtime,
              "student_weights": "unbound", "forward_passes": 0, "optimizer_steps": 0}
    verified = object.__new__(VerifiedPrepared)
    object.__setattr__(verified, "_VerifiedPrepared__state", (data, raw_payload, result.stdout, _json_bytes(replay)))
    return verified


def read_prepared(path: Path, gw: Path, tokenizer) -> VerifiedPrepared:
    """Capture a bounded build once; neither verification nor consumption reopens the path."""
    with path.open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    return verify_prepared(data, gw, tokenizer)


def save_prepared(path: Path, data: bytes) -> None:
    """Atomically publish exact producer bytes without overwriting any existing destination."""
    _unframe(data)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".prepared-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink()
