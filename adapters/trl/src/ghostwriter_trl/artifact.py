"""Rust verifies integrity; PyArrow consumes exactly the same immutable bytes."""
import json
from pathlib import Path
import subprocess
from typing import Any

import blake3
import pyarrow as pa
import pyarrow.parquet as pq


class ContractError(ValueError):
    """An unsupported or unverifiable input cannot cross the adapter boundary."""


def strict_json(text: str) -> Any:
    """Reject duplicate keys and nonstandard JSON numbers in reports and source messages."""
    def pairs(items):
        output = {}
        for key, value in items:
            if key in output:
                raise ContractError("duplicate JSON key")
            output[key] = value
        return output

    def constant(_):
        raise ContractError("nonstandard JSON number")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, TypeError) as error:
        raise ContractError("invalid JSON") from error


class VerifiedSnapshot:
    """Opaque immutable result created only by ``verify_snapshot`` or ``read_snapshot``.

    The captured bytes and original Rust success report form one private immutable state.
    A public report is a detached copy; editing it cannot rewrite verified authority.
    """
    __slots__ = ("__state",)

    def __new__(cls, *args, **kwargs):
        raise TypeError("VerifiedSnapshot must be created by verify_snapshot or read_snapshot")

    def __setattr__(self, name, value):
        raise AttributeError("verified snapshot state is immutable")

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    @property
    def data(self) -> bytes:
        """The immutable byte snapshot that Rust actually verified."""
        return self.__state[0]

    @property
    def report(self) -> dict:
        """A fresh, defensive copy of the verified Rust report."""
        return strict_json(self.__state[1].decode("utf-8"))

    def rows(self) -> list[dict]:
        """Decode the already-verified buffer; never reopen a source path."""
        rows = pq.read_table(pa.BufferReader(self.data)).to_pylist()
        if self.report["artifact"]["manifest"]["column_schema_version"] in {"record_origins", "tool_definitions"}:
            from .origin import row_origin
            for row in rows:
                row_origin(row)
        return rows


def verify_snapshot(data: bytes, gw: Path) -> VerifiedSnapshot:
    """Send this snapshot to the explicit local Rust verifier and validate its receipt."""
    if type(data) is not bytes:
        raise ContractError("snapshot must be immutable bytes")
    result = subprocess.run(
        [str(gw.resolve(strict=True)), "artifact", "verify", "--stdin"],
        input=data, capture_output=True, check=False,
    )
    if result.returncode != 0:
        raise ContractError("Rust artifact verification failed")
    try:
        report = strict_json(result.stdout.decode("utf-8"))
    except UnicodeError as error:
        raise ContractError("invalid verifier report encoding") from error
    if not isinstance(report, dict) or set(report) != {
        "report_version", "artifact", "byte_length", "snapshot_blake3",
    }:
        raise ContractError("unexpected verifier report fields")
    if type(report["report_version"]) is not int or report["report_version"] != 1:
        raise ContractError("unsupported verifier report version")
    if type(report["byte_length"]) is not int or report["byte_length"] != len(data):
        raise ContractError("verifier byte length mismatch")
    if report["snapshot_blake3"] != blake3.blake3(data).hexdigest():
        raise ContractError("verifier snapshot digest mismatch")
    artifact = report["artifact"]
    if not isinstance(artifact, dict) or type(artifact.get("metadata_version")) is not int:
        raise ContractError("unexpected verified artifact envelope")
    version = artifact["metadata_version"]
    keys = {"metadata_version", "artifact_id", "scope", "manifest"}
    if version == 3:
        keys.add("screening")
    elif version != 1:
        raise ContractError("unsupported artifact metadata version")
    if set(artifact) != keys:
        raise ContractError("unexpected version-specific artifact envelope")
    if version == 3:
        from .screening import validate_screening_shape
        validate_screening_shape(artifact)
    identity = artifact["artifact_id"]
    if not isinstance(identity, str) or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
        raise ContractError("invalid artifact identity")
    manifest = artifact["manifest"]
    if not isinstance(manifest, dict) or manifest.get("column_schema_version") not in {
        "canonical_messages", "reviewed_tasks", "record_origins", "tool_definitions",
    }:
        raise ContractError("unsupported artifact columns")
    for key in ("n_records", "n_admitted"):
        if type(manifest.get(key)) is not int or manifest[key] < 0:
            raise ContractError("invalid artifact counts")
    # Public construction is disabled: only successful Rust verification reaches this factory.
    snapshot = object.__new__(VerifiedSnapshot)
    object.__setattr__(snapshot, "_VerifiedSnapshot__state", (data, result.stdout))
    return snapshot


def read_snapshot(path: Path, gw: Path) -> VerifiedSnapshot:
    """Read once, then verify and decode only that snapshot, even if the path changes."""
    return verify_snapshot(path.read_bytes(), gw)
