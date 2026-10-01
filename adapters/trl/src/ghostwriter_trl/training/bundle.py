"""Bounded completed checkpoint framing, native inspection, and fresh local safe reload."""
from contextlib import contextmanager
import json
from pathlib import Path
import struct
import subprocess
import tempfile

import blake3

from ..artifact import ContractError, strict_json
from ..prepared import _json_bytes, verify_prepared
from .safe_model import load_model, read_config

MAGIC = b"GWCKPT01"
DOMAIN = "ghostwriter.completed-full-sft.v1"
MAX_BYTES = 7 * 1024**3
FILES = ("checkpoint/config.json", "checkpoint/model.safetensors", "initial/config.json",
         "initial/model.safetensors", "prepared.gwsft")


def inventory(files: dict[str, Path]) -> list[dict]:
    """Measure the exact fixed file population in framing order."""
    if set(files) != set(FILES):
        raise ContractError("checkpoint requires its complete fixed file inventory")
    result = []
    for name in FILES:
        digest = blake3.blake3()
        size = 0
        with files[name].open("rb") as stream:
            while data := stream.read(1024**2):
                size += len(data)
                if size > (3 * 1024**3 if name.endswith(".safetensors") else 256 * 1024**2):
                    raise ContractError("checkpoint source exceeds byte bound")
                digest.update(data)
        result.append({"path": name, "byte_length": size, "blake3": digest.hexdigest()})
    return result


def write_bundle(stream, manifest: dict, files: dict[str, Path]) -> str:
    """Stage complete bytes. The caller publishes only after native inspection and safe reload."""
    raw = _json_bytes(manifest)
    if len(raw) > 2 * 1024**2:
        raise ContractError("checkpoint manifest exceeds byte bound")
    digest = blake3.blake3(derive_key_context=DOMAIN)
    stream.write(MAGIC + bytes(32))
    for data in (struct.pack(">Q", len(raw)), raw):
        digest.update(data); stream.write(data)
    size = 48 + len(raw)
    for entry in manifest["files"]:
        file_hash = blake3.blake3()
        count = 0
        with files[entry["path"]].open("rb") as source:
            while data := source.read(1024**2):
                count += len(data); size += len(data)
                if count > entry["byte_length"] or size > MAX_BYTES:
                    raise ContractError("checkpoint source changed size during capture")
                file_hash.update(data); digest.update(data); stream.write(data)
        if count != entry["byte_length"] or file_hash.hexdigest() != entry["blake3"]:
            raise ContractError("checkpoint source changed during capture")
    stream.seek(8); stream.write(digest.digest()); stream.seek(0, 2)
    return digest.hexdigest()


def native_report(path: Path, gw: Path) -> dict:
    """Inspect captured bytes without importing saved declarations as model or training authority."""
    with path.open("rb") as source:
        result = subprocess.run([str(gw.resolve(strict=True)), "artifact", "verify-checkpoint", "--stdin"],
                                stdin=source, capture_output=True, check=False)
    if result.returncode:
        raise ContractError("native completed checkpoint verification failed: " + result.stderr.decode(errors="replace")[:300])
    report = strict_json(result.stdout.decode())
    if (type(report) is not dict or report.get("report_version") != 1
            or report.get("structural_validation") != "passed" or report.get("historical_training") != "declared"
            or report.get("model_reload") != "not_run" or report.get("tokenizer_replay") != "not_run"):
        raise ContractError("unexpected native checkpoint verification receipt")
    return report


@contextmanager
def _captured(path: Path):
    """Capture once into private storage so replacement of the public input cannot alter replay."""
    with tempfile.TemporaryDirectory(prefix="gw-checkpoint-reload-") as temporary:
        root = Path(temporary)
        snapshot = root / "captured.gwckpt"
        size = 0
        with path.open("rb") as source, snapshot.open("xb") as output:
            while data := source.read(1024**2):
                size += len(data)
                if size > MAX_BYTES:
                    raise ContractError("checkpoint exceeds complete byte bound")
                output.write(data)
        yield root, snapshot


def _extract(snapshot: Path, root: Path, report: dict) -> dict[str, Path]:
    # Recheck framing identity and each extracted file from the same private verified stream.
    paths = {}
    with snapshot.open("rb") as stream:
        prefix = stream.read(48)
        if prefix[:8] != MAGIC or prefix[8:40].hex() != report["completion_id"]:
            raise ContractError("checkpoint reload receipt identity mismatch")
        length = struct.unpack(">Q", prefix[40:])[0]
        if length > 2 * 1024**2:
            raise ContractError("checkpoint manifest exceeds bound")
        manifest = strict_json(stream.read(length).decode())
        if _json_bytes(manifest) != _json_bytes(report["declarations"]):
            raise ContractError("checkpoint reload manifest differs from verified capture")
        if [entry["path"] for entry in manifest["files"]] != list(FILES):
            raise ContractError("unsupported checkpoint extraction path")
        for entry in manifest["files"]:
            output = root / entry["path"]
            output.parent.mkdir(exist_ok=True)
            remaining = entry["byte_length"]
            digest = blake3.blake3()
            with output.open("xb") as destination:
                while remaining:
                    data = stream.read(min(remaining, 1024**2))
                    if not data:
                        raise ContractError("checkpoint truncated during reload")
                    destination.write(data); digest.update(data); remaining -= len(data)
            if digest.hexdigest() != entry["blake3"]:
                raise ContractError("checkpoint extracted content mismatch")
            paths[entry["path"]] = output
        if stream.read(1):
            raise ContractError("trailing checkpoint reload bytes")
    return paths


class ReloadedCheckpoint:
    """Freshly loaded inference model plus declared historical metadata, never observed training."""
    __slots__ = ("__state",)

    def __new__(cls, *args, **kwargs):
        raise TypeError("use read_checkpoint for whole captured-byte validation and fresh reload")

    def __setattr__(self, name, value):
        raise AttributeError("checkpoint receipts are immutable")

    def __init_subclass__(cls, **kwargs):
        raise TypeError("reload receipts cannot be subclassed")

    @property
    def model(self):
        """This newly loaded CPU inference model; its historical execution remains declared."""
        return self.__state[0]

    @property
    def report(self):
        """Detached native inspection and fresh load receipt."""
        return strict_json(self.__state[1])

    @property
    def prepared(self):
        """Original prepared input reverified and replayed from this same completed capture."""
        return self.__state[2]


def read_checkpoint(path: Path, gw: Path, tokenizer) -> ReloadedCheckpoint:
    """Independently verify an entire completion, replay its input, and safely load final weights."""
    with _captured(path) as (root, snapshot):
        report = native_report(snapshot, gw)
        paths = _extract(snapshot, root, report)
        prepared = verify_prepared(paths["prepared.gwsft"].read_bytes(), gw, tokenizer)
        config = read_config(paths["checkpoint/config.json"])
        model, measured = load_model(config, paths["checkpoint/model.safetensors"])
        if measured != report["checkpoint_model"] or prepared.build_id != report["prepared_build_id"]:
            raise ContractError("fresh checkpoint reload differs from native captured-byte receipt")
        report = {**report, "fresh_safe_load": "passed", "fresh_tokenizer_replay": "passed"}
    result = object.__new__(ReloadedCheckpoint)
    object.__setattr__(result, "_ReloadedCheckpoint__state", (model, json.dumps(report), prepared))
    return result
