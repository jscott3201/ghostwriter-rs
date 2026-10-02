"""Bounded completed checkpoint framing, native inspection, and fresh local safe reload."""
from contextlib import contextmanager
import json
import gc
from pathlib import Path
import struct
import subprocess
import tempfile

import blake3

from ..artifact import ContractError, strict_json
from ..prepared import _json_bytes, verify_prepared, MAX_BYTES as MAX_PREPARED
from .model import load_base, reload_adapter
from .lifecycle import clear_tracebacks, reload_failure
from ..lora.config import read_config
from ..lora.safe_tensors import regular

MAGIC = b"GWCUDA01"
DOMAIN = "ghostwriter.completed-gemma-cuda-lora.v1"
MAX_BYTES = 12 * 1024**3
FILES = ("base/config.json", "base/model.safetensors", "final/config.json",
         "final/adapter_model.safetensors", "initial/config.json", "initial/adapter_model.safetensors", "prepared.gwsft")


def inventory(files: dict[str, Path]) -> list[dict]:
    """Measure the exact fixed file population in framing order."""
    if set(files) != set(FILES):
        raise ContractError("checkpoint requires its complete fixed file inventory")
    result = []
    for name in FILES:
        digest = blake3.blake3()
        size = 0
        with regular(files[name]) as stream:
            while data := stream.read(1024**2):
                size += len(data)
                bound = (11 * 1024**3 if name == "base/model.safetensors" else
                         65536 if name.endswith(".json") else
                         64 * 1024**2 if name.endswith(".safetensors") else MAX_PREPARED)
                if size > bound:
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
        with regular(files[entry["path"]]) as source:
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
    with regular(path) as source:
        result = subprocess.run([str(gw.resolve(strict=True)), "artifact", "verify-cuda-lora", "--stdin"],
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
        with regular(path) as source, snapshot.open("xb") as output:
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
    """Fresh independent CUDA model with an explicit owned capture and close lifetime."""
    def __init__(self, model, report, prepared, capture):
        self.model = model
        self._report = json.dumps(report, allow_nan=False)
        self.prepared = prepared
        self._capture = capture

    @property
    def report(self):
        return strict_json(self._report)

    def close(self):
        """Release this receipt's model and source ownership."""
        self.model = None
        if self._capture is not None:
            capture, self._capture = self._capture, None
            capture.__exit__(None, None, None)

    def __enter__(self):
        if self._capture is None:
            raise ContractError("CUDA reload is closed")
        return self

    def __exit__(self, *args):
        self.close()


def read_checkpoint(path: Path, gw: Path, tokenizer):
    """Recheck whole captured input and all runtime state in an independent CUDA allocation.

    Saved observations retain declared status. This API cannot create a live training completion.
    The caller owns the returned model and must close it (or use its context manager).
    """
    from .state import measure
    from .runtime import runtime
    from .environment import observe as observe_environment
    capture = _captured(path)
    root, snapshot = capture.__enter__()
    base = model = None
    errors = []
    try:
        report = native_report(snapshot, gw)
        paths = _extract(snapshot, root, report)
        prepared = verify_prepared(paths["prepared.gwsft"].read_bytes(), gw, tokenizer)
        config, _ = read_config(paths["base/config.json"])
        with runtime():
            dependencies, environment = observe_environment()
            if dependencies != report["declarations"]["cuda_dependencies"] or environment != report["declarations"]["cuda_runtime"]:
                raise ContractError("CUDA reload runtime differs from the captured execution environment")
            base, measured_base = load_base(config, paths["base/model.safetensors"])
            model, measured_adapter = reload_adapter(base, config, paths["final/adapter_model.safetensors"])
            actual = measure(model)
        if (measured_base != report["base_model"] or measured_adapter != report["final_adapter"]
                or actual != report["declarations"]["final_state"]
                or prepared.build_id != report["prepared_build_id"]):
            raise ContractError("independent CUDA reload differs from captured source/state declarations")
        report.update(fresh_safe_load="passed", fresh_tokenizer_replay="passed", fresh_complete_state="passed")
        return ReloadedCheckpoint(model, report, prepared, capture)
    except BaseException as error:
        errors.append(("reload", error))
    # Leave the exception handler before cleanup so cleanup cannot mask the semantic failure.
    clear_tracebacks(error for _, error in errors)
    base = model = None
    for phase, cleanup in (("capture_cleanup", lambda: capture.__exit__(None, None, None)),
                           ("model_collection", gc.collect)):
        try:
            cleanup()
        except BaseException as error:
            errors.append((phase, error))
    reload_failure(errors)
