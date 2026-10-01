"""Independent reframing mutations of a genuinely trained checkpoint; identities are recomputed."""
from copy import deepcopy
import json
import struct
import subprocess

import blake3
import pytest

from ghostwriter_trl.artifact import ContractError
from ghostwriter_trl.prepared import read_prepared
from ghostwriter_trl.training.bundle import read_checkpoint
from .training_fixtures import qualify_training


@pytest.fixture(scope="module")
def completed(gw, tokenizer, fixture_dir, tmp_path_factory):
    path = tmp_path_factory.mktemp("completed-checkpoint") / "trained.gwckpt"
    prepared = read_prepared(fixture_dir / "prepared-all.gwsft", gw, tokenizer)
    observed = qualify_training(prepared, tokenizer, gw, path, max_steps=3, batch_size=3, accumulation=2)
    return path, observed


def split(data):
    length = struct.unpack(">Q", data[40:48])[0]
    manifest = json.loads(data[48:48 + length])
    cursor = 48 + length
    files = {}
    for entry in manifest["files"]:
        files[entry["path"]] = data[cursor:cursor + entry["byte_length"]]
        cursor += entry["byte_length"]
    assert cursor == len(data)
    return manifest, files


def frame(manifest, files, raw=None):
    manifest = deepcopy(manifest)
    for entry in manifest["files"]:
        if entry["path"] in files:
            data = files[entry["path"]]
            entry.update(byte_length=len(data), blake3=blake3.blake3(data).hexdigest())
    raw = raw if raw is not None else json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    body = struct.pack(">Q", len(raw)) + raw + b"".join(files[entry["path"]] for entry in manifest["files"])
    return b"GWCKPT01" + blake3.blake3(body, derive_key_context="ghostwriter.completed-full-sft.v1").digest() + body


@pytest.mark.parametrize("case", ["build", "initial-summary", "final-summary", "updates", "supervised",
    "rows", "missing-batch", "reorder-batch", "attention-as-supervision", "approved-forgery", "unknown-field",
    "missing-field", "duplicate-field", "float", "boolean", "trailing", "truncated", "length", "pickle-path",
    "config-code", "config-shape", "output-bytes", "input-bytes", "prepared-source", "stale-file-digest"])
def test_native_and_python_reject_corrupt_or_contradictory_completions(case, completed, gw, tokenizer, tmp_path):
    original = completed[0].read_bytes()
    manifest, files = split(original)
    raw = None
    if case == "build": manifest["prepared_build_id"] = "0" * 64
    elif case == "initial-summary": manifest["initial_model"]["tensor_content_id"] = "0" * 64
    elif case == "final-summary": manifest["checkpoint_model"]["tensor_content_id"] = "0" * 64
    elif case == "updates": manifest["observations"]["optimizer_updates"] += 1
    elif case == "supervised": manifest["observations"]["shifted_supervised_tokens"] += 1
    elif case == "rows": manifest["observations"]["consumed_examples"] += 1
    elif case == "missing-batch": manifest["observations"]["microbatches"].pop()
    elif case == "reorder-batch": manifest["observations"]["microbatches"][0]["example_ids"].reverse()
    elif case == "attention-as-supervision":
        for row in manifest["observations"]["microbatches"]:
            row["shifted_supervised_tokens"] = row["input_tokens"]
        manifest["observations"]["shifted_supervised_tokens"] = sum(row["shifted_supervised_tokens"] for row in manifest["observations"]["microbatches"])
    elif case == "approved-forgery": manifest["source_authorization"] = "qwen3_0_6b_c1899de_student_training_v1"
    elif case == "unknown-field": manifest["eligible"] = True
    elif case == "missing-field": del manifest["upstream_lineage"]
    elif case == "duplicate-field": raw = b'{"version":1,' + json.dumps(manifest).encode()[1:]
    elif case == "float": manifest["recipe"]["max_steps"] = 3.0
    elif case == "boolean": manifest["version"] = True
    elif case == "pickle-path":
        old = manifest["files"][1]["path"]
        manifest["files"][1]["path"] = "checkpoint/training_args.bin"
        files["checkpoint/training_args.bin"] = files.pop(old)
    elif case in {"config-code", "config-shape"}:
        config = json.loads(files["checkpoint/config.json"])
        config["auto_map" if case == "config-code" else "hidden_size"] = {} if case == "config-code" else 24
        files["checkpoint/config.json"] = json.dumps(config).encode()
    elif case in {"output-bytes", "input-bytes"}:
        path = "checkpoint/model.safetensors" if case == "output-bytes" else "initial/model.safetensors"
        data = files[path]
        files[path] = data[:-4] + struct.pack("<f", 0.75)
    elif case == "prepared-source": files["prepared.gwsft"] = b"not a prepared input"
    elif case == "stale-file-digest":
        manifest["files"][0]["blake3"] = "0" * 64
        raw = json.dumps(manifest).encode()
    data = frame(manifest, files, raw)
    if case == "trailing": data += b"!"
    elif case == "truncated": data = data[:-1]
    elif case == "length": data = data[:40] + b"\xff" * 8 + data[48:]
    native = subprocess.run([str(gw), "artifact", "verify-checkpoint", "--stdin"], input=data, capture_output=True)
    assert native.returncode != 0, case
    destination = tmp_path / "changed.gwckpt"
    destination.write_bytes(data)
    with pytest.raises(ContractError):
        read_checkpoint(destination, gw, tokenizer)


@pytest.mark.parametrize("case", ["missing", "extra", "shape", "dtype", "nonfinite", "overlap", "gap", "alias-conflict", "duplicate-key", "trailing"])
def test_safe_tensor_attacks_rejected_after_rehashing_outer_files(case, completed, gw, tokenizer, tmp_path):
    manifest, files = split(completed[0].read_bytes())
    data = files["checkpoint/model.safetensors"]
    size = struct.unpack("<Q", data[:8])[0]
    header, payload = json.loads(data[8:8 + size]), data[8 + size:]
    names = sorted(name for name in header if name != "__metadata__")
    name = names[0]
    if case == "missing": del header[name]
    elif case == "extra": header["unknown.weight"] = deepcopy(header[name])
    elif case == "shape": header[name]["shape"] = [1]
    elif case == "dtype": header[name]["dtype"] = "F64"
    elif case == "nonfinite": payload = struct.pack("<I", 0x7F800000) + payload[4:]
    elif case == "overlap": header[names[1]]["data_offsets"] = header[name]["data_offsets"]
    elif case == "gap": header[name]["data_offsets"] = [n + 4 for n in header[name]["data_offsets"]]
    elif case == "alias-conflict":
        head = deepcopy(header["model.embed_tokens.weight"])
        count = head["data_offsets"][1] - head["data_offsets"][0]
        head["data_offsets"] = [len(payload), len(payload) + count]
        header["lm_head.weight"] = head
        payload += b"\0" * count
    elif case == "trailing": payload += b"\0\0\0\0"
    raw = json.dumps(header, separators=(",", ":")).encode()
    if case == "duplicate-key": raw = b'{"__metadata__":{},' + raw[1:]
    raw += b" " * (-len(raw) % 8)
    files["checkpoint/model.safetensors"] = struct.pack("<Q", len(raw)) + raw + payload
    changed = frame(manifest, files)
    result = subprocess.run([str(gw), "artifact", "verify-checkpoint", "--stdin"], input=changed, capture_output=True)
    assert result.returncode != 0, case
    destination = tmp_path / "weights.gwckpt"
    destination.write_bytes(changed)
    with pytest.raises(ContractError):
        read_checkpoint(destination, gw, tokenizer)


def test_changed_historical_declaration_stays_declared(completed, gw, tokenizer, tmp_path):
    # Correctly rehashed metadata can describe a different producer; it cannot mint live observations.
    manifest, files = split(completed[0].read_bytes())
    manifest["recipe"]["training_source_sha256"] = "a" * 64
    destination = tmp_path / "declaration.gwckpt"
    destination.write_bytes(frame(manifest, files))
    reloaded = read_checkpoint(destination, gw, tokenizer)
    assert reloaded.report["historical_training"] == "declared"
    assert reloaded.report["completion_id"] != completed[1].completion_id
    assert not hasattr(reloaded, "observed")
