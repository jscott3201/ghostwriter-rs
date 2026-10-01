"""Rehashed artifacts still require complete native tensor, recipe and input consistency."""
from copy import deepcopy
from pathlib import Path
import struct
import subprocess
import pytest

from ghostwriter_trl.artifact import ContractError, strict_json
from ghostwriter_trl.prepared import read_prepared, _json_bytes
from ghostwriter_trl.lora.bundle import FILES, inventory, write_bundle, native_report
from .test_completion import qualify


@pytest.fixture(scope="module")
def completion(gw, tokenizer, fixture_dir, tmp_path_factory):
    root = tmp_path_factory.mktemp("gemma-completion-integrity")
    output = root / "original.gwlora"
    prepared = read_prepared(fixture_dir / "prepared-all.gwsft", gw, tokenizer)
    qualify(prepared, tokenizer, gw, output)
    paths = {}
    with output.open("rb") as stream:
        prefix = stream.read(48)
        manifest = strict_json(stream.read(struct.unpack(">Q", prefix[40:])[0]).decode())
        for entry in manifest["files"]:
            path = root / entry["path"]
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(stream.read(entry["byte_length"]))
            paths[entry["path"]] = path
    return output, manifest, paths


@pytest.mark.parametrize("change", ["count", "shape", "prepared", "source", "optimizer", "trainable", "order", "labels", "nonfinite_loss", "full_inference"])
def test_native_rejects_rehashed_manifest_forgeries(change, completion, gw, tmp_path):
    _, original, paths = completion
    m = deepcopy(original)
    if change == "count": m["base_model"]["parameter_count"] += 1
    elif change == "shape": m["observations"]["trainables"][0]["shape"][0] += 1
    elif change == "prepared": m["prepared_build_id"] = "f" * 64
    elif change == "source": m["recipe"]["preparation_source_sha256"] = "f" * 64
    elif change == "optimizer": m["recipe"]["optimizer"] = "adamw_torch_full_v1"
    elif change == "trainable": m["observations"]["trainables"][0]["name"] = "lm_head.weight"
    elif change == "order": m["observations"]["microbatches"].reverse()
    elif change == "labels":
        m["observations"]["microbatches"][0]["shifted_supervised_tokens"] += 1
        m["observations"]["shifted_supervised_tokens"] += 1
    elif change == "nonfinite_loss": m["observations"]["microbatches"][0]["loss_binary64"] = "7ff0000000000000"
    else: m["checkpoint_kind"] = "full_inference"
    output = tmp_path / "forged.gwlora"
    with output.open("w+b") as stream: write_bundle(stream, m, paths)
    with pytest.raises(ContractError, match="native"):
        native_report(output, gw)


@pytest.mark.parametrize("change", ["unchanged", "extra", "missing", "wrong_shape", "nan", "infinity", "adapter_config", "base_config", "base_bytes"])
def test_native_rejects_rehashed_file_forgeries(change, completion, gw, tmp_path):
    import torch
    from safetensors.torch import load_file, save_file
    _, original, original_paths = completion
    paths = dict(original_paths)
    m = deepcopy(original)
    key = "final/adapter_model.safetensors"
    if change == "unchanged":
        paths[key] = paths["initial/adapter_model.safetensors"]
        m["final_adapter"] = deepcopy(m["initial_adapter"])
    elif change in ("adapter_config", "base_config"):
        key = "final/config.json" if change == "adapter_config" else "base/config.json"
        config = strict_json(paths[key].read_text())
        config["base_model_name_or_path"] = "untrusted/automatic-resolution"
        paths[key] = tmp_path / "config.json"
        paths[key].write_bytes(_json_bytes(config))
    else:
        if change == "base_bytes": key = "base/model.safetensors"
        state = load_file(str(paths[key]))
        name = sorted(state)[0]
        if change == "extra": state["unwanted.weight"] = torch.ones(1)
        elif change == "missing": del state[name]
        elif change == "wrong_shape": state[name] = torch.ones(1)
        elif change == "nan": state[name].reshape(-1)[0] = float("nan")
        elif change == "infinity": state[name].reshape(-1)[0] = float("inf")
        else: state[name].reshape(-1)[0] += 1
        paths[key] = tmp_path / "weights.safetensors"
        save_file(state, str(paths[key]), metadata={"format":"pt"})
    m["files"] = inventory(paths)
    output = tmp_path / "forged.gwlora"
    with output.open("w+b") as stream: write_bundle(stream, m, paths)
    with pytest.raises(ContractError, match="native"):
        native_report(output, gw)


def test_lora_cannot_masquerade_as_qwen_full_sft(completion, gw):
    output, _, _ = completion
    with output.open("rb") as stream:
        result = subprocess.run([str(gw), "artifact", "verify-checkpoint", "--stdin"], stdin=stream, capture_output=True)
    assert result.returncode != 0


def test_source_replaced_during_streaming_capture_fails(completion, tmp_path):
    _, original, original_paths = completion
    paths = dict(original_paths)
    source = tmp_path / "prepared.gwsft"
    source.write_bytes(paths["prepared.gwsft"].read_bytes())
    paths["prepared.gwsft"] = source
    m = deepcopy(original); m["files"] = inventory(paths)
    source.write_bytes(b"replacement")
    with (tmp_path / "stage").open("w+b") as stream, pytest.raises(ContractError, match="changed"):
        write_bundle(stream, m, paths)
