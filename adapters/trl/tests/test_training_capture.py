"""Exact release admission and local immutable model snapshots; no pretrained bytes acquired."""
from hashlib import sha256
from pathlib import Path
import shutil
import os
import subprocess
import sys

import pytest

from ghostwriter_trl.artifact import ContractError
from ghostwriter_trl.training.capture import APPROVED_RELEASE, _MODEL_PINS, _copy_checked, load_approved_release
from ghostwriter_trl.training.safe_model import read_config, measure_model, load_model
from .training_fixtures import owned_model


def test_approved_release_record_is_fixed_and_no_caller_eligibility_override(tokenizer, tmp_path):
    assert APPROVED_RELEASE == "qwen3_0_6b_c1899de_student_training_v1"
    assert _MODEL_PINS["model.safetensors"] == (1503300328, "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b")
    with owned_model(tokenizer) as fixture:
        _, _, _, directory, _ = fixture._consume()
        root = tmp_path / "unapproved"
        shutil.copytree(directory, root)
    (root / "eligible.json").write_text('{"eligible":true,"role":"student","purpose":"training"}')
    with pytest.raises(ContractError, match="exactly the nine pinned"):
        with load_approved_release(root):
            pytest.fail("unapproved fixture reached the public model loader")
    with pytest.raises(TypeError):
        load_approved_release(root, eligible=True)


def test_pinned_capture_rejects_identity_and_bound_mismatch_without_loading(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"original")
    _copy_checked(source, tmp_path / "captured", 8, sha256(b"original").hexdigest())
    source.write_bytes(b"replaced")
    assert (tmp_path / "captured").read_bytes() == b"original"
    with pytest.raises(ContractError, match="differ"):
        _copy_checked(source, tmp_path / "changed", 8, sha256(b"original").hexdigest())
    with pytest.raises(ContractError, match="exceeds"):
        _copy_checked(source, tmp_path / "oversized", 7, sha256(b"original").hexdigest())


def test_special_file_capture_fails_without_blocking(tmp_path):
    source = tmp_path / "model.safetensors"
    os.mkfifo(source)
    command = [sys.executable, "-c", "from pathlib import Path; from ghostwriter_trl.training.capture import _copy_checked; "
               "_copy_checked(Path(__import__('sys').argv[1]), Path(__import__('sys').argv[2]), 8, '0'*64)",
               str(source), str(tmp_path / "captured")]
    result = subprocess.run(command, capture_output=True, timeout=10)
    assert result.returncode != 0
    assert b"must be a regular file" in result.stderr
    assert not (tmp_path / "captured").exists()


def test_symlink_replacement_after_open_preserves_captured_inode(tmp_path, monkeypatch):
    import ghostwriter_trl.training.capture as capture
    first, second, source = tmp_path / "first", tmp_path / "second", tmp_path / "source"
    first.write_bytes(b"original"); second.write_bytes(b"replaced")
    source.symlink_to(first)
    original_open = capture.os.open
    def replace_after_open(path, flags, *args, **kwargs):
        descriptor = original_open(path, flags, *args, **kwargs)
        if Path(path) == source:
            source.unlink(); source.symlink_to(second)
        return descriptor
    monkeypatch.setattr(capture.os, "open", replace_after_open)
    capture._copy_checked(source, tmp_path / "captured", 8, sha256(b"original").hexdigest())
    assert source.read_bytes() == b"replaced"
    assert (tmp_path / "captured").read_bytes() == b"original"


def test_direct_loader_rejects_missing_unknown_nonfinite_and_conflicting_weights(tokenizer, tmp_path):
    import torch
    from safetensors.torch import load_file, save_file
    with owned_model(tokenizer) as fixture:
        _, config, original, directory, _ = fixture._consume()
        state = load_file(str(directory / "model.safetensors"))
        assert original["parameter_count"] == sum(t.numel() for t in state.values())
        for name in ("missing", "extra", "nonfinite", "conflict"):
            changed = {key: value.clone() for key, value in state.items()}
            if name == "missing": changed.pop("model.norm.weight")
            elif name == "extra": changed["extra.weight"] = torch.ones(1)
            elif name == "nonfinite": changed["model.norm.weight"][0] = float("inf")
            else: changed["lm_head.weight"] = torch.zeros_like(changed["model.embed_tokens.weight"])
            path = tmp_path / (name + ".safetensors")
            save_file(changed, str(path))
            with pytest.raises(ContractError):
                load_model(config, path)


def test_bfloat16_conversion_and_matching_duplicate_tied_names_are_exact(tokenizer, tmp_path):
    import torch
    from safetensors.torch import load_file, save_file
    with owned_model(tokenizer) as fixture:
        _, config, _, directory, _ = fixture._consume()
        weights = load_file(str(directory / "model.safetensors"))
    weights = {name: value.to(torch.bfloat16) for name, value in weights.items()}
    weights["lm_head.weight"] = weights["model.embed_tokens.weight"].clone()
    config = {**config, "torch_dtype": "bfloat16"}
    path = tmp_path / "bf16.safetensors"
    save_file(weights, str(path))
    model, summary = load_model(config, path)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert torch.equal(model.model.embed_tokens.weight, weights["model.embed_tokens.weight"].float())
    assert summary["parameter_count"] == sum(parameter.numel() for parameter in model.parameters())
    # A separate F32 serialization must have the same normalized parameter identity.
    f32 = tmp_path / "f32.safetensors"
    save_file({name: tensor.float() for name, tensor in weights.items()}, str(f32))
    assert measure_model(config, f32) == summary


@pytest.mark.parametrize("replacement", [b"{\"auto_map\": {}}", b"{\"model_type\":\"qwen3\",\"model_type\":\"evil\"}", b" " * 65537])
def test_config_bounds_and_remote_code_are_rejected(replacement, tmp_path):
    path = tmp_path / "config.json"
    path.write_bytes(replacement)
    with pytest.raises(ContractError):
        read_config(path)


def test_cli_exposes_only_approved_training_and_safe_reload():
    result = subprocess.run([sys.executable, "-m", "ghostwriter_trl.training.cli", "train", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--release-directory" in result.stdout
    assert "--fixture" not in result.stdout and "--eligible" not in result.stdout and "--model-class" not in result.stdout
