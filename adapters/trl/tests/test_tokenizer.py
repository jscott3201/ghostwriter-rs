"""Pinned tokenizer capture, runtime mutation checks, and dependency drift detection."""
import copy
import os
from pathlib import Path
import shutil

import pytest

from ghostwriter_trl.artifact import ContractError
from ghostwriter_trl.tokenizer import check_dependencies, load_tokenizer, validate_tokenizer


def copied_fixture(tmp_path):
    target = tmp_path / "tokenizer"
    shutil.copytree(Path(os.environ["GW_TRL_TOKENIZER"]), target)
    return target


def test_local_tokenizer_loader_consumes_verified_snapshot_after_source_change(tmp_path, monkeypatch):
    from transformers import AutoTokenizer
    directory = copied_fixture(tmp_path)
    original = AutoTokenizer.from_pretrained
    seen = []

    def replace_then_load(path, **kwargs):
        seen.append(Path(path))
        assert kwargs["local_files_only"] is True and kwargs["trust_remote_code"] is False
        (directory / "tokenizer.json").write_bytes(b"replaced after capture")
        return original(path, **kwargs)

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", replace_then_load)
    actual = load_tokenizer(directory)
    assert seen[0] != directory
    validate_tokenizer(actual)


@pytest.mark.parametrize("mutation", ["file_bytes", "extra_file"])
def test_unpinned_local_directory_is_rejected(tmp_path, mutation):
    directory = copied_fixture(tmp_path)
    if mutation == "file_bytes":
        (directory / "tokenizer_config.json").write_bytes(b"{}")
    else:
        (directory / "config.json").write_bytes(b"{}")
    with pytest.raises(ContractError):
        load_tokenizer(directory)


@pytest.mark.parametrize("mutation", ["template", "padding", "added_token", "eos"])
def test_runtime_tokenizer_mutations_are_rejected(tokenizer, mutation):
    changed = copy.deepcopy(tokenizer)
    if mutation == "template":
        changed.chat_template += "changed"
    elif mutation == "padding":
        changed.padding_side = "left"
    elif mutation == "added_token":
        changed.add_tokens(["unqualified_added_token"])
    else:
        changed.eos_token = changed.pad_token
    with pytest.raises(ContractError):
        validate_tokenizer(changed)


def test_dependency_drift_is_not_reported_as_qualified(monkeypatch):
    import ghostwriter_trl.tokenizer as module
    version = module.version
    monkeypatch.setattr(module, "version", lambda name: "999" if name == "trl" else version(name))
    with pytest.raises(ContractError, match="dependencies"):
        check_dependencies()
