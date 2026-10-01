"""Actual installed source hashes distinguish preparation from optimizer and shared helper edits."""
import shutil
from pathlib import Path
import pytest

from ghostwriter_trl.artifact import ContractError
from ghostwriter_trl.prepared import read_prepared
from ghostwriter_trl.lora.capture import owned_fixture
from ghostwriter_trl.lora.producer import _run_loaded


def test_actual_hashes_bind_lora_recursively_and_consumed_shared_helpers(tmp_path, monkeypatch):
    import ghostwriter_trl.build as build
    import ghostwriter_trl.lora.producer as producer
    copy = tmp_path / "package"
    shutil.copytree(build.PACKAGE, copy, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    monkeypatch.setattr(build, "PACKAGE", copy)
    monkeypatch.setattr(producer, "__file__", str(copy / "lora/producer.py"))
    preparation = build.source_identity()
    training = producer.training_source_identity()
    with (copy / "lora/execution.py").open("a") as stream: stream.write("\n# optimizer-only identity control\n")
    assert build.source_identity() == preparation
    changed = producer.training_source_identity()
    assert changed != training
    with (copy / "training/capture.py").open("a") as stream: stream.write("\n# consumed capture-helper identity control\n")
    assert build.source_identity() == preparation
    assert producer.training_source_identity() != changed


def test_source_replacement_during_actual_execution_prevents_publication(gw, tokenizer, fixture_dir, tmp_path, monkeypatch):
    import ghostwriter_trl.lora.producer as producer
    original = producer.training_source_identity
    count = 0
    def replaced():
        nonlocal count
        count += 1
        return original() if count == 1 else "f" * 64
    monkeypatch.setattr(producer, "training_source_identity", replaced)
    prepared = read_prepared(fixture_dir / "prepared-all.gwsft", gw, tokenizer)
    output = tmp_path / "no.gwlora"
    with owned_fixture() as loaded, pytest.raises(ContractError, match="source changed"):
        _run_loaded(prepared, tokenizer, loaded, gw, output)
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_owned_base_mutation_before_training_is_rejected(gw, tokenizer, fixture_dir, tmp_path):
    import torch
    prepared = read_prepared(fixture_dir / "prepared-all.gwsft", gw, tokenizer)
    with owned_fixture() as loaded:
        model = loaded._LoadedBase__state[0]
        with torch.no_grad(): model.model.language_model.layers[0].layer_scalar.add_(1)
        with pytest.raises(ContractError, match="changed after capture"):
            _run_loaded(prepared, tokenizer, loaded, gw, tmp_path / "no.gwlora")
    assert list(tmp_path.iterdir()) == []
