"""Release capture uses application pins and never infers authority from local filenames."""
from collections import Counter
import os
from pathlib import Path
import pytest

from ghostwriter_trl.artifact import ContractError
from ghostwriter_trl.lora.capture import load_approved_release


def test_exact_release_capture_reads_shared_config_once_and_rejects_unapproved_weights(tmp_path, monkeypatch):
    import ghostwriter_trl.lora.capture as capture
    root = Path(os.environ["GW_TRL_GEMMA_TOKENIZER"])
    release = tmp_path / "release"
    release.mkdir()
    for source in root.iterdir():
        (release / source.name).symlink_to(source)
    (release / "model.safetensors").write_bytes(b"not approved release weights")
    calls = Counter()
    original = capture._copy_checked
    def copied(source, *args):
        calls[source.name] += 1
        return original(source, *args)
    monkeypatch.setattr(capture, "_copy_checked", copied)
    monkeypatch.setattr(capture, "_owned", lambda *a: pytest.fail("unapproved weights reached model loading"))
    with pytest.raises(ContractError, match="approved exact release"):
        with load_approved_release(release):
            pytest.fail("unapproved release acquired ownership")
    assert calls["config.json"] == 1
    assert calls["model.safetensors"] == 1
    assert all(count == 1 for count in calls.values())


def test_unknown_release_files_fail_before_capture(tmp_path):
    (tmp_path / "model.pkl").write_bytes(b"untrusted pickle")
    with pytest.raises(ContractError, match="inventory"):
        with load_approved_release(tmp_path):
            pytest.fail("unexpected release files admitted")


def test_wrong_dependency_stack_fails_before_release_capture(tmp_path, monkeypatch):
    import importlib.metadata
    import ghostwriter_trl.lora.capture as capture
    original = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.0.0" if name == "peft" else original(name))
    monkeypatch.setattr(capture, "_copy_checked", lambda *args: pytest.fail("wrong dependency stack reached capture"))
    with pytest.raises(ContractError, match="dependencies"):
        with load_approved_release(tmp_path):
            pytest.fail("wrong stack acquired model authority")
