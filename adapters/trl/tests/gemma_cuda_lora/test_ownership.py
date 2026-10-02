"""Host controls for single-consumption model-free source ownership."""
from pathlib import Path
import gc
import weakref

import pytest

from ghostwriter_trl.artifact import ContractError
from ghostwriter_trl.cuda_lora.ownership import ObservedCompletion, _Capture, _mint, consume
from ghostwriter_trl.cuda_lora.producer import training_source_identity, train
from ghostwriter_trl.cuda_lora.publication import Publication


def mint(capture, completion_id, observed, source_id, directory):
    # Real host publication for lifetime controls; this is never training evidence.
    output = directory / (capture.path.parent.name + ".published")
    staged = directory / "mint-staged"
    staged.write_bytes(capture.path.read_bytes())
    publication = Publication(completion_id, observed.get("prepared_build_id", "b" * 64), output)
    publication.commit(staged)
    publication.close()
    staged.unlink()
    return _mint(capture, completion_id, observed, source_id, publication)



def test_saved_values_and_forged_objects_cannot_enter_live_consumer():
    for saved in ({"completion_id": "a" * 64}, Path("saved.gwckpt"), "a" * 64, object.__new__(ObservedCompletion)):
        with pytest.raises(ContractError, match="live owned"):
            with consume(saved):
                pytest.fail("saved declaration entered the live gate")
    with pytest.raises(TypeError):
        ObservedCompletion({})


def test_owned_capture_survives_path_replacement_and_consumes_once(tmp_path):
    public = tmp_path / "public"
    public.write_bytes(b"owned captured bytes")
    capture = _Capture(public, 1024)
    owned_path = capture.path
    # This private factory represents successful producer issuance only within this unit control.
    receipt = mint(capture, "a" * 64, {"fresh_reload": "passed"}, training_source_identity(), tmp_path)
    public.write_bytes(b"replacement")
    assert not hasattr(receipt, "model")
    with consume(receipt) as (path, identity, observed):
        assert path == owned_path and path.read_bytes() == b"owned captured bytes"
        assert identity == "a" * 64
        with pytest.raises(ContractError):
            with consume(receipt):
                pass
    assert not owned_path.exists()
    receipt.close()
    with pytest.raises(ContractError):
        receipt.__enter__()


def test_close_and_collection_release_the_owned_capture(tmp_path):
    public = tmp_path / "public"
    public.write_bytes(b"x")
    for explicit in (True, False):
        capture = _Capture(public, 32)
        path = capture.path
        receipt = mint(capture, "b" * 64, {}, training_source_identity(), tmp_path)
        if explicit:
            receipt.close()
        reference = weakref.ref(receipt)
        del receipt
        gc.collect()
        assert reference() is None and not path.exists()


def test_public_producer_rejects_unsupported_host_before_source_access(tmp_path, monkeypatch):
    import ghostwriter_trl.cuda_lora.runtime as runtime
    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    with pytest.raises(ContractError, match="one Linux"):
        train(tmp_path / "missing", tmp_path / "missing-release", tmp_path / "gw", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_source_and_prepared_declarations_cannot_mint_completion(tmp_path):
    from ghostwriter_trl.cuda_lora.producer import _train
    from ghostwriter_trl.cuda_lora.capture import _CapturedSource
    for source in ((tmp_path, {}, "owned", None), object.__new__(_CapturedSource)):
        with pytest.raises(ContractError, match="actual verified input"):
            _train(source, {"examples": []}, tmp_path / "gw", tmp_path / "out", {})
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("operation", ["close", "consume", "consumer_error", "cli"])
def test_cleanup_failure_retains_identity_revokes_and_allows_cleanup_retry(tmp_path, monkeypatch, capsys, operation):
    from ghostwriter_trl.cuda_lora import cli
    from ghostwriter_trl.training.publication import PublishedCheckpointError
    public = tmp_path / "published"
    public.write_bytes(b"completed")
    capture = _Capture(public, 1024)
    receipt = mint(capture, "a" * 64, {"prepared_build_id": "b" * 64}, training_source_identity(), tmp_path)
    original = capture.close
    def fail():
        raise OSError("capture cleanup failure")
    monkeypatch.setattr(capture, "close", fail)
    if operation == "cli":
        monkeypatch.setattr(cli, "train", lambda *a, **k: receipt)
        status = cli.main(["train", "--prepared", "input", "--release-directory", "release", "--gw", "gw", "--output", str(public)])
        assert status == 3
        import json
        report = json.loads(capsys.readouterr().out)
    else:
        with pytest.raises(PublishedCheckpointError) as held:
            if operation == "close":
                receipt.close()
            else:
                with consume(receipt):
                    if operation == "consumer_error":
                        raise ValueError("consumer failure")
        report = held.value.report
    assert report["completion_id"] == "a" * 64
    assert report["prepared_build_id"] == "b" * 64
    assert report["status"] == "published_cleanup_failed"
    assert {"phase": "capture_cleanup", "kind": "OSError"} in report["errors"]
    if operation == "consumer_error":
        assert {"phase": "consumer", "kind": "ValueError"} in report["errors"]
    with pytest.raises(ContractError):
        with consume(receipt):
            pass
    monkeypatch.setattr(capture, "close", original)
    receipt.close()
    assert not capture.path.exists() and public.read_bytes() == b"completed"
    assert Path(report["output"]).read_bytes() == b"completed"


@pytest.mark.parametrize("fail", [False, True])
def test_fixture_scopes_cpu_threads_and_restores_them(monkeypatch, fail):
    import torch
    from ghostwriter_trl.cuda_lora.capture import fixture
    from ghostwriter_trl.lora import config, safe_model
    previous = torch.get_num_threads()
    def create(_):
        assert torch.get_num_threads() == 1
        if fail:
            raise ContractError("fixture creation failed")
        return object()
    monkeypatch.setattr(config, "create", create)
    monkeypatch.setattr(safe_model, "save_owned_base", lambda *args: None)
    if fail:
        with pytest.raises(ContractError, match="fixture creation"):
            with fixture(None):
                pass
    else:
        with fixture(None):
            assert torch.get_num_threads() == previous
    assert torch.get_num_threads() == previous
