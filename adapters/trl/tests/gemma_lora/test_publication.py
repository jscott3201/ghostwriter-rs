"""Publication acknowledgments distinguish retained complete output from an ordinary failure."""
import os
import stat
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from ghostwriter_trl.prepared import read_prepared
from ghostwriter_trl.artifact import ContractError
from ghostwriter_trl.lora.bundle import read_checkpoint
from ghostwriter_trl.training.publication import PublishedCheckpointError
from .test_completion import qualify as qualify_training
from ghostwriter_trl.lora.capture import owned_fixture

def owned_model(_tokenizer):
    return owned_fixture()


def test_post_link_directory_sync_failure_retains_identity_without_success(gw, tokenizer, fixture_dir, tmp_path, monkeypatch):
    import ghostwriter_trl.lora.producer as producer
    original = os.fsync
    def fail_directory_sync(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("injected directory synchronization failure")
        return original(descriptor)
    monkeypatch.setattr(producer.os, "fsync", fail_directory_sync)
    prepared = read_prepared(fixture_dir / "prepared-all.gwsft", gw, tokenizer)
    output = tmp_path / "retained.gwckpt"
    with pytest.raises(PublishedCheckpointError) as caught:
        qualify_training(prepared, tokenizer, gw, output)
    assert output.is_file()
    report = getattr(caught.value, "report", {})
    assert report.get("status") == "published_durability_unknown"
    assert report["completion_id"] == output.read_bytes()[8:40].hex()
    assert report["prepared_build_id"] == prepared.build_id
    assert report["durability"] == "unknown"
    assert report["errors"] == [{"phase": "directory_sync", "kind": "OSError"}]
    assert not hasattr(caught.value, "observed")
    inspected = read_checkpoint(output, gw, tokenizer)
    assert inspected.report["completion_id"] == report["completion_id"]
    assert inspected.report["historical_training"] == "declared"
    assert not hasattr(inspected, "observed")
    report["status"] = "success"
    assert caught.value.report["status"] == "published_durability_unknown"
    monkeypatch.setattr(producer, "run", lambda *a, **kw: pytest.fail("retry reached optimization"))
    with pytest.raises(ContractError, match="already exists"):
        qualify_training(prepared, tokenizer, gw, output)


@pytest.mark.parametrize("failure", ["directory_open", "directory_close", "staged_file_cleanup", "workspace_cleanup"])
def test_other_post_publication_failures_preserve_output_and_cleanup_state(failure, gw, tokenizer, fixture_dir, tmp_path, monkeypatch):
    import ghostwriter_trl.lora.producer as producer
    output = tmp_path / "retained.gwckpt"
    if failure == "directory_open":
        original = os.open
        def fail_open(path, flags, *args, **kwargs):
            if Path(path) == output.parent and flags == os.O_RDONLY:
                raise OSError("injected directory open failure")
            return original(path, flags, *args, **kwargs)
        monkeypatch.setattr(producer.os, "open", fail_open)
    elif failure == "directory_close":
        original = os.close
        parent = output.parent.stat()
        def fail_close(descriptor):
            observed = os.fstat(descriptor)
            is_directory = (observed.st_dev, observed.st_ino) == (parent.st_dev, parent.st_ino)
            original(descriptor)
            if is_directory:
                raise OSError("injected directory close failure")
        monkeypatch.setattr(producer.os, "close", fail_close)
    elif failure == "staged_file_cleanup":
        original = Path.unlink
        def fail_unlink(path, *args, **kwargs):
            if path.name.startswith(".checkpoint-"):
                raise OSError("injected staging cleanup failure")
            return original(path, *args, **kwargs)
        monkeypatch.setattr(Path, "unlink", fail_unlink)
    else:
        original = producer.tempfile.TemporaryDirectory.cleanup
        def fail_cleanup(directory):
            original(directory)
            if Path(directory.name).name.startswith("gw-gemma-lora-"):
                raise OSError("injected workspace cleanup failure")
        monkeypatch.setattr(producer.tempfile.TemporaryDirectory, "cleanup", fail_cleanup)
    prepared = read_prepared(fixture_dir / "prepared-all.gwsft", gw, tokenizer)
    with pytest.raises(PublishedCheckpointError) as caught:
        qualify_training(prepared, tokenizer, gw, output)
    report = caught.value.report
    assert report["completion_id"] == output.read_bytes()[8:40].hex()
    unknown = failure == "directory_open"
    assert report["status"] == ("published_durability_unknown" if unknown else "published_cleanup_failed")
    assert report["durability"] == ("unknown" if unknown else "confirmed")
    assert report["errors"] == [{"phase": failure, "kind": "OSError"}]
    assert not hasattr(caught.value, "observed")
    monkeypatch.undo()
    for stage in tmp_path.glob(".checkpoint-*"):
        stage.unlink()
    assert list(tmp_path.iterdir()) == [output]


def test_cleanup_failure_cannot_mask_uncertain_durability(gw, tokenizer, fixture_dir, tmp_path, monkeypatch):
    import ghostwriter_trl.lora.producer as producer
    original_sync, original_unlink = os.fsync, Path.unlink
    def fail_sync(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("directory failed")
        return original_sync(descriptor)
    def fail_unlink(path, *args, **kwargs):
        if path.name.startswith(".checkpoint-"):
            raise OSError("staging cleanup failed")
        return original_unlink(path, *args, **kwargs)
    monkeypatch.setattr(producer.os, "fsync", fail_sync)
    monkeypatch.setattr(Path, "unlink", fail_unlink)
    prepared = read_prepared(fixture_dir / "prepared-all.gwsft", gw, tokenizer)
    output = tmp_path / "retained.gwckpt"
    with pytest.raises(PublishedCheckpointError) as caught:
        qualify_training(prepared, tokenizer, gw, output)
    assert caught.value.report["status"] == "published_durability_unknown"
    assert caught.value.report["completion_id"] == output.read_bytes()[8:40].hex()
    assert [error["phase"] for error in caught.value.report["errors"]] == ["directory_sync", "staged_file_cleanup"]
    monkeypatch.undo()
    for stage in tmp_path.glob(".checkpoint-*"):
        stage.unlink()


def test_post_link_failure_does_not_delete_another_actors_replacement(gw, tokenizer, fixture_dir, tmp_path, monkeypatch):
    import ghostwriter_trl.lora.producer as producer
    output = tmp_path / "replaced.gwckpt"
    original = os.fsync
    committed_ids = []
    def replace_and_fail(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            committed_ids.append(output.read_bytes()[8:40].hex())
            output.unlink()
            output.write_bytes(b"other actor's replacement")
            raise OSError("injected post-link failure")
        return original(descriptor)
    monkeypatch.setattr(producer.os, "fsync", replace_and_fail)
    prepared = read_prepared(fixture_dir / "prepared-all.gwsft", gw, tokenizer)
    with pytest.raises(PublishedCheckpointError) as caught:
        qualify_training(prepared, tokenizer, gw, output)
    assert committed_ids == [caught.value.report["completion_id"]]
    assert output.read_bytes() == b"other actor's replacement"
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("sync_failed", [False, True])
def test_surrounding_release_cleanup_preserves_publication_outcome(sync_failed, gw, tokenizer, fixture_dir, tmp_path, monkeypatch):
    import ghostwriter_trl.lora.producer as producer
    @contextmanager
    def release(_directory):
        with owned_model(tokenizer) as loaded:
            try:
                yield loaded, tokenizer
            finally:
                raise OSError("injected release capture cleanup failure")
    monkeypatch.setattr(producer, "load_approved_release", release)
    original = os.fsync
    def fail_sync(descriptor):
        if sync_failed and stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("injected directory synchronization failure")
        return original(descriptor)
    monkeypatch.setattr(producer.os, "fsync", fail_sync)
    output = tmp_path / "retained.gwckpt"
    with pytest.raises(PublishedCheckpointError) as caught:
        producer.train(fixture_dir / "prepared-all.gwsft", tmp_path, gw, output, max_steps=2)
    report = caught.value.report
    assert report["completion_id"] == output.read_bytes()[8:40].hex()
    assert report["status"] == ("published_durability_unknown" if sync_failed else "published_cleanup_failed")
    assert [error["phase"] for error in report["errors"]] == (["directory_sync"] if sync_failed else []) + ["release_cleanup"]
    assert not hasattr(caught.value, "observed")


def test_cli_emits_publication_diagnostic_and_nonzero_exit(gw, tokenizer, fixture_dir, tmp_path, monkeypatch, capsys):
    import ghostwriter_trl.lora.cli as cli
    import ghostwriter_trl.lora.producer as producer
    original = os.fsync
    def fail_sync(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("injected directory synchronization failure")
        return original(descriptor)
    monkeypatch.setattr(producer.os, "fsync", fail_sync)
    prepared = read_prepared(fixture_dir / "prepared-all.gwsft", gw, tokenizer)
    def fixture_train(_prepared, _release, native, output, **options):
        options.pop("max_steps", None)
        return qualify_training(prepared, tokenizer, native, output, **options)
    monkeypatch.setattr(cli, "train", fixture_train)
    output = tmp_path / "retained.gwckpt"
    status = cli.main(["train", "--prepared", "unused", "--release-directory", "unused",
                       "--gw", str(gw), "--output", str(output)])
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert status == 3
    assert report["status"] == "published_durability_unknown"
    assert report["completion_id"] == output.read_bytes()[8:40].hex()
    assert "observed" not in report
    assert "Inspect the output against its completion_id" in captured.err
    assert "training again at the same path will be rejected" in captured.err
