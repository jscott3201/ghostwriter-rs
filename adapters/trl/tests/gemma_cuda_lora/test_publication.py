"""Real filesystem failure controls for the publication helper consumed by the CUDA producer."""
import os
import stat

import pytest

from ghostwriter_trl.cuda_lora.publication import Publication
from ghostwriter_trl.training.publication import PublishedCheckpointError


def inputs(tmp_path):
    source, output = tmp_path / "staged", tmp_path / "out"
    source.write_bytes(b"complete captured bytes")
    return source, output, Publication("a" * 64, "b" * 64, output)


def test_publication_does_not_overwrite_and_commits_complete_bytes(tmp_path):
    staged, output, publication = inputs(tmp_path)
    publication.commit(staged)
    publication.close()
    assert publication.published and publication.directory_synced
    assert output.read_bytes() == staged.read_bytes()
    retry = Publication("c" * 64, "b" * 64, output)
    with pytest.raises(FileExistsError):
        retry.commit(staged)
    assert not retry.published and output.read_bytes() == b"complete captured bytes"


def test_failed_directory_sync_retains_identity_and_uncertain_durability(tmp_path, monkeypatch):
    staged, output, publication = inputs(tmp_path)
    original = os.fsync
    def fail_directory(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("injected directory sync failure")
        original(fd)
    monkeypatch.setattr(os, "fsync", fail_directory)
    with pytest.raises(OSError) as failed:
        publication.commit(staged)
    publication.close()
    with pytest.raises(BaseException) as caught:
        publication.raise_failure([(publication.phase, failed.value)])
    assert isinstance(caught.value, PublishedCheckpointError)
    assert output.read_bytes() == b"complete captured bytes"
    assert caught.value.report["completion_id"] == "a" * 64
    assert caught.value.report["status"] == "published_durability_unknown"
    assert not hasattr(caught.value, "observed")


@pytest.mark.parametrize("phase", ["directory_close", "workspace_cleanup", "runtime_cleanup", "capture_cleanup"])
def test_post_commit_failure_never_unlinks_replaced_public_path(tmp_path, phase):
    staged, output, publication = inputs(tmp_path)
    publication.commit(staged)
    publication.close()
    # Another actor can replace this public name after the commit point.
    output.unlink()
    output.write_bytes(b"new owner")
    with pytest.raises(PublishedCheckpointError) as caught:
        publication.raise_failure([(phase, OSError("injected cleanup failure"))])
    assert output.read_bytes() == b"new owner"
    assert caught.value.report["status"] == "published_cleanup_failed"
    assert caught.value.report["completion_id"] == "a" * 64
    assert caught.value.report["errors"] == [{"phase": phase, "kind": "OSError"}]


def test_interrupt_after_real_link_retains_publication_identity(tmp_path, monkeypatch):
    staged, output, publication = inputs(tmp_path)
    link = os.link
    def interrupted(source, destination):
        link(source, destination)
        raise KeyboardInterrupt("after link")
    monkeypatch.setattr(os, "link", interrupted)
    with pytest.raises(KeyboardInterrupt) as failed:
        publication.commit(staged)
    with pytest.raises(BaseException) as caught:
        publication.raise_failure([(publication.phase, failed.value)])
    assert isinstance(caught.value, PublishedCheckpointError)
    assert output.read_bytes() == staged.read_bytes()
    assert caught.value.report["completion_id"] == "a" * 64
    assert caught.value.report["durability"] == "unknown"


@pytest.mark.parametrize("replace", [False, True])
def test_interrupted_link_preserves_foreign_target(tmp_path, monkeypatch, replace):
    staged, output, publication = inputs(tmp_path)
    link = os.link
    def interrupted(source, destination):
        if replace:
            link(source, destination)
            destination.unlink()
        destination.write_bytes(b"foreign")
        raise KeyboardInterrupt("foreign target")
    monkeypatch.setattr(os, "link", interrupted)
    with pytest.raises(KeyboardInterrupt):
        publication.commit(staged)
    assert not publication.published and output.read_bytes() == b"foreign"


def test_uninspectable_interrupted_link_reports_unknown_outcome(tmp_path, monkeypatch):
    from ghostwriter_trl.artifact import ContractError
    staged, output, publication = inputs(tmp_path)
    link, inspect = os.link, os.stat
    def interrupted(source, destination):
        link(source, destination)
        raise KeyboardInterrupt("after link")
    def inaccessible(path, *args, **kwargs):
        if path == output:
            raise PermissionError("settlement inspection denied")
        return inspect(path, *args, **kwargs)
    monkeypatch.setattr(os, "link", interrupted)
    monkeypatch.setattr(os, "stat", inaccessible)
    with pytest.raises(ContractError, match="publication outcome unknown") as caught:
        publication.commit(staged)
    assert "a" * 64 in str(caught.value)
    assert output.read_bytes() == b"complete captured bytes"
