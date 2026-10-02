"""Real accepted populations with large public metadata must fit the explicit bounded bridge."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from ghostwriter_trl.artifact import read_snapshot
from ghostwriter_trl.comparison.bridge import capture_population
from ghostwriter_trl.comparison.generation import cpu_runtime, recipe
from ghostwriter_trl.comparison.separation import check_separation
from ghostwriter_trl.lora.config import FIXTURE
from .test_review_repairs import prepared_reference


@pytest.mark.parametrize("metadata", ["train", "heldout"])
def test_large_accepted_metadata_captures_and_matches_exact_train(metadata, gw, tokenizer, tmp_path):
    root = os.environ.get("GW_PAIR_LARGE_TEST_DIRECTORY")
    if not root:
        pytest.fail("GW_PAIR_LARGE_TEST_DIRECTORY must hold train and heldout large-metadata native fixtures")
    directory = Path(root) / metadata
    registration = json.loads((directory / "registration.json").read_text())["registration_id"]
    database = directory / "reference.sqlite"
    # Independent native capture proves the registered/committed population exists and records
    # actual UTF-8 frame size, rather than constructing a Python-only population assertion.
    process = subprocess.Popen([str(gw), "eval", "coding-pair", "--db", str(database),
                                "--registration", registration, "--split", "test", "--stdio"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        raw = process.stdout.readline(32 * 1024**2 + 1)
        assert raw.endswith(b"\n")
        native = json.loads(raw)["population"]
    finally:
        process.stdin.close()
        process.stdin = None
        process.communicate(timeout=30)
        process.stdout.close()
        process.stderr.close()
    assert native["registration_id"] == registration and len(native["training_members"]) == 64
    if metadata == "heldout":
        assert len(raw) > 4 * 1024**2 and len(raw.decode()) < len(raw)
    with capture_population(gw, database, registration, "test") as bridge:
        population = bridge.population
    assert population == native
    if metadata == "train":
        assert len(raw) < 1024**2, "accepted Train metadata must travel as compact native-derived bindings"
    source = tmp_path / "accepted.parquet"
    result = subprocess.run([str(gw), "reference", "export", "--db", str(database),
                             "--batch-id", population["batch_id"], "--out", str(source)], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
    prepared = prepared_reference(read_snapshot(source, gw), gw, tokenizer)
    if metadata == "train":
        assert sum(len(example["source"]["task_json"].encode()) for example in prepared.examples) > 4 * 1024**2
    with cpu_runtime():
        separation = check_separation(prepared, population, tokenizer, recipe(tokenizer, 3, 2048, ""),
                                      {"source_authorization": FIXTURE})
    assert separation["training_population"] == "registered_reference_train"
    assert len(separation["training_member_ids"]) == 64
    assert separation["effective_prompt_separation"] == "generation_recipe_prompt_ids_disjoint"


@pytest.mark.parametrize("fits", [True, False])
def test_opening_frame_counts_utf8_newline_and_settles_owned_child(fits, gw, monkeypatch):
    import sys
    import ghostwriter_trl.comparison.bridge as bridge
    registration = "a" * 64
    raw = (json.dumps({"protocol_version": 1, "population": {"registration_id": registration,
                       "split": "test", "metadata": "é" * 64}}, ensure_ascii=False) + "\n").encode()
    assert len(raw.decode()) < len(raw)
    original = subprocess.Popen
    children = []
    def launch(_args, **kwargs):
        program = f"import sys; sys.stdout.buffer.write({raw!r}); sys.stdout.buffer.flush(); sys.stdin.buffer.read()"
        child = original([sys.executable, "-c", program], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(bridge.subprocess, "Popen", launch)
    monkeypatch.setattr(bridge, "MAX_BYTES", len(raw) if fits else len(raw) - 1)
    if fits:
        with capture_population(gw, Path("unused"), registration, "test") as captured:
            assert captured.population["metadata"] == "é" * 64
    else:
        from ghostwriter_trl.artifact import ContractError
        with pytest.raises(ContractError, match="bounded captured population"):
            with capture_population(gw, Path("unused"), registration, "test"):
                pytest.fail("opening frame exceeded its byte bound after counting newline")
    assert len(children) == 1
    assert children[0].poll() is not None and children[0].stdin is None and children[0].stdout.closed
