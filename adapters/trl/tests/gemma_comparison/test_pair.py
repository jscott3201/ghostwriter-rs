"""Complete live CPU pair and fresh native execution over the registered owned Test population."""
from copy import deepcopy
import json
import os
from pathlib import Path
import pytest

from ghostwriter_trl.artifact import ContractError
from ghostwriter_trl.prepared import read_prepared, _json_bytes
from ghostwriter_trl.comparison.producer import compare
from ghostwriter_trl.comparison.artifact import read, verify
from ghostwriter_trl.lora.producer import ObservedCompletion, _comparison_source
from ..gemma_lora.test_completion import qualify


def fixture_population():
    directory = os.environ.get("GW_PAIR_TEST_DIRECTORY")
    if not directory:
        pytest.fail("GW_PAIR_TEST_DIRECTORY must hold the explicit synthetic Rust coding_pair fixture")
    directory = Path(directory)
    registration = json.loads((directory / "registration.json").read_text())["registration_id"]
    return directory / "reference.sqlite", registration


@pytest.fixture(scope="module")
def completed_pair(gw, tokenizer, fixture_dir, tmp_path_factory):
    directory = tmp_path_factory.mktemp("complete-pair")
    database, registration = fixture_population()
    prepared = read_prepared(fixture_dir / "prepared-all.gwsft", gw, tokenizer)
    completed = qualify(prepared, tokenizer, gw, directory / "complete.gwlora")
    output = directory / "pair.json"
    observed = compare(completed, tokenizer, gw, database, registration, output,
                       max_new_tokens=3, max_prompt_tokens=2048)
    data, report = read(output, gw, tokenizer)
    artifact = json.loads(data)
    evidence = os.environ.get("GW_PAIR_EVIDENCE_OUT")
    if evidence: Path(evidence).write_bytes(data)
    return completed, artifact, report, observed, output


def test_actual_live_pair_full_test_coverage(completed_pair):
    completed, artifact, report, observed, _ = completed_pair
    assert artifact["request"]["models"]["completion_id"] == completed.completion_id
    assert len(artifact["population"]["members"]) == 32
    assert len(artifact["rows"]) == 64
    assert artifact["base"]["items"] == artifact["candidate"]["items"] == 32
    assert artifact["base"]["unknown"] == artifact["candidate"]["unknown"] == 0
    assert artifact["comparable"] and artifact["passed_difference"] == 0
    assert artifact["automatic_promotion"] is False
    assert report["tokenizer_replay"] == "passed"
    assert observed.report["fresh_generation"] == "both_independent_models"
    for row in artifact["request"]["rows"]:
        assert row["generation"] is not None and row["failure"] is None
        assert len(row["generation"]["output"]["suffix_ids"]) == 3
    assert all(row["reason"] == "native" and row["outcome"] == "failed" for row in artifact["rows"])
    assert "PRIVATE_REVIEW_CANARY" not in json.dumps(artifact)
    assert "PRIVATE_ZERO_ORACLE" not in json.dumps(artifact)


def test_copying_live_state_does_not_create_producer_authority(completed_pair):
    completed, *_ = completed_pair
    forged = object.__new__(ObservedCompletion)
    object.__setattr__(forged, "_ObservedCompletion__state", completed._ObservedCompletion__state)
    with pytest.raises(ContractError, match="live owned"):
        _comparison_source(forged)


def reseal(artifact):
    import blake3
    artifact["artifact_id"] = ""
    artifact["artifact_id"] = blake3.blake3(_json_bytes(artifact), derive_key_context="ghostwriter.paired-coding-artifact.v1").hexdigest()


@pytest.mark.parametrize("mutation", ["omit", "swap", "outcome", "score", "authority", "coverage", "decode", "population", "config", "parent", "foreign"])
def test_rehashed_saved_declarations_reject_changed_contracts(mutation, completed_pair, gw, tokenizer):
    _, original, *_ = completed_pair
    artifact = deepcopy(original)
    if mutation == "omit": artifact["rows"].pop()
    elif mutation == "swap": artifact["request"]["rows"][0], artifact["request"]["rows"][1] = artifact["request"]["rows"][1], artifact["request"]["rows"][0]
    elif mutation == "outcome": artifact["rows"][0]["outcome"] = "passed"
    elif mutation == "score": artifact["base"]["passed"] += 1
    elif mutation == "authority": artifact["historical_generation"] = "observed"
    elif mutation == "coverage": artifact["rows"][0]["native"]["cases"].clear()
    elif mutation == "population": artifact["request"]["population_id"] = "0" * 64
    elif mutation == "config": artifact["request"]["recipe"]["max_new_tokens"] += 1
    elif mutation == "parent": artifact["request"]["models"]["parent_revision"] = "invented"
    elif mutation == "foreign": artifact["request"]["rows"][0]["member_id"] = "0" * 64
    else: artifact["request"]["rows"][0]["generation"]["output"]["body_text"] += " altered"
    reseal(artifact)
    with pytest.raises(ContractError): verify(_json_bytes(artifact), gw, tokenizer)


def test_atomic_no_overwrite_and_post_link_failure(completed_pair, gw, tokenizer, tmp_path, monkeypatch):
    import stat
    from ghostwriter_trl.comparison.artifact import publish, PublishedComparisonError
    _, artifact, *_ = completed_pair
    target = tmp_path / "pair.json"
    target.write_bytes(b"another actor")
    with pytest.raises(ContractError, match="already exists"):
        publish(artifact, target, gw, tokenizer)
    assert target.read_bytes() == b"another actor"
    target.unlink()
    original = os.fsync
    def fail_directory(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode): raise OSError("injected directory sync failure")
        return original(fd)
    monkeypatch.setattr(os, "fsync", fail_directory)
    with pytest.raises(PublishedComparisonError) as error:
        publish(artifact, target, gw, tokenizer)
    assert error.value.report["artifact_id"] == artifact["artifact_id"]
    assert error.value.report["durability"] == "unknown"
    assert json.loads(target.read_bytes()) == artifact
    assert list(tmp_path.iterdir()) == [target]


def test_publication_race_never_replaces_existing_target(completed_pair, gw, tokenizer, tmp_path, monkeypatch):
    from ghostwriter_trl.comparison.artifact import publish
    _, artifact, *_ = completed_pair
    target = tmp_path / "pair.json"
    original = os.link
    def competing_link(source, destination):
        target.write_bytes(b"racing actor")
        return original(source, destination)
    monkeypatch.setattr(os, "link", competing_link)
    with pytest.raises(FileExistsError): publish(artifact, target, gw, tokenizer)
    assert target.read_bytes() == b"racing actor"
    assert list(tmp_path.iterdir()) == [target]


def test_declared_control_pair_pass_wrong_syntax_unknown_and_saved_replay(completed_pair, gw, tokenizer):
    from ghostwriter_trl.comparison.bridge import capture_population, run_saved
    from ghostwriter_trl.comparison.generation import config_data
    from ghostwriter_trl.comparison.protocol import capture_output, render_prompt
    _, actual, *_ = completed_pair
    database, registration = fixture_population()
    with capture_population(gw, database, registration, "validation") as bridge:
        request = deepcopy(actual["request"])
        request["population_id"] = bridge.population["population_id"]
        request["recipe"]["max_new_tokens"] = 64
        request["recipe"]["effective_config_json"] = json.dumps(config_data(64), sort_keys=True, separators=(",", ":"))
        request["rows"] = []
        for member in bridge.population["members"]:
            prompt = render_prompt(tokenizer, [{"role": "user", "content": member["prompt"]}], 2048)
            for side in ("base", "candidate"):
                index = len(request["rows"]) % 4
                row = deepcopy(actual["request"]["rows"][0 if side == "base" else 1])
                row.update(member_id=member["member_id"], generation=None, failure=None, failed_prompt=None)
                if index == 3:
                    row.update(failure="generation_error", failed_prompt=prompt)
                else:
                    code = ["def probe():\n    return 0\n", "def probe():\n    return 1\n", "def probe(:\n"][index]
                    suffix = tokenizer.encode(code, add_special_tokens=False) + [106]
                    row["generation"] = {"prompt": prompt, "output": capture_output(tokenizer, prompt["input_ids"], prompt["input_ids"] + suffix, 64),
                                         "effective_max_length": len(prompt["input_ids"]) + 64,
                                         "cache_type": "transformers.cache_utils.DynamicCache"}
                request["rows"].append(row)
        artifact = bridge.execute(request)
    assert len(artifact["rows"]) == 32 and len(artifact["population"]["members"]) == 16
    assert artifact["population"]["population_id"] != actual["population"]["population_id"]
    assert artifact["base"] == {"items": 16, "passed": 8, "failed": 8, "unknown": 0}
    assert artifact["candidate"] == {"items": 16, "passed": 0, "failed": 8, "unknown": 8}
    assert artifact["comparable"] is False and artifact["passed_difference"] is None
    assert [row["outcome"] for row in artifact["rows"]] == ["passed", "failed", "failed", "unknown"] * 8
    assert [row["native"]["cases"][0]["reason"] for row in artifact["rows"] if row["native"]] == ["matched", "wrong_result", "candidate_exit"] * 8
    checked = verify(_json_bytes(artifact), gw, tokenizer)
    assert checked["historical_generation"] == checked["historical_training"] == "declared"
    replayed = run_saved(gw, database, _json_bytes(artifact))
    assert replayed["replayed_declaration_id"] == artifact["artifact_id"]
    assert replayed["base"] == artifact["base"] and replayed["candidate"] == artifact["candidate"]
    assert replayed["historical_generation"] == "declared"


@pytest.mark.parametrize("mutation", ["bytes", "redirect", "source", "state"])
def test_live_completion_rejects_changed_captured_origin(mutation, completed_pair, gw, tokenizer, monkeypatch):
    from ghostwriter_trl.comparison.authority import fresh_models
    import ghostwriter_trl.lora.producer as producer
    completed, *_ = completed_pair
    path, *_ = _comparison_source(completed)
    original_state = completed._ObservedCompletion__state
    backup = path.with_name("original-backup.gwlora")
    if mutation in {"bytes", "redirect"}:
        path.rename(backup)
        if mutation == "bytes": path.write_bytes(b"changed checkpoint bytes")
        else:
            changed = path.with_name("changed-checkpoint.gwlora")
            changed.write_bytes(b"changed checkpoint bytes")
            path.symlink_to(changed)
    elif mutation == "source": monkeypatch.setattr(producer, "training_source_identity", lambda: "0" * 64)
    else: object.__setattr__(completed, "_ObservedCompletion__state", tuple(list(original_state)))
    try:
        with pytest.raises((ContractError, OSError)), fresh_models(completed, tokenizer, gw):
            pytest.fail("changed source reached fresh model allocation")
    finally:
        if mutation in {"bytes", "redirect"}:
            path.unlink()
            backup.rename(path)
            if mutation == "redirect": changed.unlink()
        object.__setattr__(completed, "_ObservedCompletion__state", original_state)


def test_cancellation_during_native_execution_retains_full_unknown_coverage(completed_pair, gw):
    import signal
    import subprocess
    import threading
    import time
    from ghostwriter_trl.comparison.bridge import capture_population
    _, actual, *_ = completed_pair
    database, registration = fixture_population()
    def containers():
        result = subprocess.run(["docker", "ps", "-aq", "--filter", "label=io.ghostwriter.coding-owner"], capture_output=True, check=True)
        return set(result.stdout.splitlines())
    baseline = containers()
    observed = []
    with capture_population(gw, database, registration, "test") as bridge:
        def interrupt_owned_run():
            end = time.monotonic() + 20
            while time.monotonic() < end:
                newly_created = containers() - baseline
                if newly_created:
                    observed.extend(newly_created)
                    bridge._process.send_signal(signal.SIGINT)
                    return
                time.sleep(0.05)
        worker = threading.Thread(target=interrupt_owned_run)
        worker.start()
        try: cancelled = bridge.execute(actual["request"])
        finally: worker.join()
    assert observed, "cancellation must occur after actual owned container creation"
    assert len(cancelled["rows"]) == 64
    assert cancelled["base"]["unknown"] + cancelled["candidate"]["unknown"] > 0
    assert cancelled["comparable"] is False and cancelled["passed_difference"] is None
    assert containers() == baseline


def test_consistently_rehashed_decoded_text_still_requires_actual_token_replay(completed_pair, gw, tokenizer):
    import blake3
    import subprocess
    _, original, *_ = completed_pair
    artifact = deepcopy(original)
    answer = artifact["request"]["rows"][0]
    answer["generation"]["output"]["body_text"] = "def broken(:\n"
    artifact["rows"][0]["generation_id"] = blake3.blake3(_json_bytes(answer), derive_key_context="ghostwriter.coding-generated-answer.v1").hexdigest()
    artifact["rows"][0]["code_id"] = blake3.blake3(b"def broken(:\n", derive_key_context="ghostwriter.coding-module.v1").hexdigest()
    reseal(artifact)
    data = _json_bytes(artifact)
    native = subprocess.run([str(gw), "artifact", "verify-coding-pair", "--stdin"], input=data, capture_output=True)
    assert native.returncode == 0, native.stderr.decode()
    assert json.loads(native.stdout)["historical_generation"] == "declared"
    with pytest.raises(ContractError, match="exact decoded bytes"):
        verify(data, gw, tokenizer)


def test_same_bytes_symlink_preserves_owned_descriptor_capture(completed_pair, gw, tokenizer):
    from ghostwriter_trl.comparison.authority import fresh_models
    completed, *_ = completed_pair
    path, *_ = _comparison_source(completed)
    backup = path.with_name("same-byte-checkpoint.gwlora")
    path.rename(backup)
    path.symlink_to(backup)
    try:
        with fresh_models(completed, tokenizer, gw) as (_, _, models, _):
            assert models["completion_id"] == completed.completion_id
    finally:
        path.unlink()
        backup.rename(path)
