"""Behavioral controls for accepted-source authority, model lifetime and publication interrupts."""
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import subprocess
import weakref

import blake3
import pyarrow.parquet as pq
import pytest

from ghostwriter_trl.artifact import ContractError, read_snapshot, verify_snapshot
from ghostwriter_trl.prepared import prepare, verify_prepared, read_prepared
from ghostwriter_trl.comparison.artifact import PublishedComparisonError
from ghostwriter_trl.comparison.bridge import capture_population, NativePairBridge
from ghostwriter_trl.comparison.generation import cpu_runtime, recipe
from ghostwriter_trl.comparison.producer import compare
from ghostwriter_trl.comparison.separation import check_separation
from ghostwriter_trl.lora.config import FIXTURE
from ..gemma_lora.test_completion import qualify
from ..test_artifact import KEY, changed_column, encode
from ..test_reference_origin import canonical, rehash
from .test_pair import fixture_population


@pytest.fixture(scope="module")
def live_completion(gw, tokenizer, fixture_dir, tmp_path_factory):
    root = tmp_path_factory.mktemp("repair-live-completion")
    prepared = read_prepared(fixture_dir / "prepared-all.gwsft", gw, tokenizer)
    return qualify(prepared, tokenizer, gw, root / "complete.gwlora")


@pytest.fixture(scope="module")
def accepted_source(gw, tmp_path_factory):
    database, registration = fixture_population()
    with capture_population(gw, database, registration, "test") as bridge:
        population = bridge.population
    path = tmp_path_factory.mktemp("accepted-train") / "train.parquet"
    result = subprocess.run([str(gw), "reference", "export", "--db", str(database),
                             "--batch-id", population["batch_id"], "--out", str(path)], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
    return path, population


def prepared_reference(snapshot, gw, tokenizer):
    data = prepare(snapshot, tokenizer, cot="stripped", turns="all_assistant", max_length=2048,
                   profile="gemma4_e2b_text_v1")
    prepared = verify_prepared(data, gw, tokenizer)
    assert len(prepared.examples) == 64
    return prepared


def reference_record_hash(row):
    origin = json.loads(row["origin_json"])
    origin.pop("kind")
    fields = ("role", "content", "reasoning", "reasoning_details", "tool_calls", "tool_call_id", "name")
    messages = [{key: message.get(key) for key in fields} for message in json.loads(row["messages_json"])]
    return blake3.blake3(canonical({"schema_version": "1.0.0", "training_area": row["training_area"],
                                    "tags": [], "messages": messages, "reference_origin_v1": origin})).hexdigest()


def changed_reference(path, gw):
    table = pq.read_table(path)
    artifact = json.loads(pq.read_metadata(path).metadata[KEY])
    rows = table.to_pylist()
    assert all(reference_record_hash(row) == row["record_hash"] for row in rows)
    row = rows[0]
    messages = json.loads(row["messages_json"])
    assert "return 0" in messages[1]["content"]
    messages[1]["content"] = messages[1]["content"].replace("return 0", "return 1")
    row["messages_json"] = canonical(messages).decode()
    origin = json.loads(row["origin_json"])
    origin["reference_code_id"] = blake3.blake3(messages[1]["content"].encode(),
        derive_key_context="ghostwriter.coding-module.v1").hexdigest()
    row["origin_json"] = canonical(origin).decode()
    row["record_hash"] = reference_record_hash(row)
    for name in ("messages_json", "origin_json", "record_hash"):
        table = changed_column(table, name, [row[name] for row in rows])
    artifact["manifest"]["build_inputs_hash"] = blake3.blake3(
        b"".join(value.encode() + b"\n" for value in sorted(row["record_hash"] for row in rows))).hexdigest()
    artifact["artifact_id"] = rehash(artifact, rows)
    table = table.replace_schema_metadata({KEY: canonical(artifact)})
    return verify_snapshot(encode(table), gw)


def test_exact_accepted_train_export_is_supported(accepted_source, gw, tokenizer):
    path, population = accepted_source
    prepared = prepared_reference(read_snapshot(path, gw), gw, tokenizer)
    with cpu_runtime():
        settings = recipe(tokenizer, 3, 2048, "")
        separation = check_separation(prepared, population, tokenizer, settings, {"source_authorization": FIXTURE})
    assert separation["training_population"] == "registered_reference_train"
    assert len(separation["training_member_ids"]) == 64


def test_rehashed_changed_reference_trains_but_cannot_claim_accepted_train(accepted_source, gw, tokenizer, tmp_path, monkeypatch):
    import ghostwriter_trl.comparison.producer as producer
    path, _ = accepted_source
    changed = changed_reference(path, gw)
    prepared = prepared_reference(changed, gw, tokenizer)
    completed = qualify(prepared, tokenizer, gw, tmp_path / "changed.gwlora")
    assert completed.observed["optimizer_updates"] == 2
    assert completed.observed["fresh_reload"] == "passed"
    def must_not_generate(*args, **kwargs):
        raise AssertionError("unaccepted changed Train content reached generation")
    monkeypatch.setattr(producer, "generate_one", must_not_generate)
    database, registration = fixture_population()
    with pytest.raises(ContractError, match="accepted Train"):
        compare(completed, tokenizer, gw, database, registration, tmp_path / "rejected.json", max_new_tokens=3, max_prompt_tokens=2048)
    assert not (tmp_path / "rejected.json").exists()


def test_new_evaluation_models_are_released_before_native_execution(live_completion, gw, tokenizer, tmp_path, monkeypatch):
    import ghostwriter_trl.comparison.producer as producer
    original = producer.fresh_models
    references = []
    @contextmanager
    def measured(*args, **kwargs):
        with original(*args, **kwargs) as owned:
            references.extend(weakref.ref(model) for model in owned[:2])
            yield owned
    class NativeReached(Exception): pass
    def execute(self, request):
        gc.collect()
        assert all(reference() is None for reference in references), "fresh evaluation model remains owned at native execution"
        raise NativeReached
    def failed_generation(*args, **kwargs): raise RuntimeError("injected generation failure")
    monkeypatch.setattr(producer, "fresh_models", measured)
    monkeypatch.setattr(producer, "generate_one", failed_generation)
    monkeypatch.setattr(NativePairBridge, "execute", execute)
    database, registration = fixture_population()
    with pytest.raises(NativeReached):
        compare(live_completion, tokenizer, gw, database, registration, tmp_path / "unused.json", max_new_tokens=3, max_prompt_tokens=2048)
    assert len(references) == 2 and all(reference() is None for reference in references)


def test_real_link_then_interrupt_reports_retained_identity_and_settles_cleanup(tmp_path, monkeypatch):
    import ghostwriter_trl.comparison.artifact as artifact
    value = {"artifact_id": "a" * 64, "fixture": "publication boundary already verified"}
    monkeypatch.setattr(artifact, "verify", lambda *_: {"artifact_id": value["artifact_id"]})
    link, unlink = os.link, Path.unlink
    removed = []
    def interrupt_after_link(source, destination):
        link(source, destination)
        raise KeyboardInterrupt
    def cleanup(path, *args, **kwargs):
        removed.append(path)
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(os, "link", interrupt_after_link)
    monkeypatch.setattr(Path, "unlink", cleanup)
    output = tmp_path / "retained.json"
    try:
        artifact.publish(value, output, tmp_path / "unused-gw", None)
    except PublishedComparisonError as error:
        assert error.report["artifact_id"] == value["artifact_id"]
        assert error.report["durability"] == "unknown"
    except KeyboardInterrupt:
        pytest.fail("completed link was misreported as ordinary cancellation without retained identity")
    else:
        pytest.fail("interrupted publication returned success")
    assert json.loads(output.read_bytes()) == value
    assert len(removed) == 1 and removed[0].name.startswith(".coding-pair-")
    assert list(tmp_path.iterdir()) == [output]


def test_one_rejected_prompt_retains_complete_unknown_pair_and_saved_semantics(live_completion, gw, tokenizer, tmp_path):
    from ghostwriter_trl.comparison.artifact import read
    from ghostwriter_trl.comparison.bridge import run_saved
    directory = os.environ.get("GW_PAIR_REJECTED_TEST_DIRECTORY")
    if not directory: pytest.fail("GW_PAIR_REJECTED_TEST_DIRECTORY must contain the one-rejected-prompt native fixture")
    directory = Path(directory)
    registration = json.loads((directory / "registration.json").read_text())["registration_id"]
    output = tmp_path / "incomplete.json"
    observed = compare(live_completion, tokenizer, gw, directory / "reference.sqlite", registration,
                       output, max_new_tokens=3, max_prompt_tokens=2048)
    data, inspection = read(output, gw, tokenizer)
    artifact = json.loads(data)
    assert len(artifact["rows"]) == len(artifact["request"]["rows"]) == 64
    assert artifact["base"] == artifact["candidate"] == {"items": 32, "passed": 0, "failed": 31, "unknown": 1}
    assert artifact["comparable"] is False and artifact["passed_difference"] is None
    rejected = [row for row in artifact["request"]["rows"] if row["failure"] == "prompt_rejected"]
    assert len(rejected) == 2 and rejected[0]["member_id"] == rejected[1]["member_id"]
    assert {row["side"] for row in rejected} == {"base", "candidate"}
    separation = artifact["request"]["separation"]
    assert separation["effective_prompt_separation"] == "incomplete_unrenderable_heldout"
    assert separation["unrendered_member_ids"] == [rejected[0]["member_id"]]
    assert inspection["historical_generation"] == inspection["historical_training"] == "declared"
    assert observed.report["automatic_promotion"] is False
    replayed = run_saved(gw, directory / "reference.sqlite", data)
    assert replayed["base"] == artifact["base"] and replayed["candidate"] == artifact["candidate"]
    assert replayed["replayed_declaration_id"] == artifact["artifact_id"]


@pytest.mark.parametrize("collision", ["heldout", "training"])
def test_renderable_prompt_collisions_remain_fatal(accepted_source, gw, tokenizer, monkeypatch, collision):
    import ghostwriter_trl.comparison.separation as separation
    path, population = accepted_source
    prepared = prepared_reference(read_snapshot(path, gw), gw, tokenizer)
    original = separation.render_prompt
    first_training = first_heldout = None
    calls = 0
    def collide(*args, **kwargs):
        nonlocal first_training, first_heldout, calls
        rendered = original(*args, **kwargs)
        calls += 1
        if first_training is None: first_training = rendered
        if calls > 64:
            if first_heldout is None: first_heldout = rendered
            return first_training if collision == "training" else first_heldout
        return rendered
    monkeypatch.setattr(separation, "render_prompt", collide)
    with cpu_runtime(), pytest.raises(ContractError, match="prompt IDs collide"):
        check_separation(prepared, population, tokenizer, recipe(tokenizer, 3, 2048, ""), {"source_authorization": FIXTURE})


def test_cli_interrupted_real_publication_returns_retained_report(tmp_path, monkeypatch, capsys):
    import ghostwriter_trl.comparison.artifact as artifact
    import ghostwriter_trl.comparison.cli as cli
    value = {"artifact_id": "b" * 64}
    output = tmp_path / "retained.json"
    link = os.link
    def interrupted(source, destination):
        link(source, destination)
        raise KeyboardInterrupt
    monkeypatch.setattr(os, "link", interrupted)
    monkeypatch.setattr(artifact, "verify", lambda *_: value)
    monkeypatch.setattr(cli, "load_tokenizer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "replay", lambda *_: artifact.publish(value, output, tmp_path / "gw", None))
    status = cli.main(["replay", "--artifact", "unused", "--tokenizer-directory", "unused", "--gw", "unused",
                       "--db", "unused", "--output", str(output)])
    assert status == 3
    report = json.loads(capsys.readouterr().out)
    assert report["artifact_id"] == value["artifact_id"] and report["durability"] == "unknown"
    assert json.loads(output.read_bytes()) == value
    assert list(tmp_path.iterdir()) == [output]


def test_interrupt_with_foreign_replacement_preserves_target_without_claiming_owned_link(tmp_path, monkeypatch):
    import ghostwriter_trl.comparison.artifact as artifact
    value = {"artifact_id": "c" * 64}
    output = tmp_path / "replaced.json"
    link = os.link
    def replaced(source, destination):
        link(source, destination)
        Path(destination).unlink()
        Path(destination).write_bytes(b"foreign replacement")
        raise KeyboardInterrupt
    monkeypatch.setattr(os, "link", replaced)
    monkeypatch.setattr(artifact, "verify", lambda *_: value)
    with pytest.raises(KeyboardInterrupt):
        artifact.publish(value, output, tmp_path / "gw", None)
    assert output.read_bytes() == b"foreign replacement"
    assert list(tmp_path.iterdir()) == [output]
