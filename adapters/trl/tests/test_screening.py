"""Actual Rust publication fixtures and independent screened-consumer boundary checks."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ghostwriter_trl.artifact import ContractError, read_snapshot, verify_snapshot
from ghostwriter_trl.build import build

KEY = b"ghostwriter.export_artifact"


def encode(table, artifact):
    sink = pa.BufferOutputStream()
    table = table.replace_schema_metadata({KEY: json.dumps(artifact).encode()})
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


@pytest.mark.parametrize("prefix", ["screened", "v4-screened"])
@pytest.mark.parametrize("kind,turns,count", [
    ("all", "all_assistant", 4), ("final", "final_turn_only", 2),
    ("empty", "all_assistant", 0), ("empty-final", "final_turn_only", 0),
    ("collision-a", "final_turn_only", 1), ("collision-b", "final_turn_only", 1),
])
def test_screened_artifacts_preserve_bound_components_and_source_only_qualification(gw, fixture_dir, tokenizer, kind, turns, count, prefix):
    snapshot = read_snapshot(fixture_dir / f"{prefix}-{kind}.parquet", gw)
    artifact = snapshot.report["artifact"]
    witness = artifact["screening"]
    assert artifact["metadata_version"] == 3
    assert witness["population_check"] == "transaction_checked"
    assert witness["plan"]["population_check"] == "supplied_files_only"
    assert set(witness["plan"]["required_fields"]) == {"content", "reasoning", "reasoning_detail"}
    examples, manifest = build(snapshot, tokenizer, cot="masked", turns=turns, max_length=2048)
    assert len(examples) == count
    assert manifest["rejected_item_count"] == 0
    assert manifest["qualification_limits"]["semantic_screening"] == "not_run"
    assert manifest["qualification_limits"]["effective_prompt_separation"] == "unknown"
    expected = {member["record"]["record_id"]: member for member in witness["members"]}
    bindings = {binding["record"]["record_id"]: binding for binding in witness["plan"]["population"]}
    for example in examples:
        source = example["source"]
        member = expected[source["record_id"]]
        assert source["group_kind"] == "screened_connected_component"
        assert source["group_id"] == member["component_id"]
        assert source["screening"]["plan_id"] == witness["plan"]["plan_id"]
        assert source["screening"]["population_id"] == witness["population_id"]
        assert source["screening"]["record"] == member["record"]
        assert source["screening"]["export_projection_id"] == bindings[source["record_id"]]["export_projection_id"]
        assert source["declared_task"]["split"]["role"] == "train"
    if kind == "all":
        assert len({example["source"]["group_id"] for example in examples}) == 1
        assert len({json.dumps(example["source"]["declared_task"]["group"], sort_keys=True) for example in examples}) == 2
        assert sorted(example["target_index"] for example in examples) == [1, 1, 3, 3]


@pytest.mark.parametrize("kind,cot,turns", [
    ("empty", "supervised", "all_assistant"),
    ("empty", "stripped", "all_assistant"),
    ("empty", "masked", "final_turn_only"),
    ("empty-final", "masked", "all_assistant"),
    ("empty-gemma", "masked", "all_assistant"),
    ("all", "masked", "final_turn_only"),
])
def test_policy_mismatch_rejects_before_tokenizer_or_row_decoding_even_when_empty(gw, fixture_dir, kind, cot, turns, monkeypatch):
    snapshot = read_snapshot(fixture_dir / f"screened-{kind}.parquet", gw)
    def rows_must_not_run(_):
        pytest.fail("rows were decoded before screened policy validation")
    monkeypatch.setattr(type(snapshot), "rows", rows_must_not_run)
    with pytest.raises(ContractError, match="screened artifact target"):
        build(snapshot, object(), cot=cot, turns=turns, max_length=2048)


@pytest.mark.parametrize("mutation", [
    "metadata_version", "witness_missing", "witness_null", "witness_unknown", "witness_version",
    "validation", "population_check", "plan_version", "plan_unknown", "semantic_claim",
    "effective_claim", "component", "member_missing", "member_duplicate", "member_run",
    "population", "plan_id", "policy_id", "layout", "incomplete", "required_fields_missing",
    "required_fields_lowered", "protected_field_omission",
])
def test_screening_metadata_and_membership_tampering_rejects_in_actual_rust_verifier(gw, fixture_dir, mutation):
    path = fixture_dir / "screened-all.parquet"
    table = pq.read_table(path)
    artifact = json.loads(pq.read_metadata(path).metadata[KEY])
    witness = artifact["screening"]
    if mutation == "metadata_version": artifact["metadata_version"] = 1
    elif mutation == "witness_missing": del artifact["screening"]
    elif mutation == "witness_null": artifact["screening"] = None
    elif mutation == "witness_unknown": witness["unknown"] = True
    elif mutation == "witness_version": witness["version"] = 99
    elif mutation == "validation": witness["validation"] = "claimed"
    elif mutation == "population_check": witness["population_check"] = "supplied_files_only"
    elif mutation == "plan_version": witness["plan"]["version"] = 99
    elif mutation == "plan_unknown": witness["plan"]["extra"] = True
    elif mutation == "semantic_claim": witness["plan"]["semantic_status"] = "complete"
    elif mutation == "effective_claim": witness["plan"]["effective_prompt_separation"] = "qualified"
    elif mutation == "component": witness["members"][0]["component_id"] = "0" * 64
    elif mutation == "member_missing": witness["members"].pop()
    elif mutation == "member_duplicate": witness["members"].append(witness["members"][0])
    elif mutation == "member_run": witness["members"][0]["record"]["run_id"] = "different"
    elif mutation == "population": witness["plan"]["population"].pop()
    elif mutation == "plan_id": witness["plan"]["plan_id"] = "0" * 64
    elif mutation == "policy_id": witness["plan"]["policy_id"] = "0" * 64
    elif mutation == "layout": witness["layout"] = "full_conversation_final_v1"
    elif mutation == "incomplete": witness["plan"]["lexical_status"] = "incomplete"
    elif mutation == "required_fields_missing": del witness["plan"]["required_fields"]
    elif mutation == "required_fields_lowered": witness["plan"]["required_fields"] = ["content"]
    elif mutation == "protected_field_omission": witness["plan"]["protected_inputs"][0]["coverage"]["fields"] = ["content"]
    with pytest.raises(ContractError, match="Rust artifact verification failed"):
        verify_snapshot(encode(table, artifact), gw)


@pytest.mark.parametrize("value", [None, {}])
def test_historical_metadata_cannot_carry_a_screening_key(gw, fixture_dir, value):
    path = fixture_dir / "v3-empty.parquet"
    artifact = json.loads(pq.read_metadata(path).metadata[KEY])
    artifact["screening"] = value
    with pytest.raises(ContractError):
        verify_snapshot(encode(pq.read_table(path), artifact), gw)


def test_pinned_template_historical_reasoning_collision_does_not_upgrade_source_claim(gw, fixture_dir, tokenizer):
    snapshots = [read_snapshot(fixture_dir / f"screened-collision-{letter}.parquet", gw) for letter in ("a", "b")]
    builds = [build(snapshot, tokenizer, cot="masked", turns="final_turn_only", max_length=2048) for snapshot in snapshots]
    a, b = [examples[0] for examples, _ in builds]
    # Source final prompts differ in earlier reasoning; Qwen omits that historical reasoning.
    assert a["source"]["group_id"] != b["source"]["group_id"]
    assert a["source"]["messages_json"] != b["source"]["messages_json"]
    assert a["input_ids"] == b["input_ids"]
    assert a["example_id"] != b["example_id"]
    for _, manifest in builds:
        assert manifest["qualification_limits"]["effective_prompt_separation"] == "unknown"
        assert manifest["qualification_limits"]["semantic_screening"] == "not_run"


def test_screened_snapshot_path_and_nested_metadata_edits_cannot_rewrite_authority(gw, fixture_dir, tokenizer, tmp_path, monkeypatch):
    path = tmp_path / "source.parquet"
    original = (fixture_dir / "screened-all.parquet").read_bytes()
    replacement = (fixture_dir / "screened-empty.parquet").read_bytes()
    path.write_bytes(original)
    actual_run = subprocess.run
    def replace_path(*args, **kwargs):
        path.write_bytes(replacement)
        return actual_run(*args, **kwargs)
    monkeypatch.setattr(subprocess, "run", replace_path)
    snapshot = read_snapshot(path, gw)
    exposed = snapshot.report
    exposed["artifact"]["screening"]["members"].clear()
    exposed["artifact"]["screening"]["plan"]["policy_id"] = "forged"
    examples, manifest = build(snapshot, tokenizer, cot="masked", turns="all_assistant", max_length=2048)
    assert snapshot.data == original and path.read_bytes() == replacement
    assert len(examples) == 4
    assert manifest["recipe"]["screening"]["plan_id"] == snapshot.report["artifact"]["screening"]["plan"]["plan_id"]
    before = copy.deepcopy(manifest)
    exposed["artifact"]["screening"] = None
    assert manifest == before


@pytest.mark.parametrize("kind,turns,count", [("all", "all_assistant", 4), ("final", "final_turn_only", 2)])
def test_installed_screened_cli_reaches_actual_collator_and_trainer_dataloader(gw, fixture_dir, tmp_path, kind, turns, count):
    output = tmp_path / "prepared"
    result = subprocess.run([
        sys.executable, "-m", "ghostwriter_trl.cli", "--artifact", str(fixture_dir / f"screened-{kind}.parquet"),
        "--gw", str(gw), "--tokenizer", os.environ["GW_TRL_TOKENIZER"], "--cot", "masked",
        "--turns", turns, "--max-length", "2048", "--output", str(output), "--qualify-handoff",
    ], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    from ghostwriter_trl.prepared import read_prepared
    from ghostwriter_trl.tokenizer import load_tokenizer
    loaded = read_prepared(output / "prepared.gwsft", gw, load_tokenizer(Path(os.environ["GW_TRL_TOKENIZER"])))
    manifest = loaded.manifest
    handoff = json.loads((output / "handoff.json").read_text())
    assert json.loads(result.stdout)["examples"] == count
    assert handoff["real_collator"]["rows"] == count
    assert handoff["real_sft_trainer_dataloader"]["rows"] == count
    assert handoff["real_sft_trainer_dataloader"]["nonpadding_preserved"]
    assert handoff["forward_passes"] == handoff["optimizer_steps"] == 0
    assert manifest["qualification_limits"]["effective_prompt_separation"] == "unknown"


def test_empty_gemma_rejects_in_installed_cli_before_output_creation(gw, fixture_dir, tmp_path):
    output = tmp_path / "must-not-exist"
    result = subprocess.run([
        sys.executable, "-m", "ghostwriter_trl.cli", "--artifact", str(fixture_dir / "screened-empty-gemma.parquet"),
        "--gw", str(gw), "--tokenizer", os.environ["GW_TRL_TOKENIZER"], "--cot", "masked",
        "--turns", "all_assistant", "--max-length", "2048", "--output", str(output),
    ], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1 and "screened artifact target" in result.stderr
    assert not output.exists()

@pytest.mark.parametrize("duplicate", ["metadata_version", "binary64"])
def test_duplicate_keys_do_not_gain_authority_through_typed_screened_metadata(gw, fixture_dir, duplicate):
    path = fixture_dir / "screened-all.parquet"
    table = pq.read_table(path)
    metadata = pq.read_metadata(path).metadata[KEY].decode()
    key_value = '"metadata_version":3' if duplicate == "metadata_version" else '"binary64":"3fe999999999999a"'
    assert key_value in metadata
    metadata = metadata.replace(key_value, f"{key_value},{key_value}", 1)
    sink = pa.BufferOutputStream()
    pq.write_table(table.replace_schema_metadata({KEY: metadata.encode()}), sink)
    with pytest.raises(ContractError, match="Rust artifact verification failed"):
        verify_snapshot(sink.getvalue().to_pybytes(), gw)


def test_earlier_screened_artifacts_without_export_projection_binding_are_unsupported(gw, fixture_dir):
    with pytest.raises(ContractError, match="Rust artifact verification failed"):
        read_snapshot(fixture_dir / "screened-legacy-unbound.parquet", gw)
