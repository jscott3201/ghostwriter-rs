"""Rust-produced byte fixtures and independently specified PyArrow corruption mutations."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import blake3
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ghostwriter_trl.artifact import ContractError, read_snapshot, verify_snapshot
from ghostwriter_trl.build import build

KEY = b"ghostwriter.export_artifact"


def encode(table):
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def changed_column(table, name, values):
    index = table.schema.get_field_index(name)
    return table.set_column(index, table.schema.field(index), pa.array(values, type=table.schema.field(index).type))


@pytest.mark.parametrize("version", ["v2", "v3", "v4"])
@pytest.mark.parametrize("kind", ["empty", "text"])
def test_rust_golden_snapshot_report_and_raw_messages(gw, fixture_dir, version, kind):
    snapshot = read_snapshot(fixture_dir / f"{version}-{kind}.parquet", gw)
    assert snapshot.report["report_version"] == 1
    assert snapshot.report["byte_length"] == len(snapshot.data)
    assert snapshot.report["snapshot_blake3"] == blake3.blake3(snapshot.data).hexdigest()
    assert snapshot.report["snapshot_blake3"] != snapshot.report["artifact"]["artifact_id"]
    rows = snapshot.rows()
    assert len(rows) == (0 if kind == "empty" else 2)
    if rows:
        assert "s\\u0061me" in rows[0]["messages_json"]
        assert "🙂中e\u0301" in rows[0]["messages_json"]
        assert json.loads(rows[0]["messages_json"])[0]["content"] == "SYSTEM same 🙂中e\u0301"


def test_capture_a_replace_path_b_consume_verified_a(gw, fixture_dir, tmp_path, monkeypatch):
    source = tmp_path / "snapshot.parquet"
    a = (fixture_dir / "v3-text.parquet").read_bytes()
    b = (fixture_dir / "v3-empty.parquet").read_bytes()
    source.write_bytes(a)
    real_run = subprocess.run
    seen = []

    def replace_before_verifier(*args, **kwargs):
        seen.append(kwargs["input"])
        source.write_bytes(b)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", replace_before_verifier)
    snapshot = read_snapshot(source, gw)
    assert seen == [a]
    assert source.read_bytes() == b
    assert snapshot.data == a and len(snapshot.rows()) == 2


@pytest.mark.parametrize("mutation", [
    "metadata_missing", "metadata_version", "metadata_unknown", "column_version", "schema",
    "count", "build_hash", "artifact_id", "duplicate_id", "row_hash", "messages", "messages_raw",
    "task_malformed", "task_null", "task_declaration",
])
def test_independent_corruptions_are_rejected_by_rust(gw, fixture_dir, mutation):
    filename = "v3-task.parquet" if mutation.startswith("task_") else "v3-text.parquet"
    table = pq.read_table(fixture_dir / filename)
    metadata = {KEY: pq.read_metadata(fixture_dir / filename).metadata[KEY]}
    artifact = json.loads(metadata[KEY])
    if mutation == "metadata_missing":
        metadata.pop(KEY)
    elif mutation == "metadata_version":
        artifact["metadata_version"] = 99
    elif mutation == "metadata_unknown":
        artifact["unknown"] = True
    elif mutation == "column_version":
        artifact["manifest"]["column_schema_version"] = "role_content_text"
    elif mutation == "schema":
        table = table.rename_columns(["wrong"] + table.column_names[1:])
    elif mutation == "count":
        artifact["manifest"]["n_admitted"] += 1
    elif mutation == "build_hash":
        artifact["manifest"]["build_inputs_hash"] = "0" * 64
    elif mutation == "artifact_id":
        artifact["artifact_id"] = "0" * 64
    elif mutation == "duplicate_id":
        table = changed_column(table, "record_id", ["same"] * len(table))
    elif mutation == "row_hash":
        table = changed_column(table, "record_hash", ["changed"] * len(table))
    elif mutation in {"messages", "messages_raw"}:
        values = table["messages_json"].to_pylist()
        values[-1] = "{" if mutation == "messages" else values[-1] + " "
        table = changed_column(table, "messages_json", values)
    elif mutation.startswith("task_"):
        value = table["task_json"][0].as_py()
        if mutation == "task_malformed":
            value = "{"
        elif mutation == "task_null":
            value = None
        else:
            value = value.replace('"role":"train"', '"role":"test"')
            assert value != table["task_json"][0].as_py()
        table = changed_column(table, "task_json", [value])
    if mutation != "metadata_missing":
        metadata[KEY] = json.dumps(artifact).encode()
    table = table.replace_schema_metadata(metadata)
    with pytest.raises(ContractError, match="Rust artifact verification failed"):
        verify_snapshot(encode(table), gw)


def test_corruption_after_first_1024_rows_is_rejected(gw, fixture_dir):
    data = (fixture_dir / "v3-many.parquet").read_bytes()
    snapshot = verify_snapshot(data, gw)
    assert len(snapshot.rows()) == 1025
    table = pq.read_table(pa.BufferReader(data)).replace_schema_metadata({KEY: pq.read_metadata(pa.BufferReader(data)).metadata[KEY]})
    # Re-encoding without mutation must still verify, so rejection cannot come from dropped metadata.
    assert len(verify_snapshot(encode(table), gw).rows()) == 1025
    values = table["messages_json"].to_pylist()
    values[1024] = "[{}]"
    with pytest.raises(ContractError):
        verify_snapshot(encode(changed_column(table, "messages_json", values)), gw)


@pytest.mark.parametrize("data", [b"", b"PAR1garbagePAR1", b"not parquet"])
def test_malformed_bytes_are_operational_failure_with_no_success_report(gw, data):
    result = subprocess.run([str(gw), "artifact", "verify", "--stdin"], input=data, capture_output=True)
    assert result.returncode == 1 and result.stdout == b""
    with pytest.raises(ContractError):
        verify_snapshot(data, gw)


@pytest.mark.parametrize("field,value", [("report_version", 2), ("report_version", True), ("byte_length", 0), ("snapshot_blake3", "0" * 64), ("extra", None)])
def test_bridge_validates_version_length_digest_and_exact_report(gw, fixture_dir, monkeypatch, field, value):
    data = (fixture_dir / "v3-text.parquet").read_bytes()
    report = verify_snapshot(data, gw).report
    report[field] = value
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(report).encode()))
    with pytest.raises(ContractError):
        verify_snapshot(data, gw)


@pytest.mark.parametrize("turns,count", [("all_assistant", 4), ("final_turn_only", 2)])
@pytest.mark.parametrize("cot", ["supervised", "masked", "stripped"])
def test_build_preserves_identities_counts_policies_and_source_group(gw, fixture_dir, tokenizer, turns, count, cot):
    snapshot = read_snapshot(fixture_dir / "v3-text.parquet", gw)
    examples, manifest = build(snapshot, tokenizer, cot=cot, turns=turns, max_length=2048)
    assert len(examples) == manifest["expanded_example_count"] == count
    assert manifest["source_record_count"] == manifest["accepted_source_record_count"] == 2
    assert manifest["rejected_item_count"] == 0
    assert len(set(e["example_id"] for e in examples)) == count
    assert manifest["effective_shifted_answer_token_count"] > 0
    assert manifest["recipe"]["cot_policy"] == cot
    assert manifest["qualification_limits"]["grouped_split_qualification"] == "unknown"
    assert manifest["qualification_limits"]["rights_and_execution_lineage"] == "unknown"
    assert {e["target_index"] for e in examples} == ({2, 4} if count == 4 else {4})
    original_rows = {row["record_id"]: row for row in snapshot.rows()}
    for example in examples:
        assert example["source"]["messages_json"] == original_rows[example["source"]["record_id"]]["messages_json"]
        assert example["source"]["artifact_id"] == snapshot.report["artifact"]["artifact_id"]
    if count == 4:
        assert examples[0]["source"]["group_id"] == examples[1]["source"]["group_id"]
    again, again_manifest = build(snapshot, tokenizer, cot=cot, turns=turns, max_length=2048)
    assert examples == again and manifest == again_manifest


def test_unscreened_task_stays_unscreened_and_unqualified(gw, fixture_dir, tokenizer):
    snapshot = read_snapshot(fixture_dir / "v3-task.parquet", gw)
    examples, manifest = build(snapshot, tokenizer, cot="masked", turns="final_turn_only", max_length=2048)
    assert len(examples) == 1
    task = json.loads(examples[0]["source"]["task_json"])
    assert task["provenance"]["split"]["role"] == "train"
    assert "decontam_index_id" not in snapshot.report["artifact"]["manifest"]
    assert examples[0]["source"]["group_kind"] == "declared_task_group"
    assert manifest["qualification_limits"]["contamination_screening"] == "unknown"


def test_empty_and_all_rejected_builds_have_explicit_zero_counts(gw, fixture_dir, tokenizer):
    for name, expected_rejected in [("v3-empty.parquet", 0), ("v3-text.parquet", 4)]:
        snapshot = read_snapshot(fixture_dir / name, gw)
        examples, manifest = build(snapshot, tokenizer, cot="masked", turns="all_assistant", max_length=1)
        assert not examples and manifest["expanded_example_count"] == 0
        assert manifest["rejected_item_count"] == expected_rejected
        assert manifest["supervised_token_count"] == 0


@pytest.mark.parametrize("role", ["validation", "test"])
def test_explicit_heldout_declarations_are_counted_rejections(gw, fixture_dir, tokenizer, role):
    snapshot = read_snapshot(fixture_dir / f"v3-{role}.parquet", gw)
    assert len(snapshot.rows()) == 1  # Valid source integrity does not authorize SFT use.
    examples, manifest = build(snapshot, tokenizer, cot="masked", turns="all_assistant", max_length=2048)
    assert examples == []
    assert manifest["source_record_count"] == manifest["rejected_item_count"] == 1
    assert manifest["rejections"][0]["reason"] == "declared held-out task role is excluded from SFT training preparation"
    assert manifest["rejections"][0]["record_id"] == "heldout-task"
