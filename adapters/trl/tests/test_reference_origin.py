"""Actual owned synthetic v4 source: strict origins, privacy and complete prepared reconciliation."""
import json
import struct

import blake3
import pyarrow.parquet as pq
import pytest

from ghostwriter_trl.artifact import ContractError, read_snapshot, verify_snapshot
from ghostwriter_trl.build import identity
from ghostwriter_trl.prepared import prepare, verify_prepared, _unframe, _frame, _json_bytes
from .test_artifact import KEY, changed_column, encode


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def rehash(artifact, rows):
    """Independent transcription of the documented v4 framing, including nullable judge fields."""
    def frame(value):
        return len(value).to_bytes(8, "big") + value

    digests = []
    for row in rows:
        data = b"".join(frame(row[key].encode()) for key in ("record_id", "training_area", "record_hash", "prompt_hash"))
        data += b"\0" if row["verdict"] is None else b"\1" + frame(row["verdict"].encode())
        data += b"\0" if row["judge_aggregate"] is None else b"\1" + struct.pack(">d", row["judge_aggregate"])
        data += row["reasoning_tokens"].to_bytes(4, "big") + frame(row["messages_json"].encode())
        data += b"\0" if row["task_json"] is None else b"\1" + frame(row["task_json"].encode())
        data += frame(row["origin_json"].encode())
        digests.append(blake3.blake3(data, derive_key_context="ghostwriter.export.projected-row.v3-record-origins").hexdigest())
    body = artifact["metadata_version"].to_bytes(4, "big") + frame(canonical(artifact["scope"])) + frame(canonical(artifact["manifest"]))
    body += len(rows).to_bytes(8, "big") + b"".join(frame(digest.encode()) for digest in digests)
    return blake3.blake3(body, derive_key_context="ghostwriter.export.artifact.v1").hexdigest()


def test_reference_origin_privacy_and_complete_native_python_preparation(gw, tokenizer, profile, historical_fixture_dir):
    snapshot = read_snapshot(historical_fixture_dir / "v4-reference.parquet", gw)
    rows = snapshot.rows()
    assert len(rows) == 64
    artifact = snapshot.report["artifact"]
    assert artifact["manifest"]["column_schema_version"] == "record_origins"
    assert rehash(artifact, rows) == artifact["artifact_id"]
    for row in rows:
        origin = json.loads(row["origin_json"])
        assert origin["kind"] == "reviewed_reference"
        assert origin["authorship"] == {"author": "agent", "reviewer": "agent"}
        assert origin["permitted_use"] == "training"
        assert row["verdict"] is None and row["judge_aggregate"] is None and row["reasoning_tokens"] == 0
        assert json.loads(row["task_json"])["provenance"]["split"]["role"] == "train"
        public = json.dumps(row)
        assert "PRIVATE_REVIEW_CANARY" not in public and "synthetic-111" not in public
        assert '"train_cases"' not in public and '"protected_cases"' not in public
    data = prepare(snapshot, tokenizer, cot="stripped", turns="all_assistant", max_length=2048, profile=profile)
    loaded = verify_prepared(data, gw, tokenizer)
    assert len(loaded.examples) == 64
    by_id = {row["record_id"]: row for row in rows}
    for example in loaded.examples:
        source = example["source"]
        assert source["origin_json"] == by_id[source["record_id"]]["origin_json"]
        origin = json.loads(source["origin_json"])
        assert source["group_kind"] == "reviewed_reference_component"
        assert source["group_id"] == identity(["ghostwriter.reference-component.v1", origin["catalogue_id"], origin["component"]])
    # Change origin while recomputing the complete outer frame. Native source reconciliation rejects it.
    _, payload_bytes, source_bytes = _unframe(data)
    payload = json.loads(payload_bytes)
    payload["examples"][0]["source"]["origin_json"] = '{"kind":"generated","version":1}'
    with pytest.raises(ContractError):
        verify_prepared(_frame(_json_bytes(payload), source_bytes), gw, tokenizer)


@pytest.mark.parametrize("field", ["suite_id", "reference_code_id", "member_id", "kind", "approved"])
def test_rehashed_reference_origin_attack_is_rejected(gw, historical_fixture_dir, field):
    path = historical_fixture_dir / "v4-reference.parquet"
    table = pq.read_table(path)
    artifact = json.loads(pq.read_metadata(path).metadata[KEY])
    origins = table["origin_json"].to_pylist()
    changed = json.loads(origins[0])
    changed[field] = "generated" if field == "kind" else "0" * 64
    origins[0] = canonical(changed).decode()
    table = changed_column(table, "origin_json", origins)
    artifact["artifact_id"] = rehash(artifact, table.to_pylist())
    table = table.replace_schema_metadata({KEY: canonical(artifact)})
    with pytest.raises(ContractError, match="Rust artifact verification failed"):
        verify_snapshot(encode(table), gw)


def test_screened_reference_preparation_keeps_origin_and_component_bindings(gw, tokenizer, profile, historical_fixture_dir):
    snapshot = read_snapshot(historical_fixture_dir / "v4-screened-reference.parquet", gw)
    witness = snapshot.report["artifact"]["screening"]
    assert all(stratum["teacher"] is None for stratum in witness["plan"]["strata"])
    data = prepare(snapshot, tokenizer, cot="supervised", turns="all_assistant", max_length=2048, profile=profile)
    loaded = verify_prepared(data, gw, tokenizer)
    assert len(loaded.examples) == 64
    members = {member["record"]["record_id"]: member for member in witness["members"]}
    for example in loaded.examples:
        source = example["source"]
        assert json.loads(source["origin_json"])["kind"] == "reviewed_reference"
        assert source["group_kind"] == "screened_connected_component"
        assert source["group_id"] == members[source["record_id"]]["component_id"]
