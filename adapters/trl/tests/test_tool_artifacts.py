"""Tokenizer-independent v5 interoperability and independently authored corruption controls."""
import json
from pathlib import Path

import datasets
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ghostwriter_trl.artifact import ContractError, read_snapshot, verify_snapshot

KEY = b"ghostwriter.export_artifact"


def test_complete_tools_load_in_pyarrow_and_datasets(gw, historical_fixture_dir, tmp_path):
    path = historical_fixture_dir / "v5-tools.parquet"
    source = json.loads((Path(__file__).parents[3] / "crates/gw-format/tests/fixtures/tool-training-source.json").read_text())
    snapshot = read_snapshot(path, gw)
    assert snapshot.report["artifact"]["manifest"]["column_schema_version"] == "tool_definitions"
    row = snapshot.rows()[0]
    assert json.loads(row["tools_json"]) == source["tools"]
    assert json.loads(row["messages_json"]) == source["messages"]
    assert pq.read_table(pa.BufferReader(snapshot.data)).to_pylist() == snapshot.rows()
    loaded = datasets.Dataset.from_parquet(str(path), cache_dir=str(tmp_path / "datasets"))
    assert loaded.to_list() == snapshot.rows()
    assert loaded.features["tools_json"].dtype == "string"


@pytest.mark.parametrize("mutation", ["missing", "malformed", "object", "json_null", "changed", "empty", "null", "noncanonical", "footer", "corrupt"])
def test_independent_tools_corruptions(gw, historical_fixture_dir, mutation):
    data = (historical_fixture_dir / "v5-tools.parquet").read_bytes()
    if mutation == "corrupt":
        with pytest.raises(ContractError):
            verify_snapshot(data[:len(data) // 2], gw)
        return
    table = pq.read_table(pa.BufferReader(data))
    metadata = {KEY: pq.read_metadata(pa.BufferReader(data)).metadata[KEY]}
    if mutation == "missing":
        table = table.drop(["tools_json"])
    elif mutation == "footer":
        artifact = json.loads(metadata[KEY])
        artifact["artifact_id"] = "0" * 64
        metadata[KEY] = json.dumps(artifact).encode()
    else:
        original = table["tools_json"][0].as_py()
        replacements = {"malformed": "{", "object": "{}", "json_null": "null", "changed": original.replace("Look up", "Search for"), "empty": "[]", "null": None, "noncanonical": original + " "}
        index = table.schema.get_field_index("tools_json")
        table = table.set_column(index, table.schema.field(index), pa.array([replacements[mutation]], type=pa.string()))
    sink = pa.BufferOutputStream()
    pq.write_table(table.replace_schema_metadata(metadata), sink)
    with pytest.raises(ContractError):
        verify_snapshot(sink.getvalue().to_pybytes(), gw)


def test_new_text_artifact_preserves_explicit_null_tools(gw, historical_fixture_dir):
    rows = read_snapshot(historical_fixture_dir / "v5-text.parquet", gw).rows()
    assert len(rows) == 2
    assert all("tools_json" in row and row["tools_json"] is None for row in rows)


@pytest.mark.parametrize("case", ["all", "final", "empty", "empty-final", "empty-gemma", "collision-a", "collision-b"])
def test_v5_screened_artifacts_retain_verified_witnesses(gw, historical_fixture_dir, case):
    snapshot = read_snapshot(historical_fixture_dir / f"v5-screened-{case}.parquet", gw)
    assert snapshot.report["artifact"]["manifest"]["column_schema_version"] == "tool_definitions"
    assert snapshot.report["artifact"]["screening"]["population_check"] == "transaction_checked"
    assert all(row["tools_json"] is None for row in snapshot.rows())
