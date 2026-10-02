"""Current text preparation preserves v5 source bindings and rejects tool inputs explicitly."""
import pytest

from ghostwriter_trl.artifact import read_snapshot
from ghostwriter_trl.build import build


def test_new_text_schema_prepares_with_explicit_null_tools(gw, historical_fixture_dir, tokenizer, profile):
    snapshot = read_snapshot(historical_fixture_dir / "v5-text.parquet", gw)
    examples, manifest = build(snapshot, tokenizer, cot="masked", turns="all_assistant", max_length=4096, profile=profile)
    assert examples
    assert all(example["source"]["tools_json"] is None for example in examples)


@pytest.mark.parametrize("name", ["tools", "definitions-only"])
def test_tool_training_is_explicitly_unsupported(gw, historical_fixture_dir, tokenizer, profile, name):
    snapshot = read_snapshot(historical_fixture_dir / f"v5-{name}.parquet", gw)
    examples, manifest = build(snapshot, tokenizer, cot="masked", turns="all_assistant", max_length=4096, profile=profile)
    assert not examples
    assert any("tool" in rejection["reason"] for rejection in manifest["rejections"])


def test_explicit_empty_definitions_remain_text(gw, historical_fixture_dir, tokenizer, profile):
    snapshot = read_snapshot(historical_fixture_dir / "v5-empty-tools.parquet", gw)
    examples, manifest = build(snapshot, tokenizer, cot="masked", turns="all_assistant", max_length=4096, profile=profile)
    assert examples and not manifest["rejections"]
    assert all(example["source"]["tools_json"] == "[]" for example in examples)
