"""Actual redacted coding exports retain whole modules in immutable prepared SFT inputs."""
import json
import pytest

from ghostwriter_trl.artifact import ContractError, read_snapshot
from ghostwriter_trl.prepared import prepare, read_prepared, verify_prepared
from ghostwriter_trl.projection import project_messages
from .test_prepared_integrity import frame, split


def test_coding_source_and_prepared_inputs_exclude_private_oracles(gw, tokenizer, fixture_dir, tmp_path):
    snapshot = read_snapshot(fixture_dir / "v3-coding.parquet", gw)
    for row in snapshot.rows():
        for field in ("messages_json", "task_json"):
            assert "ORACLE_SENTINEL_HELDOUT" not in row[field]
            assert "protected_cases" not in row[field]
            assert "train_cases" not in row[field]
    data = prepare(snapshot, tokenizer, cot="masked", turns="all_assistant", max_length=2048)
    path = tmp_path / "prepared.gwsft"
    path.write_bytes(data)
    loaded = read_prepared(path, gw, tokenizer)
    assert loaded.build_id == verify_prepared(data, gw, tokenizer).build_id
    assert len(loaded.examples) == 2
    text = json.dumps(loaded.payload, ensure_ascii=False)
    assert "ORACLE_SENTINEL_HELDOUT" not in text
    assert "protected_cases" not in text
    assert "train_cases" not in text
    assert "def merge_closed(intervals):" in text
    assert "def runs(text):" in text
    assert "def common_prefix(strings):" not in json.dumps(loaded.examples)
    assert all(len(example["input_ids"]) <= 2048 for example in loaded.examples)
    for example in loaded.examples:
        messages = json.loads(example["source"]["messages_json"])
        projected = project_messages(messages, tokenizer, "masked")
        assert messages[-1]["content"] in example["rendered"]
        assert example["rendered"] == tokenizer.apply_chat_template(projected, tokenize=False, add_generation_prompt=False)
        assert example["input_ids"] == tokenizer.apply_chat_template(projected, tokenize=True, add_generation_prompt=False)
    assert any("held-out" in rejection["reason"] for rejection in loaded.manifest["rejections"])


def test_coding_complete_rendering_rejects_overflow_without_truncation(gw, tokenizer, fixture_dir):
    snapshot = read_snapshot(fixture_dir / "v3-coding.parquet", gw)
    data = prepare(snapshot, tokenizer, cot="masked", turns="all_assistant", max_length=1)
    loaded = verify_prepared(data, gw, tokenizer)
    assert loaded.examples == []
    assert len(loaded.manifest["rejections"]) == 3
    assert sum("length" in rejection["reason"] for rejection in loaded.manifest["rejections"]) == 2


def test_protected_suite_relabel_is_rejected_by_source_and_prepared_consumers(gw, tokenizer, fixture_dir):
    forged = fixture_dir / "v3-coding-relabelled.parquet"
    with pytest.raises(ContractError, match="Rust artifact verification failed"):
        read_snapshot(forged, gw)
    source = read_snapshot(fixture_dir / "v3-coding.parquet", gw)
    data = prepare(source, tokenizer, cot="masked", turns="all_assistant", max_length=2048)
    payload, _ = split(data)
    # Recompute outer framing: source semantics must fail before later payload-binding checks.
    with pytest.raises(ContractError, match="Rust prepared input verification failed"):
        verify_prepared(frame(payload, forged.read_bytes()), gw, tokenizer)
