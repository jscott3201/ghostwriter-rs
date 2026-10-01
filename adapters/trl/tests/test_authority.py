"""Public API verification authority and configuration isolation regressions."""
import copy
import os
from pathlib import Path

import blake3
import pytest

from ghostwriter_trl.artifact import ContractError, VerifiedSnapshot, read_snapshot
from ghostwriter_trl.build import build, identity
from ghostwriter_trl.tokenizer import load_tokenizer


def prepare(snapshot, tokenizer):
    return build(snapshot, tokenizer, cot="masked", turns="all_assistant", max_length=2048)


@pytest.mark.parametrize("correct_raw_digest", [False, True])
def test_equal_count_snapshot_and_report_cannot_be_substituted(gw, fixture_dir, tokenizer, correct_raw_digest):
    a = read_snapshot(fixture_dir / "v2-text.parquet", gw)
    b = read_snapshot(fixture_dir / "v3-text.parquet", gw)
    assert a.report["artifact"]["manifest"]["n_admitted"] == b.report["artifact"]["manifest"]["n_admitted"]
    assert a.report["artifact"]["artifact_id"] != b.report["artifact"]["artifact_id"]
    report = a.report
    if correct_raw_digest:
        # A digest of B does not establish that Rust verified A's claimed metadata for B.
        report["byte_length"] = len(b.data)
        report["snapshot_blake3"] = blake3.blake3(b.data).hexdigest()
    with pytest.raises((TypeError, ContractError)):
        substituted = VerifiedSnapshot(b.data, report)
        prepare(substituted, tokenizer)


def test_nested_report_edit_before_build_cannot_change_verified_authority(gw, fixture_dir, tokenizer):
    snapshot = read_snapshot(fixture_dir / "v3-text.parquet", gw)
    original = copy.deepcopy(snapshot.report)
    exposed = snapshot.report
    exposed["artifact"]["artifact_id"] = "0" * 64
    examples, manifest = prepare(snapshot, tokenizer)
    assert manifest["source_verification"] == original
    assert snapshot.report == original
    assert all(example["source"]["artifact_id"] == original["artifact"]["artifact_id"] for example in examples)


def test_report_edit_after_build_cannot_change_manifest_or_build_identity(gw, fixture_dir, tokenizer):
    snapshot = read_snapshot(fixture_dir / "v3-text.parquet", gw)
    exposed = snapshot.report
    _, manifest = prepare(snapshot, tokenizer)
    original = copy.deepcopy(manifest)
    exposed["artifact"]["manifest"]["cot_policy"] = "stripped"
    assert manifest == original
    payload = dict(manifest)
    build_id = payload.pop("build_id")
    assert identity(payload) == build_id


@pytest.mark.parametrize("mutation", ["split_special_tokens", "additional_special_tokens"])
def test_build_rejects_changed_tokenizer_wrapper_configuration(gw, fixture_dir, tokenizer, mutation):
    snapshot = read_snapshot(fixture_dir / "v3-text.parquet", gw)
    changed = copy.deepcopy(tokenizer)
    if mutation == "split_special_tokens":
        changed.split_special_tokens = True
    else:
        changed.additional_special_tokens = []
    with pytest.raises(ContractError):
        prepare(snapshot, changed)


def test_returned_manifest_cannot_change_later_or_previous_build_pins(gw, fixture_dir, tokenizer):
    snapshot = read_snapshot(fixture_dir / "v3-text.parquet", gw)
    _, first = prepare(snapshot, tokenizer)
    _, previous = prepare(snapshot, tokenizer)
    original = copy.deepcopy(previous)
    exposed = first["recipe"]["tokenizer"]
    revision = exposed["revision"]
    try:
        exposed["revision"] = "changed through returned manifest"
        _, later = prepare(snapshot, tokenizer)
        assert previous == original
        assert later == original
    finally:
        # Keep the failing pre-repair implementation from poisoning other regression tests.
        exposed["revision"] = revision


def test_returned_manifest_cannot_change_subsequent_local_tokenizer_validation(gw, fixture_dir, tokenizer):
    snapshot = read_snapshot(fixture_dir / "v3-text.parquet", gw)
    _, manifest = prepare(snapshot, tokenizer)
    exposed = manifest["recipe"]["tokenizer"]["files"][0]
    original = exposed["sha256"]
    try:
        exposed["sha256"] = "0" * 64
        loaded = load_tokenizer(Path(os.environ["GW_TRL_TOKENIZER"]))
        _, next_manifest = prepare(snapshot, loaded)
        assert next_manifest["recipe"]["tokenizer"]["files"][0]["sha256"] == original
    finally:
        exposed["sha256"] = original


def test_snapshot_public_properties_and_decoded_rows_cannot_rewrite_captured_state(gw, fixture_dir, tokenizer):
    snapshot = read_snapshot(fixture_dir / "v3-text.parquet", gw)
    original = snapshot.data
    with pytest.raises(AttributeError):
        snapshot.data = b"replacement"
    with pytest.raises(AttributeError):
        snapshot.report = {}
    decoded = snapshot.rows()
    decoded[0]["messages_json"] = "[]"
    assert snapshot.data == original
    examples, manifest = prepare(snapshot, tokenizer)
    assert examples and manifest["rejected_item_count"] == 0


def test_build_rejects_snapshot_lookalikes_even_with_a_real_report(gw, fixture_dir, tokenizer):
    from types import SimpleNamespace
    snapshot = read_snapshot(fixture_dir / "v3-text.parquet", gw)
    lookalike = SimpleNamespace(data=snapshot.data, report=snapshot.report, rows=snapshot.rows)
    with pytest.raises(ContractError, match="verified-origin"):
        prepare(lookalike, tokenizer)


@pytest.mark.parametrize("mutation", ["split", "special_map", "added_literals"])
def test_returned_policy_is_detached_from_internal_and_other_builds(gw, fixture_dir, tokenizer, mutation):
    snapshot = read_snapshot(fixture_dir / "v3-text.parquet", gw)
    _, first = prepare(snapshot, tokenizer)
    _, previous = prepare(snapshot, tokenizer)
    original = copy.deepcopy(previous)
    exposed = first["recipe"]["tokenizer_policy"]
    if mutation == "split":
        exposed["wrapper"]["split_special_tokens"] = True
    elif mutation == "special_map":
        exposed["wrapper"]["special_tokens_map"]["additional_special_tokens"].clear()
    else:
        exposed["added_tokens"][0]["content"] = "changed"
    _, later = prepare(snapshot, tokenizer)
    assert previous == later == original
