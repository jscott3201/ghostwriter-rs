"""Independent framing mutations shared with the Rust importer; no trusted producer helpers."""
from copy import deepcopy
import json
from pathlib import Path
import struct
import subprocess

import blake3
import pytest

from ghostwriter_trl.artifact import ContractError, read_snapshot
from ghostwriter_trl.prepared import prepare, verify_prepared

CASES = json.loads((Path(__file__).parent / "prepared_cases.json").read_text())


def split(data):
    payload_len, source_len = struct.unpack(">QQ", data[40:56])
    assert len(data) == 56 + payload_len + source_len
    return json.loads(data[56:56 + payload_len]), data[56 + payload_len:]


def frame(payload, source):
    raw = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    body = struct.pack(">QQ", len(raw), len(source)) + raw + source
    return b"GWSFT001" + blake3.blake3(body, derive_key_context="ghostwriter.prepared-sft-input.v1").digest() + body


def mutate(data, case):
    payload, source = split(data)
    op = case["op"]
    if op == "replace":
        parts = case["pointer"].strip("/").split("/")
        item = payload
        for part in parts[:-1]:
            item = item[int(part)] if isinstance(item, list) else item[part]
        item[int(parts[-1]) if isinstance(item, list) else parts[-1]] = deepcopy(case["value"])
    elif op == "duplicate_example":
        payload["examples"].append(deepcopy(payload["examples"][-1]))
    elif op == "missing_example":
        payload["examples"].pop()
    elif op in {"reorder_examples", "reorder_and_refs"}:
        payload["examples"].reverse()
        if op == "reorder_and_refs":
            payload["manifest"]["example_ids"].reverse()
    elif op == "missing_null":
        del payload["examples"][3]["source"]["task_json"]
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    if op == "duplicate_field":
        raw = b'{"version":1,' + raw[1:]
    elif op == "nested_duplicate_field":
        raw = raw.replace(b'"max_length":2048', b'"max_length":2048,"max_length":2048', 1)
    elif op == "float":
        raw = raw.replace(b'"version":1', b'"version":1.0', 1)
    elif op == "boolean":
        raw = raw.replace(b'"version":1', b'"version":true', 1)
    elif op == "utf8":
        raw = b'"\xff"'
    elif op == "source":
        source = b"not a parquet artifact"
    changed = frame(raw, source)
    if op == "trailing":
        changed += b"!"
    elif op == "truncated":
        changed = changed[:-1]
    elif op == "length":
        changed = changed[:40] + b"\xff" * 8 + changed[48:]
    elif op == "magic":
        changed = b"GWSFT002" + changed[8:]
    elif op == "stale":
        changed = changed[:60] + bytes([changed[60] ^ 1]) + changed[61:]
    return changed


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_shared_independent_corpus(case, gw, fixture_dir, tokenizer):
    # Frozen bytes were produced by the installed pinned Python producer. Rust consumes the exact
    # same corpus independently; altered cases recompute the complete outer digest where possible.
    data = (fixture_dir / "prepared-all.gwsft").read_bytes()
    changed = mutate(data, case)
    rust = subprocess.run([str(gw), "artifact", "verify-prepared", "--stdin"], input=changed, capture_output=True)
    assert (rust.returncode == 0) == case["rust_accept"], rust.stderr.decode()
    if case["python_accept"]:
        loaded = verify_prepared(changed, gw, tokenizer)
        assert len(loaded.examples) == 4
    else:
        with pytest.raises(ContractError):
            verify_prepared(changed, gw, tokenizer)


@pytest.mark.parametrize("fixture,cot,turns", [
    ("screened-empty", "masked", "all_assistant"),
    ("screened-empty-final", "masked", "final_turn_only"),
    ("v3-test", "masked", "all_assistant"),
])
def test_empty_and_all_rejected_builds_still_validate_source_policy(fixture, cot, turns, gw, fixture_dir, tokenizer):
    source = read_snapshot(fixture_dir / f"{fixture}.parquet", gw)
    data = prepare(source, tokenizer, cot=cot, turns=turns, max_length=2048)
    loaded = verify_prepared(data, gw, tokenizer)
    assert loaded.examples == []
    payload, raw_source = split(data)
    payload["source"]["snapshot_blake3"] = "0" * 64
    with pytest.raises(ContractError):
        verify_prepared(frame(payload, raw_source), gw, tokenizer)
    if fixture.startswith("screened"):
        payload, raw_source = split(data)
        payload["manifest"]["recipe"]["cot_policy"] = "stripped"
        with pytest.raises(ContractError):
            verify_prepared(frame(payload, raw_source), gw, tokenizer)


def test_self_consistent_removed_target_cannot_bypass_source_partition(gw, fixture_dir, tokenizer):
    data = (fixture_dir / "prepared-all.gwsft").read_bytes()
    payload, source = split(data)
    removed = payload["examples"].pop()
    manifest = payload["manifest"]
    manifest["example_ids"].pop()
    for key in ("expanded_example_count", "candidate_target_count"):
        manifest[key] -= 1
    for field, example_field in [("supervised_token_count", "supervised_tokens"), ("context_token_count", "context_tokens"),
                                 ("effective_shifted_supervised_token_count", "effective_shifted_supervised_tokens")]:
        manifest[field] -= removed[example_field]
    manifest["effective_shifted_answer_token_count"] -= len(removed["shifted_answer_token_indices"])
    changed = frame(payload, source)
    rust = subprocess.run([str(gw), "artifact", "verify-prepared", "--stdin"], input=changed, capture_output=True)
    assert rust.returncode != 0
    with pytest.raises(ContractError):
        verify_prepared(changed, gw, tokenizer)
