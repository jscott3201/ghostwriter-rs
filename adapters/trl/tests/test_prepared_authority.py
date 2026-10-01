"""Capture-once ownership, historical provenance, and real loaded trainer consumption."""
from copy import copy, deepcopy
from pathlib import Path
import platform
import subprocess

import pytest

from ghostwriter_trl.artifact import ContractError, read_snapshot
from ghostwriter_trl.build import identity
from ghostwriter_trl.handoff import qualify_prepared_handoff
from ghostwriter_trl.prepared import VerifiedPrepared, prepare, read_prepared, save_prepared, verify_prepared
from .test_prepared_integrity import frame, split


def test_saved_build_and_source_capture_survive_path_replacement(gw, fixture_dir, tokenizer, tmp_path, monkeypatch):
    source_path = tmp_path / "source.parquet"
    original = (fixture_dir / "screened-all.parquet").read_bytes()
    source_path.write_bytes(original)
    actual_run = subprocess.run

    def replace_source(*args, **kwargs):
        source_path.write_bytes((fixture_dir / "screened-empty.parquet").read_bytes())
        return actual_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", replace_source)
    source = read_snapshot(source_path, gw)
    data = prepare(source, tokenizer, cot="masked", turns="all_assistant", max_length=2048)
    saved = tmp_path / "prepared.gwsft"
    save_prepared(saved, data)

    def replace_bundle(*args, **kwargs):
        saved.write_bytes((fixture_dir / "prepared-empty.gwsft").read_bytes())
        return actual_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", replace_bundle)
    loaded = read_prepared(saved, gw, tokenizer)
    assert loaded.data == data and len(loaded.examples) == 4
    assert loaded.report["source_verification"] == source.report
    assert split(loaded.data)[1] == original
    assert saved.read_bytes() != loaded.data
    assert source_path.read_bytes() != original


def test_verified_state_is_opaque_and_all_metadata_is_defensive(gw, fixture_dir, tokenizer):
    loaded = read_prepared(fixture_dir / "prepared-all.gwsft", gw, tokenizer)
    before = loaded.data, loaded.payload, loaded.report, loaded.replay_report, loaded.build_id
    loaded.examples[3]["input_ids"][0] = 0
    loaded.manifest["recipe"]["cot_policy"] = "stripped"
    loaded.payload["source"]["artifact_id"] = "0" * 64
    loaded.report["source_verification"]["artifact"]["screening"]["members"].clear()
    loaded.replay_report["official_tokenizer_replay"] = "failed"
    assert (loaded.data, loaded.payload, loaded.report, loaded.replay_report, loaded.build_id) == before
    assert copy(loaded) is loaded and deepcopy(loaded) is loaded
    with pytest.raises(TypeError):
        VerifiedPrepared(loaded.data, loaded.report)
    with pytest.raises(AttributeError):
        loaded.data = b"forged"
    with pytest.raises(ContractError):
        qualify_prepared_handoff({"data": loaded.data, "report": loaded.report}, tokenizer)


def test_replay_preserves_historical_producer_runtime_and_identity(gw, fixture_dir, tokenizer):
    payload, source = split((fixture_dir / "prepared-all.gwsft").read_bytes())
    manifest = payload["manifest"]
    historical = {"python": "3.12.1", "implementation": "CPython", "system": "Linux", "machine": "x86_64"}
    manifest["recipe"]["runtime"] = historical
    manifest["recipe_id"] = identity(manifest["recipe"])
    for example in payload["examples"]:
        example["example_id"] = identity(["ghostwriter.sft-example.v1", manifest["recipe_id"], example["source"],
                                          example["target_index"], example["input_ids"], example["labels"]])
    manifest["example_ids"] = [example["example_id"] for example in payload["examples"]]
    data = frame(payload, source)
    loaded = verify_prepared(data, gw, tokenizer)
    assert loaded.payload == payload
    assert loaded.data == data and loaded.build_id == data[8:40].hex()
    assert loaded.manifest["recipe"]["runtime"] == historical
    assert loaded.replay_report["runtime"] == {"python": platform.python_version(), "implementation": platform.python_implementation(),
                                               "system": platform.system(), "machine": platform.machine()}
    assert loaded.replay_report["runtime"] != historical
    # Actual semantic pin changes still invalidate historical recipes.
    manifest["recipe"]["tokenizer"]["revision"] = "0" * 40
    manifest["recipe"]["tokenizer_target"]["revision"] = "0" * 40
    with pytest.raises(ContractError):
        verify_prepared(frame(payload, source), gw, tokenizer)


def test_complete_loaded_long_build_reaches_real_collator_and_trainer_without_execution(gw, fixture_dir, tokenizer, monkeypatch):
    import torch
    from transformers import GPT2LMHeadModel
    from transformers.generation.utils import GenerationMixin
    from trl import SFTTrainer

    def forbidden(*args, **kwargs):
        pytest.fail("handoff must not execute forward/generation/training/optimization")

    monkeypatch.setattr(GPT2LMHeadModel, "forward", forbidden)
    monkeypatch.setattr(GenerationMixin, "generate", forbidden)
    monkeypatch.setattr(SFTTrainer, "train", forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, "step", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    loaded = read_prepared(fixture_dir / "prepared-long.gwsft", gw, tokenizer)
    before = loaded.data, loaded.build_id
    lengths = [len(example["input_ids"]) for example in loaded.examples]
    assert min(lengths) > 1024 and len(set(lengths)) > 1
    report = qualify_prepared_handoff(loaded, tokenizer)
    assert report["build_id"] == loaded.build_id
    assert report["real_collator"]["nonpadding_preserved"]
    assert report["real_sft_trainer_dataloader"]["nonpadding_preserved"]
    assert report["real_sft_trainer_dataloader"]["shape"] == [len(lengths), max(lengths)]
    assert report["real_sft_trainer_dataloader"]["padding_masked"]
    assert report["forward_passes"] == report["optimizer_steps"] == 0
    assert (loaded.data, loaded.build_id) == before


def test_saving_never_overwrites_existing_path_or_symlink(gw, fixture_dir, tokenizer, tmp_path):
    data = (fixture_dir / "prepared-all.gwsft").read_bytes()
    destination = tmp_path / "prepared.gwsft"
    destination.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        save_prepared(destination, data)
    assert destination.read_bytes() == b"existing"
    link = tmp_path / "link.gwsft"
    link.symlink_to(destination)
    with pytest.raises(FileExistsError):
        save_prepared(link, data)
    assert destination.read_bytes() == b"existing"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["link.gwsft", "prepared.gwsft"]


def test_preparation_repeats_exact_bytes_and_all_policy_layouts_replay(gw, fixture_dir, tokenizer):
    source = read_snapshot(fixture_dir / "v3-text.parquet", gw)
    identities = set()
    for cot in ("masked", "stripped", "supervised"):
        for turns, count in (("all_assistant", 4), ("final_turn_only", 2)):
            first = prepare(source, tokenizer, cot=cot, turns=turns, max_length=2048)
            second = prepare(source, tokenizer, cot=cot, turns=turns, max_length=2048)
            assert first == second
            loaded = verify_prepared(first, gw, tokenizer)
            assert len(loaded.examples) == count
            identities.add(loaded.build_id)
            assert loaded.manifest["qualification_limits"]["student_weights"] == "unbound"
            assert loaded.manifest["qualification_limits"]["semantic_screening"] == "not_run"
    assert len(identities) == 6
