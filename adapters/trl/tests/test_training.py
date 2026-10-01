"""Actual local CPU full SFT, immutable input binding, and safe completed checkpoints."""
import json
import subprocess
import pytest

from ghostwriter_trl.artifact import ContractError
from ghostwriter_trl.prepared import read_prepared
from ghostwriter_trl.training.bundle import read_checkpoint
from ghostwriter_trl.training.producer import _run_loaded
from .training_fixtures import qualify_training
from .training_fixtures import owned_model


def test_owned_qwen3_optimizes_real_repeated_partial_batches_and_publishes(gw, tokenizer, fixture_dir, tmp_path):
    prepared = read_prepared(fixture_dir / "prepared-all.gwsft", gw, tokenizer)
    before = prepared.data, prepared.build_id
    output = tmp_path / "completed.gwckpt"
    result = qualify_training(prepared, tokenizer, gw, output, max_steps=3, batch_size=3, accumulation=2)
    # Four complete examples form one three-row batch and one one-row partial batch per epoch.
    # Each of three optimizer updates consumes a whole epoch: six microbatches, twelve rows.
    expected_shifted = 3 * sum(sum(label != -100 for label in example["labels"][1:])
                               for example in prepared.examples)
    assert result.observed["optimizer_updates"] == 3
    assert result.observed["successful_microbatches"] == 6
    assert result.observed["consumed_examples"] == 12
    assert result.observed["shifted_supervised_tokens"] == expected_shifted
    assert result.observed["parameter_content_changed"] is True
    assert result.observed["fresh_reload"] == "passed"
    assert (prepared.data, prepared.build_id) == before
    with output.open("rb") as stream:
        verified = subprocess.run([str(gw), "artifact", "verify-checkpoint", "--stdin"],
                                  stdin=stream, capture_output=True, check=False)
    assert verified.returncode == 0, verified.stderr.decode()
    report = json.loads(verified.stdout)
    assert report["prepared_build_id"] == prepared.build_id
    assert report["historical_training"] == "declared"
    assert report["model_reload"] == "not_run"
    assert report["completion_id"] == result.completion_id
    result.observed["optimizer_updates"] = 0
    assert result.observed["optimizer_updates"] == 3
    loaded = read_checkpoint(output, gw, tokenizer)
    assert loaded.prepared.data == prepared.data
    assert loaded.report["fresh_safe_load"] == "passed"
    assert loaded.report["historical_training"] == "declared"
    assert not hasattr(loaded, "observed")


def test_partial_accumulation_flushes_epoch_tail_and_repeats(gw, tokenizer, fixture_dir, tmp_path):
    prepared = read_prepared(fixture_dir / "prepared-all.gwsft", gw, tokenizer)
    result = qualify_training(prepared, tokenizer, gw, tmp_path / "partial.gwckpt",
                              max_steps=3, batch_size=1, accumulation=3)
    expected = [*prepared.examples, *prepared.examples[:3]]
    observed = result.observed
    assert observed["successful_microbatches"] == observed["consumed_examples"] == 7
    assert observed["optimizer_updates"] == 3
    assert [row["update"] for row in observed["microbatches"]] == [1, 1, 1, 2, 3, 3, 3]
    assert observed["shifted_supervised_tokens"] == sum(sum(label != -100 for label in e["labels"][1:]) for e in expected)


def test_real_long_examples_remain_complete_and_overflow_is_rejected(gw, tokenizer, fixture_dir, tmp_path, monkeypatch):
    prepared = read_prepared(fixture_dir / "prepared-long.gwsft", gw, tokenizer)
    assert min(len(example["input_ids"]) for example in prepared.examples) > 1024
    result = qualify_training(prepared, tokenizer, gw, tmp_path / "long.gwckpt",
                              max_steps=1, batch_size=1, accumulation=2)
    assert [row["input_tokens"] for row in result.observed["microbatches"]] == [len(e["input_ids"]) for e in prepared.examples[:2]]
    assert result.observed["shifted_supervised_tokens"] == sum(
        sum(label != -100 for label in e["labels"][1:]) for e in prepared.examples[:2])
    import ghostwriter_trl.training.producer as producer
    monkeypatch.setattr(producer, "run", lambda *a, **kw: pytest.fail("overflow reached trainer"))
    with pytest.raises(ContractError, match="complete sequences"):
        qualify_training(prepared, tokenizer, gw, tmp_path / "overflow.gwckpt", max_sequence_length=1024)
    assert not (tmp_path / "overflow.gwckpt").exists()


@pytest.mark.parametrize("stage", ["optimizer", "save", "native", "reload", "logits"])
def test_failures_never_publish_a_completed_checkpoint(stage, gw, tokenizer, fixture_dir, tmp_path, monkeypatch):
    import torch
    import ghostwriter_trl.training.bundle as bundle
    import ghostwriter_trl.training.producer as producer
    def fail(*args, **kwargs):
        raise RuntimeError("injected " + stage)
    targets = {"optimizer": (torch.optim.AdamW, "step"), "save": (producer, "save_model"),
               "native": (bundle, "native_report"), "reload": (bundle, "load_model"),
               "logits": (torch.testing, "assert_close")}
    monkeypatch.setattr(*targets[stage], fail)
    prepared = read_prepared(fixture_dir / "prepared-all.gwsft", gw, tokenizer)
    output = tmp_path / "failed.gwckpt"
    with pytest.raises(RuntimeError, match="injected"):
        qualify_training(prepared, tokenizer, gw, output)
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_existing_output_and_publication_race_preserve_other_file(gw, tokenizer, fixture_dir, tmp_path, monkeypatch):
    import ghostwriter_trl.training.producer as producer
    prepared = read_prepared(fixture_dir / "prepared-all.gwsft", gw, tokenizer)
    output = tmp_path / "existing.gwckpt"
    output.write_bytes(b"other owner's file")
    with pytest.raises(ContractError, match="already exists"):
        qualify_training(prepared, tokenizer, gw, output)
    assert output.read_bytes() == b"other owner's file"
    output.unlink()
    original = producer.os.link
    def race(source, destination):
        destination.write_bytes(b"race winner")
        return original(source, destination)
    monkeypatch.setattr(producer.os, "link", race)
    with pytest.raises(FileExistsError):
        qualify_training(prepared, tokenizer, gw, output)
    assert output.read_bytes() == b"race winner"
    assert list(tmp_path.iterdir()) == [output]


def test_input_path_replacement_cannot_change_already_verified_training(gw, tokenizer, fixture_dir, tmp_path):
    source = tmp_path / "source.gwsft"
    source.write_bytes((fixture_dir / "prepared-all.gwsft").read_bytes())
    prepared = read_prepared(source, gw, tokenizer)
    source.write_bytes(b"replacement")
    result = qualify_training(prepared, tokenizer, gw, tmp_path / "captured.gwckpt")
    assert result.observed["prepared_build_id"] == prepared.build_id


def test_independent_saved_reload_logits_agree_for_every_vocab_column(gw, tokenizer, fixture_dir, tmp_path, monkeypatch):
    import torch
    import ghostwriter_trl.training.producer as producer
    prepared = read_prepared(fixture_dir / "prepared-all.gwsft", gw, tokenizer)
    probe = torch.tensor([prepared.examples[1]["input_ids"]], dtype=torch.long)
    original_save = producer.save_model
    before = []
    def record_before_save(model, config, directory):
        assert all(parameter.requires_grad for parameter in model.parameters())
        with torch.no_grad():
            before.append(model(input_ids=probe, use_cache=False).logits.clone())
        return original_save(model, config, directory)
    monkeypatch.setattr(producer, "save_model", record_before_save)
    path = tmp_path / "reload.gwckpt"
    qualify_training(prepared, tokenizer, gw, path)
    independently_loaded = read_checkpoint(path, gw, tokenizer)
    with torch.no_grad():
        after = independently_loaded.model(input_ids=probe, use_cache=False).logits
    assert len(before) == 1
    assert after.shape[-1] == len(tokenizer)
    # Independent numeric oracle, over a complete different example and every vocabulary column.
    assert torch.isfinite(after).all()
    assert torch.all(torch.abs(after - before[0]) <= 1e-6 + 1e-5 * torch.abs(before[0]))


def test_success_declarations_and_subclasses_cannot_construct_capabilities(gw, tokenizer, fixture_dir, tmp_path):
    from ghostwriter_trl.training.capture import _LoadedModel
    from ghostwriter_trl.training.bundle import ReloadedCheckpoint
    from ghostwriter_trl.training.producer import ObservedCompletion
    for cls in (_LoadedModel, ReloadedCheckpoint, ObservedCompletion):
        with pytest.raises(TypeError):
            cls({"success": True})
        with pytest.raises(TypeError):
            type("Forged", (cls,), {})
    prepared = read_prepared(fixture_dir / "prepared-all.gwsft", gw, tokenizer)
    for declaration in ({"model": "approved", "eligible": True}, {"optimizer_updates": 100}, object()):
        with pytest.raises(ContractError, match="freshly loaded"):
            _run_loaded(prepared, tokenizer, declaration, gw, tmp_path / "forged.gwckpt")
    with owned_model(tokenizer) as loaded:
        _run_loaded(prepared, tokenizer, loaded, gw, tmp_path / "once.gwckpt")
        with pytest.raises(ContractError, match="already consumed"):
            _run_loaded(prepared, tokenizer, loaded, gw, tmp_path / "twice.gwckpt")


@pytest.mark.parametrize("change", ["order", "labels"])
def test_actual_collated_rows_match_recorded_order_and_labels_before_forward(change, gw, tokenizer, fixture_dir, tmp_path, monkeypatch):
    from transformers import Qwen3ForCausalLM
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    original = DataCollatorForLanguageModeling.__call__
    def corrupt(self, rows):
        batch = original(self, rows)
        if change == "order":
            batch = {name: value.roll(1, 0) for name, value in batch.items()}
        else:
            batch["labels"][0, 1] = batch["input_ids"][0, 1]
        return batch
    monkeypatch.setattr(DataCollatorForLanguageModeling, "__call__", corrupt)
    monkeypatch.setattr(Qwen3ForCausalLM, "forward", lambda *args, **kwargs: pytest.fail("changed collated input reached forward"))
    prepared = read_prepared(fixture_dir / "prepared-all.gwsft", gw, tokenizer)
    with pytest.raises(ContractError, match="changed"):
        qualify_training(prepared, tokenizer, gw, tmp_path / "changed.gwckpt", batch_size=3)
    assert list(tmp_path.iterdir()) == []
