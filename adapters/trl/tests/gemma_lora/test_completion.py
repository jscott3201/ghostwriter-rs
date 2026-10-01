"""Real producer, native captured-byte inspection and independent exact base/adapter reload."""
from pathlib import Path
import json
import pytest

from ghostwriter_trl.artifact import ContractError
from ghostwriter_trl.prepared import read_prepared
from ghostwriter_trl.lora.capture import owned_fixture, _LoadedBase
from ghostwriter_trl.lora.producer import _run_loaded, ObservedCompletion
from ghostwriter_trl.lora.bundle import read_checkpoint, native_report, ReloadedCheckpoint


def qualify(prepared, tokenizer, gw, output, **options):
    options.setdefault("accumulation", 2)
    with owned_fixture() as loaded:
        return _run_loaded(prepared, tokenizer, loaded, gw, output, max_steps=2, **options)


def test_real_completed_adapter_native_and_independent_reload(gw, tokenizer, fixture_dir, tmp_path):
    prepared = read_prepared(fixture_dir / "prepared-all.gwsft", gw, tokenizer)
    output = tmp_path / "complete.gwlora"
    completed = qualify(prepared, tokenizer, gw, output)
    report = native_report(output, gw)
    assert report["completion_id"] == completed.completion_id
    assert report["historical_training"] == "declared"
    assert report["model_reload"] == "not_run"
    assert report["base_model"]["parameter_count"] == 8_402_844
    assert report["final_adapter"]["parameter_count"] == 1728
    assert report["final_adapter"]["tensor_count"] == 12
    assert completed.observed["fresh_reload"] == "passed"
    assert completed.observed["base_unchanged"]
    assert completed.observed["optimizer_updates"] == 2
    fresh = read_checkpoint(output, gw, tokenizer)
    assert fresh.prepared.build_id == prepared.build_id
    assert fresh.report["final_adapter"] == report["final_adapter"]
    assert not hasattr(fresh, "observed")
    assert not hasattr(report, "observed")
    for kind in (_LoadedBase, ObservedCompletion, ReloadedCheckpoint):
        with pytest.raises(TypeError):
            kind(report)
    # The receipt owns an independent fresh loaded inference model for the later comparison.
    assert completed._ObservedCompletion__state[2].model is not fresh.model
    evidence = {"completion_id": completed.completion_id, "native": report,
                "observed": completed.observed, "independent_reload": "passed"}
    (tmp_path / "evidence.json").write_text(json.dumps(evidence, sort_keys=True))


def test_no_publish_when_adapter_is_unchanged(gw, tokenizer, fixture_dir, tmp_path):
    prepared = read_prepared(fixture_dir / "prepared-all.gwsft", gw, tokenizer)
    output = tmp_path / "no-completion.gwlora"
    with owned_fixture() as loaded, pytest.raises(ContractError, match="bounds"):
        _run_loaded(prepared, tokenizer, loaded, gw, output, max_steps=1)
    assert not output.exists()


@pytest.mark.parametrize("batch_size,accumulation,steps,updates,rows", [
    (3, 2, 2, [1, 1, 2, 2], [3, 1, 3, 1]),
    (1, 3, 3, [1, 1, 1, 2, 3, 3, 3], [1, 1, 1, 1, 1, 1, 1]),
])
def test_real_padded_batches_and_partial_accumulation_preserve_complete_order(
        batch_size, accumulation, steps, updates, rows, gw, tokenizer, fixture_dir, tmp_path):
    prepared = read_prepared(fixture_dir / "prepared-all.gwsft", gw, tokenizer)
    examples = prepared.examples
    assert len(examples) == 4 and len({len(example["input_ids"]) for example in examples}) > 1
    with owned_fixture() as loaded:
        completed = _run_loaded(prepared, tokenizer, loaded, gw, tmp_path / "partial.gwlora",
                                max_steps=steps, batch_size=batch_size, accumulation=accumulation)
    observed = completed.observed
    batches = observed["microbatches"]
    expected = (examples * 2)[:sum(rows)]
    assert [batch["update"] for batch in batches] == updates
    assert [len(batch["example_ids"]) for batch in batches] == rows
    assert [identity for batch in batches for identity in batch["example_ids"]] == [e["example_id"] for e in expected]
    assert observed["consumed_examples"] == sum(rows)
    assert observed["successful_microbatches"] == len(updates)
    assert observed["optimizer_updates"] == steps
    assert sum(batch["input_tokens"] for batch in batches) == sum(len(e["input_ids"]) for e in expected)
    assert observed["shifted_supervised_tokens"] == sum(sum(label != -100 for label in e["labels"][1:]) for e in expected)
    assert observed["base_unchanged"] and observed["fresh_reload"] == "passed"
