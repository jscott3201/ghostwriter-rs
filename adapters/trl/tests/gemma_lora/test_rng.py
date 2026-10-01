"""The recorded seed owns initialization and training, independent of caller random state."""
import json
import random
import pytest

from ghostwriter_trl.prepared import read_prepared
from ghostwriter_trl.lora.capture import owned_fixture
from ghostwriter_trl.lora.producer import _run_loaded
from ghostwriter_trl.lora.bundle import native_report


def test_same_owned_input_and_recipe_ignore_ambient_rng_and_restore_callers(gw, tokenizer, fixture_dir, tmp_path):
    import numpy as np
    import torch
    prepared = read_prepared(fixture_dir / "prepared-all.gwsft", gw, tokenizer)
    results = []
    for seed in (17, 941):
        with owned_fixture() as loaded:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            python_before = random.getstate()
            numpy_before = np.random.get_state()
            torch_before = torch.random.get_rng_state().clone()
            output = tmp_path / f"seed-{seed}.gwlora"
            completed = _run_loaded(prepared, tokenizer, loaded, gw, output, accumulation=2)
            report = native_report(output, gw)
            numpy_after = np.random.get_state()
            results.append({"ambient_seed": seed, "completion_id": completed.completion_id,
                "initial_adapter": report["initial_adapter"]["tensor_content_id"],
                "final_adapter": report["final_adapter"]["tensor_content_id"],
                "python_restored": random.getstate() == python_before,
                "numpy_restored": numpy_after[0] == numpy_before[0] and np.array_equal(numpy_after[1], numpy_before[1]) and numpy_after[2:] == numpy_before[2:],
                "torch_restored": torch.equal(torch.random.get_rng_state(), torch_before)})
    (tmp_path / "rng-evidence.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    assert results[0]["initial_adapter"] == results[1]["initial_adapter"]
    assert results[0]["final_adapter"] == results[1]["final_adapter"]
    assert results[0]["completion_id"] == results[1]["completion_id"]
    assert all(item[key] for item in results for key in ("python_restored", "numpy_restored", "torch_restored"))


def test_failed_actual_optimizer_restores_random_state_without_publication(gw, tokenizer, fixture_dir, tmp_path, monkeypatch):
    import numpy as np
    import torch
    prepared = read_prepared(fixture_dir / "prepared-all.gwsft", gw, tokenizer)
    def fail_step(*args, **kwargs):
        raise RuntimeError("injected optimizer failure")
    monkeypatch.setattr(torch.optim.AdamW, "step", fail_step)
    with owned_fixture() as loaded:
        random.seed(17); np.random.seed(17); torch.manual_seed(17)
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.random.get_rng_state().clone()
        with pytest.raises(RuntimeError, match="injected optimizer"):
            _run_loaded(prepared, tokenizer, loaded, gw, tmp_path / "no.gwlora")
        numpy_after = np.random.get_state()
        assert random.getstate() == python_before
        assert numpy_after[0] == numpy_before[0] and np.array_equal(numpy_after[1], numpy_before[1]) and numpy_after[2:] == numpy_before[2:]
        assert torch.equal(torch.random.get_rng_state(), torch_before)
    assert list(tmp_path.iterdir()) == []
