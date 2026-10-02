"""Opt-in real-device controls. Host runs explicitly skip, never simulate CUDA evidence."""
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from ghostwriter_trl.cuda_lora.producer import train_fixture
from ghostwriter_trl.cuda_lora.bundle import read_checkpoint
from ghostwriter_trl.cuda_lora.ownership import consume
from ghostwriter_trl.cuda_lora.runtime import runtime

pytestmark = pytest.mark.skipif(os.environ.get("GW_TRL_CUDA_QUALIFICATION") != "1", reason="requires explicit owned Linux CUDA qualification")


@pytest.mark.parametrize("batch_size,accumulation,steps,counts", [(1, 1, 2, [1, 1]), (3, 2, 2, [3, 1, 3, 1]), (1, 3, 3, [1, 1, 1, 1, 1, 1, 1])])
def test_actual_fixture_producer_reload_and_consumption(gw, tokenizer, fixture_dir, tmp_path, batch_size, accumulation, steps, counts):
    path = tmp_path / "cuda.gwckpt"
    with train_fixture(fixture_dir / "prepared-all.gwsft", tokenizer, gw, path,
                       batch_size=batch_size, accumulation=accumulation, max_steps=steps, max_sequence_length=256) as result:
        observed = result.observed
        assert observed["optimizer_updates"] == steps
        assert [len(b["example_ids"]) for b in observed["microbatches"]] == counts
        assert observed["fresh_reload"] == "passed"
        assert not hasattr(result, "model")
        assert len(observed["final_state"]["adapters"]["tensors"]) == 12
        assert all(row["process_gpu_bytes"] > 0 for row in observed["model_residency"])
        with read_checkpoint(path, gw, tokenizer) as reloaded:
            assert reloaded.report["fresh_complete_state"] == "passed"
            assert reloaded.report["historical_training"] == "declared"
        # The completion owns its bytes independently of the publication's pathname.
        path.unlink()
        with consume(result) as (owned_path, identity, _):
            assert owned_path.is_file() and identity == result.completion_id


def test_actual_cuda_rng_and_flags_restore_on_success_and_error():
    for fail in (False, True):
        python, numpy = random.getstate(), np.random.get_state()
        cpu, cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state(0).clone()
        flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                 torch.are_deterministic_algorithms_enabled(), torch.get_num_threads())
        try:
            with runtime():
                random.random(); np.random.rand(); torch.rand(2); torch.rand(2, device="cuda:0")
                if fail:
                    raise ValueError("owned qualification failure control")
        except ValueError:
            assert fail
        assert random.getstate() == python
        current = np.random.get_state()
        assert current[0] == numpy[0] and np.array_equal(current[1], numpy[1]) and current[2:] == numpy[2:]
        assert torch.equal(torch.get_rng_state(), cpu) and torch.equal(torch.cuda.get_rng_state(0), cuda)
        assert flags == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                         torch.are_deterministic_algorithms_enabled(), torch.get_num_threads())
