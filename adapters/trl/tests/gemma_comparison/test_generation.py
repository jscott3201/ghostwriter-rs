"""Real CPU Gemma generation, including repeated and A/B/A cache-isolation controls."""
import pytest

from ghostwriter_trl.artifact import ContractError
from ghostwriter_trl.comparison.generation import cpu_runtime, generate_one, recipe
from ghostwriter_trl.lora.capture import owned_fixture
from ghostwriter_trl.lora.producer import ObservedCompletion, _comparison_source


def test_unregistered_and_saved_receipts_cannot_authorize_generation():
    for value in ({"completion_id": "a" * 64}, object.__new__(ObservedCompletion), object()):
        with pytest.raises(ContractError, match="live owned"):
            _comparison_source(value)
    with pytest.raises(TypeError):
        type("ForgedCompletion", (ObservedCompletion,), {})


def test_actual_repeated_greedy_and_a_b_a_use_fresh_cache(tokenizer):
    import torch
    initial = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
               torch.is_deterministic_algorithms_warn_only_enabled())
    with owned_fixture() as loaded, cpu_runtime():
        model, *_ = loaded._consume()
        settings = recipe(tokenizer, 3, 256, "")
        first = generate_one(model, tokenizer, "Return one Python function.", settings)
        repeat = generate_one(model, tokenizer, "Return one Python function.", settings)
        other = generate_one(model, tokenizer, "Count three apples.", settings)
        again = generate_one(model, tokenizer, "Return one Python function.", settings)
        assert first == repeat == again
        assert other["prompt"]["input_ids"] != first["prompt"]["input_ids"]
        assert len(first["output"]["suffix_ids"]) == 3
        assert first["output"]["termination"] == "length_limit"
        assert first["cache_type"] == "transformers.cache_utils.DynamicCache"
    assert initial == (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
                       torch.is_deterministic_algorithms_warn_only_enabled())


def test_runtime_restores_after_exception():
    import torch
    before = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(), torch.get_rng_state().clone())
    with pytest.raises(RuntimeError, match="cancelled"), cpu_runtime():
        torch.rand(3)
        raise RuntimeError("cancelled")
    assert torch.get_num_threads() == before[0]
    assert torch.are_deterministic_algorithms_enabled() == before[1]
    assert torch.equal(torch.get_rng_state(), before[2])
