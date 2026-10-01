"""Real LoRA forward/backward on the official model and independently counted input tokens."""
import pytest

from ghostwriter_trl.lora.execution import run
from ghostwriter_trl.lora.targets import attach
from .fixtures import owned_model, examples


def test_actual_gemma_lora_updates_preserve_every_frozen_base_tensor(tokenizer, tmp_path):
    import torch
    with owned_model() as base:
        assert base.get_input_embeddings().weight is base.get_output_embeddings().weight
        assert base.config.vision_config is None and base.config.audio_config is None
        assert base.config.text_config.hidden_size_per_layer_input == 4
        model, targets = attach(base)
        frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if not parameter.requires_grad}
        adapters = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
        assert len(targets) == 6 and len(adapters) == 12
        assert sum(parameter.numel() for parameter in adapters.values()) == 1728
        observed = run(model, targets, examples(), tokenizer,
            {"max_steps": 2, "batch_size": 1, "accumulation": 2, "learning_rate_millionths": 100}, tmp_path)
        assert observed["optimizer_updates"] == 2
        assert observed["successful_microbatches"] == observed["consumed_examples"] == 4
        assert observed["shifted_supervised_tokens"] == 8
        assert sum(batch["input_tokens"] for batch in observed["microbatches"]) == 64
        assert [batch["example_ids"] for batch in observed["microbatches"]] == [["a" * 64], ["b" * 64]] * 2
        for name, parameter in model.named_parameters():
            assert torch.isfinite(parameter).all()
            if name in frozen:
                assert not parameter.requires_grad and parameter.grad is None
                assert torch.equal(parameter, frozen[name])
            else:
                assert not torch.equal(parameter, adapters[name]), name
        assert all(batch["finite_adapter_gradients"] == 12 for batch in observed["microbatches"])


@pytest.mark.parametrize("mutation", ["parameter", "buffer", "unknown_buffer", "input_substitution"])
def test_actual_execution_rejects_frozen_state_mutation_and_input_substitution(mutation, tokenizer, tmp_path, monkeypatch):
    import torch
    from ghostwriter_trl.artifact import ContractError
    original = torch.optim.AdamW.step
    with owned_model() as base:
        model, targets = attach(base)
        def changed_step(optimizer, *args, **kwargs):
            result = original(optimizer, *args, **kwargs)
            with torch.no_grad():
                if mutation == "parameter": next(p for p in model.parameters() if not p.requires_grad).reshape(-1)[0].add_(1)
                elif mutation == "buffer": base.model.language_model.layers[0].layer_scalar.add_(1)
                elif mutation == "unknown_buffer": base.register_buffer("unexpected", torch.tensor(0.0))
            return result
        monkeypatch.setattr(torch.optim.AdamW, "step", changed_step)
        handle = None
        if mutation == "input_substitution":
            def substitute(module, args, kwargs):
                kwargs["inputs_embeds"] = module.get_input_embeddings()(kwargs.pop("input_ids"))
            handle = base.register_forward_pre_hook(substitute, with_kwargs=True)
        try:
            with pytest.raises(ContractError):
                run(model, targets, examples(), tokenizer,
                    {"max_steps": 2, "batch_size": 1, "accumulation": 2, "learning_rate_millionths": 100}, tmp_path)
        finally:
            if handle is not None: handle.remove()


def test_one_actual_update_rejects_unchanged_lora_a(tokenizer, tmp_path):
    from ghostwriter_trl.artifact import ContractError
    with owned_model() as base:
        model, targets = attach(base)
        with pytest.raises(ContractError, match="did not change"):
            run(model, targets, examples(), tokenizer,
                {"max_steps": 1, "batch_size": 1, "accumulation": 1, "learning_rate_millionths": 100}, tmp_path)
