"""Runtime adapters are checked beyond declared trainable names."""
import pytest

from ghostwriter_trl.artifact import ContractError
from ghostwriter_trl.lora.targets import attach, audit_parameters
from .fixtures import owned_model


@pytest.mark.parametrize("corrupt", ["scale", "disabled", "merged", "dropout", "rank", "target", "optimizer"])
def test_actual_runtime_recipe_corruption_is_rejected(corrupt):
    import torch
    with owned_model() as base:
        model, targets = attach(base)
        layer = model.get_submodule("base_model.model." + targets[0]["path"])
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        if corrupt == "scale":
            layer.scaling["default"] = 2
        elif corrupt == "disabled":
            layer.enable_adapters(False)
            # Restoring flags does not restore effective adapter execution.
            for name, parameter in model.named_parameters():
                if ".lora_" in name:
                    parameter.requires_grad_(True)
        elif corrupt == "merged":
            layer.merged_adapters.append("default")
        elif corrupt == "dropout":
            layer.lora_dropout["default"] = torch.nn.Dropout(0.5)
        elif corrupt == "rank":
            model.peft_config["default"].r = 4
        elif corrupt == "target":
            model.peft_config["default"].target_modules.pop()
        else:
            optimizer.param_groups[0]["params"].append(next(p for p in model.parameters() if not p.requires_grad))
        with pytest.raises(ContractError):
            audit_parameters(model, targets, optimizer)


@pytest.mark.parametrize("corrupt", ["extra_trainable", "missing_trainable", "optimizer_missing", "optimizer_duplicate", "omitted_target", "wrong_shape"])
def test_measured_population_cannot_be_replaced_by_declarations(corrupt):
    import torch
    with owned_model() as base:
        model, targets = attach(base)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        if corrupt == "extra_trainable": next(p for p in model.parameters() if not p.requires_grad).requires_grad_(True)
        elif corrupt == "missing_trainable": next(p for p in model.parameters() if p.requires_grad).requires_grad_(False)
        elif corrupt == "optimizer_missing": optimizer.param_groups[0]["params"].pop()
        elif corrupt == "optimizer_duplicate": optimizer.param_groups[0]["params"].append(optimizer.param_groups[0]["params"][0])
        elif corrupt == "omitted_target": targets.pop()
        else:
            layer = model.get_submodule("base_model.model." + targets[0]["path"])
            layer.lora_A["default"].weight = torch.nn.Parameter(torch.ones(1, 1))
        with pytest.raises(ContractError):
            audit_parameters(model, targets, optimizer)
