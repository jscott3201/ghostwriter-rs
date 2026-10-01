"""Exact Gemma text-attention targets and actual parameter/optimizer membership."""
import re

from ..artifact import ContractError

RANK = 8
PREFIX = "model.language_model.layers."
TARGET = re.compile(r"model\.language_model\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)")


def resolve_targets(model):
    """Measure every required q/v projection on the actual known base before attachment."""
    from torch.nn import Linear
    from transformers import Gemma4ForConditionalGeneration
    if type(model) is not Gemma4ForConditionalGeneration:
        raise ContractError("LoRA requires the known Gemma conditional-generation base")
    config = model.config.text_config
    if config.attention_k_eq_v or config.attention_bias:
        raise ContractError("unsupported Gemma text attention for q/v LoRA")
    first_shared = config.num_hidden_layers - config.num_kv_shared_layers
    expected = {}
    for index in range(config.num_hidden_layers):
        layer = config.per_layer_config[index]
        expected[f"{PREFIX}{index}.self_attn.q_proj"] = (config.num_attention_heads * layer.head_dim, config.hidden_size)
        if index < first_shared:
            expected[f"{PREFIX}{index}.self_attn.v_proj"] = (layer.num_key_value_heads * layer.head_dim, config.hidden_size)
    actual = {name: module for name, module in model.named_modules() if TARGET.fullmatch(name)}
    if actual.keys() != expected.keys():
        raise ContractError("Gemma q/v target population differs from the complete text configuration")
    result = []
    for name, shape in sorted(expected.items()):
        module = actual[name]
        if type(module) is not Linear or tuple(module.weight.shape) != shape or module.bias is not None:
            raise ContractError("Gemma q/v target type or shape differs from the exact recipe")
        result.append({"path": name, "input_features": shape[1], "output_features": shape[0]})
    return result


def attach(model):
    """Attach rank-eight q/v adapters; automatic model or target resolution is never used."""
    from .config import check_lora_dependencies
    check_lora_dependencies()
    from peft import get_peft_model
    targets = resolve_targets(model)
    adapted = get_peft_model(model, _config(targets), adapter_name="default", autocast_adapter_dtype=False)
    adapted.peft_config["default"].base_model_name_or_path = ""
    audit_parameters(adapted, targets)
    return adapted, targets


def _config(targets):
    from peft import LoraConfig, TaskType
    return LoraConfig(r=RANK, lora_alpha=RANK, lora_dropout=0.0, bias="none", task_type=TaskType.CAUSAL_LM,
                        target_modules=[target["path"] for target in targets], modules_to_save=None,
                        init_lora_weights=True, use_dora=False, use_rslora=False, base_model_name_or_path="")


def adapter_shapes(targets):
    """Expected actual parameter names and shapes, independent of requires-grad flags."""
    result = {}
    for target in targets:
        prefix = f"base_model.model.{target['path']}.lora_"
        result[prefix + "A.default.weight"] = (RANK, target["input_features"])
        result[prefix + "B.default.weight"] = (target["output_features"], RANK)
    return result


def audit_parameters(model, targets, optimizer=None):
    """Require exactly the selected adapters to train, on CPU in finite float32."""
    import torch
    from peft import PeftModelForCausalLM
    from peft.tuners.lora.layer import Linear
    if type(model) is not PeftModelForCausalLM:
        raise ContractError("unsupported adapted Gemma model")
    if (set(model.peft_config) != {"default"} or model.active_adapters != ["default"]
            or model.peft_config["default"].to_dict() != _config(targets).to_dict()):
        raise ContractError("actual PEFT configuration differs from the exact LoRA recipe")
    for target in targets:
        layer = model.get_submodule("base_model.model." + target["path"])
        if (type(layer) is not Linear or type(layer.base_layer) is not torch.nn.Linear
                or layer.disable_adapters or layer.merged_adapters or layer.active_adapters != ["default"]
                or layer.r != {"default": RANK} or layer.lora_alpha != {"default": RANK}
                or layer.scaling != {"default": 1.0} or layer.use_dora != {"default": False}
                or layer.lora_bias != {"default": False} or layer.lora_variant or layer.fan_in_fan_out
                or set(layer.lora_dropout) != {"default"}
                or type(layer.lora_dropout["default"]) is not torch.nn.Identity
                or tuple(layer.base_layer.weight.shape) != (target["output_features"], target["input_features"])):
            raise ContractError("actual adapter layer differs from the exact enabled unmerged LoRA recipe")
    shapes = adapter_shapes(targets)
    parameters = dict(model.named_parameters())
    observed = {name for name, parameter in parameters.items() if parameter.requires_grad}
    if observed != shapes.keys() or {name for name in parameters if ".lora_" in name} != shapes.keys():
        raise ContractError("actual trainable population differs from complete Gemma q/v LoRA recipe")
    for name, parameter in parameters.items():
        if parameter.dtype != torch.float32 or parameter.device.type != "cpu":
            raise ContractError("Gemma LoRA qualification requires CPU float32 parameters")
        if name in shapes and (tuple(parameter.shape) != shapes[name] or not torch.isfinite(parameter).all()):
            raise ContractError("LoRA adapter shape or finite-content mismatch")
        if name not in shapes and parameter.grad is not None:
            raise ContractError("frozen Gemma base acquired gradients")
    if optimizer is not None:
        population = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        if (len(population) != len(shapes) or len({id(parameter) for parameter in population}) != len(population)
                or {id(parameter) for parameter in population} != {id(parameters[name]) for name in shapes}):
            raise ContractError("actual optimizer population differs from exact LoRA trainables")
    return [{"name": name, "shape": list(shapes[name]), "parameters": parameters[name].numel(),
             "dtype": "float32", "device": "cpu"} for name in sorted(shapes)]
