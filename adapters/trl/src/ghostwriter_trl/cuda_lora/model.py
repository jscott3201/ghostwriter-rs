"""Known local Gemma allocation with parameter-only BF16 conversion and FP32 adapters."""
from ..artifact import ContractError
from ..lora.config import create, check_lora_dependencies
from ..lora.safe_model import measure_base, measure_adapter, saved_adapter_shapes, adapter_config
from ..lora.shapes import EMBEDDING, HEAD
from ..lora.targets import resolve_targets, _config, adapter_shapes
from ..prepared import _json_bytes
from .runtime import require_cuda
from .state import measure


def load_base(config, weights):
    """Allocate once on CUDA from approved captured bytes, retaining FP32 initialized buffers."""
    import torch
    from accelerate import init_empty_weights
    from safetensors import safe_open
    device = require_cuda()
    measured = measure_base(config, weights)
    with init_empty_weights(include_buffers=False):
        model = create(config)
    allocated = {}
    for _, module in model.named_modules():
        for name, parameter in module._parameters.items():
            if parameter is not None:
                key = id(parameter)
                if key not in allocated:
                    allocated[key] = torch.nn.Parameter(torch.empty(parameter.shape, dtype=torch.bfloat16, device=device), requires_grad=False)
                module._parameters[name] = allocated[key]
        for name, buffer in module._buffers.items():
            if buffer is not None:
                if buffer.is_meta:
                    raise ContractError("Gemma buffer initialization was lost during empty allocation")
                module._buffers[name] = buffer.to(device=device, dtype=torch.float32 if buffer.is_floating_point() else buffer.dtype)
    model.tie_weights()
    state = model.state_dict()
    with safe_open(str(weights), framework="pt", device="cpu") as saved:
        keys = set(saved.keys())
        if keys | {HEAD} != set(state):
            raise ContractError("captured source does not match the actual complete Gemma state")
        with torch.no_grad():
            for name in sorted(keys):
                source = saved.get_tensor(name).reshape(-1)
                target = state[name].view(-1)
                for start in range(0, source.numel(), 65536):
                    target[start:start + 65536].copy_(source[start:start + 65536])
    parameters = dict(model.named_parameters(remove_duplicate=False))
    if parameters[HEAD] is not parameters[EMBEDDING]:
        raise ContractError("Gemma tied embedding alias was not preserved")
    model.eval()
    return model, measured


def attach(model):
    """Finish PEFT adapter promotion before exposing optimizer parameter identities."""
    from peft import get_peft_model
    check_lora_dependencies()
    targets = resolve_targets(model)
    model = get_peft_model(model, _config(targets), adapter_name="default", autocast_adapter_dtype=True)
    model.peft_config["default"].base_model_name_or_path = ""
    audit(model, targets)
    return model, targets


def audit(model, targets, optimizer=None):
    """Require all and only FP32 q/v adapters, BF16 frozen parameters and FP32 buffers."""
    import torch
    from peft import PeftModelForCausalLM
    from peft.tuners.lora.layer import Linear
    if (type(model) is not PeftModelForCausalLM or set(model.peft_config) != {"default"}
            or model.active_adapters != ["default"] or model.peft_config["default"].to_dict() != _config(targets).to_dict()):
        raise ContractError("CUDA model differs from the exact q/v adapter policy")
    for target in targets:
        layer = model.get_submodule("base_model.model." + target["path"])
        if (type(layer) is not Linear or type(layer.base_layer) is not torch.nn.Linear
                or layer.disable_adapters or layer.merged_adapters or layer.active_adapters != ["default"]
                or layer.r != {"default": 8} or layer.lora_alpha != {"default": 8}
                or layer.scaling != {"default": 1.0} or layer.use_dora != {"default": False}
                or layer.lora_bias != {"default": False} or layer.lora_variant or layer.fan_in_fan_out
                or set(layer.lora_dropout) != {"default"}
                or type(layer.lora_dropout["default"]) is not torch.nn.Identity
                or tuple(layer.base_layer.weight.shape) != (target["output_features"], target["input_features"])):
            raise ContractError("actual CUDA adapter layer differs from enabled unmerged q/v policy")
    expected = adapter_shapes(targets)
    params = dict(model.named_parameters())
    if ({name for name, p in params.items() if p.requires_grad} != set(expected)
            or {name for name in params if ".lora_" in name} != set(expected)):
        raise ContractError("CUDA trainable population differs from the exact adapter inventory")
    for name, p in params.items():
        if p.device != torch.device("cuda:0") or p.dtype != (torch.float32 if name in expected else torch.bfloat16):
            raise ContractError("CUDA parameter placement or precision differs from policy")
        if name in expected and (tuple(p.shape) != expected[name] or not torch.isfinite(p).all()):
            raise ContractError("CUDA adapter has invalid shape or nonfinite content")
        if name not in expected and p.grad is not None:
            raise ContractError("frozen CUDA parameter acquired gradients")
    for buffer in model.buffers():
        if buffer.device != torch.device("cuda:0") or buffer.is_floating_point() and buffer.dtype != torch.float32:
            raise ContractError("CUDA buffers must retain FP32 floating state")
    if optimizer is not None:
        if type(optimizer) is not torch.optim.AdamW or any(
                group.get(flag) is not False for group in optimizer.param_groups
                for flag in ("foreach", "fused", "capturable", "differentiable", "amsgrad")):
            raise ContractError("CUDA optimizer differs from the explicit AdamW implementation policy")
        population = [p for group in optimizer.param_groups for p in group["params"]]
        if len(population) != len(expected) or {id(p) for p in population} != {id(params[n]) for n in expected}:
            raise ContractError("optimizer must contain every final adapter object exactly once")
        for p, state in optimizer.state.items():
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state and (state[key].dtype != torch.float32 or state[key].device != p.device or not torch.isfinite(state[key]).all()):
                    raise ContractError("AdamW moments must be finite CUDA FP32")
    return [{"name": n, "shape": list(expected[n]), "parameters": params[n].numel(), "dtype": "float32", "device": "cuda:0"} for n in sorted(expected)]


def save_adapter(model, config, directory):
    """Serialize only the exact finite adapter tensors in the existing inert safe format."""
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    from ..lora.shapes import targets
    audit(model, targets(config))
    state = get_peft_model_state_dict(model, adapter_name="default", save_embedding_layers=False)
    if set(state) != set(saved_adapter_shapes(config)):
        raise ContractError("unexpected adapter serialization population")
    directory.mkdir()
    (directory / "config.json").write_bytes(_json_bytes(adapter_config(config)))
    save_file({name: value.detach().cpu().contiguous() for name, value in state.items()}, str(directory / "adapter_model.safetensors"), metadata={"format": "pt"})
    return measure_adapter(config, directory / "adapter_model.safetensors")


def reload_adapter(base, config, weights):
    """Fresh attachment from strictly measured local adapter content."""
    import torch
    from safetensors.torch import load_file
    measured = measure_adapter(config, weights)
    model, targets = attach(base)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in load_file(str(weights), device="cpu").items():
            params[name.removesuffix(".weight") + ".default.weight"].copy_(value)
    audit(model, targets)
    model.eval()
    return model, measured
