"""Fresh local Gemma base plus strictly local adapter state, with no automatic loaders."""
from pathlib import Path

from ..artifact import ContractError, strict_json
from ..prepared import _json_bytes
from .config import create
from .safe_tensors import measure, summary
from .shapes import base_shapes, parameter_count, targets, EMBEDDING, HEAD
from .targets import adapter_shapes, attach, audit_parameters
from .tensors import content_id, live_content


def adapter_config(config):
    """Persist inert exact recipe data, without executable class, base path, or hub resolution."""
    return {"version": 1, "kind": "gemma_text_qv_lora", "rank": 8, "alpha": 8, "dropout": 0,
            "bias": "none", "task": "causal_lm", "targets": targets(config), "modules_to_save": [],
            "quantization": False, "dora": False, "rslora": False, "save_embedding_layers": False}


def saved_adapter_shapes(config):
    """The exact keys produced by safe PEFT state extraction for one default adapter."""
    return {name.replace(".default.weight", ".weight"): list(shape)
            for name, shape in adapter_shapes(targets(config)).items()}


def measure_base(config, path):
    return summary(measure(path, base_shapes(config), base=True), parameter_count(config))


def measure_adapter(config, path):
    import math
    shapes = saved_adapter_shapes(config)
    return summary(measure(path, shapes), sum(math.prod(shape) for shape in shapes.values()))


def load_base(config, weights):
    """Read a complete measured safetensor into an independently allocated CPU float32 base."""
    import torch
    from safetensors.torch import load_file
    measured = measure_base(config, weights)
    model = create(config)
    state = load_file(str(weights), device="cpu")
    state.setdefault(HEAD, state[EMBEDDING])
    if set(state) != set(model.state_dict()):
        raise ContractError("actual Gemma model state differs from complete safe inventory")
    model.load_state_dict(state, strict=True, assign=False)
    if (sum(p.numel() for p in model.parameters()) != measured["parameter_count"]
            or any(p.dtype != torch.float32 or p.device.type != "cpu" or not p.requires_grad for p in model.parameters())
            or content_id(live_content(model)) != measured["tensor_content_id"]):
        raise ContractError("fresh Gemma loaded representation differs from captured state")
    model.eval()
    return model, measured


def save_owned_base(model, config, directory):
    """Safe fixture serialization; production accepts only approved release bytes."""
    import torch
    from safetensors.torch import save_file
    directory = Path(directory)
    directory.mkdir()
    state = model.state_dict()
    if not torch.equal(state[HEAD], state[EMBEDDING]):
        raise ContractError("Gemma tied head differs from its embedding")
    del state[HEAD]
    (directory / "config.json").write_bytes(_json_bytes(config))
    save_file({name: tensor.contiguous() for name, tensor in state.items()},
              str(directory / "model.safetensors"), metadata={"format": "pt"})
    return measure_base(config, directory / "model.safetensors")


def save_adapter(model, config, directory):
    """Extract only adapter safetensors; PEFT embedding auto-save is explicitly disabled."""
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    audit_parameters(model, targets(config))
    directory = Path(directory)
    directory.mkdir()
    state = get_peft_model_state_dict(model, adapter_name="default", save_embedding_layers=False)
    if set(state) != set(saved_adapter_shapes(config)):
        raise ContractError("PEFT safe state extraction included unexpected tensors")
    (directory / "config.json").write_bytes(_json_bytes(adapter_config(config)))
    save_file({name: tensor.contiguous() for name, tensor in state.items()},
              str(directory / "adapter_model.safetensors"), metadata={"format": "pt"})
    measured = measure_adapter(config, directory / "adapter_model.safetensors")
    if measured["tensor_content_id"] != content_id(live_content(model, adapter=True)):
        raise ContractError("saved adapter content differs from the actual trained state")
    return measured


def reload_adapter(base, config, configuration, weights):
    """Attach measured local tensors to a fresh known base; saved config never resolves code."""
    from safetensors.torch import load_file
    with Path(configuration).open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536 or _json_bytes(strict_json(raw.decode())) != _json_bytes(adapter_config(config)):
        raise ContractError("unsupported or altered inert adapter configuration")
    measured = measure_adapter(config, weights)
    model, selected = attach(base)
    state = load_file(str(weights), device="cpu")
    parameters = dict(model.named_parameters())
    import torch
    with torch.no_grad():
        for name, value in state.items():
            # Exact inventory/shape checks above permit no missing keys or base parameters.
            parameters[name.removesuffix(".weight") + ".default.weight"].copy_(value)
    audit_parameters(model, selected)
    if content_id(live_content(model, adapter=True)) != measured["tensor_content_id"]:
        raise ContractError("fresh attached adapter differs from captured state")
    model.eval()
    return model, measured
