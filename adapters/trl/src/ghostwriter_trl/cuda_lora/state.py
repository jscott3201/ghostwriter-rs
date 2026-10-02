"""Separate complete typed parameter and buffer observations for CUDA policy v1."""
import blake3

from ..artifact import ContractError
from ..prepared import _json_bytes

DOMAINS = {kind: f"ghostwriter.gemma-cuda-{kind}.v1" for kind in ("frozen", "adapters", "buffers")}


def logical(name):
    """Remove only PEFT's structural wrappers from frozen state names."""
    return name.removeprefix("base_model.model.").replace(".base_layer.", ".")


def chunks(tensor):
    """Synchronized bounded host copies; never copy a whole large CUDA parameter."""
    import torch
    if not tensor.is_contiguous() or tensor.layout != torch.strided:
        raise ContractError("CUDA state requires dense contiguous tensors")
    if tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)
    flat = tensor.detach().reshape(-1)
    for start in range(0, flat.numel(), 65536):
        yield flat[start:start + 65536].to(device="cpu", non_blocking=False)


def measure(model):
    """Commit every logical parameter and every buffer, including nonpersistent aliases."""
    import torch
    from ..lora.safe_tensors import check_values, check_pairs
    populations = {kind: {} for kind in DOMAINS}
    clips = {}
    parameters = list(model.named_parameters(remove_duplicate=False))
    buffers = list(model.named_buffers(remove_duplicate=False))
    persist = {}
    for module_name, module in model.named_modules(remove_duplicate=False):
        for name, tensor in module._buffers.items():
            if tensor is not None:
                persist[f"{module_name}.{name}".lstrip(".")] = name not in module._non_persistent_buffers_set
    for kind in DOMAINS:
        aliases = {}
        rows = []
        for original, tensor in (buffers if kind == "buffers" else parameters):
            adapter = ".lora_A." in original or ".lora_B." in original
            if kind != "buffers" and adapter != (kind == "adapters"):
                continue
            name = original if adapter else logical(original)
            rows.append((name, original, tensor))
        for name, original, tensor in sorted(rows):
            if name in populations[kind] or tensor.dtype not in (torch.bfloat16, torch.float32, torch.int64, torch.int32, torch.bool):
                raise ContractError("unsupported or duplicate CUDA state tensor")
            key = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tuple(tensor.shape), tensor.dtype)
            alias = aliases.setdefault(key, name)
            digest = blake3.blake3()
            for part in chunks(tensor):
                if part.is_floating_point():
                    check_values(part.float().numpy().view("<u4"), name, list(tensor.shape), clips)
                digest.update(part.view(torch.uint8).numpy().tobytes())
            populations[kind][name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch."),
                "blake3": digest.hexdigest(), "alias": alias, "persistent": persist[original] if kind == "buffers" else True}
    check_pairs(clips)
    return {kind: {"version": 1, "state_id": blake3.blake3(_json_bytes(rows), derive_key_context=DOMAINS[kind]).hexdigest(),
                   "tensors": rows} for kind, rows in populations.items()}
