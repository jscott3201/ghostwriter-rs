"""Record actual CUDA operator precision, including the implementation's FP32 islands."""
from contextlib import contextmanager

from ..artifact import ContractError


@contextmanager
def observe(model):
    """Observe operator outputs inside normalization/RoPE plus softmax/loss globally."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    scope = []
    counts = {}
    hooks = []

    class Operations(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            output = func(*args, **(kwargs or {}))
            name = str(func)
            label = scope[-1] if scope else "model"
            if "softmax" in name:
                label = "softmax"
            elif "nll_loss" in name:
                label = "loss"
            elif label == "model" and name in {"aten.mm.default", "aten.addmm.default", "aten.bmm.default"}:
                label = "matmul"
            if label in {"norm", "rope", "softmax", "loss", "matmul"}:
                tensors = [output] if isinstance(output, torch.Tensor) else output if isinstance(output, (tuple, list)) else []
                for value in tensors:
                    if isinstance(value, torch.Tensor) and value.is_floating_point():
                        key = f"{label}:{name}:{value.dtype}:{value.device.type}"
                        counts[key] = counts.get(key, 0) + 1
            return output

    def enter(label):
        return lambda *args: scope.append(label)

    def leave(*args):
        scope.pop()

    for module in model.modules():
        name = type(module).__name__
        label = "norm" if name == "Gemma4RMSNorm" else "rope" if "RotaryEmbedding" in name else None
        if label:
            hooks.extend([module.register_forward_pre_hook(enter(label)), module.register_forward_hook(leave, always_call=True)])
    try:
        with Operations():
            yield counts
    finally:
        for hook in reversed(hooks):
            hook.remove()


def validate(counts):
    """Require observed FP32 normalization, rotary multiplication, softmax and NLL operations."""
    required = {"norm": "pow", "rope": "bmm", "softmax": "softmax", "loss": "nll_loss"}
    for label, operation in required.items():
        if not any(key.startswith(label + ":") and operation in key and key.endswith(":torch.float32:cuda") and value > 0
                   for key, value in counts.items()):
            raise ContractError(f"missing actual FP32 CUDA {label} execution evidence")

    if not any(key.startswith("matmul:") and key.endswith(":torch.bfloat16:cuda") and value > 0 for key, value in counts.items()):
        raise ContractError("missing actual BF16 CUDA matrix multiplication evidence")
