"""Complete parameter and buffer identities from bounded float32 chunks."""
import blake3

from ..artifact import ContractError
from ..prepared import _json_bytes

DOMAIN = "ghostwriter.gemma-lora-tensors.v1"


def live_content(model, *, adapter=False):
    """Measure every logical base or adapter state tensor without cloning large weights."""
    import torch
    from .safe_tensors import check_pairs, check_values
    from .shapes import base_shapes, targets
    from .targets import adapter_shapes
    from .config import canonical_model_config
    config = canonical_model_config(model)
    expected = ({name.replace(".default.weight", ".weight"): list(shape)
                 for name, shape in adapter_shapes(targets(config)).items()} if adapter
                else base_shapes(config))
    content = {}
    clips = {}
    for name, tensor in sorted(model.state_dict().items()):
        is_adapter = ".lora_A." in name or ".lora_B." in name
        if is_adapter != adapter:
            continue
        if adapter:
            name = name.replace(".default.weight", ".weight")
        else:
            name = name.removeprefix("base_model.model.").replace(".base_layer.", ".")
        if (name in content or name not in expected or list(tensor.shape) != expected[name]
                or tensor.dtype != torch.float32 or tensor.device.type != "cpu" or not tensor.is_contiguous()):
            raise ContractError("unsupported logical tensor state")
        digest = blake3.blake3()
        values = tensor.detach().reshape(-1)
        for start in range(0, values.numel(), 65536):
            chunk = values[start:start + 65536]
            check_values(chunk.numpy().view("<u4"), name, list(tensor.shape), clips)
            digest.update(chunk.numpy().astype("<f4", copy=False).tobytes())
        content[name] = {"shape": list(tensor.shape), "f32_blake3": digest.hexdigest()}
    if content.keys() != expected.keys():
        raise ContractError("logical tensor population differs from the complete expected state")
    check_pairs(clips)
    return content


def content_id(content):
    """Domain-separated commitment to names, shapes and normalized F32 tensor bits."""
    return blake3.blake3(_json_bytes(content), derive_key_context=DOMAIN).hexdigest()
