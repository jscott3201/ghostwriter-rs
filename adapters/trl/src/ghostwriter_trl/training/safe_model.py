"""Bounded known Qwen3 configurations and dense safe tensor content; no automatic loaders."""
import json
import struct
from pathlib import Path

import blake3

from ..artifact import ContractError, strict_json
from ..prepared import _json_bytes

MAX_WEIGHTS = 3 * 1024**3
MAX_CONFIG = 65536
MAX_HEADER = 1024**2


def read_config(path: Path) -> dict:
    """Read the bounded execution subset; unknown model code/configuration is rejected."""
    with path.open("rb") as stream:
        raw = stream.read(MAX_CONFIG + 1)
    if len(raw) > MAX_CONFIG:
        raise ContractError("model configuration exceeds its bound")
    config = strict_json(raw.decode())
    required = {"architectures", "model_type", "vocab_size", "hidden_size", "intermediate_size",
                "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
                "max_position_embeddings", "hidden_act", "rms_norm_eps", "rope_theta", "attention_dropout",
                "attention_bias", "tie_word_embeddings", "use_cache", "use_sliding_window", "eos_token_id"}
    optional = {"bos_token_id", "pad_token_id", "sliding_window", "layer_types", "rope_scaling", "dtype",
                "torch_dtype", "initializer_range", "max_window_layers", "transformers_version"}
    if type(config) is not dict or not required <= config.keys() or config.keys() - required - optional:
        raise ContractError("unsupported model configuration fields")
    bounds = {"vocab_size": (151669, 152064), "hidden_size": (8, 4096), "intermediate_size": (8, 24576),
              "num_hidden_layers": (1, 32), "num_attention_heads": (1, 32), "num_key_value_heads": (1, 32),
              "head_dim": (2, 256), "max_position_embeddings": (2, 131072), "eos_token_id": (0, config["vocab_size"])}
    for key, (lower, upper) in bounds.items():
        if type(config[key]) is not int or not lower <= config[key] <= upper:
            raise ContractError("model shape/token exceeds its bound")
    import math
    if (config["architectures"] != ["Qwen3ForCausalLM"] or config["model_type"] != "qwen3"
            or config["hidden_act"] != "silu" or config["hidden_size"] % 8 or config["head_dim"] % 2
            or config["num_attention_heads"] % config["num_key_value_heads"]
            or any(type(config[key]) is not bool for key in ("tie_word_embeddings", "use_cache", "attention_bias", "use_sliding_window"))
            or config["attention_bias"] or config["use_sliding_window"] or config["attention_dropout"] != 0
            or config.get("sliding_window") is not None or config.get("rope_scaling") is not None
            or config.get("layer_types") not in (None, ["full_attention"] * config["num_hidden_layers"])):
        raise ContractError("unsupported Qwen3 execution configuration")
    for key, lower, upper in (("rms_norm_eps", 0, 0.1), ("rope_theta", 0, 1e9)):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not lower < value <= upper:
            raise ContractError("unsupported model numeric configuration")
    for key in ("bos_token_id", "pad_token_id", "eos_token_id"):
        value = config.get(key)
        if value is not None and (type(value) is not int or not 0 <= value < config["vocab_size"]):
            raise ContractError("model token identity is out of range")
    for key in ("dtype", "torch_dtype"):
        if config.get(key) not in (None, "float32", "bfloat16"):
            raise ContractError("unsupported weight conversion")
    if config.get("dtype") and config.get("torch_dtype") and config["dtype"] != config["torch_dtype"]:
        raise ContractError("contradictory storage precision")
    initial = config.get("initializer_range")
    if initial is not None and (type(initial) not in (int, float) or not math.isfinite(initial) or not 0 <= initial <= 1):
        raise ContractError("unsupported initializer")
    window = config.get("max_window_layers")
    if window is not None and (type(window) is not int or not 0 <= window <= 64):
        raise ContractError("unsupported disabled sliding window configuration")
    version = config.get("transformers_version")
    if version is not None and (type(version) is not str or not 0 < len(version) <= 64 or any(ord(c) < 32 for c in version)):
        raise ContractError("invalid model producer metadata")
    if parameter_count(config) > 800_000_000:
        raise ContractError("model exceeds parameter bound")
    return config


def shapes(config: dict) -> dict:
    """Exact known model state, including a logical tied output alias."""
    c = config
    h, d, v = c["hidden_size"], c["head_dim"], c["vocab_size"]
    q, k, i = c["num_attention_heads"] * d, c["num_key_value_heads"] * d, c["intermediate_size"]
    result = {"model.embed_tokens.weight": [v, h], "model.norm.weight": [h], "lm_head.weight": [v, h]}
    per_layer = {"input_layernorm.weight": [h], "post_attention_layernorm.weight": [h],
                 "self_attn.q_proj.weight": [q, h], "self_attn.k_proj.weight": [k, h], "self_attn.v_proj.weight": [k, h],
                 "self_attn.o_proj.weight": [h, q], "self_attn.q_norm.weight": [d], "self_attn.k_norm.weight": [d],
                 "mlp.gate_proj.weight": [i, h], "mlp.up_proj.weight": [i, h], "mlp.down_proj.weight": [h, i]}
    for layer in range(c["num_hidden_layers"]):
        result.update({f"model.layers.{layer}.{name}": shape for name, shape in per_layer.items()})
    return result


def parameter_count(config: dict) -> int:
    import math
    return sum(math.prod(shape) for name, shape in shapes(config).items()
               if name != "lm_head.weight" or not config["tie_word_embeddings"])


def measure_model(config: dict, path: Path, *, require_f32=False) -> dict:
    """Stream actual finite tensor values and compare aliases without trusting metadata hashes."""
    import math
    import numpy as np
    length = path.stat().st_size
    if not 8 < length <= MAX_WEIGHTS:
        raise ContractError("weight file exceeds supported bounds")
    with path.open("rb") as stream:
        header_length = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < header_length <= MAX_HEADER or 8 + header_length >= length:
            raise ContractError("invalid safe tensor header length")
        header = strict_json(stream.read(header_length).decode())
        if type(header) is not dict:
            raise ContractError("invalid safe tensor header")
        metadata = header.pop("__metadata__", {})
        if (type(metadata) is not dict or len(metadata) > 8
                or any(type(v) is not str or len(k) > 128 or len(v) > 256 for k, v in metadata.items())):
            raise ContractError("unsupported safe tensor metadata")
        expected = shapes(config)
        if config["tie_word_embeddings"] and "lm_head.weight" not in header:
            del expected["lm_head.weight"]
        if header.keys() != expected.keys():
            raise ContractError("safe tensor parameter inventory mismatch")
        entries = []
        for name, value in header.items():
            if (type(value) is not dict or value.keys() != {"dtype", "shape", "data_offsets"}
                    or value["shape"] != expected[name] or any(type(v) is not int for v in value["shape"])
                    or value["dtype"] not in ("F32", "BF16") or require_f32 and value["dtype"] != "F32"):
                raise ContractError("unsupported safe tensor shape or dtype")
            width = 4 if value["dtype"] == "F32" else 2
            offsets = value["data_offsets"]
            size = math.prod(value["shape"]) * width
            if (type(offsets) is not list or len(offsets) != 2 or any(type(n) is not int for n in offsets)
                    or offsets[0] < 0 or offsets[0] + size != offsets[1]):
                raise ContractError("invalid safe tensor offsets")
            entries.append((offsets[0], offsets[1], name, width))
        cursor = 0
        content = {}
        for start, end, name, width in sorted(entries):
            if start != cursor or end > length - header_length - 8:
                raise ContractError("safe tensors contain gaps, overlaps or excessive ranges")
            digest = blake3.blake3()
            remaining = end - start
            while remaining:
                raw = stream.read(min(remaining, 65536))
                if not raw or len(raw) % width:
                    raise ContractError("truncated safe tensor content")
                bits = np.frombuffer(raw, dtype="<u4" if width == 4 else "<u2")
                if width == 2:
                    bits = bits.astype("<u4") << 16
                if np.any((bits & 0x7F800000) == 0x7F800000):
                    raise ContractError("nonfinite safe tensor content")
                digest.update(bits.astype("<u4", copy=False).tobytes())
                remaining -= len(raw)
            content[name] = {"shape": expected[name], "f32_blake3": digest.hexdigest()}
            cursor = end
        if cursor != length - header_length - 8 or stream.read(1):
            raise ContractError("safe tensor payload length mismatch")
    if config["tie_word_embeddings"]:
        embedding = content["model.embed_tokens.weight"]
        if content.get("lm_head.weight", embedding) != embedding:
            raise ContractError("tied embedding and head contain different values")
        content["lm_head.weight"] = embedding
    return {"tensor_content_id": blake3.blake3(_json_bytes(content), derive_key_context="ghostwriter.checkpoint-tensors.v1").hexdigest(),
            "parameter_count": parameter_count(config), "tensor_count": len(content)}


def load_model(config: dict, weights: Path):
    """Load only known complete safe weights into a fresh local CPU float32 model."""
    import torch
    from safetensors.torch import load_file
    from transformers import Qwen3Config, Qwen3ForCausalLM
    summary = measure_model(config, weights)
    cfg = Qwen3Config(**config)
    cfg._attn_implementation = "eager"
    with torch.device("cpu"):
        model = Qwen3ForCausalLM(cfg).float()
    state = load_file(str(weights), device="cpu")
    if config["tie_word_embeddings"]:
        state.setdefault("lm_head.weight", state["model.embed_tokens.weight"])
    if set(state) != set(model.state_dict()):
        raise ContractError("actual model state differs from supported safe tensor keys")
    model.load_state_dict(state, strict=True, assign=False)
    if (sum(p.numel() for p in model.parameters()) != summary["parameter_count"]
            or any(p.dtype != torch.float32 or p.device.type != "cpu" or not p.requires_grad for p in model.parameters())):
        raise ContractError("loaded model is not the supported full CPU float32 model")
    model.eval()
    return model, summary


def save_model(model, config: dict, directory: Path) -> tuple[dict, dict]:
    """Save full inference weights without Trainer's pickled arguments or optimizer state."""
    import torch
    from safetensors.torch import save_file
    directory.mkdir()
    output_config = dict(config)
    output_config.pop("dtype", None)
    output_config["torch_dtype"] = "float32"
    (directory / "config.json").write_bytes(json.dumps(output_config, sort_keys=True, allow_nan=False).encode())
    # Explicit alias selection keeps the embedding and omits only its tied output head.
    state = model.state_dict()
    if config["tie_word_embeddings"]:
        if not torch.equal(state["lm_head.weight"], state["model.embed_tokens.weight"]):
            raise ContractError("model's tied output and embedding diverged")
        del state["lm_head.weight"]
    save_file({name: tensor.contiguous() for name, tensor in state.items()},
              str(directory / "model.safetensors"), metadata={"format": "pt"})
    summary = measure_model(output_config, directory / "model.safetensors", require_f32=True)
    return output_config, summary
