"""Owned CPU/eager greedy calls with fresh dynamic caches and measured effective settings."""
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import platform

from ..artifact import ContractError, strict_json
from ..build import identity
from ..lora.config import check_lora_dependencies
from ..lora.runtime import seeded_runtime
from ..profiles import GEMMA
from ..tokenizer import tokenizer_manifest, tokenizer_policy, validate_tokenizer
from .protocol import capture_output, render_prompt


def comparison_source_identity():
    """Bind this separately versioned evaluator without changing preparation identities."""
    root = Path(__file__).parent
    return identity({p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir())
                     if p.is_file() and p.suffix in {".py", ".json"}})


def config_data(maximum):
    """Complete pinned effective Transformers configuration; no caller settings are inherited."""
    if type(maximum) is not int or not 1 <= maximum <= 512:
        raise ContractError("generation token bound must be between 1 and 512")
    data = strict_json((Path(__file__).parent / "generation_config.json").read_text())
    data["max_new_tokens"] = maximum
    return data


@contextmanager
def cpu_runtime():
    """Own and restore CPU thread, RNG, deterministic-algorithm and inference settings."""
    import torch
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=False)
        with seeded_runtime(0), torch.inference_mode():
            yield
    finally:
        try:
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        finally:
            torch.set_num_threads(threads)


def configure(model, maximum):
    """Resolve the actual known model's settings and reject any inherited generation control."""
    import torch
    from peft import PeftModelForCausalLM
    from transformers import GenerationConfig, Gemma4ForConditionalGeneration
    core = model.get_base_model() if type(model) is PeftModelForCausalLM else model
    if (type(core) is not Gemma4ForConditionalGeneration
            or any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters())
            or core.config._attn_implementation != "eager"
            or core.config.text_config._attn_implementation != "eager"):
        raise ContractError("comparison requires the qualified actual CPU float32 eager Gemma models")
    expected = config_data(maximum)
    config = GenerationConfig(**expected)
    # Only these independently loaded owned models are changed. All optional settings are reset.
    core.generation_config = deepcopy(config)
    model.generation_config = deepcopy(config)
    effective, remaining = core._prepare_generation_config(config)
    if remaining or effective.to_dict() != expected:
        raise ContractError("effective generation configuration differs from the pinned recipe")
    model.eval()
    if any(module.training for module in model.modules()):
        raise ContractError("generation model did not enter evaluation mode completely")
    return effective


def recipe(tokenizer, maximum, prompt_bound, system_prompt):
    """Describe the exact pinned source, tokenizer, renderer, runtime and CPU generation recipe."""
    import torch
    validate_tokenizer(tokenizer, GEMMA)
    if type(prompt_bound) is not int or not 1 <= prompt_bound <= 2048 or type(system_prompt) is not str:
        raise ContractError("unsupported prompt bound or system text")
    if len(system_prompt.encode()) > 8192:
        raise ContractError("system prompt exceeds the generation bound")
    if (torch.get_num_threads() != 1 or not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled() or not torch.is_inference_mode_enabled()):
        raise ContractError("generation runtime has not entered its owned deterministic CPU context")
    return {"version": 1, "source_sha256": comparison_source_identity(),
            "dependencies": check_lora_dependencies(),
            "runtime": {"python": platform.python_version(), "implementation": platform.python_implementation(),
                        "system": platform.system(), "machine": platform.machine()},
            "profile": GEMMA, "tokenizer": tokenizer_manifest(GEMMA), "tokenizer_policy": tokenizer_policy(GEMMA),
            "system_prompt": system_prompt, "max_new_tokens": maximum, "max_prompt_tokens": prompt_bound,
            "add_generation_prompt": True, "enable_thinking": False, "preserve_thinking": False,
            "device": "cpu", "dtype": "float32", "attention": "eager", "threads": 1,
            "processes": 1, "deterministic_algorithms": True, "seed": 0, "padding": False,
            "truncation": False, "fresh_cache": True, "cache": "dynamic", "compile": False,
            "effective_config_json": json.dumps(config_data(maximum), sort_keys=True, separators=(",", ":"))}


def generate_one(model, tokenizer, public_prompt, settings):
    """Perform exactly one unpadded call; no past cache, conversation or custom processors enter."""
    import torch
    from transformers.cache_utils import DynamicCache
    messages = ([{"role": "system", "content": settings["system_prompt"]}]
                if settings["system_prompt"] else []) + [{"role": "user", "content": public_prompt}]
    prompt = render_prompt(tokenizer, messages, settings["max_prompt_tokens"])
    if len(prompt["input_ids"]) + settings["max_new_tokens"] > model.config.text_config.max_position_embeddings:
        raise ContractError("complete generation would exceed the model context bound")
    config = configure(model, settings["max_new_tokens"])
    if json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":")) != settings["effective_config_json"]:
        raise ContractError("base and candidate effective generation settings differ")
    ids = torch.tensor([prompt["input_ids"]], dtype=torch.long, device="cpu")
    mask = torch.tensor([prompt["attention_mask"]], dtype=torch.long, device="cpu")
    result = model.generate(input_ids=ids, attention_mask=mask, generation_config=config)
    if (result.sequences.ndim != 2 or result.sequences.shape[0] != 1
            or result.sequences.dtype != torch.long or result.sequences.device.type != "cpu"
            or type(result.past_key_values) is not DynamicCache):
        raise ContractError("actual generation result or fresh dynamic cache differs from the recipe")
    captured = capture_output(tokenizer, prompt["input_ids"], result.sequences[0].tolist(), settings["max_new_tokens"])
    return {"prompt": prompt, "output": captured,
            "effective_max_length": len(prompt["input_ids"]) + settings["max_new_tokens"],
            "cache_type": "transformers.cache_utils.DynamicCache"}
