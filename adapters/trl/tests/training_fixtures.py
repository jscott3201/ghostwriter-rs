"""Test-owned fresh random models; no installed/public command grants fixture training authority."""
from contextlib import contextmanager
from pathlib import Path
import tempfile

from ghostwriter_trl.training.capture import _LoadedModel
from ghostwriter_trl.training.producer import _run_loaded
from ghostwriter_trl.training.safe_model import save_model, read_config, load_model


@contextmanager
def owned_model(tokenizer):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    config = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "vocab_size": len(tokenizer),
              "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
              "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
              "max_position_embeddings": 2048, "hidden_act": "silu", "rms_norm_eps": 1e-6,
              "rope_theta": 1000000, "attention_dropout": 0.0, "attention_bias": False,
              "tie_word_embeddings": True, "use_cache": True, "use_sliding_window": False,
              "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
              "pad_token_id": tokenizer.pad_token_id, "torch_dtype": "float32"}
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = Qwen3ForCausalLM(Qwen3Config(**config)).float()
        with tempfile.TemporaryDirectory(prefix="gw-owned-test-model-") as temporary:
            initial = Path(temporary) / "initial"
            save_model(model, config, initial)
            config = read_config(initial / "config.json")
            model, summary = load_model(config, initial / "model.safetensors")
            loaded = object.__new__(_LoadedModel)
            object.__setattr__(loaded, "_LoadedModel__state", (model, config, summary, initial, "owned_software_fixture"))
            yield loaded
    finally:
        torch.set_num_threads(threads)


def qualify_training(prepared, tokenizer, gw, output, **options):
    with owned_model(tokenizer) as loaded:
        return _run_loaded(prepared, tokenizer, loaded, gw, output, **options)
