"""Reduced official conditional-generation model with the selected text features intact."""
from contextlib import contextmanager


@contextmanager
def owned_model():
    import torch
    from transformers import Gemma4Config, Gemma4TextConfig, Gemma4ForConditionalGeneration
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            text = Gemma4TextConfig(vocab_size=262144, vocab_size_per_layer_input=262144,
                hidden_size=16, intermediate_size=32, num_hidden_layers=4,
                num_attention_heads=2, num_key_value_heads=1, head_dim=8, global_head_dim=16,
                hidden_size_per_layer_input=4, num_kv_shared_layers=2, use_double_wide_mlp=True,
                layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
                max_position_embeddings=2048, sliding_window=32, final_logit_softcapping=30.0,
                bos_token_id=2, eos_token_id=1, pad_token_id=0, tie_word_embeddings=True)
            config = Gemma4Config(text_config=text, vision_config=None, audio_config=None, tie_word_embeddings=True)
            config._attn_implementation = "eager"
            model = Gemma4ForConditionalGeneration(config).float()
        yield model
    finally:
        torch.set_num_threads(previous)


def examples():
    """Literal Gemma question/answer tokens with and without a masked thought channel."""
    first = [2, 105, 2364, 107, 15884, 106, 107, 105, 4368, 107, 100, 45518, 107, 36425, 107, 101, 14433, 106, 107]
    second = [2, 105, 2364, 107, 15884, 106, 107, 105, 4368, 107, 14433, 106, 107]
    return [{"example_id": "a" * 64, "input_ids": first, "attention_mask": [1] * 19,
             "labels": [-100] * 16 + [14433, 106, -100]},
            {"example_id": "b" * 64, "input_ids": second, "attention_mask": [1] * 13,
             "labels": [-100] * 10 + [14433, 106, -100]}]
