"""Independent complete tensor layouts for the narrow Gemma base and q/v adapters."""
import math

EMBEDDING = "model.language_model.embed_tokens.weight"
HEAD = "lm_head.weight"
CLIPS = ("input_min", "input_max", "output_min", "output_max")


def base_shapes(config):
    """Include frozen modality weights and persistent buffers, plus the logical tied head."""
    c = config["text_config"]
    h, v, p, count = c["hidden_size"], c["vocab_size"], c["hidden_size_per_layer_input"], c["num_hidden_layers"]
    root = "model.language_model."
    result = {EMBEDDING: [v, h], HEAD: [v, h], root + "norm.weight": [h],
              root + "embed_tokens_per_layer.weight": [v, count * p],
              root + "per_layer_model_projection.weight": [count * p, h],
              root + "per_layer_projection_norm.weight": [p]}
    first_shared = count - c["num_kv_shared_layers"]
    for index, kind in enumerate(c["layer_types"]):
        d = c["global_head_dim"] if kind == "full_attention" else c["head_dim"]
        q, k = c["num_attention_heads"] * d, c["num_key_value_heads"] * d
        i = c["intermediate_size"] * (2 if index >= first_shared and c["use_double_wide_mlp"] else 1)
        layer = {"layer_scalar": [1], "mlp.gate_proj.weight": [i, h], "mlp.up_proj.weight": [i, h],
                 "mlp.down_proj.weight": [h, i], "per_layer_input_gate.weight": [p, h],
                 "per_layer_projection.weight": [h, p], "self_attn.q_norm.weight": [d],
                 "self_attn.q_proj.weight": [q, h], "self_attn.o_proj.weight": [h, q]}
        layer.update({name + ".weight": [h] for name in ("input_layernorm", "post_attention_layernorm",
                      "post_feedforward_layernorm", "post_per_layer_input_norm", "pre_feedforward_layernorm")})
        if index < first_shared:
            layer.update({"self_attn.k_norm.weight": [d], "self_attn.k_proj.weight": [k, h],
                          "self_attn.v_proj.weight": [k, h]})
        result.update({f"{root}layers.{index}.{name}": shape for name, shape in layer.items()})
    if config.get("vision_config") is not None:
        _vision(result)
        _audio(result)
    return result


def _linear(result, prefix, output, input):
    result[prefix + ".linear.weight"] = [output, input]
    result.update({prefix + "." + name: [] for name in CLIPS})


def _vision(result):
    root = "model.vision_tower."
    result[root + "patch_embedder.input_proj.weight"] = [768, 768]
    result[root + "patch_embedder.position_embedding_table"] = [2, 10240, 768]
    result["model.embed_vision.embedding_projection.weight"] = [1536, 768]
    for index in range(16):
        prefix = f"{root}encoder.layers.{index}."
        for name in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            result[prefix + name + ".weight"] = [768]
        for name in ("q_norm", "k_norm"):
            result[prefix + "self_attn." + name + ".weight"] = [64]
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            _linear(result, prefix + "self_attn." + name, 768, 768)
        for name in ("gate_proj", "up_proj"):
            _linear(result, prefix + "mlp." + name, 3072, 768)
        _linear(result, prefix + "mlp.down_proj", 768, 3072)


def _audio(result):
    root = "model.audio_tower."
    result.update({root + "output_proj.bias": [1536], root + "output_proj.weight": [1536, 1024],
                   root + "subsample_conv_projection.input_proj_linear.weight": [1024, 1024],
                   "model.embed_audio.embedding_projection.weight": [1536, 1536]})
    for index, output, input in ((0, 128, 1), (1, 32, 128)):
        prefix = f"{root}subsample_conv_projection.layer{index}."
        result[prefix + "conv.weight"] = [output, input, 3, 3]
        result[prefix + "norm.weight"] = [output]
    for index in range(12):
        prefix = f"{root}layers.{index}."
        for ffw in ("feed_forward1", "feed_forward2"):
            _linear(result, prefix + ffw + ".ffw_layer_1", 4096, 1024)
            _linear(result, prefix + ffw + ".ffw_layer_2", 1024, 4096)
            for norm in ("pre_layer_norm", "post_layer_norm"):
                result[prefix + ffw + "." + norm + ".weight"] = [1024]
        for norm in ("conv_norm", "pre_layer_norm"):
            result[prefix + "lconv1d." + norm + ".weight"] = [1024]
        result[prefix + "lconv1d.depthwise_conv1d.weight"] = [1024, 1, 5]
        _linear(result, prefix + "lconv1d.linear_start", 2048, 1024)
        _linear(result, prefix + "lconv1d.linear_end", 1024, 1024)
        for norm in ("norm_out", "norm_post_attn", "norm_pre_attn"):
            result[prefix + norm + ".weight"] = [1024]
        for name in ("q_proj", "k_proj", "v_proj", "post"):
            _linear(result, prefix + "self_attn." + name, 1024, 1024)
        result[prefix + "self_attn.relative_k_proj.weight"] = [1024, 1024]
        result[prefix + "self_attn.per_dim_scale"] = [128]


def targets(config):
    """Derive the exact eligible text projection inventory from complete base shapes."""
    return [{"path": name.removesuffix(".weight"), "input_features": shape[1], "output_features": shape[0]}
            for name, shape in sorted(base_shapes(config).items())
            if name.startswith("model.language_model.layers.") and name.endswith(("self_attn.q_proj.weight", "self_attn.v_proj.weight"))]


def parameter_count(config):
    """Distinct base parameters exclude the tied output alias and clipping buffers."""
    return sum(math.prod(shape) for name, shape in base_shapes(config).items()
               if name != HEAD and name.rsplit(".", 1)[-1] not in (*CLIPS, "layer_scalar"))
