"""Version-one non-thinking plaintext generation; exact IDs precede text interpretation."""
from ..artifact import ContractError
from ..profiles import GEMMA
from ..tokenizer import official_renderer, source_control_literals, tokenizer_policy, validate_tokenizer

STOP_IDS = (1, 106, 50)


def render_prompt(tokenizer, messages, maximum):
    """Render strict text through the actual pinned processor without truncation or padding."""
    validate_tokenizer(tokenizer, GEMMA)
    if type(maximum) is not int or not 1 <= maximum <= 2048 or type(messages) is not list or not messages:
        raise ContractError("generation requires a nonempty bounded message sequence")
    controls = source_control_literals(GEMMA)
    expected = "user"
    for index, message in enumerate(messages):
        if (type(message) is not dict or set(message) - {"role", "content", "reasoning_content"}
                or not {"role", "content"} <= set(message) or type(message["content"]) is not str
                or any(control in message["content"] for control in controls)):
            raise ContractError("generation accepts only strict text messages without literal controls")
        role = message["role"]
        if role == "system" and index == 0:
            if "reasoning_content" in message:
                raise ContractError("system reasoning is unsupported")
            continue
        if role != expected:
            raise ContractError("generation messages must alternate user and assistant turns")
        reasoning = message.get("reasoning_content")
        if reasoning is not None and (role != "assistant" or type(reasoning) is not str
                                       or any(control in reasoning for control in controls)):
            raise ContractError("historical reasoning must be plain assistant text")
        expected = "assistant" if role == "user" else "user"
    if messages[-1]["role"] != "user":
        raise ContractError("generation must end on a user turn")
    rendered = official_renderer(tokenizer, GEMMA).apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        preserve_thinking=False)
    encoded = tokenizer(rendered, add_special_tokens=False, padding=False, truncation=False,
                        return_attention_mask=True, split_special_tokens=False)
    ids, mask = encoded["input_ids"], encoded["attention_mask"]
    if (not ids or len(ids) > maximum or ids[0] != 2 or ids.count(2) != 1
            or mask != [1] * len(ids) or not rendered.endswith("<|turn>model\n")):
        raise ContractError("official generation prompt violates its bound or token contract")
    return {"rendered": rendered, "input_ids": ids, "attention_mask": mask}


def capture_output(tokenizer, prompt_ids, sequence_ids, maximum):
    """Retain every returned token and decode only the exact suffix boundary and observed ending."""
    if (type(maximum) is not int or not 1 <= maximum <= 512
            or any(type(ids) is not list or any(type(i) is not int or not 0 <= i < len(tokenizer) for i in ids)
                   for ids in (prompt_ids, sequence_ids))):
        raise ContractError("generation token capture violates the exact vocabulary or bound")
    result = {"sequence_ids": list(sequence_ids), "suffix_ids": [], "body_ids": [],
              "original_suffix_text": "", "body_text": "", "terminal_id": None,
              "at_token_bound": False, "termination": "prefix_mismatch", "representation": "unknown"}
    if not prompt_ids or sequence_ids[:len(prompt_ids)] != prompt_ids:
        return result
    suffix = sequence_ids[len(prompt_ids):]
    final = suffix[-1] if suffix and suffix[-1] in STOP_IDS else None
    body = suffix[:-1] if final in (1, 106) else suffix
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    result.update(suffix_ids=suffix, body_ids=body, original_suffix_text=decode(suffix),
                  body_text=decode(body), terminal_id=final, at_token_bound=len(suffix) == maximum)
    result["termination"] = ({1: "eos", 106: "turn_end", 50: "tool_handoff"}.get(final)
                             or ("length_limit" if len(suffix) == maximum else "short_return"))
    if len(suffix) > maximum:
        result["termination"] = "over_bound"
        return result
    if result["termination"] == "short_return":
        return result
    added = {entry["id"] for entry in tokenizer_policy(GEMMA)["added_tokens"]}
    unsupported = (final == 50 or not body or any(token in added for token in body)
                   or any(control in result["body_text"] for control in source_control_literals(GEMMA))
                   or not 1 <= len(result["body_text"].encode("utf-8")) <= 65536)
    result["representation"] = "unsupported" if unsupported else "module"
    return result
