"""Ownership ledger for the pinned Qwen3 alternating-text template subset."""


def render(messages, cot, controls, emit):
    """Emit known character regions; the common caller checks the actual official rendering."""
    target = len(messages) - 1
    for index, message in enumerate(messages):
        emit(f"<|im_start|>{message['role']}\n", "header", False, index)
        if index == target:
            emit("<think>\n", "reasoning_wrapper", cot == "supervised", index)
            emit(message["reasoning_content"].strip("\n"), "reasoning", cot == "supervised", index)
            emit("\n</think>\n\n", "reasoning_wrapper", cot == "supervised", index)
            emit(message["content"].lstrip("\n"), "answer", True, index)
        else:
            emit(message["content"], "context", False, index)
        emit("<|im_end|>", "end", index == target, index)
        emit("\n", "separator", False, index)
