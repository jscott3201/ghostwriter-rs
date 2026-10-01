"""Ownership ledger for the exact Gemma4 E2B official text-only processor template."""


def render(messages, cot, controls, emit):
    """Track source positions through BOS, optional thinking/system, trimming, and history."""
    emit("<bos>", "header", False, 0)
    has_system = messages[0]["role"] == "system"
    if has_system or controls["enable_thinking"]:
        emit("<|turn>system\n", "header", False, 0)
        if controls["enable_thinking"]:
            emit("<|think|>\n", "header", False, 0)
        if has_system:
            emit(messages[0]["content"].strip(), "context", False, 0)
        emit("<turn|>", "end", False, 0)
        emit("\n", "separator", False, 0)
    target = len(messages) - 1
    for index in range(int(has_system), len(messages)):
        message = messages[index]
        role = "model" if message["role"] == "assistant" else message["role"]
        emit(f"<|turn>{role}\n", "header", False, index)
        # The official reasoning guard drops earlier assistants before the last user.
        # Empty/stripped reasoning produces no channel wrapper; source controls were rejected.
        if index == target and message["reasoning_content"]:
            emit("<|channel>thought\n", "reasoning_wrapper", cot == "supervised", index)
            emit(message["reasoning_content"], "reasoning", cot == "supervised", index)
            emit("\n<channel|>", "reasoning_wrapper", cot == "supervised", index)
        emit(message["content"].strip(), "answer" if index == target else "context", index == target, index)
        emit("<turn|>", "end", index == target, index)
        emit("\n", "separator", False, index)
