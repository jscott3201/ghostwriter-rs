"""The exact release's 24 added-token literals and unsupported canonical source structures."""
import pytest

from ghostwriter_trl.artifact import ContractError
from ghostwriter_trl.projection import project_messages

# Transcribed independently from the release tokenizer.json added-token inventory.
CONTROLS = ("<pad>", "<eos>", "<bos>", "<unk>", "<mask>", "<|tool>", "<tool|>",
            "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>", '<|"|>',
            "<|think|>", "<|channel>", "<channel|>", "<|turn>", "<turn|>", "<|image>",
            "<|audio>", "<|image|>", "<|audio|>", "<image|>", "<audio|>", "<|video|>")


def messages():
    return [{"role": "system", "content": "system"}, {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer", "reasoning": "reason"}]


@pytest.mark.parametrize("literal", CONTROLS)
@pytest.mark.parametrize("channel", ["system", "user", "answer", "reasoning", "details"])
def test_every_control_is_rejected_before_stripping(tokenizer, profile, literal, channel):
    source = messages()
    text = "literal " + literal
    if channel in {"system", "user", "answer"}:
        source[{"system": 0, "user": 1, "answer": 2}[channel]]["content"] = text
    else:
        source[-1]["reasoning"] = text
        if channel == "details":
            source[-1]["reasoning_details"] = [{"type": "reasoning.text", "index": 0, "text": text}]
    with pytest.raises(ContractError, match="control"):
        project_messages(source, tokenizer, "stripped", profile=profile)


@pytest.mark.parametrize("case", ["tools", "tool_definition_envelope", "tool_calls", "tool_response", "name",
                                  "image", "audio", "video", "text_parts", "null", "developer",
                                  "wrong_reasoning_role", "summary", "encrypted", "mismatch", "duplicate"])
def test_tools_modalities_and_unproven_reasoning_structures_fail_closed(tokenizer, profile, case):
    source = messages()
    if case == "tools": source[-1]["tools"] = []
    elif case == "tool_definition_envelope": source = {"messages": source, "tools": []}
    elif case == "tool_calls": source[-1]["tool_calls"] = []
    elif case == "tool_response": source[-1]["role"] = "tool"
    elif case == "name": source[-1]["name"] = "named"
    elif case in {"image", "audio", "video", "text_parts"}:
        source[-1]["content"] = [{"type": "text" if case == "text_parts" else case, "text": "answer"}]
    elif case == "null": source[-1]["content"] = None
    elif case == "developer": source[0]["role"] = "developer"
    elif case == "wrong_reasoning_role": source[1]["reasoning"] = "reason"
    else:
        source[-1]["reasoning_details"] = [{"type": "reasoning.text", "index": 0, "text": "reason"}]
        if case in {"summary", "encrypted"}: source[-1]["reasoning_details"][0]["type"] = "reasoning." + case
        elif case == "mismatch": source[-1]["reasoning"] = "different"
        else:
            source[-1]["reasoning"] = "reasonreason"
            source[-1]["reasoning_details"] *= 2
    with pytest.raises(ContractError):
        project_messages(source, tokenizer, "stripped", profile=profile)
