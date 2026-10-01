"""All pinned added-token literals are rejected in source text, even before stripping."""
import copy
import pytest

from ghostwriter_trl.artifact import ContractError
from ghostwriter_trl.projection import prepare_target, project_messages
from ghostwriter_trl.tokenizer import qualify_offsets

# Independently transcribed from the pinned tokenizer.json IDs151643 through151668.
ADDED_LITERALS = (
    "<|endoftext|>", "<|im_start|>", "<|im_end|>",
    "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>",
    "<|quad_start|>", "<|quad_end|>", "<|vision_start|>", "<|vision_end|>",
    "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", "<tool_call>", "</tool_call>",
    "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
    "<|repo_name|>", "<|file_sep|>", "<tool_response>", "</tool_response>", "<think>", "</think>",
)


@pytest.mark.parametrize("literal", ADDED_LITERALS)
@pytest.mark.parametrize("channel", ["system", "user", "answer", "reasoning", "redundant_details"])
def test_every_pinned_added_literal_is_rejected_before_stripping(tokenizer, literal, channel):
    source = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer", "reasoning": "why"},
    ]
    text = f"literal {literal}"
    if channel == "system":
        source[0]["content"] = text
    elif channel == "user":
        source[1]["content"] = text
    elif channel == "answer":
        source[2]["content"] = text
    else:
        source[2]["reasoning"] = text
        if channel == "redundant_details":
            source[2]["reasoning_details"] = [{"type": "reasoning.text", "index": 0, "text": text}]
    with pytest.raises(ContractError, match="control"):
        project_messages(source, tokenizer, "stripped")


def test_wrapper_special_map_cannot_weaken_source_literal_rejection(tokenizer):
    changed = copy.deepcopy(tokenizer)
    changed.additional_special_tokens = []
    with pytest.raises(ContractError, match="control"):
        project_messages([
            {"role": "user", "content": "literal <|im_start|>"},
            {"role": "assistant", "content": "answer"},
        ], changed, "stripped")


def test_qualified_encoding_and_offset_calls_explicitly_preserve_added_tokens(tokenizer, monkeypatch):
    original = type(tokenizer).__call__
    calls = []

    def checked(self, *args, **kwargs):
        calls.append(kwargs)
        assert kwargs.get("split_special_tokens") is False
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(tokenizer), "__call__", checked)
    qualify_offsets(tokenizer)
    source = [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]
    prepare_target(project_messages(source, tokenizer, "masked"), tokenizer, "masked", 2048)
    assert len(calls) == 4
