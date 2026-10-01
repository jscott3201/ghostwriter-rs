"""Independent official prompt literals and lossless decoder-boundary controls."""
import pytest

from ghostwriter_trl.artifact import ContractError
from ghostwriter_trl.comparison.protocol import capture_output, render_prompt


@pytest.mark.parametrize("messages,expected", [
    ([{"role": "user", "content": "Hello"}],
     [2, 105, 2364, 107, 9259, 106, 107, 105, 4368, 107]),
    ([{"role": "system", "content": "Be precise."}, {"role": "user", "content": "Hi"}],
     [2, 105, 9731, 107, 3912, 18997, 236761, 106, 107, 105, 2364, 107, 10979, 106, 107, 105, 4368, 107]),
    ([{"role": "user", "content": "One"},
      {"role": "assistant", "content": "Two", "reasoning_content": "Private history"},
      {"role": "user", "content": "Three"}],
     [2, 105, 2364, 107, 4906, 106, 107, 105, 4368, 107, 11634, 106, 107,
      105, 2364, 107, 19765, 106, 107, 105, 4368, 107]),
])
def test_official_generation_prompt_literals(tokenizer, messages, expected):
    captured = render_prompt(tokenizer, messages, 256)
    assert captured["input_ids"] == expected
    assert captured["attention_mask"] == [1] * len(expected)
    assert captured["rendered"].endswith("<|turn>model\n")
    assert "Private history" not in captured["rendered"]


@pytest.mark.parametrize("terminal,reason,representation", [
    (1, "eos", "module"), (106, "turn_end", "module"),
    (50, "tool_handoff", "unsupported"),
])
def test_observed_terminal_only_and_bound_are_independent(tokenizer, terminal, reason, representation):
    prompt = [2, 105, 2364, 107]
    body = tokenizer.encode("pass\n", add_special_tokens=False)
    result = capture_output(tokenizer, prompt, prompt + body + [terminal], len(body) + 1)
    assert result["sequence_ids"] == prompt + body + [terminal]
    assert result["suffix_ids"] == body + [terminal]
    assert result["body_ids"] == body + ([50] if terminal == 50 else [])
    assert result["termination"] == reason and result["at_token_bound"] is True
    assert result["representation"] == representation
    assert result["terminal_id"] == terminal


def test_unterminated_bound_and_short_return_are_distinct(tokenizer):
    body = tokenizer.encode("pass", add_special_tokens=False)
    at_bound = capture_output(tokenizer, [2], [2] + body, len(body))
    short = capture_output(tokenizer, [2], [2] + body, len(body) + 1)
    assert at_bound["termination"] == "length_limit" and at_bound["representation"] == "module"
    assert short["termination"] == "short_return" and short["representation"] == "unknown"
    assert at_bound["body_ids"] == short["body_ids"] == body


def test_returned_prefix_must_match_exact_token_ids(tokenizer):
    result = capture_output(tokenizer, [2, 9259], [2, 10979, 106], 8)
    assert result["sequence_ids"] == [2, 10979, 106]
    assert result["termination"] == "prefix_mismatch"
    assert result["representation"] == "unknown" and result["body_ids"] == []


@pytest.mark.parametrize("suffix", [[0], [100, 45518], [46], [1, 9259], [106, 9259]])
def test_added_tokens_remaining_in_body_are_never_hidden(tokenizer, suffix):
    result = capture_output(tokenizer, [2], [2] + suffix + [106], 16)
    assert result["body_ids"] == suffix and result["representation"] == "unsupported"
    assert result["original_suffix_text"] == tokenizer.decode(
        suffix + [106], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def test_ordinary_token_spellings_cannot_smuggle_control_literals(tokenizer):
    # Separate ordinary tokens spell the reserved control without using its added-token ID.
    parts = [tokenizer.encode(part, add_special_tokens=False) for part in ["<", "|", "channel", ">"]]
    ids = sum(parts, [])
    assert 100 not in ids
    assert tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == "<|channel>"
    result = capture_output(tokenizer, [2], [2] + ids + [106], len(ids) + 1)
    assert result["representation"] == "unsupported"


def test_body_decoding_keeps_exact_unicode_and_whitespace(tokenizer):
    text = "\n\tdef f():\n    return 'é中🙂'\n\n  "
    ids = tokenizer.encode(text, add_special_tokens=False)
    result = capture_output(tokenizer, [2], [2] + ids + [106], 128)
    assert result["body_text"] == text
    assert result["original_suffix_text"] == text + "<turn|>"
    assert result["representation"] == "module"


@pytest.mark.parametrize("messages", [
    [{"role": "assistant", "content": "answer"}],
    [{"role": "user", "content": "x", "tools": []}],
    [{"role": "user", "content": [{"type": "text", "text": "x"}]}],
    [{"role": "user", "content": "<|channel>thought"}],
    [{"role": "user", "content": "x"}, {"role": "user", "content": "y"}],
])
def test_generation_input_is_strict_text_user_ending(tokenizer, messages):
    with pytest.raises(ContractError):
        render_prompt(tokenizer, messages, 256)


def test_prompt_bound_rejects_instead_of_truncating(tokenizer):
    with pytest.raises(ContractError, match="bound"):
        render_prompt(tokenizer, [{"role": "user", "content": "Hello"}], 9)
