"""Independent official rendering and literal loss-region oracles for Gemma text."""
from copy import deepcopy
import pytest

from ghostwriter_trl.artifact import ContractError
from ghostwriter_trl.projection import prepare_target, project_messages


@pytest.mark.parametrize("cot,ids,labels,supervised,answer_index", [
    ("supervised", [2, 105, 2364, 107, 15884, 106, 107, 105, 4368, 107, 100, 45518, 107, 36425, 107, 101, 14433, 106, 107],
     [-100] * 10 + [100, 45518, 107, 36425, 107, 101, 14433, 106, -100], 8, 16),
    ("masked", [2, 105, 2364, 107, 15884, 106, 107, 105, 4368, 107, 100, 45518, 107, 36425, 107, 101, 14433, 106, 107],
     [-100] * 16 + [14433, 106, -100], 2, 16),
    ("stripped", [2, 105, 2364, 107, 15884, 106, 107, 105, 4368, 107, 14433, 106, 107],
     [-100] * 10 + [14433, 106, -100], 2, 10),
])
def test_literal_token_label_and_causal_count_oracle(tokenizer, profile, cot, ids, labels, supervised, answer_index):
    source = [{"role": "user", "content": "question"},
              {"role": "assistant", "content": "answer", "reasoning": "why"}]
    example = prepare_target(project_messages(source, tokenizer, cot, profile=profile), tokenizer,
                             cot, 2048, profile=profile, enable_thinking=False)
    assert example["input_ids"] == ids
    assert example["labels"] == labels
    assert sum(label != -100 for label in example["labels"]) == supervised
    assert sum(label != -100 for label in example["labels"][1:]) == supervised
    assert example["shifted_answer_token_indices"] == [answer_index]


@pytest.mark.parametrize("cot", ["supervised", "masked", "stripped"])
@pytest.mark.parametrize("thinking", [False, True])
def test_gemma_plain_reasoning_and_thinking_controls_match_official_bytes(tokenizer, profile, cot, thinking):
    source = [{"role": "user", "content": "  question \n"},
              {"role": "assistant", "content": "\n answer  ", "reasoning": "why"}]
    projected = project_messages(source, tokenizer, cot, profile=profile)
    example = prepare_target(projected, tokenizer, cot, 2048, profile=profile, enable_thinking=thinking)
    prefix = "<bos>" + ("<|turn>system\n<|think|>\n<turn|>\n" if thinking else "")
    prefix += "<|turn>user\nquestion<turn|>\n<|turn>model\n"
    reasoning = "" if cot == "stripped" else "<|channel>thought\nwhy\n<channel|>"
    expected = prefix + reasoning + "answer<turn|>\n"
    assert example["rendered"] == expected
    official = tokenizer._ghostwriter_processor.apply_chat_template(
        projected, tokenize=True, add_generation_prompt=False, enable_thinking=thinking,
        preserve_thinking=False, processor_kwargs={"padding": False, "truncation": False})
    assert example["input_ids"] == official[0]
    start = len(prefix) if cot == "supervised" else len(prefix + reasoning)
    end = len(expected) - 1
    for token, label, (left, right) in zip(example["input_ids"], example["labels"], example["offset_mapping"], strict=True):
        assert label == (token if start <= left and right <= end else -100)
    assert example["input_ids"][0] == 2 and example["labels"][0] == -100


@pytest.mark.parametrize("cot", ["supervised", "masked", "stripped"])
@pytest.mark.parametrize("thinking", [False, True])
def test_system_history_repeated_text_and_trim_have_independent_loss_regions(tokenizer, profile, cot, thinking):
    source = [{"role": "system", "content": " \tsame\n"},
              {"role": "user", "content": " same "},
              {"role": "assistant", "content": " same ", "reasoning": "old proof"},
              {"role": "user", "content": " same "},
              {"role": "assistant", "content": " same ", "reasoning": " same "}]
    before = deepcopy(source)
    projected = project_messages(source, tokenizer, cot, profile=profile)
    result = prepare_target(projected, tokenizer, cot, 2048, profile=profile, enable_thinking=thinking)
    prefix = "<bos><|turn>system\n" + ("<|think|>\n" if thinking else "")
    prefix += "same<turn|>\n<|turn>user\nsame<turn|>\n<|turn>model\nsame<turn|>\n<|turn>user\nsame<turn|>\n<|turn>model\n"
    reasoning = "" if cot == "stripped" else "<|channel>thought\n same \n<channel|>"
    expected = prefix + reasoning + "same<turn|>\n"
    assert source == before
    assert result["rendered"] == expected
    assert "old proof" not in result["rendered"]
    answer_start = len(prefix + reasoning)
    loss_start = len(prefix) if cot == "supervised" else answer_start
    answer_indices = []
    for index, (token, label, (start, end)) in enumerate(zip(result["input_ids"], result["labels"], result["offset_mapping"], strict=True)):
        assert label == (token if loss_start <= start and end < len(expected) else -100)
        if answer_start <= start and end <= answer_start + 4:
            answer_indices.append(index)
    assert result["shifted_answer_token_indices"] == answer_indices
    assert result["labels"][-2:] == [106, -100]


@pytest.mark.parametrize("reason", [None, "", " \n"])
@pytest.mark.parametrize("cot", ["supervised", "masked", "stripped"])
def test_absent_empty_and_whitespace_reasoning_follow_official_truthiness(tokenizer, profile, reason, cot):
    source = [{"role": "user", "content": "question"},
              {"role": "assistant", "content": "answer", "reasoning": reason}]
    result = prepare_target(project_messages(source, tokenizer, cot, profile=profile), tokenizer, cot, 2048, profile=profile)
    prefix = "<bos><|turn>user\nquestion<turn|>\n<|turn>model\n"
    middle = f"<|channel>thought\n{reason}\n<channel|>" if reason and cot != "stripped" else ""
    assert result["rendered"] == prefix + middle + "answer<turn|>\n"
    assert bool(result["shifted_answer_token_indices"])


@pytest.mark.parametrize("text", ["e\u0301Z", "e\u0301\u0327Z", "e\u0327\u0301Z", "A\u0301\u0308中",
                                  "🙂漢é\u0301Z", "\u1100\u1161", "👨\u200d👩\u200d👧\u200d👦", "A\u00a0B", "same  same"])
def test_unicode_and_repetition_preserve_complete_offsets_and_loss_ownership(tokenizer, profile, text):
    source = [{"role": "user", "content": text}, {"role": "assistant", "content": text, "reasoning": text}]
    messages = project_messages(source, tokenizer, "masked", profile=profile)
    example = prepare_target(messages, tokenizer, "masked", 2048, profile=profile)
    assert example["input_ids"] == tokenizer._ghostwriter_processor.apply_chat_template(
        messages, tokenize=True, enable_thinking=False, preserve_thinking=False,
        processor_kwargs={"padding": False, "truncation": False})[0]
    assert set().union(*(set(range(a, b)) for a, b in example["offset_mapping"])) == set(range(len(example["rendered"])))
    assert set().union(*(set(range(a, b)) for a, b in example["ownership_offsets"])) == set(range(len(example["rendered"])))
    answer_start = len(f"<bos><|turn>user\n{text}<turn|>\n<|turn>model\n<|channel>thought\n{text}\n<channel|>")
    for label, token, (start, end) in zip(example["labels"], example["input_ids"], example["ownership_offsets"], strict=True):
        assert label == (token if answer_start <= start and end < len(example["rendered"]) else -100)


@pytest.mark.parametrize("field", ["content", "reasoning"])
def test_combining_mark_cannot_cross_a_wrapper_owner(tokenizer, profile, field):
    source = [{"role": "user", "content": "question"},
              {"role": "assistant", "content": "answer", "reasoning": "why"}]
    source[-1][field] = "\u0301text"
    with pytest.raises(ContractError, match="combining sequence crosses"):
        prepare_target(project_messages(source, tokenizer, "supervised", profile=profile),
                       tokenizer, "supervised", 2048, profile=profile)


def test_real_exact_limit_and_no_whole_answer_are_explicit(tokenizer, profile):
    source = [{"role": "user", "content": "context " * 1100},
              {"role": "assistant", "content": "ANSWER final", "reasoning": "REASON final"}]
    messages = project_messages(source, tokenizer, "masked", profile=profile)
    full = prepare_target(messages, tokenizer, "masked", 2048, profile=profile)
    length = len(full["input_ids"])
    assert length > 1024
    assert prepare_target(messages, tokenizer, "masked", length, profile=profile) == full
    with pytest.raises(ContractError, match="overlength"):
        prepare_target(messages, tokenizer, "masked", length - 1, profile=profile)
    for blank in ("", " \n\t", "\u00a0"):
        source[-1]["content"] = blank
        with pytest.raises(ContractError, match="answer token"):
            prepare_target(project_messages(source, tokenizer, "supervised", profile=profile),
                           tokenizer, "supervised", 2048, profile=profile)
