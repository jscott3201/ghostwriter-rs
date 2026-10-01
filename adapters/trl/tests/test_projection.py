"""Independent loss-region expectations across policies, turns, normalization, and failures."""
import copy
import pytest

from ghostwriter_trl.artifact import ContractError
from ghostwriter_trl.projection import Span, label_tokens, ownership_offsets, prepare_target, project_messages


def messages(text="same 🙂中e\u0301"):
    return [
        {"role": "system", "content": text},
        {"role": "user", "content": text},
        {"role": "assistant", "content": text, "reasoning": text},
        {"role": "user", "content": text},
        {"role": "assistant", "content": text, "reasoning": text},
    ]


@pytest.mark.parametrize("cot", ["supervised", "masked", "stripped"])
def test_repeated_identical_text_is_owned_by_position_not_substring(tokenizer, cot):
    source = messages()
    before = copy.deepcopy(source)
    projected = project_messages(source, tokenizer, cot)
    example = prepare_target(projected, tokenizer, cot, 2048)
    assert source == before
    # Independent official-token oracle, no prefix token counts and no substring boundary search.
    assert example["input_ids"] == tokenizer.apply_chat_template(projected, tokenize=True, add_generation_prompt=False)
    assert example["input_ids"].count(tokenizer.convert_tokens_to_ids("<|im_start|>")) == 5
    assert example["input_ids"].count(tokenizer.eos_token_id) == 5
    assert example["attention_mask"] == [1] * len(example["input_ids"])
    by_position = {}
    for span in example["spans"]:
        for index in range(span["start"], span["end"]):
            by_position[index] = span
    answer_count, reasoning_count, context_count = 0, 0, 0
    for index, ((start, end), label, token) in enumerate(zip(example["ownership_offsets"], example["labels"], example["input_ids"], strict=True)):
        owners = [by_position[p] for p in range(start, end)]
        is_target = all(owner["message_index"] == 4 for owner in owners)
        kinds = {owner["kind"] for owner in owners}
        expected_loss = is_target and (kinds <= {"answer", "end"} or (cot == "supervised" and kinds <= {"reasoning", "reasoning_wrapper"}))
        assert label == (token if expected_loss else -100)
        if kinds == {"answer"}:
            answer_count += 1
            assert index in example["shifted_answer_token_indices"] and index > 0
        if "reasoning" in kinds:
            reasoning_count += 1
        if not is_target:
            context_count += 1
    assert answer_count > 0 and context_count > 0
    assert (reasoning_count == 0) == (cot == "stripped")


@pytest.mark.parametrize("cot", ["supervised", "masked", "stripped"])
def test_redundant_plaintext_details_render_once_and_bind_original(tokenizer, cot):
    source = messages("abc")[:3]
    source[2]["reasoning_details"] = [
        {"type": "reasoning.text", "index": 2, "text": "a", "signature": "opaque"},
        {"type": "reasoning.text", "index": 5, "text": "bc"},
    ]
    example = prepare_target(project_messages(source, tokenizer, cot), tokenizer, cot, 2048)
    flat_only = copy.deepcopy(source)
    del flat_only[2]["reasoning_details"]
    assert example == prepare_target(project_messages(flat_only, tokenizer, cot), tokenizer, cot, 2048)
    assert len(source[2]["reasoning_details"]) == 2


@pytest.mark.parametrize("mutation", [
    "summary", "encrypted", "mixed", "missing_flat", "mismatch", "duplicate", "reversed",
    "null", "parts", "tool", "tool_calls", "name", "developer", "user_reasoning", "trailing_user",
])
def test_unsupported_source_structures_fail_closed_even_when_stripped(tokenizer, mutation):
    source = messages("abc")[:3]
    target = source[2]
    target["reasoning_details"] = [{"type": "reasoning.text", "index": 0, "text": "abc"}]
    if mutation in {"summary", "encrypted"}:
        target["reasoning_details"][0]["type"] = "reasoning." + mutation
    elif mutation == "mixed":
        target["reasoning_details"].append({"type": "reasoning.summary", "index": 1, "summary": "x"})
    elif mutation == "missing_flat":
        target.pop("reasoning")
    elif mutation == "mismatch":
        target["reasoning"] = "different"
    elif mutation in {"duplicate", "reversed"}:
        target["reasoning"] = "abcabc"
        target["reasoning_details"] *= 2
        if mutation == "reversed":
            target["reasoning_details"] = [dict(target["reasoning_details"][0], index=2), dict(target["reasoning_details"][1], index=1)]
    elif mutation == "null":
        target["content"] = None
    elif mutation == "parts":
        target["content"] = [{"type": "text", "text": "abc"}]
    elif mutation == "tool":
        source[1]["role"] = "tool"
    elif mutation == "tool_calls":
        target["tool_calls"] = []
    elif mutation == "name":
        target["name"] = "named"
    elif mutation == "developer":
        source[0]["role"] = "developer"
    elif mutation == "user_reasoning":
        source[1]["reasoning"] = "not assistant"
    else:
        source.append({"role": "user", "content": "unfinished"})
    with pytest.raises(ContractError):
        project_messages(source, tokenizer, "stripped")


@pytest.mark.parametrize("control", ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_response>"])
@pytest.mark.parametrize("field", ["content", "reasoning"])
def test_control_literals_rejected_before_stripping(tokenizer, control, field):
    source = messages("abc")[:3]
    source[2][field] = f"literal {control} repeated {control}"
    with pytest.raises(ContractError, match="control"):
        project_messages(source, tokenizer, "stripped")


def test_absent_reasoning_empty_wrappers_exact_limit_and_answer_shift(tokenizer):
    source = [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]
    for cot in ("supervised", "masked", "stripped"):
        projected = project_messages(source, tokenizer, cot)
        example = prepare_target(projected, tokenizer, cot, 2048)
        assert "<think>\n\n</think>\n\nanswer" in example["rendered"]
        length = len(example["input_ids"])
        assert prepare_target(projected, tokenizer, cot, length) == example
        with pytest.raises(ContractError, match="overlength"):
            prepare_target(projected, tokenizer, cot, length - 1)
        assert all(example["labels"][i] != -100 and i > 0 for i in example["shifted_answer_token_indices"])
    for blank in ("", "\n\n", "   ", "\t"):
        source[-1]["content"] = blank
        with pytest.raises(ContractError, match="answer token"):
            prepare_target(project_messages(source, tokenizer, "supervised"), tokenizer, "supervised", 2048)


@pytest.mark.parametrize("text", ["e\u0301Z", "e\u0301\u0327Z", "e\u0327\u0301Z", "A\u0301\u0308中", "🙂漢é\u0301Z"])
def test_combining_sequences_remain_one_owner_under_real_nfc_offsets(tokenizer, text):
    source = [{"role": "user", "content": "question"}, {"role": "assistant", "content": text, "reasoning": text}]
    example = prepare_target(project_messages(source, tokenizer, "masked"), tokenizer, "masked", 2048)
    assert set().union(*(set(range(a, b)) for a, b in example["ownership_offsets"])) == set(range(len(example["rendered"])))
    assert len(example["shifted_answer_token_indices"]) > 0


@pytest.mark.parametrize("field", ["content", "reasoning"])
def test_leading_combining_mark_crosses_wrapper_boundary_and_is_rejected(tokenizer, field):
    source = [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer", "reasoning": "why"}]
    source[-1][field] = "\u0301mark"
    with pytest.raises(ContractError, match="combining sequence crosses"):
        prepare_target(project_messages(source, tokenizer, "supervised"), tokenizer, "supervised", 2048)


def test_ambiguous_tokens_and_shift_only_end_are_rejected():
    spans = [Span(0, 1, "context", False, 0), Span(1, 2, "answer", True, 1)]
    with pytest.raises(ContractError, match="crosses"):
        label_tokens([99], [(0, 2)], spans, "ab")
    with pytest.raises(ContractError, match="answer token"):
        label_tokens([99, 100], [(0, 1), (1, 2)], [Span(0, 1, "answer", True, 0), Span(1, 2, "end", True, 0)], "ab")
    # Even a hidden combining mark must not cross an ownership boundary or answer boundary.
    for loss in (False, True):
        with pytest.raises(ContractError, match="combining sequence crosses"):
            ownership_offsets("e\u0301", [(0, 1)], [Span(0, 1, "context", loss, 0), Span(1, 2, "answer", True, 0)])
    with pytest.raises(ContractError, match="unexplained gap"):
        ownership_offsets("ab", [(0, 1)], [Span(0, 2, "answer", True, 0)])


def test_unsupported_hangul_normalization_gap_is_explicit(tokenizer):
    source = [{"role": "user", "content": "question"}, {"role": "assistant", "content": "\u1100\u1161Z"}]
    with pytest.raises(ContractError, match="unexplained gap"):
        prepare_target(project_messages(source, tokenizer, "masked"), tokenizer, "masked", 2048)


@pytest.mark.parametrize("cot,final_loss,final_answer", [
    ("supervised", list(range(27, 35)), 33),
    ("masked", [33, 34], 33),
    ("stripped", [31, 32], 31),
])
@pytest.mark.parametrize("target", [2, 4])
def test_independent_fixed_token_oracle_for_identical_role_and_channel_text(tokenizer, cot, final_loss, final_answer, target):
    # Fixed positions independently audited from the complete official encoding. No adapter
    # spans, substring search, or prefix-token-length arithmetic constructs this oracle.
    expected = {
        ("supervised", 2): (list(range(15, 23)), 21),
        ("masked", 2): ([21, 22], 21),
        ("stripped", 2): ([19, 20], 19),
    }
    loss, answer = expected.get((cot, target), (final_loss, final_answer))
    projected = project_messages(messages("same")[:target + 1], tokenizer, cot)
    example = prepare_target(projected, tokenizer, cot, 2048)
    assert [i for i, label in enumerate(example["labels"]) if label != -100] == loss
    assert example["shifted_answer_token_indices"] == [answer]
    assert tokenizer.convert_ids_to_tokens(example["input_ids"][answer]) == "same"
    assert example["input_ids"][loss[-1]] == tokenizer.eos_token_id
