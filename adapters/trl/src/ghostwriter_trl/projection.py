"""Explicit model-specific character ownership, checked against every official rendering."""
from dataclasses import asdict, dataclass
import unicodedata

from .artifact import ContractError
from .tokenizer import source_control_literals


@dataclass(frozen=True)
class Span:
    """A half-open interval in Python Unicode codepoints, independent of token boundaries."""
    start: int
    end: int
    kind: str
    supervised: bool
    message_index: int


def project_messages(messages: list, tokenizer, cot: str) -> list[dict]:
    """Accept only unambiguous text trajectories and redundant ordered plaintext reasoning."""
    if cot not in {"supervised", "masked", "stripped"}:
        raise ContractError("unsupported reasoning policy")
    if not isinstance(messages, list) or not messages:
        raise ContractError("empty or invalid conversation")
    controls = source_control_literals()
    projected = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) - {
            "role", "content", "reasoning", "reasoning_details", "tool_calls", "tool_call_id", "name",
        }:
            raise ContractError("unsupported message fields")
        role = message.get("role")
        if role not in {"system", "user", "assistant"} or (role == "system" and index != 0):
            raise ContractError("unsupported role or system position")
        if any(message.get(key) is not None for key in ("tool_calls", "tool_call_id", "name")):
            raise ContractError("tool/name semantics are unsupported")
        content, reasoning = message.get("content"), message.get("reasoning")
        if not isinstance(content, str) or (reasoning is not None and not isinstance(reasoning, str)):
            raise ContractError("only string content and flat reasoning are supported")
        details = message.get("reasoning_details")
        if role != "assistant" and (reasoning is not None or details is not None):
            raise ContractError("reasoning must belong to an assistant")
        if details is not None:
            if not isinstance(details, list) or not isinstance(reasoning, str):
                raise ContractError("reasoning details require matching flat text")
            previous, texts = -1, []
            for detail in details:
                if not isinstance(detail, dict) or set(detail) - {"type", "text", "index", "signature", "id", "format"}:
                    raise ContractError("unsupported reasoning detail")
                value = detail.get("index")
                if detail.get("type") != "reasoning.text" or type(value) is not int or not previous < value <= 2**32 - 1:
                    raise ContractError("reasoning details require strictly increasing text indices")
                if not isinstance(detail.get("text"), str):
                    raise ContractError("reasoning detail text missing")
                previous = value
                texts.append(detail["text"])
            if "".join(texts) != reasoning:
                raise ContractError("reasoning details differ from flat text")
        for text in (content, reasoning or ""):
            if any(control in text for control in controls):
                raise ContractError("literal template/control token in source text")
        projected.append({"role": role, "content": content, "reasoning_content": "" if cot == "stripped" else reasoning or ""})
    roles = [message["role"] for message in projected]
    dialogue = roles[1:] if roles[0] == "system" else roles
    if not dialogue or any(role != ("user" if index % 2 == 0 else "assistant") for index, role in enumerate(dialogue)):
        raise ContractError("expected alternating user/assistant turns after optional system")
    if roles[-1] != "assistant":
        raise ContractError("conversation must end with the final assistant target")
    return projected


def render_ledger(messages: list[dict], cot: str) -> tuple[str, list[Span]]:
    """Implement the qualified alternating text subset of the exact Qwen template."""
    pieces, spans, position = [], [], 0

    def emit(text, kind, supervised, index):
        nonlocal position
        if text:
            pieces.append(text)
            spans.append(Span(position, position + len(text), kind, supervised, index))
            position += len(text)

    target = len(messages) - 1
    for index, message in enumerate(messages):
        role = message["role"]
        emit(f"<|im_start|>{role}\n", "header", False, index)
        if index == target:
            emit("<think>\n", "reasoning_wrapper", cot == "supervised", index)
            emit(message["reasoning_content"].strip("\n"), "reasoning", cot == "supervised", index)
            emit("\n</think>\n\n", "reasoning_wrapper", cot == "supervised", index)
            emit(message["content"].lstrip("\n"), "answer", True, index)
        else:
            emit(message["content"], "context", False, index)
        emit("<|im_end|>", "end", index == target, index)
        emit("\n", "separator", False, index)
    return "".join(pieces), spans


def ownership_offsets(text: str, offsets: list[tuple[int, int]], spans: list[Span]) -> list[tuple[int, int]]:
    """Expand offsets to complete canonical combining sequences, conservatively.

    NFC may compose/reorder marks while reporting only part of their original sequence.
    Every source codepoint must be accounted for. Only omitted nonstarter marks in a
    represented sequence are accepted; normalization such as Hangul Jamo composition
    with omitted starters is explicitly unsupported. A sequence cannot cross an owner,
    answer/context, or loss boundary even when the raw offset hides that crossing.
    """
    if not offsets or offsets[0][0] != 0:
        raise ContractError("unowned leading text")
    owners = [None] * len(text)
    for span in spans:
        owners[span.start:span.end] = [(span.kind, span.supervised, span.message_index)] * (span.end - span.start)
    clusters = []
    for position, character in enumerate(text):
        if not unicodedata.combining(character) or not clusters:
            clusters.append([position, position + 1])
        else:
            clusters[-1][1] = position + 1
    cluster_at = [None] * len(text)
    for start, end in clusters:
        if len(set(owners[start:end])) != 1 or owners[start] is None:
            raise ContractError("combining sequence crosses ownership boundary")
        for position in range(start, end):
            cluster_at[position] = (start, end)
    covered = set()
    previous_start = -1
    for start, end in offsets:
        if not 0 <= start < end <= len(text) or start < previous_start:
            raise ContractError("unsupported tokenizer offset pattern")
        previous_start = start
        covered.update(range(start, end))
    for position in set(range(len(text))) - covered:
        cluster_start, _ = cluster_at[position]
        if not unicodedata.combining(text[position]) or cluster_start not in covered:
            raise ContractError("unexplained gap in tokenizer offsets")
    return [(cluster_at[start][0], cluster_at[end - 1][1]) for start, end in offsets]


def label_tokens(input_ids: list[int], offsets: list[tuple[int, int]], spans: list[Span], text: str):
    """Reject ambiguous mixed-ownership tokens, retaining only whole answer tokens as proof."""
    labels, answer_tokens, kinds = [], [], []
    for token_index, (token, (start, end)) in enumerate(zip(input_ids, offsets, strict=True)):
        if not 0 <= start < end <= len(text):
            raise ContractError("invalid/empty token offset")
        owners = [span for span in spans if span.start < end and span.end > start]
        if not owners or sum(min(end, s.end) - max(start, s.start) for s in owners) != end - start:
            raise ContractError("unowned token")
        if len({span.supervised for span in owners}) != 1:
            raise ContractError("token crosses masked/supervised boundary")
        supervised = owners[0].supervised
        labels.append(token if supervised else -100)
        kinds.append(sorted({s.kind for s in owners}))
        if token_index > 0 and supervised and all(s.kind == "answer" for s in owners) and any(not char.isspace() for char in text[start:end]):
            answer_tokens.append(token_index)
    if not answer_tokens:
        raise ContractError("no whole answer token survives causal shifting")
    return labels, answer_tokens, kinds


def prepare_target(messages: list[dict], tokenizer, cot: str, max_length: int) -> dict:
    """Render once, prove exact official equivalence, and tokenize the complete text once."""
    text, spans = render_ledger(messages, cot)
    official = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if text != official:
        raise ContractError("span ledger differs from pinned official rendering")
    encoded = tokenizer(text, add_special_tokens=False, split_special_tokens=False, truncation=False, return_offsets_mapping=True)
    if len(encoded["input_ids"]) > max_length:
        raise ContractError("overlength example; truncation is forbidden")
    effective = ownership_offsets(text, encoded["offset_mapping"], spans)
    labels, answer_tokens, kinds = label_tokens(encoded["input_ids"], effective, spans, text)
    return {
        "input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "labels": labels,
        "rendered": text, "spans": [asdict(span) for span in spans],
        "offset_mapping": encoded["offset_mapping"], "ownership_offsets": effective, "token_kinds": kinds,
        "shifted_answer_token_indices": answer_tokens,
    }
