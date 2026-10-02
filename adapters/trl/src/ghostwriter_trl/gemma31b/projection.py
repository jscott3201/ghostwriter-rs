"""Actual official rendering, Unicode ownership, and shifted call/answer supervision."""
from dataclasses import asdict
from ..artifact import ContractError
from ..projection import ownership_offsets
from .source import project_messages
from .render import render_ledger
from .tokenizer import official_renderer
from . import PROFILE


def prepare_target(messages, tokenizer, cot, max_length, *, tools, settings):
    """Prepare a selected prefix only after its complete source passed serial validation."""
    text, spans = render_ledger(messages, tools, cot, settings)
    official = official_renderer(tokenizer, PROFILE).apply_chat_template(messages, tools=tools, tokenize=False, **settings)
    if text != official:
        raise ContractError('span ledger differs from pinned official rendering')
    encoded = tokenizer(text, add_special_tokens=False, split_special_tokens=False, truncation=False, return_offsets_mapping=True)
    if len(encoded['input_ids']) > max_length:
        raise ContractError('overlength example; truncation is forbidden')
    effective = ownership_offsets(text, encoded['offset_mapping'], spans)
    labels, kinds, answers, calls = [], [], [], []
    for index, (token, (start, end)) in enumerate(zip(encoded['input_ids'], effective, strict=True)):
        owners = [s for s in spans if s.start < end and s.end > start]
        if not owners or len({s.supervised for s in owners}) != 1:
            raise ContractError('token crosses masked/supervised boundary')
        supervised = owners[0].supervised
        current = sorted({s.kind for s in owners})
        labels.append(token if supervised else -100)
        kinds.append(current)
        if index > 0 and supervised and text[start:end].strip():
            if current == ['answer']:
                answers.append(index)
            if current == ['call']:
                calls.append(index)
    is_call = bool(messages[-1].get('tool_calls'))
    if (is_call and (not calls or answers)) or (not is_call and (not answers or calls)):
        raise ContractError('target has no whole call/answer tokens surviving causal shift')
    return {'input_ids': encoded['input_ids'], 'attention_mask': encoded['attention_mask'], 'labels': labels,
            'rendered': text, 'spans': [asdict(s) for s in spans], 'offset_mapping': encoded['offset_mapping'],
            'ownership_offsets': effective, 'token_kinds': kinds, 'shifted_answer_token_indices': answers,
            'shifted_call_token_indices': calls, 'target_kind': 'tool_call' if is_call else 'text_answer'}
