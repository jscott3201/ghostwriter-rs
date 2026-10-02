"""Validate the complete serial conversation before selecting any assistant prefix."""
from copy import deepcopy
import math
from ..artifact import ContractError
from .policy import data
from . import PROFILE


def require(condition, reason):
    if not condition:
        raise ContractError(reason)


def identifier(value):
    require(isinstance(value, str) and value and (value[0].isalpha() or value[0] == '_') and all(c.isalnum() or c == '_' for c in value), 'unescaped tool names/keys require identifiers')


def clean(value, depth=0):
    require(depth <= 64, 'source nesting exceeds supported depth')
    if isinstance(value, str):
        require(not any(t['content'] in value for t in data(PROFILE, 'policy')['added_tokens']),
                'source contains a template control delimiter')
    elif isinstance(value, dict):
        for key, item in value.items():
            clean(key, depth + 1)
            clean(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            clean(item, depth + 1)
    elif type(value) is int:
        require(-(2**63) <= value <= 2**64 - 1, 'argument integer exceeds canonical native range')
    elif isinstance(value, float):
        require(math.isfinite(value) and abs(value) < 1e16 and (value == 0 or abs(value) >= 1e-4), 'unsupported nonfinite/scientific argument float')


def argument(value, depth=0):
    require(depth <= 64, 'argument nesting exceeds supported depth')
    if isinstance(value, dict):
        require(len({key.lower() for key in value}) == len(value), 'case-folded argument keys are ambiguous')
        for key, item in value.items():
            identifier(key)
            argument(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            argument(item, depth + 1)


def schema(value, *, root=False, depth=0):
    require(depth <= 64 and isinstance(value, dict), 'unsupported parameter schema nesting/shape')
    kind = value.get('type')
    require(kind in {'object', 'array', 'string', 'integer', 'number', 'boolean', 'null'}, 'unsupported parameter type')
    allowed = {'type', 'description'} | ({'properties', 'required'} if kind == 'object' else {'items'} if kind == 'array' else {'enum'} if kind == 'string' else set())
    if root:
        allowed = {'type', 'properties', 'required'}
        require(kind == 'object', 'function parameters require object type')
    require(not set(value) - allowed, 'parameter fields would be omitted or unsupported by official template')
    require('description' not in value or isinstance(value['description'], str), 'description requires text')
    if kind == 'object':
        properties = value.get('properties')
        require(isinstance(properties, dict), 'object schemas require explicit properties')
        require(len({key.lower() for key in properties}) == len(properties), 'case-folded property names are ambiguous')
        for key, item in properties.items():
            identifier(key)
            schema(item, depth=depth + 1)
        required = value.get('required', [])
        require(isinstance(required, list) and all(isinstance(k, str) for k in required), 'required needs string entries')
        require(len(set(required)) == len(required) and set(required) <= set(properties), 'required must name distinct properties')
    if kind == 'array':
        require('items' in value, 'arrays require an explicit item schema')
        schema(value['items'], depth=depth + 1)
    if 'enum' in value:
        require(isinstance(value['enum'], list) and value['enum'] and all(isinstance(x, str) for x in value['enum']), 'string enum requires nonempty string list')


def project_messages(messages, tools, cot):
    """Return a lossless supported projection, stripping only explicitly selected reasoning."""
    require(cot in {'masked', 'supervised', 'stripped'}, 'unsupported reasoning policy')
    require(isinstance(tools, list), 'tool definitions require a list')
    names = set()
    for tool in tools:
        require(isinstance(tool, dict) and set(tool) == {'type', 'function'} and tool['type'] == 'function', 'only function tool wrappers are supported')
        function = tool['function']
        require(isinstance(function, dict) and not set(function) - {'name', 'description', 'parameters'}, 'unsupported function definition fields')
        identifier(function.get('name'))
        require(function['name'] not in names, 'duplicate definition name')
        names.add(function['name'])
        require('description' not in function or isinstance(function['description'], str), 'function description requires text')
        schema(function.get('parameters'), root=True)
    require(isinstance(messages, list) and messages, 'empty conversation')
    result, ids, phase, pending = [], set(), 'user', None
    for index, original in enumerate(messages):
        require(isinstance(original, dict) and not set(original) - {'role', 'content', 'reasoning', 'reasoning_details', 'tool_calls', 'tool_call_id', 'name'}, 'unsupported message fields')
        message = deepcopy(original)
        role, content = message.get('role'), message.get('content')
        reasoning, details = message.get('reasoning'), message.get('reasoning_details')
        require(reasoning is None or isinstance(reasoning, str), 'reasoning must be flat text')
        if details is not None:
            require(role == 'assistant' and isinstance(reasoning, str) and isinstance(details, list), 'reasoning details require matching text')
            previous, pieces = -1, []
            for detail in details:
                require(isinstance(detail, dict) and not set(detail) - {'type', 'text', 'index', 'signature', 'id', 'format'}, 'unsupported reasoning detail')
                n = detail.get('index')
                require(detail.get('type') == 'reasoning.text' and type(n) is int and previous < n <= 2**32 - 1 and isinstance(detail.get('text'), str), 'reasoning indices/text invalid')
                previous = n
                pieces.append(detail['text'])
            require(''.join(pieces) == reasoning, 'reasoning details differ from flat text')
        require(role == 'assistant' or (reasoning is None and details is None), 'reasoning belongs to assistant')
        calls = message.get('tool_calls')
        if index == 0 and role == 'system':
            require(isinstance(content, str) and calls is None, 'system requires text')
        elif role == 'user' and phase == 'user':
            require(isinstance(content, str) and calls is None, 'user requires text')
            phase = 'assistant'
        elif role == 'assistant' and phase == 'assistant':
            if calls is not None:
                require(isinstance(calls, list) and len(calls) == 1 and content in (None, ''), 'serial call requires exactly one call and null/empty content')
                call = calls[0]
                require(isinstance(call, dict) and not set(call) - {'id', 'type', 'function'}, 'unsupported call fields')
                require(call.get('type', 'function') == 'function', 'unsupported call type')
                cid = call.get('id')
                require(isinstance(cid, str) and cid.strip() and cid not in ids, 'call ID must be unique explicit text')
                ids.add(cid)
                function = call.get('function')
                require(isinstance(function, dict) and not set(function) - {'name', 'arguments', 'raw_arguments'}, 'unsupported call function fields')
                require(function.get('name') in names and isinstance(function.get('arguments'), dict), 'call requires definition and object arguments')
                require(function.get('raw_arguments') is None or isinstance(function['raw_arguments'], str), 'raw arguments require text')
                function.pop('raw_arguments', None)
                argument(function['arguments'])
                pending, phase = (cid, function['name']), 'tool'
            else:
                require(isinstance(content, str) and content.strip(), 'ordinary assistant target requires nonempty answer')
                phase = 'user'
        elif role == 'tool' and phase == 'tool':
            require(isinstance(content, str) and calls is None and message.get('tool_call_id') == pending[0] and message.get('name', pending[1]) in (None, pending[1]), 'tool reply must immediately match the explicit call ID/name')
            phase = 'assistant'
        else:
            raise ContractError('unsupported serial role order or incomplete tool trajectory')
        require(role == 'tool' or (message.get('tool_call_id') is None and message.get('name') is None), 'tool reply identity requires tool role')
        clean(message)
        message.pop('reasoning_details', None)
        message['reasoning'] = '' if cot == 'stripped' else reasoning or ''
        result.append(message)
    require(phase == 'user' and result[-1]['role'] == 'assistant' and not result[-1].get('tool_calls'), 'complete trajectory must end with ordinary assistant answer')
    clean(tools)
    return result
