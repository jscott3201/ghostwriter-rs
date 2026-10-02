"""Independent ownership ledger for the qualified official serial tool subset."""
from ..projection import Span

Q = '<|"|>'


def ordered(value):
    return sorted(value.items(), key=lambda pair: pair[0].lower())


def arg(value, escaped=False):
    if value is None:
        return 'null'
    if isinstance(value, str):
        return Q + value + Q
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, dict):
        return '{' + ','.join((Q + k + Q if escaped else k) + ':' + arg(v, escaped) for k, v in ordered(value)) + '}'
    if isinstance(value, list):
        return '[' + ','.join(arg(v, escaped) for v in value) + ']'
    return str(value)


def properties(values):
    return ','.join(key + ':{' + property_body(value) + '}' for key, value in ordered(values))


def property_body(value):
    parts = []
    if value.get('description'):
        parts.append('description:' + arg(value['description']))
    kind = value['type']
    if kind == 'string' and value.get('enum'):
        parts.append('enum:' + arg(value['enum'], True))
    if kind == 'array':
        items = []
        for key, item in ordered(value['items']):
            if key == 'properties':
                rendered = '{' + properties(item) + '}'
            elif key == 'type':
                rendered = arg(item.upper())
            else:
                rendered = arg(item, True)
            items.append(key + ':' + rendered)
        parts.append('items:{' + ','.join(items) + '}')
    if kind == 'object':
        parts.append('properties:{' + properties(value['properties']) + '}')
        if value.get('required'):
            parts.append('required:' + arg(value['required']))
    parts.append('type:' + arg(kind.upper()))
    return ','.join(parts)


def definition(tool):
    function = tool['function']
    parameters = function['parameters']
    parts = []
    if parameters['properties']:
        parts.append('properties:{' + properties(parameters['properties']) + '}')
    if parameters.get('required'):
        parts.append('required:' + arg(parameters['required']))
    parts.append('type:' + arg(parameters['type'].upper()))
    return ('<|tool>declaration:' + function['name'] + '{description:' + arg(function.get('description', ''))
            + ',parameters:{' + ','.join(parts) + '}}<tool|>')


def render_ledger(messages, tools, cot, settings):
    """Attribute external observations to their source tool message, never to a call target."""
    pieces, spans, position = [], [], 0
    def emit(text, kind, supervised, index):
        nonlocal position
        if text:
            pieces.append(text)
            spans.append(Span(position, position + len(text), kind, supervised, index))
            position += len(text)
    emit('<bos>', 'header', False, 0)
    system = messages[0]['role'] == 'system'
    if system or tools or settings['enable_thinking']:
        emit('<|turn>system\n', 'header', False, 0)
        if settings['enable_thinking']:
            emit('<|think|>\n', 'header', False, 0)
        if system:
            emit(messages[0]['content'].strip(), 'context', False, 0)
        for tool in tools:
            emit(definition(tool), 'definition', False, 0)
        emit('<turn|>', 'end', False, 0)
        emit('\n', 'separator', False, 0)
    last_user = max(i for i, m in enumerate(messages) if m['role'] == 'user')
    target = len(messages) - 1
    previous = None
    for index in range(int(system), len(messages)):
        message = messages[index]
        role = message['role']
        if role == 'tool':
            call = messages[index - 1]['tool_calls'][0]['function']
            emit('<|tool_response>response:' + call['name'] + '{value:' + arg(message['content']) + '}<tool_response|>', 'observation', False, index)
            continue
        if role != 'assistant' or previous != 'assistant':
            emit('<|turn>' + ('model' if role == 'assistant' else role) + '\n', 'header', False, index)
        calls = message.get('tool_calls')
        if message.get('reasoning') and (index > last_user or (settings['preserve_thinking'] and calls)):
            supervised = index == target and cot == 'supervised'
            emit('<|channel>thought\n', 'reasoning_wrapper', supervised, index)
            emit(message['reasoning'], 'reasoning', supervised, index)
            emit('\n<channel|>', 'reasoning_wrapper', supervised, index)
        if calls:
            function = calls[0]['function']
            emit('<|tool_call>', 'call_wrapper', index == target, index)
            emit('call:' + function['name'] + arg(function['arguments']), 'call', index == target, index)
            emit('<tool_call|>', 'call_wrapper', index == target, index)
            if index == target:
                emit('<|tool_response>', 'handoff', True, index)
        else:
            emit(message['content'].strip(), 'answer' if index == target else 'context', index == target, index)
            emit('<turn|>', 'end', index == target, index)
            emit('\n', 'separator', False, index)
        previous = role
    return ''.join(pieces), spans
