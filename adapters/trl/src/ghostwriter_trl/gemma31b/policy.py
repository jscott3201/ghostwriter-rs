"""Immutable release and runtime policy for the independent serial tool consumer."""
from pathlib import Path
from ..artifact import ContractError, strict_json
from ..profiles import freeze, detached as shared_data, GEMMA as SHARED_GEMMA
from . import PROFILE

PACKAGE = Path(__file__).parent
GEMMA = PROFILE
_RAW = {part: (PACKAGE / 'manifest.json').read_text() if part == 'manifest' else None
        for part in ('manifest', 'policy', 'dependencies')}
_DATA = freeze({part: strict_json(raw) if raw is not None else shared_data(SHARED_GEMMA, part)
                for part, raw in _RAW.items()})


def data(profile, part):
    """Reject other profile names before returning immutable policy data."""
    if profile != PROFILE:
        raise ContractError('unsupported serial tool preparation profile')
    return _DATA[part]


def detached(profile, part):
    """Return detached JSON declarations for the captured recipe."""
    data(profile, part)
    return strict_json(_RAW[part]) if _RAW[part] is not None else shared_data(SHARED_GEMMA, part)


def controls(profile, enable_thinking, preserve_thinking):
    """Require both thinking decisions explicitly; complete examples have no generation prompt."""
    data(profile, 'manifest')
    if type(enable_thinking) is not bool or type(preserve_thinking) is not bool:
        raise ContractError('explicit Boolean thinking and preservation controls are required')
    return {'enable_thinking': enable_thinking, 'preserve_thinking': preserve_thinking,
            'add_generation_prompt': False}
