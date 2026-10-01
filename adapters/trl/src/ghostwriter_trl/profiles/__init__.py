"""Named, pinned text profiles share the preparation and verification implementation."""
from pathlib import Path
from types import MappingProxyType

from ..artifact import ContractError, strict_json

QWEN = "qwen3_text_v1"
GEMMA = "gemma4_e2b_text_v1"
NAMES = (QWEN, GEMMA)
PACKAGE = Path(__file__).parent.parent


def freeze(value):
    """Keep loaded application-owned profile data recursively immutable."""
    if isinstance(value, dict):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(freeze(item) for item in value)
    return value


_PATHS = {
    QWEN: {"manifest": PACKAGE / "tokenizer_manifest.json", "policy": PACKAGE / "tokenizer_policy.json",
           "dependencies": PACKAGE / "dependency_pins.json"},
    GEMMA: {part: Path(__file__).parent / f"gemma_{suffix}.json"
            for part, suffix in (("manifest", "manifest"), ("policy", "policy"), ("dependencies", "dependencies"))},
}
_RAW = freeze({name: {part: path.read_text() for part, path in paths.items()} for name, paths in _PATHS.items()})
_DATA = freeze({name: {part: strict_json(raw) for part, raw in parts.items()} for name, parts in _RAW.items()})


def data(profile: str, part: str):
    """Return a pinned immutable component for one supported explicit profile name."""
    if profile not in NAMES:
        raise ContractError("unsupported named text preparation profile")
    return _DATA[profile][part]


def detached(profile: str, part: str) -> dict:
    """Return a defensive JSON copy for persisted recipe declarations."""
    data(profile, part)
    return strict_json(_RAW[profile][part])


def controls(profile: str, enable_thinking: bool | None = None) -> dict:
    """The supported renderer options are explicit and bound into every version-two recipe."""
    data(profile, "manifest")
    enabled = profile == QWEN if enable_thinking is None else enable_thinking
    if type(enabled) is not bool or (profile == QWEN and not enabled):
        raise ContractError("unsupported thinking control for text profile")
    result = {"enable_thinking": enabled, "add_generation_prompt": False}
    if profile == GEMMA:
        result["preserve_thinking"] = False
    return result
