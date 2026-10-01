"""Exact local tokenizer and dependency identities, with Unicode offset qualification."""
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import tempfile
from types import MappingProxyType

from .artifact import ContractError, strict_json

PACKAGE = Path(__file__).parent
def _freeze(value):
    """Keep bundled policy data recursively immutable inside the package."""
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


_TOKENIZER_MANIFEST_JSON = (PACKAGE / "tokenizer_manifest.json").read_text()
_TOKENIZER_POLICY_JSON = (PACKAGE / "tokenizer_policy.json").read_text()
_TOKENIZER_MANIFEST = _freeze(strict_json(_TOKENIZER_MANIFEST_JSON))
_TOKENIZER_POLICY = _freeze(strict_json(_TOKENIZER_POLICY_JSON))
_DEPENDENCY_PINS = _freeze(strict_json((PACKAGE / "dependency_pins.json").read_text()))
_SOURCE_CONTROL_LITERALS = frozenset(token["content"] for token in _TOKENIZER_POLICY["added_tokens"])


def tokenizer_manifest() -> dict:
    """Return a detached manifest for public build results."""
    return strict_json(_TOKENIZER_MANIFEST_JSON)


def tokenizer_policy() -> dict:
    """Return a detached record of the pinned wrapper and added-token policy."""
    return strict_json(_TOKENIZER_POLICY_JSON)


def source_control_literals() -> frozenset[str]:
    """All 26 pinned added-token literals, independent of mutable tokenizer wrapper lists."""
    return _SOURCE_CONTROL_LITERALS


def check_dependencies() -> dict[str, str]:
    """Require the qualified dependency recipe; do not silently accept a new tokenizer/trainer."""
    actual = {name: version(name) for name in _DEPENDENCY_PINS}
    if actual != _DEPENDENCY_PINS:
        raise ContractError("installed dependencies differ from the qualified pins")
    return actual


def load_tokenizer(directory: Path):
    """Verify pinned local bytes before loading with remote code and networking disabled."""
    check_dependencies()
    expected = {entry["name"] for entry in _TOKENIZER_MANIFEST["files"]}
    # Extra tokenizer/config files could change AutoTokenizer resolution or special tokens.
    if {p.name for p in directory.iterdir()} != expected:
        raise ContractError("tokenizer directory must contain exactly the pinned fixture files")
    captured = {}
    for entry in _TOKENIZER_MANIFEST["files"]:
        data = (directory / entry["name"]).read_bytes()
        if len(data) != entry["bytes"] or sha256(data).hexdigest() != entry["sha256"]:
            raise ContractError("local tokenizer file identity mismatch")
        captured[entry["name"]] = data
    added_tokens = strict_json(captured["tokenizer.json"].decode("utf-8"))["added_tokens"]
    if _freeze(added_tokens) != _TOKENIZER_POLICY["added_tokens"]:
        raise ContractError("added-token policy differs from verified tokenizer source")
    from transformers import AutoTokenizer
    # Load exactly the verified bytes, even if the original directory changes after capture.
    with tempfile.TemporaryDirectory(prefix="gw-tokenizer-snapshot-") as temporary:
        for name, data in captured.items():
            (Path(temporary) / name).write_bytes(data)
        tokenizer = AutoTokenizer.from_pretrained(
            temporary, local_files_only=True, trust_remote_code=False, use_fast=True,
        )
    if not tokenizer.is_fast:
        raise ContractError("real fast-tokenizer offsets are required")
    if sha256(tokenizer.chat_template.encode()).hexdigest() != _TOKENIZER_MANIFEST["chat_template_sha256"]:
        raise ContractError("official chat template identity mismatch")
    validate_tokenizer(tokenizer)
    qualify_offsets(tokenizer)
    return tokenizer


def validate_tokenizer(tokenizer) -> None:
    """Pin both the Rust tokenizer backend and all wrapper settings used by preparation."""
    if not tokenizer.is_fast or sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest() != _TOKENIZER_POLICY["backend_sha256"]:
        raise ContractError("runtime tokenizer differs from pinned backend")
    if sha256(tokenizer.chat_template.encode()).hexdigest() != _TOKENIZER_MANIFEST["chat_template_sha256"]:
        raise ContractError("runtime chat template differs from pinned template")
    if tokenizer.split_special_tokens is not False:
        raise ContractError("runtime split_special_tokens must be False")
    for name, expected in _TOKENIZER_POLICY["wrapper"].items():
        if _freeze(getattr(tokenizer, name, None)) != expected:
            raise ContractError(f"runtime tokenizer wrapper differs: {name}")
    if len(tokenizer) != _TOKENIZER_POLICY["vocab_size"]:
        raise ContractError("runtime tokenizer vocabulary size differs")


def qualify_offsets(tokenizer) -> None:
    """Pin observed codepoint semantics, including the NFC combining-mark offset gap."""
    fixtures = {
        "A🙂中e\u0301Z": [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6)],
        "é漢🚀\nplain": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 9)],
        "<|im_start|>user\n界🙂<|im_end|>\n": [(0, 12), (12, 16), (16, 17), (17, 18), (18, 19), (19, 29), (29, 30)],
    }
    for text, expected in fixtures.items():
        offsets = tokenizer(text, add_special_tokens=False, split_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        if offsets != expected:
            raise ContractError("tokenizer Unicode offset semantics differ from qualified fixtures")
