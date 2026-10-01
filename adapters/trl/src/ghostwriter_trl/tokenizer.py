"""Exact local tokenizer/processor profiles with explicit wrapper and offset qualification."""
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import stat
import tempfile

from .artifact import ContractError, strict_json
from .profiles import GEMMA, PACKAGE, QWEN, data, detached, freeze


def tokenizer_manifest(profile: str = QWEN) -> dict:
    """Return the named profile's exact local release inventory."""
    return detached(profile, "manifest")


def tokenizer_policy(profile: str = QWEN) -> dict:
    """Return a detached record of the qualified wrapper, backend, and added-token policy."""
    return detached(profile, "policy")


def source_control_literals(profile: str = QWEN) -> frozenset[str]:
    """All pinned added-token literals, independent of mutable tokenizer wrapper lists."""
    return frozenset(token["content"] for token in data(profile, "policy")["added_tokens"])


def check_dependencies(profile: str = QWEN) -> dict[str, str]:
    """Require the complete qualified dependency profile; no implicit stack upgrades."""
    pins = data(profile, "dependencies")
    try:
        actual = {name: version(name) for name in pins}
    except PackageNotFoundError as error:
        raise ContractError("install the complete pinned dependencies for the selected profile") from error
    if actual != pins:
        raise ContractError("installed dependencies differ from the qualified profile pins")
    return actual


def _capture(directory: Path, manifest):
    if {p.name for p in directory.iterdir()} != {entry["name"] for entry in manifest["files"]}:
        raise ContractError("tokenizer directory must contain exactly the pinned fixture files")
    captured = {}
    for entry in manifest["files"]:
        descriptor = os.open(directory / entry["name"], os.O_RDONLY | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ContractError("tokenizer capture requires regular file bytes")
            value = stream.read(entry["bytes"] + 1)
        if len(value) != entry["bytes"] or sha256(value).hexdigest() != entry["sha256"]:
            raise ContractError("local tokenizer file identity mismatch")
        captured[entry["name"]] = value
    return captured


def load_tokenizer(directory: Path, *, profile: str = QWEN):
    """Load captured pinned local bytes and the official text renderer, with networking disabled."""
    check_dependencies(profile)
    manifest, policy = data(profile, "manifest"), data(profile, "policy")
    captured = _capture(directory, manifest)
    added = strict_json(captured["tokenizer.json"].decode("utf-8"))["added_tokens"]
    if freeze(added) != policy["added_tokens"]:
        raise ContractError("added-token policy differs from verified tokenizer source")
    with tempfile.TemporaryDirectory(prefix="gw-tokenizer-snapshot-") as temporary:
        for name, value in captured.items():
            (Path(temporary) / name).write_bytes(value)
        if profile == GEMMA:
            from transformers import Gemma4Processor
            processor = Gemma4Processor.from_pretrained(temporary, local_files_only=True, trust_remote_code=False)
            tokenizer = processor.tokenizer
            # The release defaults to left padding; explicit causal training uses right padding.
            tokenizer.padding_side = "right"
            tokenizer._ghostwriter_processor = processor
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(temporary, local_files_only=True, trust_remote_code=False, use_fast=True)
    validate_tokenizer(tokenizer, profile)
    qualify_offsets(tokenizer, profile)
    return tokenizer


def official_renderer(tokenizer, profile: str = QWEN):
    """Return the actual official renderer whose template and tokenizer are validated together."""
    data(profile, "manifest")
    if profile == GEMMA:
        from transformers import Gemma4Processor, GemmaTokenizer
        processor = getattr(tokenizer, "_ghostwriter_processor", None)
        if type(tokenizer) is not GemmaTokenizer or type(processor) is not Gemma4Processor or processor.tokenizer is not tokenizer:
            raise ContractError("Gemma text preparation requires its actual bound official processor")
        return processor
    return tokenizer


def validate_tokenizer(tokenizer, profile: str = QWEN) -> None:
    """Pin the Rust tokenizer backend, official renderer, and every consumed wrapper setting."""
    policy, manifest = data(profile, "policy"), data(profile, "manifest")
    if not tokenizer.is_fast or sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest() != policy["backend_sha256"]:
        raise ContractError("runtime tokenizer differs from pinned backend")
    renderer = official_renderer(tokenizer, profile)
    if not isinstance(renderer.chat_template, str) or sha256(renderer.chat_template.encode()).hexdigest() != manifest["chat_template_sha256"]:
        raise ContractError("runtime chat template differs from pinned template")
    for name, expected in policy["wrapper"].items():
        if freeze(getattr(tokenizer, name, None)) != expected:
            raise ContractError(f"runtime tokenizer wrapper differs: {name}")
    if len(tokenizer) != policy["vocab_size"]:
        raise ContractError("runtime tokenizer vocabulary size differs")


def qualify_offsets(tokenizer, profile: str = QWEN) -> None:
    """Check actual codepoint offsets for the profile's exact normalization/backend behavior."""
    data(profile, "manifest")
    fixtures = {
        "A🙂中e\u0301Z": [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6)],
        "é漢🚀\nplain": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 9)],
        "<|im_start|>user\n界🙂<|im_end|>\n": [(0, 12), (12, 16), (16, 17), (17, 18), (18, 19), (19, 29), (29, 30)],
    } if profile == QWEN else {
        "A🙂中e\u0301Z": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
        "é漢🚀\nplain": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 9)],
        "<bos><|turn>user\n界🙂<turn|>\n": [(0, 5), (5, 12), (12, 16), (16, 17), (17, 18), (18, 19), (19, 26), (26, 27)],
    }
    for text, expected in fixtures.items():
        offsets = tokenizer(text, add_special_tokens=False, split_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        if offsets != expected:
            raise ContractError("tokenizer Unicode offset semantics differ from qualified fixtures")
