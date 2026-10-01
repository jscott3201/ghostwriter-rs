"""Owned local Gemma capture, with fixed application approval and no network loader."""
from contextlib import contextmanager
from pathlib import Path
import os
import tempfile

from ..artifact import ContractError
from ..profiles import GEMMA
from ..tokenizer import load_tokenizer, tokenizer_manifest
from ..training.capture import _copy_checked
from .config import CONFIG_SHA256, WEIGHTS_BYTES, WEIGHTS_SHA256, RELEASE, FIXTURE, read_config, check_lora_dependencies
from .safe_model import load_base, save_owned_base


class _LoadedBase:
    """Single-use base ownership, never reconstructed from a saved receipt."""
    __slots__ = ("__state",)

    def __new__(cls, *args, **kwargs):
        raise TypeError("use the approved local base loader")

    def __setattr__(self, name, value):
        raise AttributeError("owned base state is immutable")

    def __init_subclass__(cls, **kwargs):
        raise TypeError("owned base capabilities cannot be subclassed")

    def _consume(self):
        state = self.__state
        if state is None:
            raise ContractError("owned Gemma base has already been consumed")
        object.__setattr__(self, "_LoadedBase__state", None)
        return state


def _owned(config, root, authorization):
    model, measured = load_base(config, root / "model.safetensors")
    result = object.__new__(_LoadedBase)
    object.__setattr__(result, "_LoadedBase__state", (model, config, measured, root, authorization))
    return result


@contextmanager
def load_approved_release(directory):
    """Capture precisely the approved local base plus tokenizer files through regular descriptors."""
    check_lora_dependencies()
    pins = {item["name"]: (item["bytes"], item["sha256"]) for item in tokenizer_manifest(GEMMA)["files"]}
    model_pins = {"config.json": (None, CONFIG_SHA256), "model.safetensors": (WEIGHTS_BYTES, WEIGHTS_SHA256)}
    directory = Path(directory)
    if {entry.name for entry in directory.iterdir()} != set(pins) | set(model_pins):
        raise ContractError("Gemma release must contain precisely the approved model and tokenizer inventory")
    with tempfile.TemporaryDirectory(prefix="gw-gemma-base-") as temporary:
        root = Path(temporary)
        base, tokens = root / "base", root / "tokenizer"
        base.mkdir(); tokens.mkdir()
        for name, (size, digest) in pins.items():
            _copy_checked(directory / name, tokens / name, size, digest)
        for name, (size, digest) in model_pins.items():
            if name in pins:
                os.link(tokens / name, base / name)
            else:
                _copy_checked(directory / name, base / name, size, digest)
        config, kind = read_config(base / "config.json")
        if kind != RELEASE:
            raise ContractError("captured Gemma base is not the approved release")
        yield _owned(config, base, RELEASE), load_tokenizer(tokens, profile=GEMMA)


@contextmanager
def owned_fixture():
    """Generate the explicit seeded reduced random CPU fixture; never accepts supplied weights."""
    import torch
    from .config import ROOT, create
    config, kind = read_config(ROOT / "fixture_config.json")
    assert kind == FIXTURE
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="gw-gemma-fixture-") as temporary:
            root = Path(temporary) / "base"
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0)
                model = create(config)
            save_owned_base(model, config, root)
            del model
            yield _owned(config, root, FIXTURE)
    finally:
        torch.set_num_threads(previous)
