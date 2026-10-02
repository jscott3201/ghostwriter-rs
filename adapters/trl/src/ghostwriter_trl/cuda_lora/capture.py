"""Owned source bytes independent of model allocation and release-path lifetime."""
from contextlib import contextmanager
from pathlib import Path
import os
import tempfile
from weakref import WeakKeyDictionary

from ..artifact import ContractError
from ..lora.config import CONFIG_SHA256, WEIGHTS_BYTES, WEIGHTS_SHA256, RELEASE, FIXTURE, ROOT, read_config
from ..profiles import GEMMA
from ..tokenizer import load_tokenizer, tokenizer_manifest
from ..training.capture import _copy_checked


_SOURCES = WeakKeyDictionary()


class _CapturedSource:
    """Single-use ownership issued only inside an actual source-capture lifetime."""
    __slots__ = ("__weakref__",)

    def __new__(cls, *args, **kwargs):
        raise TypeError("use the approved release or owned fixture capture")

    def __init_subclass__(cls, **kwargs):
        raise TypeError("CUDA source ownership cannot be subclassed")

    @property
    def tokenizer(self):
        if self not in _SOURCES:
            raise ContractError("CUDA source capture is closed or consumed")
        return _SOURCES[self][3]

    def _consume(self):
        if self not in _SOURCES:
            raise ContractError("requires a live unconsumed owned CUDA source capture")
        return _SOURCES.pop(self)


@contextmanager
def _issued(state):
    result = object.__new__(_CapturedSource)
    _SOURCES[result] = state
    try:
        yield result
    finally:
        _SOURCES.pop(result, None)


@contextmanager
def approved(directory):
    """Copy exactly approved local source bytes without loading model objects."""
    pins = {item["name"]: (item["bytes"], item["sha256"]) for item in tokenizer_manifest(GEMMA)["files"]}
    model_pins = {"config.json": (None, CONFIG_SHA256), "model.safetensors": (WEIGHTS_BYTES, WEIGHTS_SHA256)}
    directory = Path(directory)
    if {entry.name for entry in directory.iterdir()} != set(pins) | set(model_pins):
        raise ContractError("CUDA release requires exactly the approved source inventory")
    with tempfile.TemporaryDirectory(prefix="gw-cuda-source-") as temporary:
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
            raise ContractError("CUDA source is not the approved release")
        with _issued((base, config, RELEASE, load_tokenizer(tokens, profile=GEMMA))) as source:
            yield source


@contextmanager
def fixture(tokenizer):
    """Create the owned random reduced official fixture; supplied weights are never accepted."""
    import torch
    from ..lora.config import create
    from ..lora.safe_model import save_owned_base
    config, kind = read_config(ROOT / "fixture_config.json")
    if kind != FIXTURE:
        raise ContractError("owned fixture configuration identity mismatch")
    with tempfile.TemporaryDirectory(prefix="gw-cuda-fixture-") as temporary:
        root = Path(temporary) / "base"
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(0)
                model = create(config)
            save_owned_base(model, config, root)
            del model
        finally:
            torch.set_num_threads(previous_threads)
        with _issued((root, config, FIXTURE, tokenizer)) as source:
            yield source
