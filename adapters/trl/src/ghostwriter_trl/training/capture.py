"""Approved release capture and fresh model ownership; saved declarations confer no authority."""
from contextlib import contextmanager
from hashlib import sha256
import os
from pathlib import Path
import stat
import tempfile
from types import MappingProxyType

from ..artifact import ContractError
from ..tokenizer import check_dependencies, load_tokenizer, tokenizer_manifest
from .safe_model import load_model, read_config

APPROVED_RELEASE = "qwen3_0_6b_c1899de_student_training_v1"
# Publisher release commitments, checked against the pinned Hugging Face tree and Git blobs.
# This specific Student/Training approval does not resolve the publisher's missing parent revision.
_MODEL_PINS = MappingProxyType({
    "config.json": (726, "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"),
    "generation_config.json": (239, "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2"),
    "model.safetensors": (1503300328, "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"),
})


class _LoadedModel:
    """Single-use owned model capability. Only fresh approved loading supplies public authority."""
    __slots__ = ("__state",)

    def __new__(cls, *args, **kwargs):
        raise TypeError("loaded models are created only by the approved local loader")

    def __setattr__(self, name, value):
        raise AttributeError("loaded model state is private")

    def __init_subclass__(cls, **kwargs):
        raise TypeError("loaded model capabilities cannot be subclassed")

    def _consume(self):
        state = self.__state
        if state is None:
            raise ContractError("loaded model was already consumed")
        object.__setattr__(self, "_LoadedModel__state", None)
        return state


def _copy_checked(source: Path, destination: Path, size, digest: str):
    bound = size if size is not None else 65536
    hasher = sha256()
    count = 0
    descriptor = os.open(source, os.O_RDONLY | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ContractError("local release input must be a regular file")
        # The descriptor fixes the captured inode even if a symlink/path changes afterwards.
        with os.fdopen(descriptor, "rb", closefd=False) as reader, destination.open("xb") as writer:
            while data := reader.read(min(1024**2, bound + 1 - count)):
                count += len(data)
                if count > bound:
                    raise ContractError("local release file exceeds its pinned bound")
                hasher.update(data)
                writer.write(data)
    finally:
        os.close(descriptor)
    if (size is not None and size != count) or hasher.hexdigest() != digest:
        raise ContractError("local release bytes differ from the approved exact release")


@contextmanager
def load_approved_release(directory: Path):
    """Capture the exact approved nine-file local release once, then safely load those bytes.

    There is no network, caller manifest, arbitrary model-object input, or fixture CLI flag.
    The returned tokenizer and model both come from this private captured snapshot.
    """
    check_dependencies()
    tokenizer_pins = {item["name"]: (item["bytes"], item["sha256"]) for item in tokenizer_manifest()["files"]}
    expected = set(tokenizer_pins) | set(_MODEL_PINS)
    if {entry.name for entry in directory.iterdir()} != expected:
        raise ContractError("approved local release must contain exactly the nine pinned files")
    with tempfile.TemporaryDirectory(prefix="gw-student-capture-") as temporary:
        root = Path(temporary)
        initial = root / "initial"
        tokenized = root / "tokenizer"
        initial.mkdir(); tokenized.mkdir()
        for name, (size, digest) in tokenizer_pins.items():
            _copy_checked(directory / name, tokenized / name, size, digest)
        for name, (size, digest) in _MODEL_PINS.items():
            _copy_checked(directory / name, initial / name, size, digest)
        tokenizer = load_tokenizer(tokenized)
        config = read_config(initial / "config.json")
        model, summary = load_model(config, initial / "model.safetensors")
        loaded = object.__new__(_LoadedModel)
        object.__setattr__(loaded, "_LoadedModel__state", (model, config, summary, initial, APPROVED_RELEASE))
        yield loaded, tokenizer
