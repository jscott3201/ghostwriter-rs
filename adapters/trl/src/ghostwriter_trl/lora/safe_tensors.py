"""Stream bounded complete safetensors; hashes describe normalized loaded float32 values."""
from contextlib import contextmanager
from functools import lru_cache
import math
import os
import stat
import struct

import blake3

from ..artifact import ContractError, strict_json
from .shapes import CLIPS, EMBEDDING, HEAD
from .tensors import content_id

MAX_HEADER = 1024**2
MAX_BASE = 11 * 1024**3
MAX_ADAPTER = 64 * 1024**2


@contextmanager
def regular(path):
    """Own one regular descriptor, including when a supplied path names a symlink."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ContractError("Gemma artifact inputs must be regular files")
        yield stream


def clip_bound(name, shape):
    """Only scalar bounds from the known frozen modality clippable linear inventory."""
    return shape == [] and name in _clip_names()


@lru_cache(maxsize=1)
def _clip_names():
    from .config import ROOT, read_config
    from .shapes import base_shapes
    return frozenset(name for name, shape in base_shapes(read_config(ROOT / "release_config.json")[0]).items()
                     if shape == [] and name.rsplit(".", 1)[-1] in CLIPS)


def check_values(bits, name, shape, clips):
    """Permit directed infinite clipping sentinels, never NaNs or nonfinite weights."""
    import numpy as np
    nonfinite = (bits & 0x7F800000) == 0x7F800000
    if clip_bound(name, shape):
        permitted = 0xFF800000 if name.endswith("_min") else 0x7F800000
        if bits.size != 1 or np.any(nonfinite & (bits != permitted)):
            raise ContractError("invalid scalar Gemma clipping bound")
        clips[name] = float(bits.astype("<u4", copy=False).view("<f4")[0])
    elif np.any(nonfinite):
        raise ContractError("nonfinite Gemma weight or adapter tensor")


def check_pairs(clips):
    """Captured lower and upper clipping bounds must describe a nonempty interval."""
    for name, value in clips.items():
        if name.endswith("_min") and (name[:-3] + "max" not in clips or value > clips[name[:-3] + "max"]):
            raise ContractError("inverted or missing Gemma clipping interval")


def measure(path, shapes, *, base=False):
    """Validate every serialized key, shape and byte before hashing the complete population."""
    import numpy as np
    with regular(path) as stream:
        length = os.fstat(stream.fileno()).st_size
        if not 8 < length <= (MAX_BASE if base else MAX_ADAPTER):
            raise ContractError("Gemma safetensor file exceeds supported bounds")
        header_length = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < header_length <= MAX_HEADER or 8 + header_length >= length:
            raise ContractError("invalid Gemma safetensor header length")
        header = strict_json(stream.read(header_length).decode())
        if type(header) is not dict:
            raise ContractError("invalid Gemma safetensor header")
        metadata = header.pop("__metadata__", {})
        if metadata not in ({}, {"format": "pt"}):
            raise ContractError("unsupported Gemma safetensor metadata")
        expected = dict(shapes)
        if base and HEAD not in header:
            del expected[HEAD]
        if header.keys() != expected.keys():
            raise ContractError("Gemma safetensor complete inventory mismatch")
        entries = []
        for name, value in header.items():
            if (type(value) is not dict or value.keys() != {"dtype", "shape", "data_offsets"}
                    or value["shape"] != expected[name] or any(type(v) is not int for v in value["shape"])
                    or value["dtype"] not in (("F32", "BF16") if base else ("F32",))):
                raise ContractError("unsupported Gemma safetensor shape or dtype")
            width = 4 if value["dtype"] == "F32" else 2
            offsets = value["data_offsets"]
            if (type(offsets) is not list or len(offsets) != 2 or any(type(v) is not int for v in offsets)
                    or offsets[0] < 0 or offsets[0] + math.prod(expected[name]) * width != offsets[1]):
                raise ContractError("invalid Gemma safetensor offsets")
            entries.append((*offsets, name, width))
        cursor, content, clips = 0, {}, {}
        for start, end, name, width in sorted(entries):
            if start != cursor or end > length - 8 - header_length:
                raise ContractError("Gemma safetensors contain gaps, overlaps or excessive ranges")
            digest = blake3.blake3()
            remaining = end - start
            while remaining:
                raw = stream.read(min(remaining, 65536))
                if not raw or len(raw) % width:
                    raise ContractError("truncated Gemma safetensor content")
                bits = np.frombuffer(raw, dtype="<u4" if width == 4 else "<u2")
                if width == 2:
                    bits = bits.astype("<u4") << 16
                check_values(bits, name, expected[name], clips)
                digest.update(bits.astype("<u4", copy=False).tobytes())
                remaining -= len(raw)
            content[name] = {"shape": expected[name], "f32_blake3": digest.hexdigest()}
            cursor = end
        if cursor != length - 8 - header_length or stream.read(1):
            raise ContractError("Gemma safetensor payload length mismatch")
    check_pairs(clips)
    if base:
        embedding = content[EMBEDDING]
        if content.get(HEAD, embedding) != embedding:
            raise ContractError("Gemma tied output and embedding differ")
        content[HEAD] = embedding
    return content


def summary(content, parameters):
    """The measured loaded representation is distinct from the release file identity."""
    return {"tensor_content_id": content_id(content), "parameter_count": parameters, "tensor_count": len(content)}
