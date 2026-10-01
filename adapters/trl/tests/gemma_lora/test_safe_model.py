"""Safe state boundaries include persistent buffers and exact clipping exceptions."""
import struct
import pytest

from ghostwriter_trl.artifact import ContractError
from ghostwriter_trl.lora.config import ROOT, create, read_config
from ghostwriter_trl.lora.shapes import base_shapes, parameter_count, targets
from ghostwriter_trl.lora.safe_model import save_owned_base, load_base, save_adapter, reload_adapter
from ghostwriter_trl.lora.safe_tensors import check_values, check_pairs, regular
from ghostwriter_trl.lora.targets import attach


def test_official_meta_inventory_matches_independent_bounded_layouts():
    for filename, count in (("fixture_config.json", 6), ("release_config.json", 50)):
        config, _ = read_config(ROOT / filename)
        model = create(config, device="meta")
        assert {name: list(value.shape) for name, value in model.state_dict().items()} == base_shapes(config)
        assert sum(p.numel() for p in model.parameters()) == parameter_count(config)
        assert len(targets(config)) == count


def test_exact_directed_clipping_sentinels_only():
    import numpy as np
    prefix = "model.vision_tower.encoder.layers.0.self_attn.q_proj."
    clips = {}
    check_values(np.array([0xff800000], dtype="<u4"), prefix + "input_min", [], clips)
    check_values(np.array([0x7f800000], dtype="<u4"), prefix + "input_max", [], clips)
    check_pairs(clips)
    for name, shape, bits in [(prefix + "input_min", [], 0x7f800000),
                              (prefix + "input_max", [], 0xff800000),
                              (prefix + "input_min", [], 0x7fc00000),
                              (prefix + "input_min", [1], 0xff800000),
                              ("unexpected.input_min", [], 0xff800000),
                              (prefix + "linear.weight", [], 0x7f800000)]:
        with pytest.raises(ContractError):
            check_values(np.array([bits], dtype="<u4"), name, shape, {})


def test_regular_capture_follows_one_owned_symlink_descriptor_and_rejects_fifo(tmp_path):
    import os
    first, second, link = (tmp_path / name for name in ("first", "second", "link"))
    first.write_bytes(b"first"); second.write_bytes(b"second"); link.symlink_to(first)
    with regular(link) as stream:
        link.unlink(); link.symlink_to(second)
        assert stream.read() == b"first"
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ContractError, match="regular"):
        with regular(fifo):
            pytest.fail("FIFO admitted")
