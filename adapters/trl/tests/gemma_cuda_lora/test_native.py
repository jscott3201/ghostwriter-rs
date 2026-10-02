"""Synthetic saved declarations exercise real native byte verification, never live CUDA issuance."""
import copy
from pathlib import Path
import struct
import subprocess

import blake3
import pytest
import torch

from ghostwriter_trl.cuda_lora import model as cuda_model
from ghostwriter_trl.cuda_lora.bundle import inventory, write_bundle, native_report
from ghostwriter_trl.cuda_lora.capture import fixture
from ghostwriter_trl.cuda_lora.producer import recipe
from ghostwriter_trl.cuda_lora.state import measure, DOMAINS
from ghostwriter_trl.artifact import ContractError
from ghostwriter_trl.lora.safe_model import adapter_config, measure_adapter
from ghostwriter_trl.lora.targets import resolve_targets, _config, adapter_shapes
from ghostwriter_trl.prepared import _json_bytes, read_prepared


def rehash_states(manifest):
    for state in (manifest["initial_state"], manifest["final_state"]):
        for kind, population in state.items():
            population["state_id"] = blake3.blake3(_json_bytes(population["tensors"]), derive_key_context=DOMAINS[kind]).hexdigest()


@pytest.fixture
def declared_checkpoint(gw, tokenizer, fixture_dir, tmp_path, monkeypatch):
    from peft import get_peft_model, get_peft_model_state_dict
    from safetensors.torch import save_file
    monkeypatch.setattr(cuda_model, "require_cuda", lambda: torch.device("cpu"))
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with fixture(tokenizer) as source:
            base_path, config, authorization, _ = source._consume()
            model, original = cuda_model.load_base(config, base_path / "model.safetensors")
            targets = resolve_targets(model)
            model = get_peft_model(model, _config(targets), autocast_adapter_dtype=True)
            model.peft_config["default"].base_model_name_or_path = ""
            initial_state = measure(model)
            summaries = {}
            for phase in ("initial", "final"):
                if phase == "final":
                    with torch.no_grad():
                        for parameter in model.parameters():
                            if parameter.requires_grad:
                                parameter.add_(0.001)
                directory = tmp_path / phase
                directory.mkdir()
                (directory / "config.json").write_bytes(_json_bytes(adapter_config(config)))
                state = get_peft_model_state_dict(model, save_embedding_layers=False)
                save_file({n: t.detach().contiguous() for n, t in state.items()}, str(directory / "adapter_model.safetensors"), metadata={"format": "pt"})
                summaries[phase] = measure_adapter(config, directory / "adapter_model.safetensors")
            final_state = measure(model)
            del model
            prepared = read_prepared(fixture_dir / "prepared-all.gwsft", gw, tokenizer)
            policy = recipe(max_sequence_length=2048)
            policy["runtime"].update(system="Linux", machine="x86_64")
            batches = [{"update": index + 1, "example_ids": [e["example_id"]], "input_tokens": len(e["input_ids"]),
                       "shifted_supervised_tokens": sum(label != -100 for label in e["labels"][1:]),
                       "finite_adapter_gradients": 12, "loss_binary64": struct.pack(">d", 1.0).hex()}
                      for index, e in enumerate(prepared.examples[:2])]
            observations = {"successful_microbatches": 2, "consumed_examples": 2,
                            "shifted_supervised_tokens": sum(b["shifted_supervised_tokens"] for b in batches),
                            "optimizer_updates": 2, "successful_forwards": 2, "microbatches": batches,
                            "trainables": [{"name": n, "shape": list(s), "parameters": s[0] * s[1], "dtype": "float32", "device": "cuda:0"}
                                           for n, s in sorted(adapter_shapes(targets).items())],
                            "base_unchanged": True, "adapter_content_changed": True}
            files = {"base/config.json": base_path / "config.json", "base/model.safetensors": base_path / "model.safetensors",
                     "prepared.gwsft": fixture_dir / "prepared-all.gwsft",
                     **{f"{phase}/{name}": tmp_path / phase / name for phase in ("initial", "final") for name in ("config.json", "adapter_model.safetensors")}}
            manifest = {"version": 1, "prepared_build_id": prepared.build_id, "recipe": policy, "base_model": original,
                        "initial_adapter": summaries["initial"], "final_adapter": summaries["final"], "observations": observations,
                        "source_authorization": authorization, "upstream_lineage": "unknown", "checkpoint_kind": "gemma_cuda_qv_lora_v1",
                        "initial_state": initial_state, "final_state": final_state, "probe_policy": "cuda_bf16_exact_probe_v1",
                        "precision_operations": {f"{label}:aten.{op}.default:torch.float32:cuda": 1 for label, op in
                                                 (("norm", "pow"), ("rope", "bmm"), ("softmax", "_softmax"), ("loss", "nll_loss"))} | {"matmul:aten.mm.default:torch.bfloat16:cuda": 1},
                        "cuda_runtime": {"torch_cuda": "12.8", "torch_build": "2.8.0+cu128", "device_name": "synthetic saved declaration", "capability": "0.0", "installed_wheel_records_sha256": "a" * 64},
                        "cuda_dependencies": __import__("json").loads((Path(cuda_model.__file__).parent / "dependencies.json").read_text()),
                        "files": inventory(files)}
            yield manifest, files
    finally:
        torch.set_num_threads(previous)


def saved(tmp_path, manifest, files):
    path = tmp_path / "declared.gwckpt"
    with path.open("wb") as stream:
        write_bundle(stream, manifest, files)
    return path


def test_native_accepts_measured_bytes_but_never_grants_execution_authority(declared_checkpoint, gw, tmp_path):
    manifest, files = declared_checkpoint
    path = saved(tmp_path, manifest, files)
    report = native_report(path, gw)
    assert report["historical_training"] == "declared" and report["model_reload"] == "not_run"
    assert report["materialized_state"] == "persistent_source_derived_nonpersistent_declared"
    assert report["declarations"]["final_state"] == manifest["final_state"]
    with path.open("rb") as stream:
        old = subprocess.run([str(gw), "artifact", "verify-lora", "--stdin"], stdin=stream, capture_output=True)
    assert old.returncode != 0


@pytest.mark.parametrize("mutation", ["frozen_digest", "buffer_digest", "adapter_dtype", "alias", "missing_buffer", "cpu_recipe", "probe_policy", "precision", "counter", "extra_tensor", "missing_nonpersistent"])
def test_rehashed_invalid_declarations_fail_native_consumption(mutation, declared_checkpoint, gw, tmp_path):
    original, files = declared_checkpoint
    manifest = copy.deepcopy(original)
    for state in (manifest["initial_state"], manifest["final_state"]):
        if mutation == "frozen_digest":
            for row in state["frozen"]["tensors"].values():
                row["blake3"] = "0" * 64
        elif mutation == "buffer_digest":
            next(row for row in state["buffers"]["tensors"].values() if row["persistent"])["blake3"] = "0" * 64
        elif mutation == "adapter_dtype":
            next(iter(state["adapters"]["tensors"].values()))["dtype"] = "bfloat16"
        elif mutation == "alias":
            state["frozen"]["tensors"]["model.language_model.embed_tokens.weight"]["alias"] = "model.language_model.embed_tokens.weight"
        elif mutation == "missing_buffer":
            name = next(n for n, row in state["buffers"]["tensors"].items() if row["persistent"])
            del state["buffers"]["tensors"][name]
        elif mutation == "missing_nonpersistent":
            name = next(n for n, row in state["buffers"]["tensors"].items() if not row["persistent"])
            del state["buffers"]["tensors"][name]
        elif mutation == "extra_tensor":
            state["frozen"]["tensors"]["extra"] = {"shape": [1], "dtype": "bfloat16", "blake3": "a" * 64, "alias": "extra", "persistent": True}
    if mutation == "cpu_recipe":
        manifest["recipe"]["device"] = "cpu"
    elif mutation == "probe_policy":
        manifest["probe_policy"] = "cpu_tolerance"
    elif mutation == "precision":
        manifest["precision_operations"] = {}
    elif mutation == "counter":
        manifest["observations"]["optimizer_updates"] = 1
    rehash_states(manifest)
    with pytest.raises(ContractError, match="native"):
        native_report(saved(tmp_path, manifest, files), gw)


def test_nonpersistent_values_remain_declared_until_fresh_runtime_reload(declared_checkpoint, gw, tmp_path):
    original, files = declared_checkpoint
    manifest = copy.deepcopy(original)
    for state in (manifest["initial_state"], manifest["final_state"]):
        for row in state["buffers"]["tensors"].values():
            if not row["persistent"]:
                row["blake3"] = "0" * 64
    rehash_states(manifest)
    report = native_report(saved(tmp_path, manifest, files), gw)
    assert report["materialized_state"] == "persistent_source_derived_nonpersistent_declared"
    assert report["model_reload"] == "not_run"


@pytest.mark.parametrize("build,supported", [("2.8.0", True), ("2.8.0+cu128", True), ("99.0.0", False)])
def test_native_torch_build_allowlist(build, supported, declared_checkpoint, gw, tmp_path):
    manifest, files = declared_checkpoint
    manifest["cuda_runtime"]["torch_build"] = build
    path = saved(tmp_path, manifest, files)
    if supported:
        assert native_report(path, gw)["structural_validation"] == "passed"
    else:
        with pytest.raises(ContractError, match="native"):
            native_report(path, gw)
