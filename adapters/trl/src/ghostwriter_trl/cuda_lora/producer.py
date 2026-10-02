"""A consumed CUDA checkpoint path with fresh reload and model-free live ownership."""
from hashlib import sha256
import gc
import os
from pathlib import Path
import platform
import tempfile
import weakref

from ..artifact import ContractError
from ..build import identity, source_identity
from ..lora.config import check_lora_dependencies
from ..lora.safe_tensors import regular
from ..prepared import MAX_BYTES, verify_prepared, VerifiedPrepared
from ..profiles import GEMMA
from ..tokenizer import validate_tokenizer
from ..training.publication import PublishedCheckpointError, publication_cause
from . import capture
from .bundle import inventory, read_checkpoint, write_bundle, native_report, MAX_BYTES as MAX_COMPLETION
from .execution import run
from .environment import observe as observe_environment
from .model import load_base, attach, save_adapter
from .ownership import ObservedCompletion, _Capture, _mint
from .runtime import runtime, require_cuda, memory
from .state import measure
from .publication import Publication
from .lifecycle import clear_tracebacks


def training_source_identity():
    """Bind the complete consumed CUDA, LoRA, preparation and shared helper closure."""
    root = Path(__file__).parent.parent
    paths = [*(root / "cuda_lora").rglob("*"), *(root / "lora").rglob("*"),
             root / "training/capture.py", root / "training/publication.py", root / "training/__init__.py", root / "training/safe_model.py"]
    return identity({str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
                     for p in sorted(paths) if p.is_file() and p.suffix in {".py", ".json"}})


def recipe(max_steps=2, batch_size=1, accumulation=1, learning_rate_millionths=100, max_sequence_length=256):
    """Bound the single-device policy before any model allocation."""
    limits = {"max_steps": (max_steps, 2, 32), "batch_size": (batch_size, 1, 8), "accumulation": (accumulation, 1, 8),
              "learning_rate_millionths": (learning_rate_millionths, 1, 10000), "max_sequence_length": (max_sequence_length, 2, 2048)}
    if any(type(v) is not int or not lower <= v <= upper for v, lower, upper in limits.values()):
        raise ContractError("CUDA recipe exceeds the explicit execution bounds")
    return {"version": 1, "training_source_sha256": training_source_identity(), "preparation_source_sha256": source_identity(),
            "dependencies": check_lora_dependencies(), "runtime": {"python": platform.python_version(),
            "implementation": platform.python_implementation(), "system": platform.system(), "machine": platform.machine()},
            "device": "cuda:0", "precision": "bf16_frozen_fp32_adapter_buffers_v1", "processes": 1, "threads": 1,
            **{k: v for k, (v, _, _) in limits.items()}, "seed": 0, "optimizer": "adamw_cuda_lora_v1", "scheduler": "constant",
            "sampler": "sequential_epoch_v1", "packing": False, "truncation": False}


def _release(reference, baseline, phase):
    gc.collect()
    observed = memory()
    if reference() is not None or observed["allocated"] > baseline:
        raise ContractError(f"CUDA model or tensor allocation remained after {phase}")
    return {"phase": phase, **observed}


def _probe(model, examples):
    import torch
    inputs = torch.tensor([examples[0]["input_ids"][:16]], dtype=torch.long, device="cuda:0")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(input_ids=inputs, use_cache=False).logits
    if not torch.isfinite(output).all():
        raise ContractError("CUDA independent reload probe has nonfinite logits")
    return output.detach().cpu()


def _train(source, prepared, gw, output, policy):
    import torch
    if type(source) is not capture._CapturedSource or type(prepared) is not VerifiedPrepared:
        raise ContractError("CUDA training requires actual verified input and live owned captured source")
    base_path, config, authorization, tokenizer = source._consume()
    examples = prepared.examples
    validate_tokenizer(tokenizer, GEMMA)
    if (prepared.manifest["recipe"].get("preparation_profile", {}).get("name") != GEMMA
            or prepared.manifest["recipe"]["adapter_source_sha256"] != source_identity()
            or not examples or any(len(e["input_ids"]) > policy["max_sequence_length"] for e in examples)):
        raise ContractError("CUDA training requires complete verified Gemma sequences within the bound")
    if os.path.lexists(output):
        raise ContractError("CUDA checkpoint destination already exists")
    cuda_dependencies, cuda_runtime = observe_environment()
    owned = None
    staged = None
    publication = None
    completion_id = None
    errors = []
    model = loaded = None
    phase = "training"
    workspace = tempfile.TemporaryDirectory(prefix="gw-cuda-training-")
    try:
        with runtime(policy["seed"]):
            baseline = memory()["allocated"]
            model, original = load_base(config, base_path / "model.safetensors")
            model, targets = attach(model)
            initial_state = measure(model)
            work = Path(workspace.name)
            initial_adapter = save_adapter(model, config, work / "initial")
            observations = run(model, targets, examples, tokenizer, policy, work / "trainer")
            model.eval()
            final_state = measure(model)
            final_adapter = save_adapter(model, config, work / "final")
            expected_probe = _probe(model, examples)
            if measure(model) != final_state:
                raise ContractError("CUDA checkpoint probe changed runtime state")
            reference = weakref.ref(model)
            model = None
            residency = [_release(reference, baseline, "training_released")]
            prepared_file = work / "prepared.gwsft"
            prepared_file.write_bytes(prepared.data)
            files = {"base/config.json": base_path / "config.json", "base/model.safetensors": base_path / "model.safetensors",
                     "initial/config.json": work / "initial/config.json", "initial/adapter_model.safetensors": work / "initial/adapter_model.safetensors",
                     "final/config.json": work / "final/config.json", "final/adapter_model.safetensors": work / "final/adapter_model.safetensors",
                     "prepared.gwsft": prepared_file}
            manifest = {"version": 1, "prepared_build_id": prepared.build_id, "recipe": policy, "base_model": original,
                        "initial_adapter": initial_adapter, "final_adapter": final_adapter,
                        "precision_operations": observations.pop("precision_operations"), "observations": observations,
                        "source_authorization": authorization, "upstream_lineage": "unknown", "checkpoint_kind": "gemma_cuda_qv_lora_v1",
                        "initial_state": initial_state, "final_state": final_state, "probe_policy": "cuda_bf16_exact_probe_v1",
                        "cuda_runtime": cuda_runtime, "cuda_dependencies": cuda_dependencies, "files": inventory(files)}
            with tempfile.NamedTemporaryFile(mode="w+b", dir=output.parent, prefix=".cuda-checkpoint-", delete=False) as stream:
                staged = Path(stream.name)
                completion_id = write_bundle(stream, manifest, files)
                stream.flush(); os.fsync(stream.fileno())
            phase = "independent_reload"
            with read_checkpoint(staged, gw, tokenizer) as loaded:
                if loaded.report["completion_id"] != completion_id:
                    raise ContractError("CUDA reload completion identity mismatch")
                actual_probe = _probe(loaded.model, examples)
                if not torch.equal(actual_probe, expected_probe) or measure(loaded.model) != final_state:
                    raise ContractError("fresh CUDA reload failed the exact versioned probe/state policy")
                reference = weakref.ref(loaded.model)
            loaded = None
            residency.append(_release(reference, baseline, "verification_released"))
            observed = {**observations, "prepared_build_id": prepared.build_id, "recipe_id": identity(policy),
                        "fresh_reload": "passed", "probe_policy": manifest["probe_policy"], "model_residency": residency,
                        "initial_state": initial_state, "final_state": final_state}
            if policy["training_source_sha256"] != training_source_identity() or policy["preparation_source_sha256"] != source_identity():
                raise ContractError("CUDA producer source changed during execution")
            # Preserve a separate complete private capture before publishing the public path.
            owned = _Capture(staged, MAX_COMPLETION)
            if native_report(owned.path, gw)["completion_id"] != completion_id:
                raise ContractError("owned CUDA completion capture changed")
            phase = "publication"
            publication = Publication(completion_id, prepared.build_id, output)
            publication.commit(staged)
            phase = "runtime_cleanup"
    except BaseException as error:
        errors.append((publication.phase if phase == "publication" and publication is not None else phase, error))
    finally:
        # Exception tracebacks may retain completed trainer frames and CUDA tensors.
        clear_tracebacks(error for _, error in errors)
        model = None
        cleanups = [("model_collection", gc.collect), ("workspace_cleanup", workspace.cleanup)]
        if loaded is not None:
            cleanups.insert(0, ("reload_cleanup", loaded.close))
        if staged is not None:
            cleanups.append(("staged_file_cleanup", lambda: staged.unlink(missing_ok=True)))
        if publication is not None:
            cleanups.append(("directory_close", publication.close))
        for name, cleanup in cleanups:
            try:
                cleanup()
            except BaseException as error:
                errors.append((name, error))
    if errors:
        if owned is not None:
            try:
                owned.close()
            except BaseException as error:
                errors.append(("capture_cleanup", error))
        if publication is not None:
            publication.raise_failure(errors)
        raise errors[0][1]
    try:
        return _mint(owned, completion_id, observed, policy["training_source_sha256"], publication)
    except BaseException as error:
        errors = [("receipt_issuance", error)]
        try:
            owned.close()
        except BaseException as cleanup_error:
            errors.append(("capture_cleanup", cleanup_error))
        clear_tracebacks(error for _, error in errors)
        publication.raise_failure(errors)


def _execute(prepared_path, source_context, gw, output, options):
    policy = recipe(**options)
    require_cuda()
    observe_environment()
    completed = None
    try:
        with source_context as source:
            with regular(prepared_path) as stream:
                prepared = verify_prepared(stream.read(MAX_BYTES + 1), gw, source.tokenizer)
            completed = _train(source, prepared, gw, Path(output), policy)
    except BaseException as error:
        recorded = publication_cause(error)
        if completed is not None:
            failures = [("source_cleanup", error)]
            try:
                completed.close()
            except PublishedCheckpointError as cleanup_error:
                raise cleanup_error.with_cleanup_error("source_cleanup", error) from error
            except BaseException as cleanup_error:
                failures.append(("capture_cleanup", cleanup_error))
            raise PublishedCheckpointError(completed.completion_id, completed.observed["prepared_build_id"], output, True,
                                           failures) from error
        if recorded is not None and recorded is not error:
            raise recorded.with_cleanup_error("source_cleanup", error) from error
        raise
    return completed


def train(prepared_path: Path, release_directory: Path, gw: Path, output: Path, **options) -> ObservedCompletion:
    """Train only exact approved local release bytes and the whole verified prepared input."""
    return _execute(prepared_path, capture.approved(release_directory), gw, output, options)


def train_fixture(prepared_path: Path, tokenizer, gw: Path, output: Path, **options) -> ObservedCompletion:
    """Explicit reduced official random fixture entry point for later CUDA qualification."""
    return _execute(prepared_path, capture.fixture(tokenizer), gw, output, options)
