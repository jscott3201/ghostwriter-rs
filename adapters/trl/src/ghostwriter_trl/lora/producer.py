"""Fresh owned training observations, separate from saved completion declarations."""
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import tempfile
from weakref import WeakKeyDictionary

from ..artifact import ContractError, strict_json
from ..build import identity, source_identity
from ..prepared import VerifiedPrepared, verify_prepared, MAX_BYTES
from .safe_tensors import regular
from ..tokenizer import validate_tokenizer
from .bundle import inventory, read_checkpoint, write_bundle
from .capture import _LoadedBase, load_approved_release
from .execution import run
from ..training.publication import PublishedCheckpointError, publication_cause
from .safe_model import save_adapter
from .targets import attach
from .runtime import seeded_runtime
from .config import check_lora_dependencies
from ..profiles import GEMMA

_LIVE_COMPLETIONS = WeakKeyDictionary()


class ObservedCompletion:
    """Receipt from this actual successful producer call; loading a bundle cannot create it."""
    __slots__ = ("__state", "__weakref__")

    def __new__(cls, *args, **kwargs):
        raise TypeError("observed completion is created only by successful local training")

    def __setattr__(self, name, value):
        raise AttributeError("observed completion state is immutable")

    def __init_subclass__(cls, **kwargs):
        raise TypeError("observed completion receipts cannot be subclassed")

    @property
    def completion_id(self):
        """Complete final checkpoint identity, separate from the original prepared identity."""
        return self.__state[0]

    @property
    def observed(self):
        """Defensive copy of completed batch/update counters and fresh reload evidence."""
        return strict_json(self.__state[1])


def training_source_identity():
    """Bind every installed training source/policy, preserving the older preparation identity."""
    root = Path(__file__).parent.parent
    paths = [*(root / "lora").rglob("*"), root / "training/capture.py", root / "training/publication.py",
             root / "training/__init__.py", root / "training/safe_model.py"]
    return identity({str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
                     for path in sorted(paths) if path.is_file() and path.suffix in {".py", ".json"}})



def _recipe(max_steps, batch_size, accumulation, learning_rate_millionths, max_sequence_length):
    limits = {"max_steps": (max_steps, 2, 32), "batch_size": (batch_size, 1, 8),
              "accumulation": (accumulation, 1, 8), "learning_rate_millionths": (learning_rate_millionths, 1, 10000),
              "max_sequence_length": (max_sequence_length, 2, 2048)}
    if any(type(value) is not int or not lower <= value <= upper for value, lower, upper in limits.values()):
        raise ContractError("Gemma LoRA recipe exceeds supported bounds")
    return {"version": 1, "training_source_sha256": training_source_identity(),
            "preparation_source_sha256": source_identity(), "dependencies": check_lora_dependencies(),
            "runtime": {"python": platform.python_version(), "implementation": platform.python_implementation(),
                        "system": platform.system(), "machine": platform.machine()},
            "device": "cpu", "precision": "float32", "processes": 1, "threads": 1,
            **{key: value for key, (value, _, _) in limits.items()}, "seed": 0,
            "optimizer": "adamw_torch_lora_v1", "scheduler": "constant", "sampler": "sequential_epoch_v1",
            "packing": False, "truncation": False}


def _run_loaded(prepared, tokenizer, loaded, gw, output, *, max_steps=2, batch_size=1,
                accumulation=1, learning_rate_millionths=100, max_sequence_length=2048):
    if type(prepared) is not VerifiedPrepared or type(loaded) is not _LoadedBase:
        raise ContractError("training requires actual verified input and a freshly loaded owned model")
    validate_tokenizer(tokenizer, GEMMA)
    recipe = _recipe(max_steps, batch_size, accumulation, learning_rate_millionths, max_sequence_length)
    if os.path.lexists(output):
        raise ContractError("checkpoint destination already exists")
    examples = prepared.examples
    if not examples or any(len(e["input_ids"]) > max_sequence_length for e in examples):
        raise ContractError("training requires nonempty complete sequences within the explicit bound")
    model, config, initial_summary, initial, authorization = loaded._consume()
    from .tensors import content_id, live_content
    if content_id(live_content(model)) != initial_summary["tensor_content_id"]:
        raise ContractError("owned Gemma base changed after capture")
    text = config["text_config"]
    if (prepared.manifest["recipe"].get("preparation_profile", {}).get("name") != GEMMA
            or prepared.manifest["recipe"]["adapter_source_sha256"] != source_identity()
            or text["vocab_size"] < len(tokenizer)
            or any(len(e["input_ids"]) > text["max_position_embeddings"]
                   or max(e["input_ids"]) >= text["vocab_size"] for e in examples)):
        raise ContractError("Gemma base or preparation identity does not support the complete input")
    import torch
    old_threads = torch.get_num_threads()
    rng = None
    staged = None
    workspace = None
    directory = None
    published = directory_synced = False
    errors = []
    phase = "training"
    try:
        torch.set_num_threads(1)
        rng = seeded_runtime(recipe["seed"])
        rng.__enter__()
        workspace = tempfile.TemporaryDirectory(prefix="gw-gemma-lora-")
        work = Path(workspace.name)
        model, targets = attach(model)
        adapter_initial = save_adapter(model, config, work / "initial")
        observations = run(model, targets, examples, tokenizer, recipe, work / "trainer")
        model.eval()
        final_summary = save_adapter(model, config, work / "final")
        if final_summary["tensor_content_id"] == adapter_initial["tensor_content_id"]:
            raise ContractError("optimization did not change parameter content")
        source = work / "prepared.gwsft"
        source.write_bytes(prepared.data)
        files = {"base/config.json": initial / "config.json", "base/model.safetensors": initial / "model.safetensors",
                 "initial/config.json": work / "initial/config.json",
                 "initial/adapter_model.safetensors": work / "initial/adapter_model.safetensors",
                 "final/config.json": work / "final/config.json",
                 "final/adapter_model.safetensors": work / "final/adapter_model.safetensors", "prepared.gwsft": source}
        manifest = {"version": 1, "prepared_build_id": prepared.build_id, "recipe": recipe,
                    "base_model": initial_summary, "initial_adapter": adapter_initial,
                    "final_adapter": final_summary, "observations": observations,
                    "source_authorization": authorization, "upstream_lineage": "unknown",
                    "checkpoint_kind": "gemma_qv_lora", "files": inventory(files)}
        with tempfile.NamedTemporaryFile(mode="w+b", dir=output.parent, prefix=".checkpoint-", delete=False) as stream:
            staged = Path(stream.name)
            completion_id = write_bundle(stream, manifest, files)
            stream.flush(); os.fsync(stream.fileno())
        reloaded = read_checkpoint(staged, gw, tokenizer)
        if reloaded.report["completion_id"] != completion_id:
            raise ContractError("completion reload identity mismatch")
        probe = torch.tensor([examples[0]["input_ids"][:16]], dtype=torch.long, device="cpu")
        with torch.no_grad():
            before = model(input_ids=probe, use_cache=False).logits
            after = reloaded.model(input_ids=probe, use_cache=False).logits
        if not torch.isfinite(before).all() or not torch.isfinite(after).all():
            raise ContractError("checkpoint reload produced nonfinite logits")
        torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-6)
        observed = json.dumps({**observations, "fresh_reload": "passed", "logit_rtol": 1e-5, "logit_atol": 1e-6,
                               "prepared_build_id": prepared.build_id, "base_model": initial_summary,
                               "final_adapter": final_summary, "recipe_id": identity(recipe)})
        if recipe["training_source_sha256"] != training_source_identity() or recipe["preparation_source_sha256"] != source_identity():
            raise ContractError("producer source changed during execution")
        # This is the commit point. After it, errors report the retained identity and never
        # unlink the destination: another actor may already have replaced that pathname.
        phase = "publication"
        os.link(staged, output)
        published = True
        phase = "directory_open"
        directory = os.open(output.parent, os.O_RDONLY)
        phase = "directory_sync"
        os.fsync(directory)
        directory_synced = True
    except BaseException as error:
        errors.append((phase, error))
    finally:
        # Attempt every cleanup independently. None can mask a prior committed publication.
        cleanups = [("runtime_cleanup", lambda: torch.set_num_threads(old_threads))]
        if rng is not None:
            cleanups.insert(0, ("rng_cleanup", lambda: rng.__exit__(None, None, None)))
        if workspace is not None:
            cleanups.insert(0, ("workspace_cleanup", workspace.cleanup))
        if staged is not None:
            cleanups.insert(0, ("staged_file_cleanup", lambda: staged.unlink(missing_ok=True)))
        if directory is not None:
            cleanups.insert(0, ("directory_close", lambda: os.close(directory)))
        for cleanup_phase, cleanup in cleanups:
            try:
                cleanup()
            except BaseException as error:
                errors.append((cleanup_phase, error))
    if errors:
        if published:
            raise PublishedCheckpointError(completion_id, prepared.build_id, output, directory_synced, errors) from errors[0][1]
        raise errors[0][1]
    result = object.__new__(ObservedCompletion)
    object.__setattr__(result, "_ObservedCompletion__state", (completion_id, observed, reloaded))
    _LIVE_COMPLETIONS[result] = (result._ObservedCompletion__state, Path(output).resolve(),
                                 recipe["training_source_sha256"], recipe["preparation_source_sha256"])
    return result


def _comparison_source(completed):
    """Resolve only an actual live successful producer receipt and its original publication."""
    if type(completed) is not ObservedCompletion or completed not in _LIVE_COMPLETIONS:
        raise ContractError("comparison requires the live owned Gemma LoRA completion")
    state, path, training_source, preparation_source = _LIVE_COMPLETIONS[completed]
    if (getattr(completed, "_ObservedCompletion__state", None) is not state
            or training_source != training_source_identity() or preparation_source != source_identity()):
        raise ContractError("live completion state or producer source changed")
    return path, state[0], strict_json(state[1]), state[2].report


def train(prepared_path: Path, release_directory: Path, gw: Path, output: Path, **options) -> ObservedCompletion:
    """Train a CPU LoRA adapter from the exact approved locally supplied release and whole input.

    No acquisition occurs. The release must already exist locally with the exact pinned files.
    Arbitrary local weights, loaded objects and saved eligibility assertions are rejected.
    """
    completed = None
    try:
        with load_approved_release(release_directory) as (loaded, tokenizer):
            with regular(prepared_path) as source:
                data = source.read(MAX_BYTES + 1)
            prepared = verify_prepared(data, gw, tokenizer)
            completed = _run_loaded(prepared, tokenizer, loaded, gw, output, **options)
    except BaseException as error:
        recorded = publication_cause(error)
        if recorded is not None:
            if recorded is error:
                raise
            raise recorded.with_cleanup_error("release_cleanup", error) from error
        if completed is not None:
            raise PublishedCheckpointError(completed.completion_id, completed.observed["prepared_build_id"],
                output, True, [("release_cleanup", error)]) from error
        raise
    return completed
