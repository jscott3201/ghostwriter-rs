"""Fresh owned training observations, separate from saved completion declarations."""
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import tempfile

from ..artifact import ContractError, strict_json
from ..build import identity, source_identity
from ..prepared import VerifiedPrepared, read_prepared
from ..tokenizer import check_dependencies, validate_tokenizer
from .bundle import inventory, read_checkpoint, write_bundle
from .capture import _LoadedModel, load_approved_release
from .execution import run
from .publication import PublishedCheckpointError, publication_cause
from .safe_model import save_model


class ObservedCompletion:
    """Receipt from this actual successful producer call; loading a bundle cannot create it."""
    __slots__ = ("__state",)

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
    root = Path(__file__).parent
    return identity({str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
                     for path in sorted(root.rglob("*")) if path.suffix in {".py", ".json"}})


def _recipe(max_steps, batch_size, accumulation, learning_rate_millionths, max_sequence_length):
    limits = {"max_steps": (max_steps, 1, 32), "batch_size": (batch_size, 1, 8),
              "accumulation": (accumulation, 1, 8), "learning_rate_millionths": (learning_rate_millionths, 1, 10000),
              "max_sequence_length": (max_sequence_length, 2, 2048)}
    if any(type(value) is not int or not lower <= value <= upper for value, lower, upper in limits.values()):
        raise ContractError("full-SFT recipe exceeds supported bounds")
    return {"version": 1, "training_source_sha256": training_source_identity(),
            "preparation_source_sha256": source_identity(), "dependencies": check_dependencies(),
            "runtime": {"python": platform.python_version(), "implementation": platform.python_implementation(),
                        "system": platform.system(), "machine": platform.machine()},
            "device": "cpu", "precision": "float32", "processes": 1, "threads": 1,
            **{key: value for key, (value, _, _) in limits.items()}, "seed": 0,
            "optimizer": "adamw_torch_full_v1", "scheduler": "constant", "sampler": "sequential_epoch_v1",
            "packing": False, "truncation": False}


def _run_loaded(prepared, tokenizer, loaded, gw, output, *, max_steps=1, batch_size=1,
                accumulation=1, learning_rate_millionths=100, max_sequence_length=2048):
    if type(prepared) is not VerifiedPrepared or type(loaded) is not _LoadedModel:
        raise ContractError("training requires actual verified input and a freshly loaded owned model")
    validate_tokenizer(tokenizer)
    recipe = _recipe(max_steps, batch_size, accumulation, learning_rate_millionths, max_sequence_length)
    if os.path.lexists(output):
        raise ContractError("checkpoint destination already exists")
    examples = prepared.examples
    if not examples or any(len(e["input_ids"]) > max_sequence_length for e in examples):
        raise ContractError("training requires nonempty complete sequences within the explicit bound")
    model, config, initial_summary, initial, authorization = loaded._consume()
    if (config["vocab_size"] < len(tokenizer) or config["eos_token_id"] != tokenizer.eos_token_id
            or config.get("pad_token_id") not in (None, tokenizer.pad_token_id)
            or any(len(e["input_ids"]) > config["max_position_embeddings"]
                   or max(e["input_ids"]) >= config["vocab_size"] for e in examples)):
        raise ContractError("initial model does not support the full prepared examples/tokenizer")
    import torch
    old_threads = torch.get_num_threads()
    staged = None
    workspace = None
    directory = None
    published = directory_synced = False
    errors = []
    phase = "training"
    try:
        torch.set_num_threads(1)
        workspace = tempfile.TemporaryDirectory(prefix="gw-full-sft-")
        work = Path(workspace.name)
        observations = run(model, examples, tokenizer, recipe, work / "trainer")
        model.eval()
        _, final_summary = save_model(model, config, work / "checkpoint")
        if final_summary["tensor_content_id"] == initial_summary["tensor_content_id"]:
            raise ContractError("optimization did not change parameter content")
        observations["parameter_content_changed"] = True
        source = work / "prepared.gwsft"
        source.write_bytes(prepared.data)
        files = {"initial/config.json": initial / "config.json", "initial/model.safetensors": initial / "model.safetensors",
                 "checkpoint/config.json": work / "checkpoint/config.json",
                 "checkpoint/model.safetensors": work / "checkpoint/model.safetensors", "prepared.gwsft": source}
        manifest = {"version": 1, "prepared_build_id": prepared.build_id, "recipe": recipe,
                    "initial_model": initial_summary, "checkpoint_model": final_summary, "observations": observations,
                    "source_authorization": authorization, "upstream_lineage": "unknown",
                    "checkpoint_kind": "full_inference", "files": inventory(files)}
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
                               "prepared_build_id": prepared.build_id})
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
    object.__setattr__(result, "_ObservedCompletion__state", (completion_id, observed))
    return result


def train(prepared_path: Path, release_directory: Path, gw: Path, output: Path, **options) -> ObservedCompletion:
    """Train a full CPU model from the exact approved locally supplied release and whole input.

    No acquisition occurs. The release must already exist locally with all nine pinned files.
    Arbitrary local weights, loaded objects and saved eligibility assertions are rejected.
    """
    completed = None
    try:
        with load_approved_release(release_directory) as (loaded, tokenizer):
            prepared = read_prepared(prepared_path, gw, tokenizer)
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
