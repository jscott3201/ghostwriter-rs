"""The live producer-derived model path; saved checkpoints and caller checksums are ineligible."""
from pathlib import Path
import os

import blake3

from ..artifact import ContractError, strict_json
from ..lora.producer import _comparison_source
from ..prepared import _json_bytes
from .artifact import publish, read
from .authority import fresh_models
from .bridge import capture_population, run_saved
from .generation import comparison_source_identity, cpu_runtime, generate_one, recipe
from .protocol import render_prompt
from .separation import check_separation


class ObservedComparison:
    """Receipt from this completed fresh generation and settled native execution only."""
    __slots__ = ("__state",)

    def __new__(cls, *args, **kwargs):
        raise TypeError("observed comparisons are created only by the live paired controller")

    def __setattr__(self, name, value):
        raise AttributeError("observed comparison receipt is immutable")

    def __init_subclass__(cls, **kwargs):
        raise TypeError("observed comparison receipts cannot be subclassed")

    @property
    def report(self):
        """Detached result identity and actual fresh ownership scope; not an automatic promotion."""
        return strict_json(self.__state)


def model_id(models, side):
    """Match the native model identity over the complete base and optional measured adapter."""
    value = {"base": models["base_model"]["tensor_content_id"],
             "adapter": models["final_adapter"]["tensor_content_id"] if side == "candidate" else None}
    return blake3.blake3(_json_bytes(value), derive_key_context="ghostwriter.coding-comparison-model.v1").hexdigest()


def compare(completed, tokenizer, gw: Path, database: Path, registration: str, output: Path, *,
            split="test", max_new_tokens=128, max_prompt_tokens=1024, system_prompt=""):
    """Compare a fresh approved base and freshly reloaded live producer-derived LoRA candidate.

    The complete registered held-out split is captured once by Rust. Only public prompts and compact accepted Train bindings cross
    into this process. No directory, saved receipt or source-authorization string replaces the
    live completion. The saved paired artifact always retains declared historical provenance.
    """
    _comparison_source(completed)
    if os.path.lexists(output):
        raise ContractError("paired artifact destination already exists")
    source = comparison_source_identity()
    with capture_population(gw, database, registration, split) as bridge:
        with cpu_runtime(), fresh_models(completed, tokenizer, gw) as (base, candidate, models, prepared):
            settings = recipe(tokenizer, max_new_tokens, max_prompt_tokens, system_prompt)
            separation = check_separation(prepared, bridge.population, tokenizer, settings, models)
            rows = []
            for member in bridge.population["members"]:
                for side, model in (("base", base), ("candidate", candidate)):
                    row = {"side": side, "member_id": member["member_id"], "model_id": model_id(models, side),
                           "generation": None, "failure": None, "failed_prompt": None}
                    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
                    messages += [{"role": "user", "content": member["prompt"]}]
                    try:
                        captured_prompt = render_prompt(tokenizer, messages, max_prompt_tokens)
                    except ContractError:
                        row["failure"] = "prompt_rejected"
                    else:
                        try:
                            row["generation"] = generate_one(model, tokenizer, member["prompt"], settings)
                        except (RuntimeError, ValueError, ContractError):
                            row["failure"] = "generation_error"
                            row["failed_prompt"] = captured_prompt
                    rows.append(row)
            request = {"version": 1, "population_id": bridge.population["population_id"], "models": models,
                       "recipe": settings, "separation": separation, "rows": rows}
        # Release the caller bindings as well as the loader scope before native execution.
        del base, candidate, model
        result = bridge.execute(request)
    if comparison_source_identity() != source:
        raise ContractError("comparison implementation changed during execution")
    _comparison_source(completed)
    report = publish(result, output, gw, tokenizer)
    observed = object.__new__(ObservedComparison)
    object.__setattr__(observed, "_ObservedComparison__state", _json_bytes({
        **report, "fresh_training_completion": completed.completion_id,
        "fresh_generation": "both_independent_models", "fresh_native_execution": "complete_controller_call",
        "software_profile": models["source_authorization"], "automatic_promotion": False}).decode())
    return observed


def replay(path, tokenizer, gw, database, output):
    """Verify saved bytes and freshly execute their modules; never load or regenerate a model."""
    data, _ = read(path, gw, tokenizer)
    result = run_saved(gw, database, data)
    return publish(result, output, gw, tokenizer)


def train_and_compare(prepared_path, release_directory, gw, database, registration, checkpoint_output, output, *,
                      max_steps=2, **generation):
    """Settle approved training and its capture before consuming the live receipt into a pair."""
    from ..lora.capture import load_approved_release
    from ..lora.producer import _run_loaded
    from ..prepared import read_prepared
    from ..training.publication import PublishedCheckpointError, publication_cause
    if os.path.lexists(output):
        raise ContractError("paired artifact destination already exists")
    completed = None
    try:
        with load_approved_release(release_directory) as (loaded, tokenizer):
            prepared = read_prepared(prepared_path, gw, tokenizer)
            completed = _run_loaded(prepared, tokenizer, loaded, gw, checkpoint_output, max_steps=max_steps)
    except BaseException as error:
        recorded = publication_cause(error)
        if recorded is not None:
            if recorded is error:
                raise
            raise recorded.with_cleanup_error("release_cleanup", error) from error
        if completed is not None:
            raise PublishedCheckpointError(completed.completion_id, completed.observed["prepared_build_id"],
                checkpoint_output, True, [("release_cleanup", error)]) from error
        raise
    return compare(completed, tokenizer, gw, database, registration, output, **generation)
