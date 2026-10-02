"""Fresh independently allocated evaluation models from an actual live producer completion."""
from contextlib import contextmanager

from ..artifact import ContractError
from ..lora.bundle import _captured, _extract, native_report
from ..lora.config import read_config
from ..lora.producer import _comparison_source
from ..lora.safe_model import load_base, reload_adapter
from ..lora.tensors import content_id, live_content
from ..prepared import verify_prepared
from ..tokenizer import tokenizer_manifest
from ..profiles import GEMMA


def measure_models(base, candidate, binding):
    """Recheck the complete independently loaded base and adapter inventories before publication."""
    if (content_id(live_content(base)) != binding["base_model"]["tensor_content_id"]
            or content_id(live_content(candidate)) != binding["base_model"]["tensor_content_id"]
            or content_id(live_content(candidate, adapter=True)) != binding["final_adapter"]["tensor_content_id"]):
        raise ContractError("evaluation model state changed from the captured producer completion")


@contextmanager
def fresh_models(completed, tokenizer, gw):
    """Capture the original measured publication and allocate both models anew, without a hub."""
    path, completion_id, observed, original = _comparison_source(completed)
    with _captured(path) as (root, snapshot):
        report = native_report(snapshot, gw)
        if (report["completion_id"] != completion_id or report["declarations"] != original["declarations"]
                or report["prepared_build_id"] != observed["prepared_build_id"]
                or report["base_model"] != observed["base_model"]
                or report["final_adapter"] != observed["final_adapter"]):
            raise ContractError("captured checkpoint bytes differ from the live completion")
        paths = _extract(snapshot, root, report)
        prepared = verify_prepared(paths["prepared.gwsft"].read_bytes(), gw, tokenizer)
        if prepared.build_id != report["prepared_build_id"]:
            raise ContractError("captured prepared build differs from the live completion")
        config, authorization = read_config(paths["base/config.json"])
        if authorization != original["declarations"]["source_authorization"]:
            raise ContractError("captured model kind differs from the actual producer origin")
        base, measured_base = load_base(config, paths["base/model.safetensors"])
        second_base, second_measured = load_base(config, paths["base/model.safetensors"])
        candidate, measured_adapter = reload_adapter(second_base, config, paths["final/config.json"],
                                                      paths["final/adapter_model.safetensors"])
        if measured_base != report["base_model"] or second_measured != measured_base or measured_adapter != report["final_adapter"]:
            raise ContractError("fresh evaluation models differ from the captured checkpoint inventories")
        manifest = tokenizer_manifest(GEMMA)
        binding = {"completion_id": completion_id, "prepared_build_id": prepared.build_id,
                   "base_model": measured_base, "final_adapter": measured_adapter,
                   "producer_recipe_id": observed["recipe_id"], "source_authorization": authorization,
                   "training_source_sha256": original["declarations"]["recipe"]["training_source_sha256"],
                   "preparation_source_sha256": original["declarations"]["recipe"]["preparation_source_sha256"],
                   "checkpoint_files": original["declarations"]["files"],
                   "publisher_model": manifest["repository"], "publisher_revision": manifest["revision"],
                   "declared_parent": manifest["declared_parent"], "parent_revision": manifest["parent_revision"]}
        measure_models(base, candidate, binding)
        try:
            yield base, candidate, binding, prepared
            measure_models(base, candidate, binding)
            _comparison_source(completed)
        finally:
            del base, candidate, second_base
