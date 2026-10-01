"""Only the selected release and one explicit owned software-qualification architecture."""
from hashlib import sha256
from pathlib import Path

from ..artifact import ContractError, strict_json

ROOT = Path(__file__).parent
RELEASE = "gemma4_e2b_3e22461_student_lora_v1"
FIXTURE = "owned_gemma4_text_fixture_v1"
CONFIG_SHA256 = "1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330"
WEIGHTS_SHA256 = "2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550"
WEIGHTS_BYTES = 10_246_621_918
MAX_CONFIG = 65536


def read_config(path):
    """Check inert data before constructing the one known Transformers model class."""
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_CONFIG + 1)
    if len(raw) > MAX_CONFIG:
        raise ContractError("Gemma configuration exceeds the narrow bound")
    config = strict_json(raw.decode())
    if sha256(raw).hexdigest() == CONFIG_SHA256:
        return config, RELEASE
    # Exact canonical data also rejects bool/int equality and unknown loader properties.
    from ..prepared import _json_bytes
    fixture = strict_json((ROOT / "fixture_config.json").read_text())
    if _json_bytes(config) != _json_bytes(fixture):
        raise ContractError("unsupported Gemma configuration or release identity")
    return config, FIXTURE


def create(config, *, device="cpu"):
    """Construct the official conditional-generation class without automatic resolution."""
    check_lora_dependencies()
    import torch
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration
    cfg = Gemma4Config(**config)
    cfg._attn_implementation = "eager"
    with torch.device(device):
        return Gemma4ForConditionalGeneration(cfg).float()


def canonical_model_config(model):
    """Resolve Transformers' normalized per-layer configuration to one reviewed input."""
    from transformers import Gemma4Config
    actual = model.config.to_dict()
    actual.pop("_name_or_path", None)
    if actual.get("use_cache") is False:
        actual.pop("use_cache")
    if actual.get("architectures") is None:
        actual["architectures"] = ["Gemma4ForConditionalGeneration"]
    for filename in ("fixture_config.json", "release_config.json"):
        candidate, _ = read_config(ROOT / filename)
        expected = Gemma4Config(**candidate).to_dict()
        expected.pop("_name_or_path", None)
        if actual == expected:
            return candidate
    raise ContractError("actual Gemma architecture differs from both reviewed configurations")


def check_lora_dependencies():
    """Require every qualified preparation and PEFT pin before constructing model code."""
    from importlib.metadata import version, PackageNotFoundError
    from ..profiles import GEMMA
    from ..tokenizer import check_dependencies
    expected = strict_json((ROOT / "dependencies.json").read_text())
    try:
        actual = {name: version(name) for name in expected}
    except PackageNotFoundError as error:
        raise ContractError("install all pinned Gemma LoRA dependencies") from error
    if actual != expected:
        raise ContractError("Gemma LoRA dependencies differ from exact qualified pins")
    check_dependencies(GEMMA)
    return actual
