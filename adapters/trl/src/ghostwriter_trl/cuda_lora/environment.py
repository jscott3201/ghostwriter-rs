"""Additional exact Linux CUDA dependency policy; immutable image admission is separate."""
from hashlib import sha256
from importlib.metadata import distribution, PackageNotFoundError
from pathlib import Path

from ..artifact import ContractError, strict_json
from ..build import identity


def observe():
    """Check extra pins and bind installed distribution metadata/RECORD plus runtime build facts.

    RECORD commitments identify installed-wheel declarations. They are not independent image
    attestation or fresh hashes of every installed library; a later controller owns image admission.
    """
    import torch
    pins = strict_json((Path(__file__).parent / "dependencies.json").read_text())
    records = {}
    try:
        for name, expected in {"torch": "2.8.0", "torchvision": "0.23.0", **pins}.items():
            installed = distribution(name)
            if installed.version != expected:
                raise ContractError("CUDA dependencies differ from the exact additional policy")
            metadata, record = installed.read_text("METADATA"), installed.read_text("RECORD")
            if metadata is None or record is None:
                raise ContractError("CUDA distribution lacks installed metadata/RECORD")
            records[name] = {"version": installed.version, "metadata_sha256": sha256(metadata.encode()).hexdigest(),
                             "record_sha256": sha256(record.encode()).hexdigest()}
    except PackageNotFoundError as error:
        raise ContractError("CUDA policy requires all pinned NVIDIA/Triton packages") from error
    if torch.version.cuda != "12.8" or str(torch.__version__) not in {"2.8.0", "2.8.0+cu128"}:
        raise ContractError("CUDA policy requires the exact Torch 2.8 CUDA 12.8 runtime")
    return pins, {"torch_cuda": torch.version.cuda, "torch_build": str(torch.__version__),
                  "device_name": torch.cuda.get_device_name(0), "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
                  "installed_wheel_records_sha256": identity(records)}
