"""Saved inspection, exact tokenizer replay and atomic publication without overwrite or authority."""
from pathlib import Path
import os
import subprocess
import tempfile

from ..artifact import ContractError, strict_json
from ..lora.safe_tensors import regular
from ..prepared import _json_bytes
from .bridge import MAX_BYTES
from .protocol import capture_output, render_prompt


def verify(data, gw, tokenizer):
    """Recompute native bindings/arithmetic and actual token decoding; history remains declared."""
    if len(data) > MAX_BYTES:
        raise ContractError("paired artifact exceeds bound")
    result = subprocess.run([str(gw.resolve(strict=True)), "artifact", "verify-coding-pair", "--stdin"],
                            input=data, capture_output=True, check=False)
    if result.returncode:
        raise ContractError("native paired artifact inspection rejected its bindings or arithmetic")
    report = strict_json(result.stdout.decode())
    artifact = strict_json(data.decode())
    if (report.get("report_version") != 1 or report.get("structural_validation") != "passed"
            or report.get("artifact_id") != artifact.get("artifact_id")
            or any(report.get("historical_" + name) != "declared" for name in ("training", "generation", "execution"))
            or report.get("automatic_promotion") is not False):
        raise ContractError("native paired inspection receipt differs from captured artifact")
    settings = artifact["request"]["recipe"]
    for answer, member in zip(artifact["request"]["rows"],
                              [m for m in artifact["population"]["members"] for _ in range(2)], strict=True):
        messages = ([{"role": "system", "content": settings["system_prompt"]}]
                    if settings["system_prompt"] else []) + [{"role": "user", "content": member["prompt"]}]
        generation = answer["generation"]
        prompt = generation["prompt"] if generation is not None else answer["failed_prompt"]
        if prompt is not None and prompt != render_prompt(tokenizer, messages, settings["max_prompt_tokens"]):
            raise ContractError("saved generation prompt IDs differ from the actual pinned processor")
        if generation is not None and generation["output"] != capture_output(
                tokenizer, prompt["input_ids"], generation["output"]["sequence_ids"], settings["max_new_tokens"]):
            raise ContractError("saved generation output IDs and exact decoded bytes disagree")
    return {**report, "tokenizer_replay": "passed"}


def read(path, gw, tokenizer):
    """Capture a bounded regular saved file once; never recreate producer or candidate eligibility."""
    with regular(Path(path)) as stream:
        data = stream.read(MAX_BYTES + 1)
    return data, verify(data, gw, tokenizer)


class PublishedComparisonError(ContractError):
    """A complete comparison was linked but final durability or cleanup could not be confirmed."""
    def __init__(self, artifact_id, output, synced, errors):
        self.report = {"artifact_id": artifact_id, "output": str(output),
                       "publication": "complete_comparison_linked", "durability": "confirmed" if synced else "unknown",
                       "errors": [{"phase": phase, "kind": type(error).__name__} for phase, error in errors]}
        super().__init__("complete paired artifact retained after publication settlement failure")


def publish(artifact, output, gw, tokenizer):
    """Validate first, then atomically link complete synced bytes; never overwrite or unlink a target."""
    output = Path(output)
    if os.path.lexists(output):
        raise ContractError("paired artifact destination already exists")
    data = _json_bytes(artifact)
    report = verify(data, gw, tokenizer)
    staged = directory = None
    linked = synced = False
    staged_identity = None
    errors = []
    phase = "staging"
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", dir=output.parent, prefix=".coding-pair-", delete=False) as stream:
            staged = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            stat = os.fstat(stream.fileno())
            staged_identity = (stat.st_dev, stat.st_ino)
        phase = "publication"
        os.link(staged, output)
        linked = True
        phase = "directory_open"
        directory = os.open(output.parent, os.O_RDONLY)
        phase = "directory_sync"
        os.fsync(directory)
        synced = True
    except BaseException as error:
        errors.append((phase, error))
        if phase == "publication" and not linked and staged_identity is not None:
            try:
                target = os.stat(output, follow_symlinks=False)
                linked = (target.st_dev, target.st_ino) == staged_identity
            except FileNotFoundError:
                pass
            except BaseException as settlement_error:
                errors.append(("publication_identity", settlement_error))
    finally:
        cleanups = []
        if staged is not None:
            cleanups.append(("staged_file_cleanup", lambda: staged.unlink(missing_ok=True)))
        if directory is not None:
            cleanups.append(("directory_close", lambda: os.close(directory)))
        for phase, cleanup in cleanups:
            try:
                cleanup()
            except BaseException as error:
                errors.append((phase, error))
    if errors:
        if linked:
            raise PublishedComparisonError(artifact["artifact_id"], output, synced, errors) from errors[0][1]
        raise errors[0][1]
    return report
