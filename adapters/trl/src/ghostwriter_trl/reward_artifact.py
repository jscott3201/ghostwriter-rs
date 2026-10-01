"""Captured, Rust-verified numeric corpora and prompt-only trainer rows."""
import json
import math
from pathlib import Path
import subprocess

import blake3

from .artifact import ContractError, strict_json
from .tokenizer import source_control_literals, validate_tokenizer


def json_bytes(value) -> bytes:
    """Canonical UTF-8 JSON for the integer/string-only completion binding contract."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError) as error:
        raise ContractError("reward data is not valid finite UTF-8 JSON") from error


def reward_identity(domain: str, value) -> str:
    """Match the Rust domain-separated binding encoding; this performs no numeric evaluation."""
    return blake3.blake3(domain.encode("ascii") + b"\0" + json_bytes(value)).hexdigest()


def same_typed(actual, expected) -> bool:
    """Keep bool/int and missing/null distinct while comparing an exact wire binding."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(same_typed(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(same_typed(a, e) for a, e in zip(actual, expected, strict=True))
    return actual == expected


def snapshot_identity(data: bytes) -> dict:
    """Bind the complete captured input, including whitespace and fresh attempt identifiers."""
    return {"byte_length": len(data), "blake3": blake3.blake3(data).hexdigest()}


def run_reward_command(gw: Path, command: str, data: bytes, timeout_seconds: float) -> bytes:
    """Run one local evaluator process; timeout, failure, or partial output is never a reward."""
    if type(data) is not bytes or type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ContractError("reward input must be immutable bytes and timeout must be positive and finite")
    try:
        result = subprocess.run(
            [str(gw.resolve(strict=True)), "reward", command, "--stdin"],
            input=data, capture_output=True, check=False, timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise ContractError("numeric reward evaluator timed out; the complete batch is aborted") from error
    except OSError as error:
        raise ContractError("numeric reward evaluator could not be started") from error
    if result.returncode != 0:
        raise ContractError("numeric reward evaluator failed; the complete batch is aborted")
    return result.stdout


def report_json(data: bytes) -> dict:
    """Require one complete strict UTF-8 JSON object from the evaluator."""
    try:
        value = strict_json(data.decode("utf-8"))
    except UnicodeError as error:
        raise ContractError("invalid numeric reward report encoding") from error
    if type(value) is not dict:
        raise ContractError("numeric reward report must be an object")
    return value


class VerifiedNumericCorpus:
    """Immutable captured corpus and receipt, constructed only by ``verify_numeric_corpus``."""
    __slots__ = ("__state",)

    def __new__(cls, *args, **kwargs):
        raise TypeError("use verify_numeric_corpus or read_numeric_corpus")

    def __setattr__(self, name, value):
        raise AttributeError("verified numeric corpus is immutable")

    @property
    def data(self) -> bytes:
        """Exactly the bytes verified by Rust, never a reopened path."""
        return self.__state[0]

    @property
    def artifact(self) -> dict:
        """A detached copy of the complete task/oracle/provenance artifact."""
        return report_json(self.data)

    @property
    def report(self) -> dict:
        """A detached copy of the captured-byte verification receipt."""
        return report_json(self.__state[1])


def verify_numeric_corpus(data: bytes, gw: Path, *, timeout_seconds: float = 30.0) -> VerifiedNumericCorpus:
    """Verify one self-contained corpus through the strict Rust boundary without reading keys."""
    output = run_reward_command(gw, "verify", data, timeout_seconds)
    report = report_json(output)
    artifact = report_json(data)
    if set(report) != {"report_version", "snapshot", "artifact_id", "reward_contract_id", "task_count"}:
        raise ContractError("unexpected numeric corpus report fields")
    if type(report["report_version"]) is not int or report["report_version"] != 1:
        raise ContractError("unsupported numeric corpus report version")
    if not same_typed(report["snapshot"], snapshot_identity(data)):
        raise ContractError("numeric corpus snapshot binding mismatch")
    if set(artifact) != {"artifact_version", "artifact_id", "reward_contract", "reward_contract_id", "tasks"}:
        raise ContractError("unexpected numeric corpus fields")
    if type(artifact["artifact_version"]) is not int or artifact["artifact_version"] != 1:
        raise ContractError("unsupported numeric corpus version")
    if type(artifact["tasks"]) is not list or not artifact["tasks"]:
        raise ContractError("numeric corpus has no training tasks")
    for name in ("artifact_id", "reward_contract_id"):
        value = artifact[name]
        if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value) or report[name] != value:
            raise ContractError("numeric corpus identity mismatch")
    if type(report["task_count"]) is not int or report["task_count"] != len(artifact["tasks"]):
        raise ContractError("numeric corpus task count mismatch")
    corpus = object.__new__(VerifiedNumericCorpus)
    object.__setattr__(corpus, "_VerifiedNumericCorpus__state", (data, output))
    return corpus


def read_numeric_corpus(path: Path, gw: Path, *, timeout_seconds: float = 30.0) -> VerifiedNumericCorpus:
    """Read once and retain exactly the bytes the pure Rust verifier accepted."""
    return verify_numeric_corpus(path.read_bytes(), gw, timeout_seconds=timeout_seconds)


def reward_metadata(artifact: dict, entry: dict) -> dict:
    """References identify the separate reward authority without copying its oracle into prompts."""
    return {
        "artifact_id": artifact["artifact_id"], "reward_contract_id": artifact["reward_contract_id"],
        "task_id": entry["task"]["task_id"], "semantic_task_digest": entry["task_identity"]["digest"],
    }


def reward_rows(corpus: VerifiedNumericCorpus, tokenizer) -> list[dict]:
    """Produce only one typed user prompt plus separate reward references for each unique task."""
    if type(corpus) is not VerifiedNumericCorpus:
        raise ContractError("a verified captured numeric corpus is required")
    validate_tokenizer(tokenizer)
    artifact = corpus.artifact
    rows = []
    for entry in artifact["tasks"]:
        prompt = entry["task"]["prompt"]
        if any(literal in prompt["content"] for literal in source_control_literals()):
            raise ContractError("numeric reward source prompt contains reserved control syntax")
        # Exercise the actual immutable template with the selected explicit nonthinking setting.
        # The trainer receives these messages, not the rendered string or any oracle/provenance.
        tokenizer.apply_chat_template([prompt], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        rows.append({"prompt": [prompt.copy()], "gw_reward": reward_metadata(artifact, entry)})
    return rows
