"""One synchronous, single-process TRL callback for fresh numeric completions."""
import math
from pathlib import Path
import threading
import uuid
import weakref

from .artifact import ContractError
from .reward_artifact import (
    VerifiedNumericCorpus, json_bytes, report_json, reward_identity,
    reward_rows, run_reward_command, same_typed, snapshot_identity,
)
from .tokenizer import check_dependencies, tokenizer_manifest, tokenizer_policy, validate_tokenizer


class NumericRewardCallback:
    """Validate an entire Rust-evaluated batch before returning any numeric rewards.

    Construct with a captured verified corpus, then call ``bind_trainer`` on the actual
    GRPOTrainer before dispatch. The dataset must be produced by ``reward_rows``.
    Unknown factual results and infrastructure failures abort the complete batch.
    """

    def __init__(self, corpus: VerifiedNumericCorpus, tokenizer, gw: Path, *, timeout_seconds: float = 30.0):
        if type(corpus) is not VerifiedNumericCorpus:
            raise ContractError("a verified captured numeric corpus is required")
        if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ContractError("reward timeout must be positive and finite")
        check_dependencies()
        self._artifact = corpus.artifact
        self._rows = {row["gw_reward"]["task_id"]: row for row in reward_rows(corpus, tokenizer)}
        self._tokenizer = tokenizer
        self._gw = Path(gw)
        self._timeout = timeout_seconds
        self._trainer = None
        self._run_id = uuid.uuid4().hex
        self._sequence = 0
        self._lock = threading.Lock()
        self._last_report = None
        policy = tokenizer_policy()
        self._policy = {
            "version": 1,
            "tokenizer_id": reward_identity("gw-numeric-reward-tokenizer-v1", [tokenizer_manifest(), policy]),
            "decode": "plain_assistant_skip_special_no_cleanup_v1",
            "vocab_size": policy["vocab_size"], "eos_token_id": tokenizer.eos_token_id,
            "control_tokens": sorted(
                ({"id": token["id"], "literal": token["content"]} for token in policy["added_tokens"]),
                key=lambda token: token["id"],
            ),
        }
        self._controls = {token["id"] for token in self._policy["control_tokens"]}
        self._validate_tokenizer()

    @property
    def last_report(self) -> dict | None:
        """Detached receipt for the last successful batch; cleared at every callback entry."""
        return None if self._last_report is None else report_json(self._last_report)

    def _validate_tokenizer(self):
        validate_tokenizer(self._tokenizer)
        if self._tokenizer.clean_up_tokenization_spaces is not False:
            raise ContractError("numeric reward decoding requires cleanup disabled")

    def bind_trainer(self, trainer) -> None:
        """Bind the actual pinned trainer and reject unsupported execution configurations."""
        from trl import GRPOTrainer
        if type(trainer) is not GRPOTrainer:
            raise ContractError("numeric reward callback requires the pinned GRPOTrainer")
        if self._trainer is not None and self._trainer() is not trainer:
            raise ContractError("numeric reward callback is already bound to another trainer")
        self._check_trainer(trainer)
        self._trainer = weakref.ref(trainer)

    def _check_trainer(self, trainer):
        from accelerate.utils import DistributedType
        accelerator = trainer.accelerator
        if accelerator.num_processes != 1 or accelerator.process_index != 0 or accelerator.distributed_type != DistributedType.NO or trainer.args.world_size != 1:
            raise ContractError("numeric reward callback supports one local process only")
        if trainer.reward_funcs != [self] or trainer.processing_class is not self._tokenizer:
            raise ContractError("numeric reward callback requires its pinned tokenizer and one reward function")
        if trainer.args.remove_unused_columns is not False:
            raise ContractError("numeric reward metadata requires remove_unused_columns=False")
        if trainer.beta != 0 or trainer.use_vllm or trainer.use_transformers_continuous_batching:
            raise ContractError("numeric reward trainer configuration is not qualified")
        if trainer.tools or trainer.environments is not None or trainer.rollout_func is not None:
            raise ContractError("numeric reward tools, environments, and custom rollouts are unsupported")
        if not same_typed(trainer.chat_template_kwargs, {"enable_thinking": False}):
            raise ContractError("numeric reward prompts require explicitly nonthinking rendering")
        if type(trainer.mask_truncated_completions) is not bool or not same_typed(trainer.mask_truncated_completions, trainer.args.mask_truncated_completions):
            raise ContractError("numeric reward truncation setting differs from the trainer configuration")
        self._validate_tokenizer()

    def __call__(self, prompts, completions, completion_ids, gw_reward, **kwargs) -> list[float]:
        """Allocate fresh attempts even for failed calls, then return one fully validated batch."""
        if not self._lock.acquire(blocking=False):
            raise ContractError("concurrent numeric reward callbacks are unsupported")
        try:
            sequence = self._sequence
            self._sequence += 1
            self._last_report = None
            if self._trainer is None or (trainer := self._trainer()) is None:
                raise ContractError("numeric reward callback must be bound to its live trainer")
            self._check_trainer(trainer)
            if set(kwargs) - {"trainer_state", "log_extra", "log_metric"}:
                raise ContractError("unexpected numeric reward batch metadata")
            values = (prompts, completions, completion_ids, gw_reward)
            if any(type(value) is not list for value in values) or not prompts or any(len(value) != len(prompts) for value in values):
                raise ContractError("numeric reward input cardinality mismatch")
            items = [
                self._item(prompt, completion, ids, metadata, sequence, position)
                for position, (prompt, completion, ids, metadata) in enumerate(zip(*values, strict=True))
            ]
            request = {
                "version": 1, "artifact": self._artifact, "completion_policy": self._policy,
                "mask_truncated_completions": trainer.mask_truncated_completions, "items": items,
            }
            data = json_bytes(request)
            output = run_reward_command(self._gw, "evaluate", data, self._timeout)
            rewards = self._validate_report(report_json(output), request, data)
            self._last_report = output
            return rewards
        finally:
            self._lock.release()

    def _item(self, prompt, completion, ids, metadata, sequence, position):
        if type(metadata) is not dict or type(metadata.get("task_id")) is not str:
            raise ContractError("invalid numeric reward task metadata")
        row = self._rows.get(metadata["task_id"])
        if row is None or not same_typed(metadata, row["gw_reward"]) or not same_typed(prompt, row["prompt"]):
            raise ContractError("numeric reward task/prompt binding mismatch")
        if type(completion) is not list or len(completion) != 1 or type(completion[0]) is not dict or set(completion[0]) != {"role", "content"} or completion[0]["role"] != "assistant" or type(completion[0]["content"]) is not str:
            raise ContractError("numeric reward requires one plain assistant completion")
        if type(ids) is not list or any(type(token) is not int or token < 0 or token >= self._policy["vocab_size"] for token in ids):
            raise ContractError("invalid numeric reward completion token IDs")
        eos = self._policy["eos_token_id"]
        if any(token in self._controls and not (token == eos and position + 1 == len(ids)) for position, token in enumerate(ids)):
            raise ContractError("numeric reward completion contains unsupported control IDs")
        text = completion[0]["content"]
        if any(token["literal"] in text for token in self._policy["control_tokens"]):
            raise ContractError("numeric reward completion contains unsupported control syntax")
        decoded = self._tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if decoded != text:
            raise ContractError("numeric reward token/text binding mismatch")
        evidence = {"token_ids": ids.copy(), "text": text}
        binding = {
            **metadata,
            "attempt": {"callback_run_id": self._run_id, "batch_sequence": sequence, "position": position},
            "completion_digest": reward_identity("gw-numeric-reward-completion-v1", [self._policy, evidence]),
        }
        return {"binding": binding, "completion": evidence}

    def _validate_report(self, report, request, data):
        if set(report) != {"report_version", "request", "completion_policy_id", "mask_truncated_completions", "results"} or not same_typed(report["report_version"], 1):
            raise ContractError("unsupported numeric reward report")
        if not same_typed(report["request"], snapshot_identity(data)) or report["completion_policy_id"] != reward_identity("gw-numeric-reward-policy-v1", self._policy) or not same_typed(report["mask_truncated_completions"], request["mask_truncated_completions"]):
            raise ContractError("numeric reward report request/policy binding mismatch")
        if type(report["results"]) is not list or len(report["results"]) != len(request["items"]):
            raise ContractError("numeric reward response cardinality mismatch")
        rewards = []
        for result, item in zip(report["results"], request["items"], strict=True):
            if type(result) is not dict or set(result) != {"binding", "outcome", "reward", "termination"} or not same_typed(result["binding"], item["binding"]):
                raise ContractError("numeric reward result binding mismatch")
            ids = item["completion"]["token_ids"]
            termination = "observed_eos" if ids and ids[-1] == self._policy["eos_token_id"] else "unknown"
            if result["termination"] != termination:
                raise ContractError("numeric reward termination evidence mismatch")
            expected = {"pass": 1.0, "fail": 0.0}.get(result["outcome"]) if type(result["outcome"]) is str else None
            reward = result["reward"]
            if expected is None or type(reward) not in (int, float) or reward != expected or not math.isfinite(reward):
                raise ContractError("numeric reward unavailable or invalid; the complete batch is aborted")
            rewards.append(float(reward))
        return rewards
