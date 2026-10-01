"""Actual pinned GRPO reward dispatch with synthetic completions and no model computation."""
import copy
import json
from pathlib import Path
import subprocess

import pytest

from ghostwriter_trl.artifact import ContractError
from ghostwriter_trl.reward_artifact import VerifiedNumericCorpus, json_bytes, read_numeric_corpus, reward_rows, verify_numeric_corpus
from ghostwriter_trl.rewards import NumericRewardCallback

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def corpus(gw, tmp_path_factory):
    path = tmp_path_factory.mktemp("reward-corpus") / "corpus.json"
    result = subprocess.run([str(gw), "reward", "export", "--tasks", str(REPO / "examples/reviewed-numeric-tasks.json")], capture_output=True, check=True)
    path.write_bytes(result.stdout)
    return read_numeric_corpus(path, gw)


@pytest.fixture(scope="module")
def dispatch(corpus, tokenizer, gw, tmp_path_factory):
    import torch
    from datasets import Dataset
    from transformers import GPT2Config, GPT2LMHeadModel, Trainer
    from trl import GRPOConfig, GRPOTrainer
    calls = []

    def forbidden(*args, **kwargs):
        calls.append("forbidden model/training/optimizer operation")
        raise AssertionError(calls[-1])

    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(tokenizer), n_positions=512, n_embd=16, n_layer=1, n_head=2,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
    ))
    guard = pytest.MonkeyPatch()
    guard.setattr(model, "forward", forbidden)
    guard.setattr(model, "generate", forbidden)
    for method in ("train", "training_step", "_prepare_inputs", "_generate_and_score_completions", "predict", "evaluate", "create_optimizer", "create_optimizer_and_scheduler"):
        guard.setattr(GRPOTrainer, method, forbidden)
    guard.setattr(Trainer, "create_optimizer", forbidden)
    guard.setattr(Trainer, "create_optimizer_and_scheduler", forbidden)
    guard.setattr(torch.optim.Optimizer, "__init__", forbidden)
    callback = NumericRewardCallback(corpus, tokenizer, gw)
    settings = GRPOConfig(
        output_dir=str(tmp_path_factory.mktemp("grpo-no-execution")), use_cpu=True,
        bf16=False, fp16=False, beta=0.0, use_vllm=False,
        use_transformers_continuous_batching=False, report_to="none", push_to_hub=False,
        gradient_checkpointing=False, remove_unused_columns=False,
        chat_template_kwargs={"enable_thinking": False}, per_device_train_batch_size=4,
        num_generations=2, steps_per_generation=1, gradient_accumulation_steps=1,
        dataloader_num_workers=0, dataloader_pin_memory=False, shuffle_dataset=False,
        mask_truncated_completions=True, max_completion_length=8, seed=0, data_seed=0,
    )
    rows = reward_rows(corpus, tokenizer)
    trainer = GRPOTrainer(model=model, args=settings, processing_class=tokenizer,
                          reward_funcs=[callback], train_dataset=Dataset.from_list(rows))
    callback.bind_trainer(trainer)
    inputs = next(iter(trainer.get_train_dataloader()))
    assert [row["gw_reward"]["task_id"] for row in inputs] == ["addition-001", "addition-001", "division-001", "division-001"]
    assert all(set(row) == {"prompt", "gw_reward"} for row in inputs)
    assert trainer.accelerator.num_processes == 1
    assert trainer.args.device.type == next(model.parameters()).device.type == "cpu"
    yield trainer, callback, inputs, calls
    guard.undo()
    assert calls == []


def completion_batch(tokenizer, texts=("FINAL: 5", "FINAL: 4", "3", "4"), eos=True):
    ids = [tokenizer.encode(text, add_special_tokens=False) + ([tokenizer.eos_token_id] if eos else []) for text in texts]
    # Exact Transformers 4.56 path used by the pinned trainer; no response parser.
    decoded = tokenizer.batch_decode(ids, skip_special_tokens=True)
    assert decoded == list(texts)
    return [[{"role": "assistant", "content": text}] for text in decoded], ids


def calculate(dispatch, tokenizer, *, texts=("FINAL: 5", "FINAL: 4", "3", "4"), eos=True, inputs=None):
    trainer, _, original, _ = dispatch
    selected = copy.deepcopy(original if inputs is None else inputs)
    completions, ids = completion_batch(tokenizer, texts, eos)
    return trainer._calculate_rewards(selected, [row["prompt"] for row in selected], completions, ids)


def forbid_reward_assignment_and_gather(monkeypatch):
    import trl.trainer.grpo_trainer as module
    def forbidden(*args, **kwargs):
        raise AssertionError("failed batch reached numeric tensor assignment or reward gather")
    # The real dispatcher allocates torch.zeros before calling the callback. No reward values
    # may reach the subsequent torch.tensor assignment, and no failed batch may reach gather.
    monkeypatch.setattr(module.torch, "tensor", forbidden)
    monkeypatch.setattr(module, "gather", forbidden)


def test_real_dispatch_finite_cpu_float32_order_and_fresh_attempts(dispatch, tokenizer):
    import torch
    _, callback, _, calls = dispatch
    tensor = calculate(dispatch, tokenizer)
    assert tensor.dtype == torch.float32 and tensor.device.type == "cpu" and tuple(tensor.shape) == (4, 1)
    assert torch.isfinite(tensor).all() and tensor.tolist() == [[1.0], [0.0], [1.0], [0.0]]
    first = callback.last_report
    assert [row["termination"] for row in first["results"]] == ["observed_eos"] * 4
    assert first["mask_truncated_completions"] is True
    calculate(dispatch, tokenizer)
    second = callback.last_report
    for position, (a, b) in enumerate(zip(first["results"], second["results"], strict=True)):
        assert b["binding"]["attempt"]["position"] == position
        assert b["binding"]["attempt"]["batch_sequence"] == a["binding"]["attempt"]["batch_sequence"] + 1
        assert a["binding"]["attempt"]["callback_run_id"] == b["binding"]["attempt"]["callback_run_id"]
        assert a["binding"]["completion_digest"] == b["binding"]["completion_digest"]
    first["results"].clear()
    assert len(callback.last_report["results"]) == 4 and calls == []


@pytest.mark.parametrize("mask", [False, True])
def test_unknown_termination_preserves_numeric_results_and_mask(dispatch, tokenizer, monkeypatch, mask):
    trainer, callback, _, _ = dispatch
    monkeypatch.setattr(trainer, "mask_truncated_completions", mask)
    monkeypatch.setattr(trainer.args, "mask_truncated_completions", mask)
    completions, ids = completion_batch(tokenizer, eos=False)
    monkeypatch.setattr(trainer.args, "max_completion_length", len(ids[0]))
    monkeypatch.setattr(trainer, "max_completion_length", len(ids[0]))
    result = calculate(dispatch, tokenizer, eos=False)
    assert result.tolist() == [[1.0], [0.0], [1.0], [0.0]]
    assert callback.last_report["mask_truncated_completions"] is mask
    assert [row["termination"] for row in callback.last_report["results"]] == ["unknown"] * 4


@pytest.mark.parametrize("bad", ["", "FINAL: 5\nFINAL: 5", "FINAL: five", "some explanation", "<think>FINAL: 5</think>"])
def test_later_unknown_or_reasoning_aborts_whole_actual_dispatch(dispatch, tokenizer, monkeypatch, bad):
    _, callback, _, _ = dispatch
    calculate(dispatch, tokenizer)
    before = callback.last_report["results"][0]["binding"]["attempt"]["batch_sequence"]
    forbid_reward_assignment_and_gather(monkeypatch)
    with pytest.raises(ContractError):
        calculate(dispatch, tokenizer, texts=("FINAL: 5", bad, "3", "4"))
    assert callback.last_report is None
    assert callback._sequence == before + 2


@pytest.mark.parametrize("control", list(range(151643, 151669)))
def test_hidden_or_nonterminal_reserved_ids_rejected(dispatch, tokenizer, monkeypatch, control):
    trainer, callback, inputs, _ = dispatch
    completions, ids = completion_batch(tokenizer)
    ids[1] = [control] + ids[1]
    completions[1][0]["content"] = tokenizer.decode(ids[1], skip_special_tokens=True)
    forbid_reward_assignment_and_gather(monkeypatch)
    with pytest.raises(ContractError, match="control"):
        trainer._calculate_rewards(inputs, [row["prompt"] for row in inputs], completions, ids)
    assert callback.last_report is None


@pytest.mark.parametrize("bad", [True, -1, 151669, 2**40, "7", 7.0])
def test_invalid_token_ids_fail_before_decoding(dispatch, tokenizer, monkeypatch, bad):
    trainer, _, inputs, _ = dispatch
    completions, ids = completion_batch(tokenizer)
    ids[1].append(bad)
    forbid_reward_assignment_and_gather(monkeypatch)
    with pytest.raises(ContractError, match="token IDs"):
        trainer._calculate_rewards(inputs, [row["prompt"] for row in inputs], completions, ids)


@pytest.mark.parametrize("mutation", ["reasoning", "tool", "media", "wrong_role", "extra_turn", "stale_text", "raw_control", "prompt", "metadata", "input_length"])
def test_unsupported_or_stale_inputs_abort(dispatch, tokenizer, monkeypatch, mutation):
    trainer, _, original, _ = dispatch
    inputs = copy.deepcopy(original)
    prompts = [row["prompt"] for row in inputs]
    completions, ids = completion_batch(tokenizer)
    target = completions[1][0]
    if mutation == "reasoning": target["reasoning"] = "FINAL: 5"
    elif mutation == "tool": target["tool_calls"] = []
    elif mutation == "media": target["content"] = [{"type": "text", "text": "FINAL: 5"}]
    elif mutation == "wrong_role": target["role"] = "user"
    elif mutation == "extra_turn": completions[1].append({"role": "assistant", "content": "FINAL: 5"})
    elif mutation == "stale_text": target["content"] = "FINAL: 5"
    elif mutation == "raw_control": target["content"] = "<tool_call>FINAL: 5</tool_call>"
    elif mutation == "prompt": prompts[1][0]["content"] += " FINAL: 5"
    elif mutation == "metadata": inputs[1]["gw_reward"]["task_id"] = "division-001"
    elif mutation == "input_length": ids.pop()
    forbid_reward_assignment_and_gather(monkeypatch)
    with pytest.raises(ContractError):
        trainer._calculate_rewards(inputs, prompts, completions, ids)


@pytest.mark.parametrize("mutation", ["partial", "short", "extra", "reorder", "stale_digest", "stale_attempt", "bool_position", "unknown", "missing_reward", "bool_reward", "nan_reward", "bad_outcome", "mask", "policy", "request", "extra_field", "duplicate_field", "termination"])
def test_later_invalid_evaluator_result_never_reaches_assignment_or_gather(dispatch, tokenizer, monkeypatch, mutation):
    import ghostwriter_trl.rewards as module
    real = module.run_reward_command
    def corrupt(*args):
        output = real(*args)
        if mutation == "partial": return output[:-8]
        if mutation == "duplicate_field": return output.replace(b'"report_version":1', b'"report_version":1,"report_version":1')
        report = json.loads(output)
        result = report["results"][1]
        if mutation == "short": report["results"].pop()
        elif mutation == "extra": report["results"].append(copy.deepcopy(result))
        elif mutation == "reorder": report["results"].reverse()
        elif mutation == "stale_digest": result["binding"]["completion_digest"] = "0" * 64
        elif mutation == "stale_attempt": result["binding"]["attempt"]["batch_sequence"] += 1
        elif mutation == "bool_position": result["binding"]["attempt"]["position"] = True
        elif mutation == "unknown": result.update(outcome="unknown", reward=None)
        elif mutation == "missing_reward": del result["reward"]
        elif mutation == "bool_reward": result["reward"] = False
        elif mutation == "nan_reward": result["reward"] = float("inf")
        elif mutation == "bad_outcome": result["outcome"] = "passed"
        elif mutation == "mask": report["mask_truncated_completions"] = 1
        elif mutation == "policy": report["completion_policy_id"] = "0" * 64
        elif mutation == "request": report["request"]["byte_length"] += 1
        elif mutation == "extra_field": result["teacher_evidence"] = True
        elif mutation == "termination": result["termination"] = "unknown"
        return json.dumps(report).encode()
    monkeypatch.setattr(module, "run_reward_command", corrupt)
    forbid_reward_assignment_and_gather(monkeypatch)
    with pytest.raises(ContractError):
        calculate(dispatch, tokenizer)
    assert dispatch[1].last_report is None


@pytest.mark.parametrize("mode", ["timeout", "nonzero", "missing_executable"])
def test_evaluator_infrastructure_failure_aborts_actual_dispatch(dispatch, tokenizer, monkeypatch, mode):
    import ghostwriter_trl.reward_artifact as module
    def fail(*args, **kwargs):
        if mode == "timeout": raise subprocess.TimeoutExpired("gw", 0.01)
        if mode == "missing_executable": raise OSError("synthetic unavailable evaluator")
        return subprocess.CompletedProcess([], 1, b"", b"private diagnostics")
    monkeypatch.setattr(module.subprocess, "run", fail)
    forbid_reward_assignment_and_gather(monkeypatch)
    with pytest.raises(ContractError):
        calculate(dispatch, tokenizer)


def test_shuffled_aligned_batch_uses_each_actual_task(dispatch, tokenizer):
    inputs = [copy.deepcopy(dispatch[2][i]) for i in (2, 0, 3, 1)]
    assert calculate(dispatch, tokenizer, inputs=inputs, texts=("3", "FINAL: 4", "4", "FINAL: 5")).tolist() == [[1.0], [0.0], [0.0], [1.0]]


def test_oracle_and_provenance_sentinels_never_enter_trainer_prompt(corpus, tokenizer, gw, tmp_path):
    document = json.loads((REPO / "examples/reviewed-numeric-tasks.json").read_text())
    task = document["tasks"][0]
    task["verification"]["oracle"]["expected"] = "9123456789"
    task["source"]["citation"] = "SOURCE_SENTINEL"
    task["rights"]["evidence"] = ["RIGHTS_SENTINEL"]
    task["observations"]["qc"]["evidence"] = ["QC_SENTINEL"]
    source = tmp_path / "tasks.json"
    source.write_text(json.dumps(document))
    output = subprocess.run([str(gw), "reward", "export", "--tasks", str(source)], capture_output=True, check=True).stdout
    selected = verify_numeric_corpus(output, gw)
    rows = reward_rows(selected, tokenizer)
    assert len(rows) == 2
    for row in rows:
        assert set(row) == {"prompt", "gw_reward"}
        assert set(row["gw_reward"]) == {"artifact_id", "reward_contract_id", "task_id", "semantic_task_digest"}
        rendered = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert all(sentinel not in rendered for sentinel in ("9123456789", "SOURCE_SENTINEL", "RIGHTS_SENTINEL", "QC_SENTINEL"))
    detached = selected.artifact
    detached["tasks"].clear()
    assert len(selected.artifact["tasks"]) == 2
    with pytest.raises(AttributeError): selected.data = b"changed"


def test_answer_in_prompt_cannot_rescue_empty_completion(dispatch, tokenizer, gw, tmp_path, monkeypatch):
    document = json.loads((REPO / "examples/reviewed-numeric-tasks.json").read_text())
    document["tasks"][0]["prompt"]["content"] = "Prompt-only answer sentinel: FINAL: 5"
    source = tmp_path / "prompt-only.json"
    source.write_text(json.dumps(document))
    output = subprocess.run([str(gw), "reward", "export", "--tasks", str(source)], capture_output=True, check=True).stdout
    corpus = verify_numeric_corpus(output, gw)
    callback = NumericRewardCallback(corpus, tokenizer, gw)
    trainer = dispatch[0]
    monkeypatch.setattr(trainer, "reward_funcs", [callback])
    callback.bind_trainer(trainer)
    rows = reward_rows(corpus, tokenizer)
    inputs = [copy.deepcopy(rows[i]) for i in (0, 0, 1, 1)]
    completions, ids = completion_batch(tokenizer, ("FINAL: 5", "", "3", "4"))
    forbid_reward_assignment_and_gather(monkeypatch)
    with pytest.raises(ContractError, match="unavailable"):
        trainer._calculate_rewards(inputs, [row["prompt"] for row in inputs], completions, ids)


@pytest.mark.parametrize("change", ["processes", "mask", "thinking", "cleanup", "columns", "beta"])
def test_effective_trainer_changes_are_rejected(dispatch, tokenizer, monkeypatch, change):
    trainer, _, _, _ = dispatch
    if change == "processes": monkeypatch.setattr(trainer.accelerator.state, "num_processes", 2)
    elif change == "mask": monkeypatch.setattr(trainer, "mask_truncated_completions", False)
    elif change == "thinking": monkeypatch.setattr(trainer, "chat_template_kwargs", {"enable_thinking": True})
    elif change == "cleanup": monkeypatch.setattr(tokenizer, "clean_up_tokenization_spaces", True)
    elif change == "columns": monkeypatch.setattr(trainer.args, "remove_unused_columns", True)
    elif change == "beta": monkeypatch.setattr(trainer, "beta", 0.1)
    forbid_reward_assignment_and_gather(monkeypatch)
    with pytest.raises(ContractError): calculate(dispatch, tokenizer)


def test_callback_namespace_is_fresh_and_requires_real_trainer(corpus, tokenizer, gw):
    first, second = (NumericRewardCallback(corpus, tokenizer, gw) for _ in range(2))
    assert first._run_id != second._run_id
    with pytest.raises(ContractError): first.bind_trainer(object())
    with pytest.raises(ContractError): first([], [], [], [])


@pytest.mark.parametrize("mutation", ["missing_oracle", "unknown_field", "version", "duplicate_field"])
def test_corpus_wire_rejected_by_rust(corpus, gw, mutation):
    artifact = corpus.artifact
    if mutation == "missing_oracle": del artifact["tasks"][0]["task"]["verification"]["oracle"]
    elif mutation == "unknown_field": artifact["teacher_execution"] = {"passed": True}
    elif mutation == "version": artifact["artifact_version"] = 2
    data = json_bytes(artifact)
    if mutation == "duplicate_field": data = data.replace(b'"artifact_version":1', b'"artifact_version":1,"artifact_version":1')
    with pytest.raises(ContractError): verify_numeric_corpus(data, gw)


@pytest.mark.parametrize("consumer", ["rows", "callback"])
def test_factory_bypass_subclass_is_rejected_before_tokenizer_or_projection(corpus, monkeypatch, gw, consumer):
    import ghostwriter_trl.reward_artifact as module
    class UnverifiedSubclass(VerifiedNumericCorpus):
        def __new__(cls):
            return object.__new__(cls)
        @property
        def artifact(self):
            return corpus.artifact
    class NeverUsedTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError("unverified subclass reached prompt projection")
    def forbidden(*args, **kwargs):
        raise AssertionError("unverified subclass reached tokenizer validation")
    monkeypatch.setattr(module, "validate_tokenizer", forbidden)
    forged = UnverifiedSubclass()
    with pytest.raises(ContractError, match="verified captured"):
        if consumer == "rows": reward_rows(forged, NeverUsedTokenizer())
        else: NumericRewardCallback(forged, NeverUsedTokenizer(), gw)


def test_actual_dispatch_forwards_exact_tolerance_objects_without_float_reparsing(dispatch, tokenizer, gw, tmp_path, monkeypatch):
    import ghostwriter_trl.rewards as module
    document = json.loads((REPO / "examples/reviewed-numeric-tasks.json").read_text())
    document["tasks"][0]["verification"]["numeric"]["tolerance"] = {"absolute": 20 / 13, "relative": -0.0}
    source = tmp_path / "finite-tolerances.json"
    source.write_text(json.dumps(document))
    output = subprocess.run([str(gw), "reward", "export", "--tasks", str(source)], capture_output=True, check=True).stdout
    corpus = verify_numeric_corpus(output, gw)
    tolerance = corpus.artifact["tasks"][0]["task"]["verification"]["numeric"]["tolerance"]
    assert set(tolerance["absolute"]) == {"binary64"}
    assert tolerance["relative"] == {"binary64": "8000000000000000"}
    callback = NumericRewardCallback(corpus, tokenizer, gw)
    trainer = dispatch[0]
    monkeypatch.setattr(trainer, "reward_funcs", [callback])
    callback.bind_trainer(trainer)
    real = module.run_reward_command
    calls = []
    def capture(executable, command, data, timeout):
        request = json.loads(data)
        assert request["artifact"] == corpus.artifact
        calls.append(command)
        return real(executable, command, data, timeout)
    monkeypatch.setattr(module, "run_reward_command", capture)
    rows = reward_rows(corpus, tokenizer)
    inputs = [copy.deepcopy(rows[i]) for i in (0, 0, 1, 1)]
    completions, ids = completion_batch(tokenizer)
    result = trainer._calculate_rewards(inputs, [row["prompt"] for row in inputs], completions, ids)
    assert result.tolist() == [[1.0], [1.0], [1.0], [0.0]]
    assert calls == ["evaluate"]
