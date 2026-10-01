"""Rehash adversarial complete builds; native policy checks and full replay remain distinct."""
from copy import deepcopy
import shutil
import subprocess

import pytest

from ghostwriter_trl.artifact import ContractError, read_snapshot
from ghostwriter_trl.prepared import prepare, verify_prepared
from ..test_prepared_integrity import frame, split


@pytest.fixture(scope="module")
def prepared(gw, tokenizer, profile, fixture_dir):
    snapshot = read_snapshot(fixture_dir / "v3-text.parquet", gw)
    data = prepare(snapshot, tokenizer, cot="masked", turns="all_assistant", max_length=2048,
                   profile=profile, enable_thinking=True)
    verify_prepared(data, gw, tokenizer)
    return data


@pytest.mark.parametrize("case", ["unknown_profile", "other_profile", "recipe_version", "template", "file_pin",
                                 "backend", "added_control", "padding", "bos", "eos", "pad", "dependencies",
                                 "thinking", "preserve_thinking", "generation_prompt", "missing_controls",
                                 "source", "source_projection", "label", "bos_id", "turn_end", "preamble"])
def test_native_and_replay_reject_changed_profile_source_controls_and_labels(case, prepared, gw, tokenizer):
    payload, source = split(prepared)
    recipe = payload["manifest"]["recipe"]
    selection = recipe["preparation_profile"]
    example = payload["examples"][-1]
    if case == "unknown_profile": selection["name"] = "unqualified"
    elif case == "other_profile": selection["name"] = "qwen3_text_v1"
    elif case == "recipe_version": recipe["version"] = 1
    elif case == "template": recipe["tokenizer"]["chat_template_sha256"] = "0" * 64
    elif case == "file_pin": recipe["tokenizer"]["files"][0]["sha256"] = "0" * 64
    elif case == "backend": recipe["tokenizer_policy"]["backend_sha256"] = "0" * 64
    elif case == "added_control": recipe["tokenizer_policy"]["added_tokens"].pop()
    elif case == "padding": recipe["tokenizer_policy"]["wrapper"]["padding_side"] = "left"
    elif case in {"bos", "eos", "pad"}: recipe["tokenizer_policy"]["wrapper"][case + "_token_id"] = 99
    elif case == "dependencies": recipe["dependencies"]["tokenizers"] = "999"
    elif case == "thinking": selection["controls"]["enable_thinking"] = False
    elif case == "preserve_thinking": selection["controls"]["preserve_thinking"] = True
    elif case == "generation_prompt": selection["controls"]["add_generation_prompt"] = True
    elif case == "missing_controls": del selection["controls"]
    elif case == "source": source = b"not a canonical verified artifact"
    elif case == "source_projection": example["source"]["messages_json"] = "[]"
    elif case == "label": example["labels"][1] = example["input_ids"][1]
    elif case == "bos_id": example["input_ids"][0] = 3
    elif case == "turn_end": example["labels"][-2] = -100
    elif case == "preamble": example["rendered"] = example["rendered"].replace("<|think|>", "<|other|>")
    changed = frame(payload, source)
    native = subprocess.run([str(gw), "artifact", "verify-prepared", "--stdin"], input=changed, capture_output=True)
    assert native.returncode != 0, case
    with pytest.raises(ContractError):
        verify_prepared(changed, gw, tokenizer)


@pytest.mark.parametrize("case", ["adapter_source", "self_consistent_token"])
def test_structural_inspection_does_not_claim_installed_source_or_tokenizer_replay(case, prepared, gw, tokenizer):
    payload, source = split(prepared)
    if case == "adapter_source":
        payload["manifest"]["recipe"]["adapter_source_sha256"] = "0" * 64
    else:
        example = payload["examples"][-1]
        index = example["shifted_answer_token_indices"][0]
        example["input_ids"][index] = example["labels"][index] = 777
    changed = frame(payload, source)
    native = subprocess.run([str(gw), "artifact", "verify-prepared", "--stdin"], input=changed, capture_output=True)
    assert native.returncode == 0, native.stderr.decode()
    with pytest.raises(ContractError):
        verify_prepared(changed, gw, tokenizer)


def test_recursive_profile_source_is_bound_and_optimizer_source_is_separate(prepared, gw, tokenizer, tmp_path, monkeypatch):
    import ghostwriter_trl.build as module
    original = module.source_identity()
    copied = tmp_path / "package"
    shutil.copytree(module.PACKAGE, copied)
    monkeypatch.setattr(module, "PACKAGE", copied)
    assert module.source_identity() == original
    with (copied / "training" / "producer.py").open("a") as stream:
        stream.write("\n# unrelated optimizer implementation\n")
    assert module.source_identity() == original
    with (copied / "profiles" / "gemma.py").open("a") as stream:
        stream.write("\n# changed consumed profile\n")
    assert module.source_identity() != original
    with pytest.raises(ContractError, match="incompatible installed preparation recipe/source"):
        verify_prepared(prepared, gw, tokenizer)


def test_redundant_reasoning_details_replay_without_mutating_original_source(gw, tokenizer, profile, fixture_dir):
    from ghostwriter_trl.projection import prepare_target, project_messages
    source = [{"role": "user", "content": "question"},
              {"role": "assistant", "content": "answer", "reasoning": "why this",
               "reasoning_details": [{"type": "reasoning.text", "index": 2, "text": "why "},
                                     {"type": "reasoning.text", "index": 5, "text": "this"}]}]
    original = deepcopy(source)
    projected = project_messages(source, tokenizer, "masked", profile=profile)
    result = prepare_target(projected, tokenizer, "masked", 2048, profile=profile)
    assert source == original
    del source[-1]["reasoning_details"]
    assert result == prepare_target(project_messages(source, tokenizer, "masked", profile=profile),
                                     tokenizer, "masked", 2048, profile=profile)
