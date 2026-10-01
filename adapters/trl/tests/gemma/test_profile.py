"""The exact official processor owns rendering; the training wrapper is explicit."""
import os
from pathlib import Path
from copy import deepcopy
import shutil

import pytest

from ghostwriter_trl.artifact import ContractError
from ghostwriter_trl.tokenizer import load_tokenizer, validate_tokenizer, check_dependencies


def test_exact_gemma_release_loads_official_processor_with_right_padding(profile):
    from transformers import Gemma4Processor, GemmaTokenizer
    tokenizer = load_tokenizer(Path(os.environ["GW_TRL_GEMMA_TOKENIZER"]), profile=profile)
    assert type(tokenizer) is GemmaTokenizer
    assert tokenizer.padding_side == "right"
    processor = tokenizer._ghostwriter_processor
    assert type(processor) is Gemma4Processor
    assert processor.tokenizer is tokenizer
    source = [{"role": "user", "content": "  question \n"},
              {"role": "assistant", "content": "\n answer  "}]
    expected = "<bos><|turn>user\nquestion<turn|>\n<|turn>model\nanswer<turn|>\n"
    assert processor.apply_chat_template(source, tokenize=False, add_generation_prompt=False,
                                         enable_thinking=False, preserve_thinking=False) == expected


@pytest.mark.parametrize("name", ["README.md", "config.json", "generation_config.json", "processor_config.json",
                                 "tokenizer_config.json", "chat_template.jinja", "tokenizer.json", "extra"])
def test_every_consumed_release_file_is_pinned_before_loading(name, profile, tmp_path, monkeypatch):
    from transformers import Gemma4Processor
    directory = tmp_path / "release"
    shutil.copytree(os.environ["GW_TRL_GEMMA_TOKENIZER"], directory)
    (directory / name).write_bytes(b"changed")
    monkeypatch.setattr(Gemma4Processor, "from_pretrained", lambda *a, **k: pytest.fail("unverified bytes reached processor"))
    with pytest.raises(ContractError):
        load_tokenizer(directory, profile=profile)


def test_loader_consumes_captured_bytes_when_original_release_is_replaced(profile, tmp_path, monkeypatch):
    from transformers import Gemma4Processor
    directory = tmp_path / "release"
    shutil.copytree(os.environ["GW_TRL_GEMMA_TOKENIZER"], directory)
    original = Gemma4Processor.from_pretrained
    def replace_and_load(path, **kwargs):
        assert Path(path) != directory
        assert kwargs["local_files_only"] is True and kwargs["trust_remote_code"] is False
        (directory / "chat_template.jinja").write_bytes(b"replacement")
        return original(path, **kwargs)
    monkeypatch.setattr(Gemma4Processor, "from_pretrained", replace_and_load)
    validate_tokenizer(load_tokenizer(directory, profile=profile), profile)


@pytest.mark.parametrize("change", ["template", "processor_tokenizer", "padding", "truncation", "bos", "eos", "pad", "add_bos", "add_eos", "backend"])
def test_runtime_wrapper_backend_and_official_processor_mutations_are_rejected(tokenizer, profile, change):
    altered = deepcopy(tokenizer)
    if change == "template": altered._ghostwriter_processor.chat_template += "changed"
    elif change == "processor_tokenizer": altered._ghostwriter_processor.tokenizer = tokenizer
    elif change in {"padding", "truncation"}: setattr(altered, change + "_side", "left")
    elif change in {"bos", "eos", "pad"}: setattr(altered, change + "_token", "<unk>")
    elif change in {"add_bos", "add_eos"}: setattr(altered, change + "_token", True)
    else: altered.add_tokens(["unqualified_added_token"])
    with pytest.raises(ContractError):
        validate_tokenizer(altered, profile)


def test_dependency_profiles_are_exact_and_cannot_be_substituted(profile, monkeypatch):
    pins = check_dependencies(profile)
    assert pins["transformers"] == "5.18.0" and pins["tokenizers"] == "0.23.2"
    with pytest.raises(ContractError, match="dependencies"):
        check_dependencies("qwen3_text_v1")
    import ghostwriter_trl.tokenizer as module
    original = module.version
    monkeypatch.setattr(module, "version", lambda name: "999" if name == "pillow" else original(name))
    with pytest.raises(ContractError, match="dependencies"):
        check_dependencies(profile)
