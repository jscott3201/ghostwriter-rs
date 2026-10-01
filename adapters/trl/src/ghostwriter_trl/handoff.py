"""Inspect real collated batches and a real SFTTrainer dataloader without model execution."""
from collections import Counter
from pathlib import Path
import tempfile

from .artifact import ContractError
from .profiles import QWEN
from .tokenizer import check_dependencies, validate_tokenizer

FEATURE_KEYS = ("input_ids", "attention_mask", "labels")


def features(example: dict) -> dict:
    """Keep audit fields outside the trainer's feature namespace."""
    return {key: example[key] for key in FEATURE_KEYS}


def audit_batch(batch, expected: list[dict], pad_token_id: int) -> dict:
    """Compare every nonpadding ID/label and every padding value, independent of sampler order."""
    observed = []
    for ids, attention, labels in zip(*(batch[key].tolist() for key in FEATURE_KEYS), strict=True):
        length = sum(attention)
        if attention != [1] * length + [0] * (len(attention) - length):
            raise ContractError("trainer changed attention or right padding")
        if ids[length:] != [pad_token_id] * (len(ids) - length) or labels[length:] != [-100] * (len(labels) - length):
            raise ContractError("trainer changed padding labels or IDs")
        observed.append((tuple(ids[:length]), tuple(labels[:length])))
    wanted = [(tuple(e["input_ids"]), tuple(e["labels"])) for e in expected]
    if Counter(observed) != Counter(wanted):
        raise ContractError("trainer changed or dropped audited nonpadding IDs/labels")
    return {"rows": len(observed), "shape": list(batch["input_ids"].shape), "nonpadding_preserved": True, "padding_masked": True}


def qualify_handoff(examples: list[dict], tokenizer, *, profile: str = QWEN) -> dict:
    """Construct a small random CPU model and inspect batches; never run forward or optimization."""
    if len(examples) < 2 or len({len(e["input_ids"]) for e in examples}) < 2:
        raise ContractError("handoff qualification requires at least two unequal-length examples")
    check_dependencies(profile)
    validate_tokenizer(tokenizer, profile)
    import torch
    from datasets import Dataset
    from transformers import GPT2Config, GPT2LMHeadModel
    from trl import SFTConfig, SFTTrainer
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    torch.manual_seed(0)
    batch_features = [features(e) for e in examples]
    collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id, padding_free=False)
    collator_report = audit_batch(collator(batch_features), examples, tokenizer.pad_token_id)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(tokenizer), n_positions=max(len(e["input_ids"]) for e in examples),
        n_embd=16, n_layer=1, n_head=2, bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
    ))
    with tempfile.TemporaryDirectory(prefix="gw-trl-handoff-") as output:
        settings = dict(
            output_dir=str(Path(output)), use_cpu=True, bf16=False, fp16=False,
            loss_type="nll", gradient_checkpointing=False, report_to="none", push_to_hub=False,
            dataloader_num_workers=0, dataloader_pin_memory=False,
            dataset_kwargs={"skip_prepare_dataset": True}, max_length=None, packing=False, padding_free=False,
            assistant_only_loss=False, completion_only_loss=False,
            per_device_train_batch_size=len(examples), seed=0, data_seed=0,
        )
        trainer = SFTTrainer(
            model=model, args=SFTConfig(**settings), processing_class=tokenizer,
            train_dataset=Dataset.from_list(batch_features), data_collator=collator,
        )
        if trainer.args.device.type != "cpu" or next(model.parameters()).device.type != "cpu":
            raise ContractError("qualification requires CPU-only model and trainer")
        trainer_report = audit_batch(next(iter(trainer.get_train_dataloader())), examples, tokenizer.pad_token_id)
    settings.pop("output_dir")
    return {
        "qualification_version": 1, "real_collator": collator_report,
        "real_sft_trainer_dataloader": trainer_report, "settings": settings,
        "random_model": {"architecture": "GPT2LMHeadModel", "vocab_size": len(tokenizer), "n_embd": 16, "n_layer": 1, "n_head": 2},
        "pretrained_weights": False, "forward_passes": 0, "optimizer_steps": 0,
        "shifted_answer_tokens": sum(len(e["shifted_answer_token_indices"]) for e in examples),
    }


def qualify_prepared_handoff(prepared, tokenizer) -> dict:
    """Consume only a whole immutable build loaded through actual verification and replay.

    Handoff evidence cites the already-fixed input identity and never changes it.
    """
    from .prepared import VerifiedPrepared
    if type(prepared) is not VerifiedPrepared:
        raise ContractError("handoff source must be an actual verified and replayed prepared input")
    profile = prepared.manifest["recipe"]["preparation_profile"]["name"]
    return {"build_id": prepared.build_id, **qualify_handoff(prepared.examples, tokenizer, profile=profile)}
