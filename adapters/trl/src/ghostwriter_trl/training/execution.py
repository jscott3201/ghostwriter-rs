"""Actual CPU full SFT, with observations committed only after successful operations."""
import math
import os

from ..artifact import ContractError
from ..handoff import audit_batch, features
from ..tokenizer import check_dependencies, validate_tokenizer


def run(model, examples, tokenizer, recipe, directory):
    """Run the bounded sampler and count completed forward/backward and raw optimizer calls."""
    check_dependencies()
    validate_tokenizer(tokenizer)
    import torch
    from datasets import Dataset
    from torch.utils.data import SequentialSampler
    from transformers import get_constant_schedule
    from trl import SFTConfig, SFTTrainer
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    if any(os.environ.get(key, default) != default for key, default in (
            ("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "-1"), ("ACCELERATE_USE_CPU", "true"))):
        raise ContractError("full SFT requires a standalone single CPU process")
    if any(not parameter.requires_grad or parameter.dtype != torch.float32 or parameter.device.type != "cpu"
           for parameter in model.parameters()):
        raise ContractError("all model parameters must train on CPU in float32")
    observations = {"successful_microbatches": 0, "consumed_examples": 0, "shifted_supervised_tokens": 0,
                    "optimizer_updates": 0, "microbatches": [], "parameter_content_changed": False}
    collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id, padding_free=False)

    def collate(rows):
        indices = [row["_gw_index"] for row in rows]
        batch = collator([{key: row[key] for key in ("input_ids", "attention_mask", "labels")} for row in rows])
        batch["_gw_indices"] = indices
        return batch

    class ObservedTrainer(SFTTrainer):
        def _get_train_sampler(self, train_dataset=None):
            return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

        def training_step(self, model, inputs, num_items_in_batch=None):
            indices = inputs.pop("_gw_indices")
            expected = [examples[index] for index in indices]
            audit_batch(inputs, expected, tokenizer.pad_token_id)
            for row, example in enumerate(expected):
                length = len(example["input_ids"])
                if any(inputs[key][row, :length].tolist() != example[key]
                       for key in ("input_ids", "attention_mask", "labels")):
                    raise ContractError("trainer changed the recorded example order or labels")
            # Count the actual labels and padding supplied to this successful model call.
            shifted = int(inputs["labels"][..., 1:].ne(-100).sum().item())
            tokens = int(inputs["attention_mask"].sum().item())
            inputs["use_cache"] = False
            loss = super().training_step(model, inputs, num_items_in_batch)
            if not math.isfinite(float(loss)):
                raise ContractError("training produced a nonfinite loss")
            observations["microbatches"].append({"update": observations["optimizer_updates"] + 1,
                "example_ids": [example["example_id"] for example in expected], "input_tokens": tokens,
                "shifted_supervised_tokens": shifted})
            observations["successful_microbatches"] += 1
            observations["consumed_examples"] += len(expected)
            observations["shifted_supervised_tokens"] += shifted
            return loss

    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate_millionths"] / 1e6,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, foreach=False, fused=False)
    def stepped(*args):
        observations["optimizer_updates"] += 1
    hook = optimizer.register_step_post_hook(stepped)
    try:
        args = SFTConfig(output_dir=str(directory), use_cpu=True, bf16=False, fp16=False,
            loss_type="nll", gradient_checkpointing=False, report_to="none", push_to_hub=False,
            save_strategy="no", logging_strategy="no", eval_strategy="no", disable_tqdm=True,
            dataloader_num_workers=0, dataloader_pin_memory=False, dataloader_drop_last=False,
            dataset_kwargs={"skip_prepare_dataset": True}, max_length=None, packing=False, padding_free=False,
            assistant_only_loss=False, completion_only_loss=False, remove_unused_columns=False,
            per_device_train_batch_size=recipe["batch_size"], gradient_accumulation_steps=recipe["accumulation"],
            max_steps=recipe["max_steps"], seed=0, data_seed=0, lr_scheduler_type="constant",
            optim="adamw_torch", learning_rate=recipe["learning_rate_millionths"] / 1e6,
            max_grad_norm=1.0, full_determinism=True)
        trainer = ObservedTrainer(model=model, args=args, processing_class=tokenizer, data_collator=collate,
            train_dataset=Dataset.from_list([{**features(example), "_gw_index": index} for index, example in enumerate(examples)]),
            optimizers=(optimizer, get_constant_schedule(optimizer)))
        if trainer.accelerator.num_processes != 1 or trainer.args.device.type != "cpu":
            raise ContractError("unsupported distributed/device trainer")
        trainer.train()
    finally:
        hook.remove()
    if observations["optimizer_updates"] != recipe["max_steps"]:
        raise ContractError("trainer did not complete the requested successful optimizer updates")
    return observations
