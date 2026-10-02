"""Actual bounded BF16 CUDA training with separate complete frozen/buffer evidence."""
import math
import os
import struct

from ..artifact import ContractError
from ..handoff import audit_batch, features
from ..profiles import GEMMA
from ..tokenizer import validate_tokenizer
from ..lora.config import check_lora_dependencies
from .model import audit as audit_parameters
from .state import measure
from .runtime import require_cuda
from .precision import observe, validate


def run(model, targets, examples, tokenizer, recipe, directory):
    """Run the bounded LoRA recipe and retain actual successful-operation observations."""
    import torch
    from datasets import Dataset
    from torch.utils.data import SequentialSampler
    from transformers import get_constant_schedule
    from trl import SFTConfig, SFTTrainer
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    check_lora_dependencies()
    validate_tokenizer(tokenizer, GEMMA)
    require_cuda()
    inventory = audit_parameters(model, targets)
    initial = measure(model)
    observations = {"successful_microbatches": 0, "consumed_examples": 0, "shifted_supervised_tokens": 0,
                    "optimizer_updates": 0, "successful_forwards": 0, "microbatches": [],
                    "trainables": inventory, "base_unchanged": False, "adapter_content_changed": False}
    collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id, padding_free=False)

    def collate(rows):
        batch = collator([{key: row[key] for key in ("input_ids", "attention_mask", "labels")} for row in rows])
        batch["_gw_indices"] = [row["_gw_index"] for row in rows]
        return batch

    def gradients():
        count = 0
        for parameter in model.parameters():
            if parameter.requires_grad:
                if parameter.grad is None or parameter.grad.dtype != torch.float32 or not torch.isfinite(parameter.grad).all():
                    raise ContractError("missing or nonfinite LoRA gradient")
                count += 1
            elif parameter.grad is not None:
                raise ContractError("frozen Gemma base acquired gradients")
        return count

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
            shifted = int(inputs["labels"][..., 1:].ne(-100).sum().item())
            tokens = int(inputs["attention_mask"].sum().item())
            audit_parameters(model, targets, optimizer)
            inputs["use_cache"] = False
            before = observations["successful_forwards"]
            loss = super().training_step(model, inputs, num_items_in_batch)
            if not math.isfinite(float(loss)) or observations["successful_forwards"] != before + 1:
                raise ContractError("LoRA training requires one successful finite-loss forward per microbatch")
            observed_gradients = gradients()
            observations["microbatches"].append({"update": observations["optimizer_updates"] + 1,
                "example_ids": [example["example_id"] for example in expected], "input_tokens": tokens,
                "shifted_supervised_tokens": shifted, "finite_adapter_gradients": observed_gradients,
                "loss_binary64": struct.pack(">d", float(loss)).hex()})
            observations["successful_microbatches"] += 1
            observations["consumed_examples"] += len(expected)
            observations["shifted_supervised_tokens"] += shifted
            return loss

    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=recipe["learning_rate_millionths"] / 1e6, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.0, foreach=False, fused=False, capturable=False, differentiable=False, amsgrad=False)
    audit_parameters(model, targets, optimizer)

    def before_step(*args):
        audit_parameters(model, targets, optimizer)
        gradients()

    def stepped(*args):
        audit_parameters(model, targets, optimizer)
        observations["optimizer_updates"] += 1

    def real_input_ids(module, args, kwargs):
        if (not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.bfloat16
                or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
                or not torch.are_deterministic_algorithms_enabled()):
            raise ContractError("actual forward requires deterministic BF16 CUDA autocast with TF32 disabled")
        if kwargs.get("input_ids") is None or kwargs.get("inputs_embeds") is not None or kwargs.get("per_layer_inputs") is not None:
            raise ContractError("Gemma LoRA must preserve actual token IDs through per-layer embeddings")

    def forward_completed(*args):
        observations["successful_forwards"] += 1

    hooks = [optimizer.register_step_pre_hook(before_step), optimizer.register_step_post_hook(stepped),
             model.get_base_model().register_forward_pre_hook(real_input_ids, with_kwargs=True),
             model.get_base_model().register_forward_hook(forward_completed)]
    try:
        args = SFTConfig(output_dir=str(directory), use_cpu=False, bf16=True, fp16=False, bf16_full_eval=False, fp16_full_eval=False,
            tf32=False, auto_find_batch_size=False, torch_compile=False,
            loss_type="nll", gradient_checkpointing=False, report_to="none", push_to_hub=False,
            save_strategy="no", logging_strategy="no", eval_strategy="no", disable_tqdm=True,
            dataloader_num_workers=0, dataloader_pin_memory=False, dataloader_drop_last=False,
            dataset_kwargs={"skip_prepare_dataset": True}, max_length=None, packing=False, padding_free=False,
            assistant_only_loss=False, completion_only_loss=False, remove_unused_columns=False,
            per_device_train_batch_size=recipe["batch_size"], gradient_accumulation_steps=recipe["accumulation"],
            max_steps=recipe["max_steps"], seed=0, data_seed=0, lr_scheduler_type="constant",
            optim="adamw_torch", learning_rate=recipe["learning_rate_millionths"] / 1e6,
            max_grad_norm=1.0, full_determinism=False)
        trainer = ObservedTrainer(model=model, args=args, processing_class=tokenizer, data_collator=collate,
            train_dataset=Dataset.from_list([{**features(example), "_gw_index": index} for index, example in enumerate(examples)]),
            optimizers=(optimizer, get_constant_schedule(optimizer)))
        if trainer.accelerator.num_processes != 1 or trainer.args.device != torch.device("cuda:0"):
            raise ContractError("unsupported distributed/device CUDA LoRA trainer")
        require_cuda()
        audit_parameters(model, targets, optimizer)
        with observe(model) as precision:
            trainer.train()
        validate(precision)
        observations["precision_operations"] = precision
    finally:
        for hook in reversed(hooks):
            hook.remove()
    if observations["optimizer_updates"] != recipe["max_steps"]:
        raise ContractError("LoRA trainer did not complete the requested successful updates")
    final = measure(model)
    if final["frozen"] != initial["frozen"] or final["buffers"] != initial["buffers"]:
        raise ContractError("CUDA execution changed frozen parameters or any registered buffer")
    if any(value["blake3"] == initial["adapters"]["tensors"][name]["blake3"]
           for name, value in final["adapters"]["tensors"].items()):
        raise ContractError("CUDA optimization did not change every adapter tensor")
    observations.update(base_unchanged=True, adapter_content_changed=True)
    return observations
