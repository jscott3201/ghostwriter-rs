"""Real TRL behavior, with no inference or training."""
from ghostwriter_trl.projection import prepare_target, project_messages


def test_collator_preserves_explicit_reasoning_mask(tokenizer):
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    example = prepare_target(project_messages([
        {"role": "user", "content": "USER sentinel"},
        {"role": "assistant", "content": "ANSWER sentinel", "reasoning": "REASON sentinel"},
    ], tokenizer, "masked"), tokenizer, "masked", 2048)
    collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id)
    from ghostwriter_trl.handoff import features
    batch = collator([features(example)])
    assert batch["labels"][0].tolist() == example["labels"]


def test_real_trainer_handoff_preserves_unequal_length_audited_examples(tokenizer, gw, fixture_dir):
    from ghostwriter_trl.artifact import read_snapshot
    from ghostwriter_trl.build import build
    from ghostwriter_trl.handoff import qualify_handoff
    snapshot = read_snapshot(fixture_dir / "v3-text.parquet", gw)
    for policy in ("supervised", "masked", "stripped"):
        examples, manifest = build(snapshot, tokenizer, cot=policy, turns="all_assistant", max_length=2048)
        assert manifest["expanded_example_count"] == 4
        report = qualify_handoff(examples, tokenizer)
        assert report["real_collator"]["rows"] == 4
        assert report["real_sft_trainer_dataloader"]["rows"] == 4
        assert report["shifted_answer_tokens"] > 0
        assert report["forward_passes"] == report["optimizer_steps"] == 0


def test_real_trainer_does_not_truncate_examples_above_its_usual_default(tokenizer):
    from ghostwriter_trl.handoff import qualify_handoff
    examples = []
    for count in (1050, 1060):
        example = prepare_target(project_messages([
            {"role": "user", "content": "context " * count},
            {"role": "assistant", "content": "ANSWER final", "reasoning": "REASON final"},
        ], tokenizer, "masked"), tokenizer, "masked", 2048)
        assert len(example["input_ids"]) > 1024
        examples.append(example)
    report = qualify_handoff(examples, tokenizer)
    assert report["real_sft_trainer_dataloader"]["shape"][1] > 1024
    assert report["settings"]["max_length"] is None
