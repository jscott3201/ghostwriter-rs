"""Real canonical-source preparation, native inspection, complete replay, and trainer consumption."""
from ghostwriter_trl.artifact import read_snapshot
from ghostwriter_trl.prepared import prepare, save_prepared, read_prepared
from ghostwriter_trl.handoff import qualify_prepared_handoff


def test_complete_gemma_saved_input_replays_and_reaches_actual_dataloader(gw, tokenizer, profile, fixture_dir, tmp_path):
    snapshot = read_snapshot(fixture_dir / "v3-text.parquet", gw)
    data = prepare(snapshot, tokenizer, cot="masked", turns="all_assistant", max_length=2048,
                   profile=profile, enable_thinking=True)
    path = tmp_path / "prepared.gwsft"
    save_prepared(path, data)
    loaded = read_prepared(path, gw, tokenizer)
    assert loaded.data == data
    assert loaded.manifest["recipe"]["version"] == 2
    assert loaded.manifest["recipe"]["preparation_profile"]["name"] == profile
    assert len(loaded.examples) == 4
    report = qualify_prepared_handoff(loaded, tokenizer)
    assert report["real_collator"]["rows"] == report["real_sft_trainer_dataloader"]["rows"] == 4
    assert report["forward_passes"] == report["optimizer_steps"] == 0
