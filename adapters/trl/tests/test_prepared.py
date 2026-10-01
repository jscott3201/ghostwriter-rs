"""Saved input builds cross the actual producer, Rust verifier, and pinned Python consumer."""
import json
import os
import subprocess
import sys


def test_cli_saves_consumed_input_build_before_separate_handoff(gw, fixture_dir, tmp_path):
    output = tmp_path / "prepared"
    result = subprocess.run([
        sys.executable, "-m", "ghostwriter_trl.cli", "--artifact", str(fixture_dir / "screened-all.parquet"),
        "--gw", str(gw), "--tokenizer", os.environ["GW_TRL_TOKENIZER"],
        "--cot", "masked", "--turns", "all_assistant", "--max-length", "2048",
        "--output", str(output), "--qualify-handoff",
    ], capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    data = (output / "prepared.gwsft").read_bytes()
    assert data[:8] == b"GWSFT001"
    summary = json.loads(result.stdout)
    verification = json.loads((output / "verification.json").read_text())
    handoff = json.loads((output / "handoff.json").read_text())
    assert summary["build_id"] == verification["build_id"] == handoff["build_id"]
    assert handoff["real_sft_trainer_dataloader"]["nonpadding_preserved"] is True
