"""Exercise the installed Gemma entry point and both explicit thinking controls."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ghostwriter_trl.prepared import read_prepared


@pytest.mark.parametrize("thinking", ["on", "off"])
def test_installed_cli_saves_replays_and_qualifies_gemma_without_execution(
        tokenizer, profile, gw, historical_fixture_dir, tmp_path, thinking):
    output = tmp_path / "prepared"
    command = [str(Path(sys.executable).with_name("ghostwriter-trl")),
               "--artifact", str(historical_fixture_dir / "v3-text.parquet"),
               "--gw", str(gw), "--tokenizer", os.environ["GW_TRL_GEMMA_TOKENIZER"],
               "--profile", profile, "--thinking", thinking,
               "--cot", "masked", "--turns", "all_assistant", "--max-length", "2048",
               "--output", str(output), "--qualify-handoff"]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    loaded = read_prepared(output / "prepared.gwsft", gw, tokenizer)
    assert json.loads(result.stdout)["build_id"] == loaded.build_id
    assert len(loaded.examples) == 4
    assert loaded.manifest["recipe"]["preparation_profile"] == {
        "name": profile, "controls": {"enable_thinking": thinking == "on",
                                     "add_generation_prompt": False, "preserve_thinking": False}}
    reports = {name: json.loads((output / f"{name}.json").read_text())
               for name in ("verification", "replay", "handoff")}
    assert all(report["build_id"] == loaded.build_id for report in reports.values())
    assert reports["handoff"]["forward_passes"] == reports["handoff"]["optimizer_steps"] == 0
    assert reports["handoff"]["real_sft_trainer_dataloader"]["nonpadding_preserved"]
    before = {file.name: file.read_bytes() for file in output.iterdir()}
    again = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert again.returncode == 1 and "already exists" in again.stderr
    assert before == {file.name: file.read_bytes() for file in output.iterdir()}
