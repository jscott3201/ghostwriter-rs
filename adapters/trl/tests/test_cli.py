"""The installed module writes auditable offline outputs without a training step."""
import json
import os
from pathlib import Path
import subprocess
import sys

from ghostwriter_trl.build import identity


def test_cli_writes_manifest_examples_and_real_handoff_without_overwrite(gw, fixture_dir, tmp_path):
    output = tmp_path / "prepared"
    args = [
        sys.executable, "-m", "ghostwriter_trl.cli", "--artifact", str(fixture_dir / "v3-text.parquet"),
        "--gw", str(gw), "--tokenizer", os.environ["GW_TRL_TOKENIZER"],
        "--cot", "masked", "--turns", "all_assistant", "--max-length", "2048",
        "--output", str(output), "--qualify-handoff",
    ]
    result = subprocess.run(args, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["examples"] == 4 and summary["rejected_items"] == 0
    manifest = json.loads((output / "manifest.json").read_text())
    examples = [json.loads(line) for line in (output / "examples.jsonl").read_text().splitlines()]
    assert [example["example_id"] for example in examples] == manifest["example_ids"]
    build_id = manifest.pop("build_id")
    assert identity(manifest) == build_id == summary["build_id"]
    assert manifest["trainer_handoff"]["real_sft_trainer_dataloader"]["nonpadding_preserved"]
    assert str(tmp_path) not in json.dumps(manifest)
    before = {file.name: file.read_bytes() for file in output.iterdir()}
    again = subprocess.run(args, cwd=tmp_path, capture_output=True, text=True)
    assert again.returncode == 1 and "Preparation failed" in again.stderr
    assert before == {file.name: file.read_bytes() for file in output.iterdir()}
