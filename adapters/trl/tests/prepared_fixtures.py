"""Prepare current test inputs without rewriting historical interoperability artifacts."""
from pathlib import Path
import shutil

from ghostwriter_trl.artifact import read_snapshot
from ghostwriter_trl.prepared import prepare, verify_prepared


def current_fixtures(destination: Path, source: Path, gw: Path, tokenizer, profile: str):
    """Copy canonical synthetic inputs and produce three independently verified current builds."""
    for path in source.glob("*.parquet"):
        shutil.copyfile(path, destination / path.name)
    for source_name, output_name in [("screened-all", "prepared-all"), ("screened-empty", "prepared-empty"),
                                     ("v3-long", "prepared-long")]:
        snapshot = read_snapshot(source / f"{source_name}.parquet", gw)
        data = prepare(snapshot, tokenizer, cot="masked", turns="all_assistant", max_length=2048, profile=profile)
        loaded = verify_prepared(data, gw, tokenizer)
        (destination / f"{output_name}.gwsft").write_bytes(loaded.data)
    return destination
