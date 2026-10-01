"""Explicitly regenerate synthetic interoperability inputs using the installed adapter."""
import os
from pathlib import Path

from ghostwriter_trl.artifact import read_snapshot
from ghostwriter_trl.prepared import prepare, verify_prepared
from ghostwriter_trl.tokenizer import load_tokenizer


def main():
    """Verify each produced build before replacing its intentionally regenerated fixture."""
    root = Path(__file__).parent / "fixtures"
    gw = Path(os.environ["GW_TRL_GW"]).resolve(strict=True)
    tokenizer = load_tokenizer(Path(os.environ["GW_TRL_TOKENIZER"]))
    for source, name in [("screened-all", "prepared-all"), ("screened-empty", "prepared-empty"), ("v3-long", "prepared-long")]:
        snapshot = read_snapshot(root / f"{source}.parquet", gw)
        data = prepare(snapshot, tokenizer, cot="masked", turns="all_assistant", max_length=2048)
        loaded = verify_prepared(data, gw, tokenizer)
        (root / f"{name}.gwsft").write_bytes(loaded.data)
        print(name, loaded.build_id, len(loaded.examples))


if __name__ == "__main__":
    main()
