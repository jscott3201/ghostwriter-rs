"""Offline command-line preparation with explicit policies and auditable outputs."""
import argparse
import json
from pathlib import Path

from .artifact import ContractError, read_snapshot
from .build import build, identity
from .handoff import qualify_handoff
from .tokenizer import load_tokenizer


def main() -> None:
    """Write examples and a manifest; optionally inspect the actual trainer handoff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--gw", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--cot", required=True, choices=("supervised", "masked", "stripped"))
    parser.add_argument("--turns", required=True, choices=("final_turn_only", "all_assistant"))
    parser.add_argument("--max-length", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--qualify-handoff", action="store_true")
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ContractError("output directory already exists")
        snapshot = read_snapshot(args.artifact, args.gw)
        tokenizer = load_tokenizer(args.tokenizer)
        examples, manifest = build(snapshot, tokenizer, cot=args.cot, turns=args.turns, max_length=args.max_length)
        if args.qualify_handoff:
            manifest["trainer_handoff"] = qualify_handoff(examples, tokenizer)
            manifest.pop("build_id")
            manifest["build_id"] = identity(manifest)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "examples.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False, allow_nan=False) + "\n" for e in examples))
        (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    except (ContractError, OSError) as error:
        parser.exit(1, f"Preparation failed: {error}\n")
    print(json.dumps({"examples": len(examples), "rejected_items": manifest["rejected_item_count"], "build_id": manifest["build_id"]}))


if __name__ == "__main__":
    main()
