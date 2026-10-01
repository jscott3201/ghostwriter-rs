"""Offline command-line preparation with explicit policies and auditable outputs."""
import argparse
import json
from pathlib import Path

from .artifact import ContractError, read_snapshot
from .prepared import prepare, read_prepared, save_prepared
from .handoff import qualify_prepared_handoff
from .tokenizer import load_tokenizer
from .profiles import NAMES, QWEN


def main() -> None:
    """Save a consumed immutable input; keep replay and actual trainer reports separate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--gw", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--profile", choices=NAMES, default=QWEN)
    parser.add_argument("--thinking", choices=("on", "off"), help="explicit Gemma thinking preamble; Qwen requires on")
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
        tokenizer = load_tokenizer(args.tokenizer, profile=args.profile)
        thinking = None if args.thinking is None else args.thinking == "on"
        data = prepare(snapshot, tokenizer, cot=args.cot, turns=args.turns, max_length=args.max_length,
                       profile=args.profile, enable_thinking=thinking)
        args.output.mkdir(parents=True, exist_ok=False)
        save_prepared(args.output / "prepared.gwsft", data)
        loaded = read_prepared(args.output / "prepared.gwsft", args.gw, tokenizer)
        reports = {"verification.json": loaded.report, "replay.json": loaded.replay_report}
        if args.qualify_handoff:
            reports["handoff.json"] = qualify_prepared_handoff(loaded, tokenizer)
        for name, report in reports.items():
            (args.output / name).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    except (ContractError, OSError) as error:
        parser.exit(1, f"Preparation failed: {error}\n")
    print(json.dumps({"examples": len(loaded.examples), "rejected_items": loaded.manifest["rejected_item_count"], "build_id": loaded.build_id}))


if __name__ == "__main__":
    main()
