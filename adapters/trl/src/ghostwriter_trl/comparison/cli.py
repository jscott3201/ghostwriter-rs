"""Explicit live training/comparison and saved inspection/replay entry points."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

from ..artifact import ContractError
from ..tokenizer import load_tokenizer
from ..training.publication import PublishedCheckpointError
from .artifact import PublishedComparisonError, read
from .producer import train_and_compare, replay


def main(argv=None):
    """Run local CPU software workflows; saved files never re-create a live training receipt."""
    parser = argparse.ArgumentParser(description="Gemma CPU base/LoRA paired coding software evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    live = commands.add_parser("train-and-compare", help="train once, then consume that live receipt into one whole held-out pair")
    live.add_argument("--prepared", type=Path, required=True)
    live.add_argument("--release-directory", type=Path, required=True)
    live.add_argument("--checkpoint-output", type=Path, required=True)
    live.add_argument("--registration", required=True)
    live.add_argument("--split", choices=["validation", "test"], default="test")
    live.add_argument("--max-steps", type=int, default=2)
    live.add_argument("--max-new-tokens", type=int, default=128)
    live.add_argument("--max-prompt-tokens", type=int, default=1024)
    live.add_argument("--system-prompt", default="")
    inspect = commands.add_parser("inspect", help="check saved bindings, arithmetic and actual token decoding")
    fresh = commands.add_parser("replay", help="freshly execute saved modules against the registered private oracles")
    for command in (live, inspect, fresh): command.add_argument("--gw", type=Path, required=True)
    for command in (live, fresh):
        command.add_argument("--db", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    for command in (inspect, fresh):
        command.add_argument("--artifact", type=Path, required=True)
        command.add_argument("--tokenizer-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        with redirect_stdout(sys.stderr):
            if args.command == "train-and-compare":
                result = train_and_compare(args.prepared, args.release_directory, args.gw, args.db,
                    args.registration, args.checkpoint_output, args.output, max_steps=args.max_steps,
                    split=args.split, max_new_tokens=args.max_new_tokens,
                    max_prompt_tokens=args.max_prompt_tokens, system_prompt=args.system_prompt).report
            else:
                tokenizer = load_tokenizer(args.tokenizer_directory, profile="gemma4_e2b_text_v1")
                result = (read(args.artifact, args.gw, tokenizer)[1] if args.command == "inspect" else
                          replay(args.artifact, tokenizer, args.gw, args.db, args.output))
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    except (PublishedCheckpointError, PublishedComparisonError) as error:
        print(json.dumps(error.report, sort_keys=True, allow_nan=False))
        print(f"publication settlement: {error}", file=sys.stderr)
        return 3
    except (ContractError, OSError, ValueError, RuntimeError) as error:
        print(f"paired comparison error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("paired comparison cancelled; owned native execution settled", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
