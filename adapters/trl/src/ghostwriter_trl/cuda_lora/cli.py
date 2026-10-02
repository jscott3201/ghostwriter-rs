"""Separate Gemma LoRA entry point; preparation source/build identities remain unchanged."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

from ..artifact import ContractError
from ..tokenizer import load_tokenizer
from .bundle import read_checkpoint
from .producer import train, train_fixture
from ..training.publication import PublishedCheckpointError


def main(argv=None):
    """Train the approved local release or safely inspect/reload one completed checkpoint."""
    parser = argparse.ArgumentParser(description="Bounded local CUDA Gemma LoRA and safe checkpoint reload")
    subcommands = parser.add_subparsers(dest="command", required=True)
    execute = subcommands.add_parser("train", help="train the exact approved Gemma4-E2B local release")
    execute.add_argument("--prepared", type=Path, required=True)
    source = execute.add_mutually_exclusive_group(required=True)
    source.add_argument("--release-directory", type=Path, help="exact approved model and tokenizer files")
    source.add_argument("--owned-fixture-tokenizer", type=Path, help="explicit random reduced official fixture using this approved tokenizer")
    execute.add_argument("--gw", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--max-steps", type=int, default=2)
    execute.add_argument("--batch-size", type=int, default=1)
    execute.add_argument("--accumulation", type=int, default=1)
    execute.add_argument("--learning-rate-millionths", type=int, default=100)
    execute.add_argument("--max-sequence-length", type=int, default=256)
    inspect = subcommands.add_parser("reload", help="verify captured bytes and safely reload inference weights")
    inspect.add_argument("--checkpoint", type=Path, required=True)
    inspect.add_argument("--tokenizer-directory", type=Path, required=True)
    inspect.add_argument("--gw", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        # Trainer progress belongs on stderr; stdout is a single machine-readable receipt.
        with redirect_stdout(sys.stderr):
            if args.command == "train":
                options = dict(max_steps=args.max_steps, batch_size=args.batch_size, accumulation=args.accumulation,
                    learning_rate_millionths=args.learning_rate_millionths, max_sequence_length=args.max_sequence_length)
                if args.release_directory is not None:
                    result = train(args.prepared, args.release_directory, args.gw, args.output, **options)
                else:
                    tokenizer = load_tokenizer(args.owned_fixture_tokenizer, profile="gemma4_e2b_text_v1")
                    result = train_fixture(args.prepared, tokenizer, args.gw, args.output, **options)
                with result:
                    report = {"completion_id": result.completion_id, "observed": result.observed}
            else:
                tokenizer = load_tokenizer(args.tokenizer_directory, profile="gemma4_e2b_text_v1")
                with read_checkpoint(args.checkpoint, args.gw, tokenizer) as loaded:
                    report = loaded.report
        print(json.dumps(report, sort_keys=True, allow_nan=False))
    except PublishedCheckpointError as error:
        print(json.dumps(error.report, sort_keys=True, allow_nan=False))
        print(f"checkpoint publication: {error}. Inspect the output against its completion_id; "
              "training again at the same path will be rejected.", file=sys.stderr)
        return 3
    except (ContractError, OSError, ValueError, RuntimeError, AssertionError) as error:
        print(f"checkpoint error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
