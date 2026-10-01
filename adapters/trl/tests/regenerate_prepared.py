"""Produce current synthetic inputs in a new directory; preserve historical saved identities."""
import argparse
import os
from pathlib import Path

from ghostwriter_trl.tokenizer import load_tokenizer
from ghostwriter_trl.profiles import NAMES
from .prepared_fixtures import current_fixtures


def main():
    """Create and verify new fixtures without replacing historical committed artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=NAMES, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parent / "fixtures"
    gw = Path(os.environ["GW_TRL_GW"]).resolve(strict=True)
    tokenizer = load_tokenizer(args.tokenizer, profile=args.profile)
    args.output.mkdir(parents=True, exist_ok=False)
    current_fixtures(args.output, root, gw, tokenizer, args.profile)
    for path in sorted(args.output.glob("*.gwsft")):
        print(path.name, path.read_bytes()[8:40].hex())


if __name__ == "__main__":
    main()
