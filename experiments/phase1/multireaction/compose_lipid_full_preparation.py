"""Build or independently reproduce full-universe protected preparation accounting."""

import argparse
from pathlib import Path

from forge.corpus.compose_lipid_full_preparation import (
    build_full_preparation,
    verify_full_preparation,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    if args.verify and not (args.config or args.output):
        result, _, _ = verify_full_preparation(repo, args.verify)
    elif args.config and args.output and not args.verify:
        result = build_full_preparation(repo, args.config, args.output)
    else:
        parser.error("provide --config and --output, or --verify")
    print(result["status"], flush=True)
    print(result["summary"]["totals"], flush=True)


if __name__ == "__main__":
    main()
