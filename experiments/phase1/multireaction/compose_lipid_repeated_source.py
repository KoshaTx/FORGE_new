"""Run or authenticate the protected-TRAIN repeated source-program diagnostic."""

import argparse
from pathlib import Path

from forge.corpus.compose_lipid_repeated_source import run_repeated_source, verify_repeated_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        if args.config or args.output_dir:
            parser.error("--verify cannot be combined with execution arguments")
        verify_repeated_source(args.repo_root, args.verify)
        print("verified")
    else:
        if not args.config or not args.output_dir:
            parser.error("--config and --output-dir are required")
        result = run_repeated_source(args.repo_root, args.config, args.output_dir)
        print(result["status"])


if __name__ == "__main__":
    main()
