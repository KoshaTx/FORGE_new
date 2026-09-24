"""Run or independently verify source-qualified sequential corpus programs."""

import argparse
from pathlib import Path

from forge.corpus.compose_lipid_sequential import run_sequential_program, verify_sequential_program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        if args.config or args.output_dir:
            parser.error("--verify cannot be combined with execution options")
        verify_sequential_program(args.repo_root, args.verify)
        print("verified")
    else:
        if not args.config or not args.output_dir:
            parser.error("execution requires --config and --output-dir")
        print(run_sequential_program(args.repo_root, args.config, args.output_dir)["status"])


if __name__ == "__main__":
    main()
