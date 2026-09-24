"""Run or independently verify source-scaffold assembly-event qualification."""

import argparse
from pathlib import Path

from forge.corpus.compose_lipid_scaffold_event import run_scaffold_event, verify_scaffold_event


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        if args.config or args.output_dir:
            parser.error("verification cannot be combined with execution arguments")
        verify_scaffold_event(args.repo_root, args.verify)
        print("verified")
    else:
        if not args.config or not args.output_dir:
            parser.error("execution requires --config and --output-dir")
        print(run_scaffold_event(args.repo_root, args.config, args.output_dir)["status"])


if __name__ == "__main__":
    main()
