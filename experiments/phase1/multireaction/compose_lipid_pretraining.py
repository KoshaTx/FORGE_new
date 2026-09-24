"""Run or authenticate the local COMPOSE pre-training qualification audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from forge.corpus.compose_lipid_pretraining import (
    run_pretraining_checks,
    verify_pretraining_checks,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        if args.config or args.output_dir:
            parser.error("verification cannot be combined with execution")
        result = verify_pretraining_checks(args.repo_root, args.verify)
        print(f"verified: {result['status']}")
    else:
        if not args.config or not args.output_dir:
            parser.error("execution requires --config and --output-dir")
        result = run_pretraining_checks(args.repo_root, args.config, args.output_dir)
        print(result["status"])


if __name__ == "__main__":
    main()
