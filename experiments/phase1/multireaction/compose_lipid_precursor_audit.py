"""Build or independently recount the COMPOSE precursor-identity audit."""

import argparse
from pathlib import Path

from forge.corpus.compose_lipid_precursor_audit import (
    build_precursor_audit,
    verify_precursor_audit,
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
            parser.error("verification cannot be combined with execution arguments")
        verify_precursor_audit(args.repo_root, args.verify)
        print("verified")
    else:
        if not args.config or not args.output_dir:
            parser.error("execution requires --config and --output-dir")
        print(build_precursor_audit(args.repo_root, args.config, args.output_dir)["status"])


if __name__ == "__main__":
    main()
