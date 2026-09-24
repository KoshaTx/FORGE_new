"""CLI for qualifying twelve-library atom semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.corpus.combinatorial_program_semantics import (
    qualify_combinatorial_program_semantics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = qualify_combinatorial_program_semantics(args.repo_root, args.config, args.output_dir)
    print(json.dumps({"status": result["status"], "gates": result["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
