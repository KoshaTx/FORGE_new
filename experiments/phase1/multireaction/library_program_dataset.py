"""Recover multi-step library programs and build partitions preserving frozen identity guards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.corpus.library_program_dataset import build_library_program_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_library_program_dataset(args.repo_root, args.config, args.output_dir)
    print(json.dumps({"status": result["status"], "gates": result["gates"]}, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
