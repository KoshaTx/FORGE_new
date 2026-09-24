"""Qualify the existing constitutional Kekule encoding on protected COMPOSE TRAIN."""

import argparse
from pathlib import Path

from forge.corpus.compose_lipid_representation import run_representation_check


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(run_representation_check(args.repo_root, args.config, args.output_dir)["status"])


if __name__ == "__main__":
    main()
