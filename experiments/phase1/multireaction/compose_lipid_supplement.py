"""Verify the full supplemental target/component join without admitting training rows."""

import argparse
from pathlib import Path

from forge.corpus.compose_lipid_supplement import build_intake


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    result = build_intake(repo, args.config.resolve(), args.output.resolve())
    print(result["status"])


if __name__ == "__main__":
    main()
