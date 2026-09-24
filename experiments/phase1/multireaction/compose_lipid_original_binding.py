"""Bind current eligible recipes to restored original source records."""

import argparse
import json
from pathlib import Path

from forge.corpus.compose_lipid_original_binding import run_original_binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_original_binding(Path.cwd(), args.config, args.output)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
