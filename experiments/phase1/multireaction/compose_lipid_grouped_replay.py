"""Replay source-qualified transforms on the current protected preparation view."""

import argparse
import json
from pathlib import Path

from forge.corpus.compose_lipid_grouped_replay import run_grouped_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_grouped_replay(Path.cwd(), args.config, args.output)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
