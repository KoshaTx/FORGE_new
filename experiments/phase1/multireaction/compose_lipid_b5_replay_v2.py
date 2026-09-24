"""Run protected B5 original-task/source-profile replay."""

import argparse
import json
from pathlib import Path

from forge.corpus.compose_lipid_b5_replay_v2 import run_b5_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    result = run_b5_replay(root, args.config, args.output)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
