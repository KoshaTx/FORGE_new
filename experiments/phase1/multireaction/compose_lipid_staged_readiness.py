"""Merge additional fixed and repeated replay evidence into full-universe accounting."""

import argparse
import json
from pathlib import Path

from forge.corpus.compose_lipid_staged_readiness import merge_readiness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = merge_readiness(Path.cwd(), args.config, args.output)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
