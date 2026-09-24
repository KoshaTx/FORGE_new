"""Append B5 profile-qualified evidence without changing full-universe protections."""

import argparse
import json
from pathlib import Path

from forge.corpus.compose_lipid_b5_readiness import merge_b5_readiness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = merge_b5_readiness(Path(__file__).resolve().parents[3], args.config, args.output)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
