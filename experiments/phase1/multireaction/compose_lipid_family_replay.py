"""Run supplied-component checks for all currently qualified reaction programs."""

import argparse
from pathlib import Path

from forge.corpus.compose_lipid_family_replay import run_family_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_family_replay(Path(__file__).resolve().parents[3], args.config, args.output)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
