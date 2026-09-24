"""Qualify and replay the supplied Ugi-3 role namespace using exact precursor witnesses."""

import argparse
from pathlib import Path

from forge.corpus.compose_lipid_role_binding import run_role_binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_role_binding(Path(__file__).resolve().parents[3], args.config, args.output)
    print(result["summary"], flush=True)


if __name__ == "__main__":
    main()
