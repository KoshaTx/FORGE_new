"""Build a protected source-component view or replay its frozen repeated reactions."""

import argparse
from pathlib import Path

from forge.corpus.compose_lipid_source_view import build_source_view
from forge.corpus.compose_lipid_supplied_replay import run_supplied_replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("view", "replay"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    run = build_source_view if args.operation == "view" else run_supplied_replay
    result = run(repo, args.config, args.output)
    print(result["status"])


if __name__ == "__main__":
    main()
