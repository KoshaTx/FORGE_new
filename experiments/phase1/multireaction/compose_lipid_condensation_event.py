"""Build or independently verify a source-qualified single condensation event."""

import argparse
import json
from pathlib import Path

from forge.corpus.compose_lipid_condensation_event import (
    run_condensation_event,
    verify_condensation_event,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.action == "build":
        if args.config is None or args.output is None:
            parser.error("build requires --config and --output")
        result = run_condensation_event(args.repo, args.config, args.output)
    else:
        if args.result is None:
            parser.error("verify requires --result")
        result = verify_condensation_event(args.repo, args.result)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
