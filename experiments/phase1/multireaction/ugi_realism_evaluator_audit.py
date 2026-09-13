"""CLI for the bounded train-only realism evaluator diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path

from forge.core.io import stable_json
from forge.model.ugi_realism_evaluator_audit import run_ugi_realism_evaluator_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_ugi_realism_evaluator_audit(args.repo_root, args.config, args.output_dir)
    print(
        stable_json(
            {
                "status": result["status"],
                "output_dir": str(args.output_dir),
                "descriptor_collision_summary": result["descriptor_collision_summary"],
            }
        )
    )


if __name__ == "__main__":
    main()
