"""Run a local, train-only audit of measured Ugi admission and reference coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.model.ugi_realism_support_audit import run_ugi_realism_support_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_ugi_realism_support_audit(args.repo_root, args.config, args.output_dir)
    print(
        json.dumps(
            {
                "status": result["status"],
                "support": {
                    key: result["support"][key]
                    for key in (
                        "measured_training_products",
                        "admitted_products",
                        "excluded_products",
                    )
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
