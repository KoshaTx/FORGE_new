#!/usr/bin/env python3
"""Run the optimistic AGILE-template saturation stress test."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.route.ugi3_agile_template_saturation_stress import (
    build_agile_template_saturation_stress,
)

REPO = Path(__file__).resolve().parents[1]


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/route/phase1_ugi3_agile_template_saturation_stress_v1.json",
    )
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else REPO / args.config
    result = build_agile_template_saturation_stress(REPO, config)
    output = REPO / "results/phase1/ugi3_agile_template_saturation_stress_v1/result.json"
    _atomic_write(output, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    print(
        json.dumps(
            {
                "component_census": result["component_census"],
                "product_census": result["product_census"],
                "counterfactual": result["counterfactual"],
                "decision": result["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
