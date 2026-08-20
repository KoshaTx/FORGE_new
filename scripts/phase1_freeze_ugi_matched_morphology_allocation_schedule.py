#!/usr/bin/env python3
"""Freeze the three-arm matched morphology-allocation schedule."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from forge.design.schedule.ugi_matched_morphology_allocation_schedule import (
    build_matched_morphology_allocation_schedule,
)

REPO = Path(__file__).resolve().parents[1]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model/phase1_ugi_matched_morphology_allocation_schedule_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1/ugi_matched_morphology_allocation_schedule_v1"),
    )
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else REPO / args.config
    output = args.output_dir if args.output_dir.is_absolute() else REPO / args.output_dir
    result, schedule = build_matched_morphology_allocation_schedule(REPO, config)
    _atomic_json(output / "schedule.json", schedule)
    _atomic_json(output / "result.json", result)
    print(json.dumps(result["draw_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
