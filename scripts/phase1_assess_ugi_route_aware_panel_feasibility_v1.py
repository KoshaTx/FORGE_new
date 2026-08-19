#!/usr/bin/env python3
"""Run the frozen Ugi route-aware prospective-panel feasibility census."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.product.ugi_route_aware_panel_feasibility import (
    run_route_aware_panel_feasibility,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model/phase1_ugi_route_aware_panel_feasibility_v1.json"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("results/phase1/.ugi_route_aware_panel_feasibility_v1.cache"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1/ugi_route_aware_panel_feasibility_v1"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    result = run_route_aware_panel_feasibility(
        repo,
        (repo / args.config).resolve() if not args.config.is_absolute() else args.config,
        (
            (repo / args.cache_root).resolve()
            if not args.cache_root.is_absolute()
            else args.cache_root
        ),
        (
            (repo / args.output_dir).resolve()
            if not args.output_dir.is_absolute()
            else args.output_dir
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
