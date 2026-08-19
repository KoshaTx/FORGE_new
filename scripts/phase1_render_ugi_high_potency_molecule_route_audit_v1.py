#!/usr/bin/env python3
"""Render the frozen conservative-high Ugi molecule/route-readiness audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.route.ugi_high_potency_molecule_audit import (
    build_audit,
    render_pdf,
    write_audit_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/route/phase1_ugi_high_potency_molecule_route_audit_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results/phase1/ugi_high_potency_molecule_route_audit_v1",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=REPO_ROOT / "output/pdf/FORGE_high_potency_route_molecule_audit.pdf",
    )
    arguments = parser.parse_args()
    result, selected = build_audit(config_path=arguments.config, repo_root=REPO_ROOT)
    write_audit_artifacts(result=result, selected=selected, output_dir=arguments.output_dir)
    render_pdf(result=result, selected=selected, output_path=arguments.output_pdf)
    print(
        json.dumps(
            {
                "result": str(arguments.output_dir / "result.json"),
                "pdf": str(arguments.output_pdf),
                "summary": result["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
