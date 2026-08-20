#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.design.audit.ugi_support_bounds_audit import audit_ugi_support_bounds

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/model/phase1_ugi_support_bounds_audit.json",
    )
    args = parser.parse_args()
    result = audit_ugi_support_bounds(args.config, REPO)
    print(
        json.dumps(
            {
                "status": result["status"],
                "exact_ugi": result["exact_ugi"],
                "lnpdb_hydrophobic_component_proxy": result["lnpdb_hydrophobic_component_proxy"],
                "cross_platform_transfer_pilot": result["cross_platform_transfer_pilot"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
