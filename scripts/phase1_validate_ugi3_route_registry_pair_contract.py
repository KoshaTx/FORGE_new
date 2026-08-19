#!/usr/bin/env python3
"""Validate the prereveal paired route-registry contract without reading molecules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.route.ugi3_route_registry_pair_contract import (
    validate_binding,
    validate_protocol,
)

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO / "configs/route/phase1_ugi3_route_registry_pair_protocol_v1.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        help="Optional immutable R0/without-C18 and R1/with-C18 binding to validate.",
    )
    args = parser.parse_args()
    result = (
        validate_binding(REPO, args.protocol, args.binding)
        if args.binding is not None
        else validate_protocol(REPO, args.protocol)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
