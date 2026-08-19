#!/usr/bin/env python3
"""Audit exact supplier terminals for high-leverage generated Ugi heads."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.route.ugi3_high_leverage_head_terminals import (
    build_high_leverage_head_terminal_audit,
)

REPO = Path(__file__).resolve().parents[1]
INPUT_PATHS = {
    "audit_source": REPO / "src/forge/route/ugi3_high_leverage_head_terminals.py",
    "evidence_pack": REPO / "configs/route/phase1_ugi3_high_leverage_head_terminals_v1.json",
    "prior_product_impact": REPO
    / "results/phase1/ugi3_targeted_role_gap_evidence_audit_v1/product_impact_ledger.csv.gz",
    "prior_result": REPO / "results/phase1/ugi3_targeted_role_gap_evidence_audit_v1/result.json",
    "qualifier_source": REPO / "scripts/phase1_audit_ugi3_high_leverage_head_terminals.py",
    "readiness_ledger": REPO
    / "results/phase1/ugi3_production_registry_route_readiness/component_readiness_ledger.csv.gz",
}


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/route/phase1_ugi3_high_leverage_head_terminal_audit_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/phase1/ugi3_high_leverage_head_terminal_audit_v1",
    )
    args = parser.parse_args()
    result, ledger = build_high_leverage_head_terminal_audit(
        config_path=args.config,
        input_paths=INPUT_PATHS,
    )
    _write_atomic(args.output_dir / "product_impact_ledger.csv.gz", ledger)
    _write_atomic(
        args.output_dir / "result.json",
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
