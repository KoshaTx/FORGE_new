#!/usr/bin/env python3
"""Build versioned Ugi synthesis values after the second exact-terminal wave."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.value.ugi3_synthesis_value_audit_v2 import (
    build_ugi3_synthesis_value_audit_v2,
)

REPO = Path(__file__).resolve().parents[1]

INPUT_PATHS = {
    "audit_source": REPO / "src/forge/value/ugi3_synthesis_value_audit_v2.py",
    "base_component_values": REPO
    / "results/phase1/ugi3_synthesis_value_audit_v1/component_synthesis_values.json.gz",
    "base_config": REPO / "configs/route/phase1_ugi3_synthesis_value_audit_v1.json",
    "base_product_values": REPO
    / "results/phase1/ugi3_synthesis_value_audit_v1/product_synthesis_values.json.gz",
    "base_result": REPO / "results/phase1/ugi3_synthesis_value_audit_v1/result.json",
    "generated_component_provenance": REPO
    / "results/phase1/ugi_postselection_provenance_audit_v1/component_provenance_ledger.csv.gz",
    "generated_product_provenance": REPO
    / "results/phase1/ugi_postselection_provenance_audit_v1/product_provenance_ledger.csv.gz",
    "qualifier_source": REPO / "scripts/phase1_build_ugi3_synthesis_values_v2.py",
    "second_wave_config": REPO
    / "configs/route/phase1_ugi3_second_wave_head_terminal_audit_v1.json",
    "second_wave_product_impact": REPO
    / "results/phase1/ugi3_second_wave_head_terminal_audit_v1/product_impact_ledger.csv.gz",
    "second_wave_result": REPO
    / "results/phase1/ugi3_second_wave_head_terminal_audit_v1/result.json",
    "value_source": REPO / "src/forge/value/synthesis.py",
}

SECOND_WAVE_INPUT_PATHS = {
    "audit_source": REPO / "src/forge/route/ugi3_second_wave_head_terminals.py",
    "evidence_pack": REPO / "configs/route/phase1_ugi3_second_wave_head_terminals_v1.json",
    "prior_product_impact": REPO
    / "results/phase1/ugi3_high_leverage_head_terminal_audit_v1/product_impact_ledger.csv.gz",
    "prior_result": REPO / "results/phase1/ugi3_high_leverage_head_terminal_audit_v1/result.json",
    "qualifier_source": REPO / "scripts/phase1_audit_ugi3_second_wave_head_terminals.py",
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
        default=REPO / "configs/route/phase1_ugi3_synthesis_value_audit_v2.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/phase1/ugi3_synthesis_value_audit_v2",
    )
    args = parser.parse_args()
    result, component_ledger, product_ledger = build_ugi3_synthesis_value_audit_v2(
        config_path=args.config,
        input_paths=INPUT_PATHS,
        second_wave_input_paths=SECOND_WAVE_INPUT_PATHS,
    )
    _write_atomic(args.output_dir / "component_synthesis_values.json.gz", component_ledger)
    _write_atomic(args.output_dir / "product_synthesis_values.json.gz", product_ledger)
    _write_atomic(
        args.output_dir / "result.json",
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
