#!/usr/bin/env python3
"""Qualify bounded fail-closed hybrid evidence search for Ugi components."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.synthesis.engine.ugi3_hybrid_search import build_hybrid_search_diagnostic

REPO = Path(__file__).resolve().parents[3]

INPUT_PATHS = {
    "component_dossier": REPO
    / "results/phase1/ugi3_complete_computational_dossiers/component_dossier_ledger.csv.gz",
    "component_dossier_result": REPO
    / "results/phase1/ugi3_complete_computational_dossiers/result.json",
    "component_program": REPO / "results/m0_09/agile_virtual_ugi3_component_program_ledger.csv.gz",
    "component_program_result": REPO / "results/m0_09/agile_virtual_ugi3_component_programs.json",
    "exact_adapter_source": REPO / "src/forge/route/ugi3_exact_evidence_source.py",
    "exact_evidence_ledger": REPO
    / "results/phase1/ugi3_exact_evidence_source/assessment_ledger.json.gz",
    "exact_evidence_result": REPO / "results/phase1/ugi3_exact_evidence_source/result.json",
    "hybrid_adapter_source": REPO / "src/forge/route/ugi3_hybrid_search.py",
    "l1_variant": REPO / "configs/assembly/ugi_variant.yaml",
    "oxidation_variant": REPO
    / "configs/route/variants/ugi3_upstream_primary_alcohol_oxidation_exact_source_v1.json",
    "planner_cache_source": REPO / "src/forge/route/planner_cache.py",
    "planner_contract_source": REPO / "src/forge/route/planner.py",
    "qualified_forward_source": REPO / "src/forge/route/qualified_forward.py",
    "readiness_config": REPO / "configs/route/phase1_ugi3_production_registry_route_readiness.json",
    "readiness_ledger": REPO
    / "results/phase1/ugi3_production_registry_route_readiness/component_readiness_ledger.csv.gz",
    "readiness_result": REPO
    / "results/phase1/ugi3_production_registry_route_readiness/result.json",
    "step_ledger": REPO
    / "results/phase1/ugi3_exact_source_forward_verification/step_verification_ledger.csv.gz",
    "step_result": REPO / "results/phase1/ugi3_exact_source_forward_verification/result.json",
    "terminal_procurement": REPO / "configs/route/m0_09_ugi3_virtual_terminal_procurement.json",
    "transfer_config": REPO / "configs/route/m0_09_hydrophobic_motif_transfer.json",
    "transfer_ledger": REPO / "results/m0_09/hydrophobic_motif_transfer_ledger.csv.gz",
    "transfer_result": REPO / "results/m0_09/hydrophobic_motif_transfer.json",
    "upstream_registry": REPO / "configs/route/phase1_ugi3_upstream_qualified_reactions_v1.json",
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
        default=REPO / "configs/route/phase1_ugi3_hybrid_search.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/phase1/ugi3_hybrid_search",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="forge-ugi3-hybrid-search-cache-") as cache_dir:
        result, ledger = build_hybrid_search_diagnostic(
            config_path=args.config,
            input_paths=INPUT_PATHS,
            cache_root=Path(cache_dir),
        )
    _write_atomic(args.output_dir / "assessment_ledger.json.gz", ledger)
    _write_atomic(
        args.output_dir / "result.json",
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
