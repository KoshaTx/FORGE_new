#!/usr/bin/env python3
"""Qualify the exact C18 alkynyl-aldehyde route through current L3 evidence."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.route.evidence.ugi3_exact_c18_route import build_exact_c18_route_audit

REPO = Path(__file__).resolve().parents[1]

INPUT_PATHS = {
    "audit_source": REPO / "src/forge/route/ugi3_exact_c18_route.py",
    "audit_runner": REPO / "scripts/phase1_qualify_ugi3_exact_c18_route.py",
    "audit_tests": REPO / "tests/test_ugi3_exact_c18_route.py",
    "qualified_forward_source": REPO / "src/forge/route/qualified_forward.py",
    "planner_source": REPO / "src/forge/route/planner.py",
    "synthesis_value_source": REPO / "src/forge/value/synthesis.py",
    "reaction_registry": REPO / "configs/route/phase1_ugi3_exact_c18_qualified_reactions_v1.json",
    "reduction_variant": REPO
    / "configs/route/variants/ugi3_exact_c18_carboxylic_acid_reduction_v1.json",
    "zipper_variant": REPO / "configs/route/variants/ugi3_exact_c18_alkyne_zipper_v1.json",
    "oxidation_variant": REPO
    / "configs/route/variants/ugi3_exact_c18_primary_alcohol_oxidation_v1.json",
    "amara_primary_article": REPO
    / "data/source_cache/phase1_ugi3_exact_alkynyl_aldehydes/amara_cell_chem_biol_2019.pdf",
    "oppolzer_primary_si": REPO
    / "data/source_cache/phase1_ugi3_exact_alkynyl_aldehydes/oppolzer_jo000463n_si_001.pdf",
    "oppolzer_article_transcription": REPO
    / "data/source_cache/phase1_ugi3_exact_alkynyl_aldehydes/oppolzer_article_transcription_lookchem_2026-08-03.html",
    "terminal_procurement_observation": REPO
    / "data/source_cache/phase1_ugi3_exact_alkynyl_aldehydes/mce_hy_w341997_product_observation_2026-08-03.json",
    "terminal_identity_datasheet": REPO
    / "data/source_cache/phase1_ugi3_exact_alkynyl_aldehydes/mce_hy_w341997_datasheet_2026-08-03.pdf",
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
        default=REPO / "configs/route/phase1_ugi3_exact_c18_route_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/phase1/ugi3_exact_c18_route_v1",
    )
    args = parser.parse_args()
    result, steps, assessment = build_exact_c18_route_audit(
        config_path=args.config,
        input_paths=INPUT_PATHS,
    )
    _write_atomic(args.output_dir / "step_verification_ledger.json.gz", steps)
    _write_atomic(args.output_dir / "assessment.json.gz", assessment)
    _write_atomic(
        args.output_dir / "result.json",
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
