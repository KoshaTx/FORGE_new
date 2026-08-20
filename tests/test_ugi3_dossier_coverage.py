from __future__ import annotations

import csv
import gzip
import io
import json
from pathlib import Path

from experiments.archive.phase1.synthesis_audits.ugi3_dossier_coverage import (
    build_ugi3_dossier_coverage,
)
from forge.corpus.r1_prime_audit import sha256_file

REPO = Path(__file__).resolve().parents[1]


def test_frozen_dossier_coverage_is_conservative_and_reproducible() -> None:
    result, component_bytes, product_bytes = build_ugi3_dossier_coverage(
        REPO / "configs/route/phase1_ugi3_dossier_coverage.json",
        REPO / "results/m0_09/agile_virtual_ugi3_component_program_ledger.csv.gz",
        REPO / "results/m0_09/agile_virtual_ugi3_product_ledger.csv.gz",
        REPO / "configs/route/m0_09_ugi3_virtual_terminal_procurement.json",
    )
    assert result["summary"] == {
        "components": 93,
        "components_by_status": {
            "accepted_terminal": 17,
            "exact_source_leaf_closed": 23,
            "family_projected_leaf_closed": 47,
            "incomplete": 6,
        },
        "products": 12276,
        "products_by_upstream_tier": {
            "exact_source_upstream_closed": 1734,
            "family_supported_upstream_closed": 6698,
            "incomplete": 3844,
        },
        "complete_forward_verified_dossiers": 0,
    }
    assert result["claims_boundary"]["complete_forward_verified_dossier_claimed"] is False
    component_rows = _read_rows(component_bytes)
    product_rows = _read_rows(product_bytes)
    assert len(component_rows) == 93
    assert len(product_rows) == 12276
    assert not any(row["complete_forward_verified_dossier"] == "true" for row in product_rows)


def _read_rows(payload: bytes) -> list[dict[str, str]]:
    with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as compressed:
        with io.TextIOWrapper(compressed) as text:
            return list(csv.DictReader(text))


def test_config_hashes_match_frozen_inputs() -> None:
    config = json.loads((REPO / "configs/route/phase1_ugi3_dossier_coverage.json").read_text())
    for record in config["inputs"].values():
        assert sha256_file(REPO / record["asset"]) == record["expected_sha256"]
