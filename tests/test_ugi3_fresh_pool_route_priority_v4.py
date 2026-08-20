from __future__ import annotations

import csv
import gzip
import io
from collections import Counter
from pathlib import Path

from forge.data.r1_prime_audit import sha256_bytes, sha256_file
from forge.value.coverage.ugi3_fresh_pool_route_priority import build_fresh_pool_route_priority

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/phase1_ugi3_fresh_pool_route_priority_v4.json"


def _rows(payload: bytes) -> list[dict[str, str]]:
    with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as compressed:
        with io.TextIOWrapper(compressed) as handle:
            return list(csv.DictReader(handle))


def test_v5_priority_refresh_is_deterministic_nonselecting_and_evidence_preserving() -> None:
    first = build_fresh_pool_route_priority(REPO, CONFIG)
    second = build_fresh_pool_route_priority(REPO, CONFIG)

    assert first == second
    result, ledger = first
    assert result["status"] == "complete_nonselecting_one_gap_route_priority"
    assert result["artifacts"]["priority_ledger.csv.gz"]["sha256"] == sha256_bytes(ledger)
    assert result["adjudication"] == {
        "route_planning_performed": False,
        "synthesis_guidance_authorized": False,
        "prospective_candidate_selection_changed": False,
        "targeted_evidence_mining_authorized": True,
    }
    assert result["summary"]["one_gap_products"] == 897
    assert result["summary"]["unique_one_gap_components"] == 401
    assert result["summary"]["unique_one_gap_components_by_outcome"] == {
        "missing_knowledge": 377,
        "outside_support": 24,
    }
    assert result["summary"]["one_gap_products_by_component_outcome"] == {
        "missing_knowledge": 802,
        "outside_support": 95,
    }
    assert result["summary"]["one_gap_products_by_missing_component_role"] == {
        "amine_head": 305,
        "isocyanide_tail": 184,
        "oxoester_aldehyde_body_tail": 408,
    }
    assert result["summary"]["greedy_missing_component_counts_for_coverage"] == {
        "25pct": 16,
        "50pct": 59,
        "75pct": 177,
        "90pct": 297,
    }
    rows = _rows(ledger)
    assert len(rows) == result["summary"]["unique_one_gap_components"]
    assert (
        sum(int(row["one_gap_product_count"]) for row in rows)
        == result["summary"]["one_gap_products"]
    )
    assert {row["assessment_outcome"] for row in rows} <= {
        "missing_knowledge",
        "outside_support",
    }
    assert all(row["route_mining_lane"] != "reaction_family_promoted" for row in rows)
    assert Counter(row["value_source_class"] for row in rows) == {
        "fresh_component_missing_route_knowledge": 227,
        "hybrid_registry_assessment": 39,
        "v3_exact_overlay_or_replay": 135,
    }
    completed_exact_targets = {
        ("oxoester_aldehyde_body_tail", "C#CCCCCCCCCCCCCCC=O"),
        ("oxoester_aldehyde_body_tail", "C#CCCCCCCCCCCCCCCCC=O"),
    }
    assert not completed_exact_targets.intersection(
        (row["role"], row["canonical_smiles"]) for row in rows
    )


def test_v4_priority_config_pins_frozen_v5_inputs() -> None:
    import json

    config = json.loads(CONFIG.read_text())
    for specification in config["inputs"].values():
        assert sha256_file(REPO / specification["path"]) == specification["sha256"]
