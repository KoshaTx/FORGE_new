from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.synthesis.assessment.ugi3_l2_coverage_priority import (
    CATALOG_ABSENT_NOT_ASSESSED,
    ROUTE_COMPLETE,
    UgiL2CoveragePriorityError,
    build_l2_coverage_priority_audit,
    rank_component_priorities,
    route_work_state,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/phase1_ugi3_l2_coverage_priority_audit_v1.json"


def _production_paths(config_path: Path = CONFIG) -> dict[str, Path]:
    config = json.loads(config_path.read_text())
    return {name: REPO / specification["path"] for name, specification in config["inputs"].items()}


def test_route_states_keep_missing_knowledge_distinct_from_closure() -> None:
    assert route_work_state("accepted_terminal") == ROUTE_COMPLETE
    assert route_work_state("not_assessed_missing_route_knowledge") == (CATALOG_ABSENT_NOT_ASSESSED)
    with pytest.raises(UgiL2CoveragePriorityError, match="unsupported static route tier"):
        route_work_state("chemically_incompatible")


def test_component_priority_uses_guaranteed_single_gap_unlocks() -> None:
    common = {
        "structural_provenance_stratum": "graph_absent_from_frozen_424_component_catalog",
        "catalog_provenance_substratum": "graph_absent_from_frozen_catalog",
        "catalog_membership": "false",
        "static_route_readiness_tier": "not_assessed_missing_route_knowledge",
        "static_route_evidence_category": "missing_route_knowledge",
        "route_work_state": CATALOG_ABSENT_NOT_ASSESSED,
        "gap_class": "route_evidence_not_assessed",
    }
    rows = [
        {
            **common,
            "sample_index": "0",
            "role": "amine_head",
            "canonical_component_smiles": "CCN",
        },
        {
            **common,
            "sample_index": "1",
            "role": "amine_head",
            "canonical_component_smiles": "CCN",
        },
        {
            **common,
            "sample_index": "2",
            "role": "isocyanide_tail",
            "canonical_component_smiles": "[C-]#[N+]CC",
        },
    ]
    head_key = "amine_head\tCCN"
    tail_key = "isocyanide_tail\t[C-]#[N+]CC"
    gaps = {
        "0": frozenset({head_key}),
        "1": frozenset({head_key}),
        "2": frozenset({head_key, tail_key}),
    }
    priorities = rank_component_priorities(rows, gaps, minimum_occurrences=2)
    assert priorities[0]["canonical_smiles"] == "CCN"
    assert priorities[0]["single_gap_products_guaranteed_unlocked"] == 2
    assert priorities[0]["chemical_incompatibility_evidenced"] is False


@pytest.mark.skipif(not CONFIG.exists(), reason="production audit config is absent")
def test_production_audit_is_deterministic_and_fail_closed() -> None:
    paths = _production_paths()
    first_result, first_ledgers = build_l2_coverage_priority_audit(CONFIG, paths)
    second_result, second_ledgers = build_l2_coverage_priority_audit(CONFIG, paths)

    assert first_result == second_result
    assert first_ledgers == second_ledgers
    assert first_result["summary"]["registry_components"] == 424
    assert first_result["summary"]["registry_route_complete_components"] == 41
    assert first_result["summary"]["generated_products"] == 1007
    assert first_result["summary"]["baseline_static_route_complete_products"] == 3
    assert first_result["summary"]["chemical_incompatibility_evidenced_components"] == 0
    assert first_result["scope"]["route_planning_performed"] is False
    assert first_result["claims_boundary"]["structural_analogy_is_route_evidence"] is False


@pytest.mark.skipif(not CONFIG.exists(), reason="production audit config is absent")
def test_production_audit_rejects_hash_drift(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    config["inputs"]["component_registry"]["sha256"] = "0" * 64
    bad_config = tmp_path / "bad.json"
    bad_config.write_text(json.dumps(config))
    with pytest.raises(UgiL2CoveragePriorityError, match="component_registry hash mismatch"):
        build_l2_coverage_priority_audit(bad_config, _production_paths())
