from __future__ import annotations

import csv
import gzip
import io
import json
from pathlib import Path

import pytest

from forge.data.r1_prime_audit import sha256_file
from forge.route.ugi3_production_registry_route_readiness import (
    ACCEPTED_TERMINAL,
    EXACT_CLOSED,
    EXACT_OPEN,
    FAMILY_ONLY,
    MISSING_KNOWLEDGE,
    OUTSIDE_SUPPORT,
    PROVENANCE_ONLY,
    ProductionRegistryRouteReadinessError,
    build_production_registry_route_readiness,
    classify_component_evidence,
)

REPO = Path(__file__).resolve().parents[1]


def _production_paths() -> tuple[Path, ...]:
    return (
        REPO / "configs/route/phase1_ugi3_production_registry_route_readiness.json",
        REPO / "results/phase1/ugi_component_expansion/component_registry.csv.gz",
        REPO / "results/phase1/ugi_component_expansion/result.json",
        REPO
        / "results/phase1/ugi3_complete_computational_dossiers/component_dossier_ledger.csv.gz",
        REPO / "results/phase1/ugi3_complete_computational_dossiers/result.json",
        REPO / "results/m0_09/hydrophobic_motif_transfer_ledger.csv.gz",
        REPO / "configs/route/phase1_ugi3_upstream_qualified_reactions_v1.json",
        REPO
        / "configs/route/variants/ugi3_upstream_primary_alcohol_oxidation_exact_source_v1.json",
    )


def _read_rows(payload: bytes) -> list[dict[str, str]]:
    with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as compressed:
        with io.TextIOWrapper(compressed) as text:
            return list(csv.DictReader(text))


def _row(route_states: list[str], *, e2: bool = False) -> dict[str, str]:
    return {
        "component_id": "component-test",
        "route_evidence_states_json": json.dumps(route_states, separators=(",", ":")),
        "procurement_evidence_states_json": "[]",
        "route_closure_states_json": "[]",
        "e2_route_closed": str(e2).lower(),
        "source_classes_json": '["high_similarity_motif"]',
        "source_record_ids_json": '["source-1"]',
        "passes_registry_handle_policy": "true",
    }


def test_frozen_424_component_census_is_exact_and_role_stratified() -> None:
    result, component_bytes, gap_bytes = build_production_registry_route_readiness(
        *_production_paths()
    )

    summary = result["summary"]
    assert summary["components"] == 424
    assert summary["route_complete_components"] == 41
    assert summary["by_evidence_category"] == {
        "accepted_terminal": 17,
        "exact_source_forward_verified_l3_closed": 24,
        "exact_source_l3_open": 1,
        "family_projected_only": 87,
        "handle_qualified_provenance_only": 0,
        "outside_route_support": 111,
        "missing_route_knowledge": 184,
    }
    assert {
        role: values["route_complete_components"] for role, values in summary["by_role"].items()
    } == {
        "amine_head": 17,
        "isocyanide_tail": 6,
        "oxoester_aldehyde_body_tail": 18,
    }
    assert summary["gap_evidence_status_groups"] == {
        "exact_substrate_route_evidence_absent_or_unreported": {
            "components": 198,
            "by_role": {
                "amine_head": 63,
                "isocyanide_tail": 46,
                "oxoester_aldehyde_body_tail": 89,
            },
            "includes_categories": [
                "family_projected_only",
                "outside_route_support",
                "handle_qualified_provenance_only",
            ],
        },
        "route_evidence_not_yet_assessed": {
            "components": 184,
            "by_role": {
                "amine_head": 184,
                "isocyanide_tail": 0,
                "oxoester_aldehyde_body_tail": 0,
            },
            "includes_categories": ["missing_route_knowledge"],
        },
        "exact_source_route_present_but_l3_open": {
            "components": 1,
            "by_role": {
                "amine_head": 0,
                "isocyanide_tail": 1,
                "oxoester_aldehyde_body_tail": 0,
            },
            "includes_categories": ["exact_source_l3_open"],
        },
    }
    assert summary["smallest_recurring_gap_classes"][0] == {
        "priority_rank": 1,
        "role": "isocyanide_tail",
        "gap_class": "upstream_route_not_reported",
        "evidence_category": "outside_route_support",
        "component_count": 10,
    }
    components = _read_rows(component_bytes)
    gaps = _read_rows(gap_bytes)
    assert len(components) == 424
    assert sum(row["route_complete_component"] == "true" for row in components) == 41
    transfer = [row for row in components if row["e2_route_closed_input_flag"] == "true"]
    assert len(transfer) == 1
    assert transfer[0]["evidence_category"] == EXACT_CLOSED
    assert transfer[0]["l2_forward_status"] == "exact_source_product_uniquely_forward_verified"
    assert [int(row["component_count"]) for row in gaps if row["recurring"] == "true"] == [
        10,
        36,
        38,
        51,
        63,
        184,
    ]
    assert result["exact_transfer_verification"]["verification_status"] == (
        "verified_exact_product_unique"
    )
    assert result["claims_boundary"]["structural_l1_admission_is_route_closure"] is False
    assert str(REPO) not in json.dumps(result["inputs"])


def test_production_audit_is_byte_deterministic() -> None:
    first = build_production_registry_route_readiness(*_production_paths())
    second = build_production_registry_route_readiness(*_production_paths())

    assert first == second


def test_frozen_config_hashes_match_all_inputs() -> None:
    config = json.loads(_production_paths()[0].read_text())
    for record in config["inputs"].values():
        assert sha256_file(REPO / record["asset"]) == record["expected_sha256"]


@pytest.mark.parametrize(
    ("states", "expected"),
    [
        ([], PROVENANCE_ONLY),
        (["not_assessed"], MISSING_KNOWLEDGE),
        (["no_upstream_route_reported"], OUTSIDE_SUPPORT),
        (["reaction_family_precedent"], FAMILY_ONLY),
        (["bounded_family_applicability"], FAMILY_ONLY),
        (["exact_source_route"], EXACT_OPEN),
    ],
)
def test_explicit_evidence_states_never_receive_unearned_promotion(
    states: list[str], expected: str
) -> None:
    result = classify_component_evidence(
        _row(states),
        original_component=None,
        exact_transfer_forward_verified=False,
    )

    assert result["evidence_category"] == expected
    assert result["route_complete_component"] == "false"


def test_handle_or_motif_provenance_alone_cannot_become_route_evidence() -> None:
    row = _row([])
    row["passes_registry_handle_policy"] = "true"
    row["source_classes_json"] = '["exact_motif_match","active_lipid_source"]'

    result = classify_component_evidence(
        row,
        original_component=None,
        exact_transfer_forward_verified=False,
    )

    assert result["evidence_category"] == PROVENANCE_ONLY
    assert result["route_complete_component"] == "false"


def test_e2_flag_without_independent_reexecution_fails_closed() -> None:
    row = _row(["exact_source_route"], e2=True)
    row["procurement_evidence_states_json"] = '["current_item_level_procurement_closed"]'
    row["route_closure_states_json"] = '["computationally_complete"]'

    with pytest.raises(
        ProductionRegistryRouteReadinessError,
        match="without independent exact forward verification",
    ):
        classify_component_evidence(
            row,
            original_component=None,
            exact_transfer_forward_verified=False,
        )


def test_original_dossier_evidence_takes_precedence_over_weak_registry_states() -> None:
    result = classify_component_evidence(
        _row(["no_upstream_route_reported", "not_assessed"]),
        original_component={
            "evidence_tier": "accepted_terminal_l3_closed",
            "l2_forward_status": "not_applicable",
            "l3_terminal_status": "closed",
        },
        exact_transfer_forward_verified=False,
    )

    assert result["evidence_category"] == ACCEPTED_TERMINAL
    assert result["route_complete_component"] == "true"


def test_unknown_route_state_fails_closed() -> None:
    with pytest.raises(ProductionRegistryRouteReadinessError, match="unknown route evidence"):
        classify_component_evidence(
            _row(["inferred_from_tanimoto_similarity"]),
            original_component=None,
            exact_transfer_forward_verified=False,
        )
