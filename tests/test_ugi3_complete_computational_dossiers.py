from __future__ import annotations

import csv
import gzip
import io
import json
from pathlib import Path

from forge.data.r1_prime_audit import sha256_file
from forge.route.evidence.ugi3_complete_computational_dossiers import (
    EXACT_CLOSED,
    FAMILY_CLOSED,
    build_complete_computational_dossiers,
    classify_component_program,
)

REPO = Path(__file__).resolve().parents[1]


def _read_rows(payload: bytes) -> list[dict[str, str]]:
    with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as compressed:
        with io.TextIOWrapper(compressed) as text:
            return list(csv.DictReader(text))


def _production_paths() -> tuple[Path, ...]:
    return (
        REPO / "configs/route/phase1_ugi3_complete_computational_dossiers.json",
        REPO / "results/m0_09/agile_virtual_ugi3_component_program_ledger.csv.gz",
        REPO / "results/m0_09/agile_virtual_ugi3_product_ledger.csv.gz",
        REPO / "results/phase1/ugi3_dossier_coverage/component_dossier_ledger.csv.gz",
        REPO / "results/phase1/ugi3_dossier_coverage/product_dossier_ledger.csv.gz",
        REPO / "results/phase1/ugi3_dossier_coverage/result.json",
        REPO
        / "results/phase1/ugi3_exact_source_forward_verification/step_verification_ledger.csv.gz",
        REPO / "results/phase1/ugi3_exact_source_forward_verification/result.json",
        REPO / "configs/route/m0_09_ugi3_virtual_terminal_procurement.json",
    )


def test_frozen_complete_dossier_census_is_exact_and_conservative() -> None:
    result, component_bytes, product_bytes = build_complete_computational_dossiers(
        *_production_paths()
    )

    assert result["summary"] == {
        "components": 93,
        "components_by_evidence_tier": {
            "accepted_terminal_l3_closed": 17,
            "exact_source_l2_verified_l3_closed": 23,
            "exact_source_l2_verified_l3_open": 1,
            "family_projected_l3_closed_not_exact_source": 47,
            "unresolved_component": 5,
        },
        "products": 12276,
        "products_by_dossier_tier": {
            "complete_exact_source_computational_dossier": 1734,
            "family_projected_support_not_exact_source": 6698,
            "incomplete_computational_dossier": 3844,
        },
        "complete_exact_source_computational_dossiers": 1734,
    }
    assert str(REPO) not in json.dumps(result["inputs"])
    assert (
        result["claims_boundary"][
            "complete_computational_dossier_is_observed_exact_product_synthesis"
        ]
        is False
    )
    components = _read_rows(component_bytes)
    products = _read_rows(product_bytes)
    assert len(components) == 93
    assert len(products) == 12276
    complete = [
        row for row in products if row["complete_exact_source_computational_dossier"] == "true"
    ]
    assert len(complete) == 1734
    assert all(
        row["observed_exact_product_synthesis_status"] == "not_assessed_by_this_gate"
        and row["experimental_success_status"] == "not_assessed_by_this_gate"
        for row in complete
    )


def test_frozen_config_hashes_match_inputs() -> None:
    config = json.loads(_production_paths()[0].read_text())
    for record in config["inputs"].values():
        assert sha256_file(REPO / record["asset"]) == record["expected_sha256"]


def _program(*, status: str = "exact_source_program") -> dict[str, str]:
    return {
        "component_id": "component-1",
        "role": "isocyanide_tail",
        "canonical_smiles": "CCN#[C]",
        "program_status": status,
        "program_steps_json": json.dumps(
            [
                {
                    "step_index": 1,
                    "transformation": "step_one",
                    "reactants": ["CCN"],
                    "product": "CCNC=O",
                },
                {
                    "step_index": 2,
                    "transformation": "step_two",
                    "reactants": ["CCNC=O"],
                    "product": "CCN#[C]",
                },
            ],
            separators=(",", ":"),
        ),
        "proposed_leaf_candidates_json": '["CCN"]',
    }


def _step(index: int) -> dict[str, str]:
    values = {
        1: ("step_one", '["CCN"]', "CCNC=O"),
        2: ("step_two", '["CCNC=O"]', "CCN#[C]"),
    }
    transformation, reactants, product = values[index]
    return {
        "component_id": "component-1",
        "component_role": "isocyanide_tail",
        "component_smiles": "CCN#[C]",
        "step_index": str(index),
        "transformation": transformation,
        "reactants_json": reactants,
        "expected_product": product,
        "verification_status": "verified_exact_product_unique",
        "forward_product_count": "1",
        "expected_product_in_outputs": "true",
    }


def test_missing_or_ambiguous_step_fails_closed() -> None:
    missing = classify_component_program(_program(), [_step(1)], {"CCN"})
    ambiguous = classify_component_program(
        _program(),
        [_step(1), _step(2), _step(2)],
        {"CCN"},
    )

    assert missing["evidence_tier"] == "exact_source_l2_missing_step"
    assert missing["exact_source_computational_component"] == "false"
    assert ambiguous["evidence_tier"] == "exact_source_l2_ambiguous_step"
    assert ambiguous["exact_source_computational_component"] == "false"


def test_open_terminal_leaf_fails_l3_closure() -> None:
    result = classify_component_program(_program(), [_step(1), _step(2)], set())

    assert result["evidence_tier"] == "exact_source_l2_verified_l3_open"
    assert result["l2_forward_status"] == "all_steps_uniquely_verified"
    assert result["l3_terminal_status"] == "open"
    assert result["exact_source_computational_component"] == "false"


def test_family_projection_cannot_leak_into_exact_source_tier() -> None:
    result = classify_component_program(
        _program(status="reaction_family_projected_program"),
        [_step(1), _step(2)],
        {"CCN"},
    )

    assert result["evidence_tier"] == FAMILY_CLOSED
    assert result["evidence_tier"] != EXACT_CLOSED
    assert result["family_projected_only"] == "true"
    assert result["exact_source_computational_component"] == "false"
