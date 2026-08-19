from __future__ import annotations

import csv
import gzip
import io
import json
from pathlib import Path

import pytest

from forge.data.r1_prime_audit import sha256_file
from forge.route.ugi3_exact_source_forward_verification import (
    AMBIGUOUS_MATCH_STATUS,
    EXPECTED_ABSENT_STATUS,
    INVALID_INPUT_STATUS,
    VERIFIED_STATUS,
    Ugi3ExactSourceForwardVerificationError,
    build_exact_source_forward_verification,
)

REPO = Path(__file__).resolve().parents[1]


def _read_rows(payload: bytes) -> list[dict[str, str]]:
    with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as compressed:
        with io.TextIOWrapper(compressed) as text:
            return list(csv.DictReader(text))


def test_frozen_exact_source_gate_uniquely_reconstructs_all_source_steps() -> None:
    result, ledger_bytes = build_exact_source_forward_verification(
        REPO / "configs/route/phase1_ugi3_exact_source_forward_verification.json",
        REPO / "results/m0_09/agile_virtual_ugi3_component_program_ledger.csv.gz",
        REPO / "results/m0_09/agile_component_routes.json",
        REPO / "configs/route/phase1_ugi3_upstream_qualified_reactions_v1.json",
        REPO / "data/vendor/qualified_reaction_families_v1.json",
    )

    assert result["summary"]["exact_source_component_programs"] == 24
    assert result["summary"]["upstream_steps"] == 46
    assert result["summary"]["steps_by_verification_status"] == {
        "verified_exact_product_unique": 46
    }
    assert result["summary"]["fully_forward_verified_component_programs"] == 24
    assert len(result["registry_inventory"]["bound_upstream_transformations"]) == 4
    assert result["claims_boundary"]["complete_forward_verified_dossier_claimed"] is False
    assert str(REPO) not in json.dumps(result["inputs"])
    rows = _read_rows(ledger_bytes)
    assert len(rows) == 46
    assert all(row["expected_product_in_outputs"] == "true" for row in rows)
    assert all(int(row["forward_product_count"]) == 1 for row in rows)
    assert all(row["qualified_reaction_id"] for row in rows)


def test_frozen_config_hashes_match_all_inputs() -> None:
    config = json.loads(
        (REPO / "configs/route/phase1_ugi3_exact_source_forward_verification.json").read_text()
    )
    for record in config["inputs"].values():
        assert sha256_file(REPO / record["asset"]) == record["expected_sha256"]


def _write_fixture(
    tmp_path: Path,
    *,
    expected_product: str,
    expected_status: str | None = None,
    reactant_smiles: str = "CC=O",
    reaction_smarts: str = "[CX3:1]=[OX1:2]>>[CX4:1][OX2:2]",
) -> tuple[Path, Path, Path, Path, Path]:
    component_ledger = tmp_path / "component_programs.csv.gz"
    fields = (
        "component_id",
        "role",
        "canonical_smiles",
        "program_status",
        "program_family",
        "exact_source_route_ids_json",
        "program_steps_json",
    )
    row = {
        "component_id": "component-1",
        "role": "oxoester_aldehyde_body_tail",
        "canonical_smiles": expected_product,
        "program_status": "exact_source_program",
        "program_family": "test_reduction",
        "exact_source_route_ids_json": '["route-1"]',
        "program_steps_json": json.dumps(
            [
                {
                    "step_index": 1,
                    "transformation": "test_carbonyl_reduction",
                    "reactants": [reactant_smiles],
                    "product": expected_product,
                }
            ],
            separators=(",", ":"),
        ),
    }
    with gzip.open(component_ledger, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)

    source_routes = tmp_path / "source_routes.json"
    source_routes.write_text(
        json.dumps(
            {
                "schema_version": "m0_09_agile_component_routes.v1",
                "routes": [
                    {
                        "route_id": "route-1",
                        "target": {"canonical_smiles": expected_product},
                        "steps": [
                            {
                                "step_index": 1,
                                "transformation": "test_carbonyl_reduction",
                                "reactants": [{"canonical_smiles": reactant_smiles}],
                                "product": {"canonical_smiles": expected_product},
                            }
                        ],
                    }
                ],
            }
        )
    )
    executable_registry = tmp_path / "qualified.json"
    executable_registry.write_text(
        json.dumps(
            {
                "reactions": [
                    {
                        "reaction_id": "test_carbonyl_reduction",
                        "reactant_roles": [{"name": "carbonyl"}],
                        "atom_mapped_reaction_smarts": reaction_smarts,
                    }
                ]
            }
        )
    )
    family_registry = tmp_path / "families.json"
    family_registry.write_text(json.dumps({"reactions": []}))
    variant = tmp_path / "variant.json"
    variant.write_text(
        json.dumps(
            {
                "variant": "test_carbonyl_reduction",
                "reaction_id": "test_carbonyl_reduction",
                "expected_reactant_count": 1,
                "roles": [{"name": "carbonyl"}],
            }
        )
    )
    config = tmp_path / "config.json"
    if expected_status is None:
        expected_status = VERIFIED_STATUS if expected_product == "CCO" else EXPECTED_ABSENT_STATUS
    config.write_text(
        json.dumps(
            {
                "schema_version": ("phase1_ugi3_exact_source_forward_verification_config.v1"),
                "task": "fixture",
                "inputs": {
                    "component_program_ledger": {"expected_sha256": sha256_file(component_ledger)},
                    "source_routes": {"expected_sha256": sha256_file(source_routes)},
                    "executable_registry": {"expected_sha256": sha256_file(executable_registry)},
                    "family_registry": {"expected_sha256": sha256_file(family_registry)},
                },
                "policy": {
                    "max_products_per_step": 10,
                    "isomeric_smiles": True,
                },
                "qualified_transform_bindings": {
                    "test_carbonyl_reduction": {
                        "reaction_id": "test_carbonyl_reduction",
                        "variant_asset": str(variant),
                        "variant_expected_sha256": sha256_file(variant),
                        "reactant_order": [0],
                    }
                },
                "expected_counts": {
                    "exact_source_component_programs": 1,
                    "unique_source_routes": 1,
                    "upstream_steps": 1,
                    "forward_verified_steps": (1 if expected_status == VERIFIED_STATUS else 0),
                    "steps_by_transformation": {"test_carbonyl_reduction": 1},
                    "steps_by_verification_status": {expected_status: 1},
                    "component_programs_by_family": {"test_reduction": 1},
                    "component_programs_by_role": {"oxoester_aldehyde_body_tail": 1},
                    "fully_forward_verified_component_programs": (
                        1 if expected_status == VERIFIED_STATUS else 0
                    ),
                },
            }
        )
    )
    return config, component_ledger, source_routes, executable_registry, family_registry


def test_qualified_binding_executes_and_verifies_exact_product(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path, expected_product="CCO")
    result, ledger_bytes = build_exact_source_forward_verification(*paths)

    assert result["summary"]["fully_forward_verified_component_programs"] == 1
    row = _read_rows(ledger_bytes)[0]
    assert row["verification_status"] == VERIFIED_STATUS
    assert row["forward_products_json"] == '["CCO"]'
    assert row["expected_product_in_outputs"] == "true"


def test_qualified_binding_records_exact_product_miss(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path, expected_product="CCN")
    result, ledger_bytes = build_exact_source_forward_verification(*paths)

    assert result["summary"]["fully_forward_verified_component_programs"] == 0
    row = _read_rows(ledger_bytes)[0]
    assert row["verification_status"] == EXPECTED_ABSENT_STATUS
    assert row["forward_products_json"] == '["CCO"]'
    assert row["expected_product_in_outputs"] == "false"


def test_qualified_binding_records_invalid_step_input(tmp_path: Path) -> None:
    paths = _write_fixture(
        tmp_path,
        expected_product="not_smiles",
        expected_status=INVALID_INPUT_STATUS,
    )
    result, ledger_bytes = build_exact_source_forward_verification(*paths)

    assert result["summary"]["forward_verified_steps"] == 0
    row = _read_rows(ledger_bytes)[0]
    assert row["verification_status"] == INVALID_INPUT_STATUS
    assert "invalid SMILES" in row["reason"]


def test_qualified_binding_records_multiproduct_ambiguity(tmp_path: Path) -> None:
    paths = _write_fixture(
        tmp_path,
        reactant_smiles="CC(CO)CCO",
        reaction_smarts="[CH2:1][OX2H1:2]>>[CH1:1]=[OX1:2]",
        expected_product="CC(CO)CC=O",
        expected_status=AMBIGUOUS_MATCH_STATUS,
    )
    result, ledger_bytes = build_exact_source_forward_verification(*paths)

    assert result["summary"]["forward_verified_steps"] == 0
    row = _read_rows(ledger_bytes)[0]
    assert row["verification_status"] == AMBIGUOUS_MATCH_STATUS
    assert int(row["forward_product_count"]) == 2
    assert row["expected_product_in_outputs"] == "true"


def test_gate_fails_on_source_program_mismatch(tmp_path: Path) -> None:
    paths = list(_write_fixture(tmp_path, expected_product="CCO"))
    source_routes = paths[2]
    artifact = json.loads(source_routes.read_text())
    artifact["routes"][0]["steps"][0]["transformation"] = "different_transform"
    source_routes.write_text(json.dumps(artifact))
    config = json.loads(paths[0].read_text())
    config["inputs"]["source_routes"]["expected_sha256"] = sha256_file(source_routes)
    paths[0].write_text(json.dumps(config))

    with pytest.raises(
        Ugi3ExactSourceForwardVerificationError,
        match="does not exactly reproduce source route",
    ):
        build_exact_source_forward_verification(*paths)
