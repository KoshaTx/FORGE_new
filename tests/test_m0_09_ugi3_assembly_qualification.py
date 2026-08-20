from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from rdkit import Chem

from forge.chemistry import audit_reactive_site_multiplicity
from forge.route.evidence.ugi3_assembly_qualification import (
    Ugi3AssemblyQualificationError,
    build_ugi3_assembly_qualification,
    write_ugi3_assembly_qualification,
)

REPO = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True))


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    agile = tmp_path / "agile.csv"
    agile.write_text(
        "id,label,combined_mol_SMILES,A_smiles,B_smiles,C_smiles\n"
        "0,A5B1C1,"
        "CCCCCCCCCCCCNC(=O)C(CCCCCOC(=O)CCCCCCCC)NCCN(CCN)CCN,"
        "NCCN(CCN)CCN,"
        "O=C(CCCCCCCC)OCCCCCC=O,"
        "CCCCCCCCCCCC[N+]#[C-]\n"
    )
    registry = tmp_path / "registry.json"
    _write_json(
        registry,
        {
            "reactions": [
                {
                    "reaction_id": "ugi_3cr_agile",
                    "reactant_roles": [
                        {
                            "name": "amine_head",
                            "required_handle_smarts": "[NX3;H2,H1]",
                            "forbidden_smarts": [],
                            "allowed_site_multiplicity": [1, 2],
                        },
                        {"name": "oxoester_aldehyde_body_tail"},
                        {"name": "isocyanide_tail"},
                    ],
                    "atom_mapped_reaction_smarts": (
                        "[NX3;H2,H1:1].[CX3H1:2]=[OX1]."
                        "[C;-1,+0;X1:3]#[N;+1,+0;X2:4]>>"
                        "[N:1][CH1:2][C+0:3](=O)[NH1+0:4]"
                    ),
                    "selectivity_policy": (
                        "Multiple N-H amines enumerate products and deduplicate."
                    ),
                }
            ]
        },
    )
    variant = tmp_path / "variant.yaml"
    variant.write_text(
        yaml.safe_dump(
            {
                "variant": "ugi_3cr_agile",
                "reaction_id": "ugi_3cr_agile",
                "expected_reactant_count": 3,
                "roles": [
                    {"name": "amine_head"},
                    {"name": "oxoester_aldehyde_body_tail"},
                    {"name": "isocyanide_tail"},
                ],
                "compatibility": {
                    "amine_head_site_multiplicity_semantics": (
                        "symmetry_distinct_required_handle_matches"
                    ),
                    "deduplicate_forward_products": True,
                    "multiple_unique_products_require_explicit_site_selection": True,
                },
            },
            sort_keys=True,
        )
    )
    config = tmp_path / "config.json"
    _write_json(
        config,
        {
            "schema_version": "m0_09_ugi3_assembly_qualification_config.v1",
            "task": "fixture qualification",
            "generated_utc": "2026-07-29T00:00:00+00:00",
            "inputs": {
                "agile_measured_library": {"expected_sha256": _sha256(agile)},
                "qualified_reactions": {"expected_sha256": _sha256(registry)},
                "ugi_variant": {"expected_sha256": _sha256(variant)},
            },
            "scope": {
                "reaction_id": "ugi_3cr_agile",
                "amine_role": "amine_head",
                "multiplicity_semantics": (
                    "symmetry_distinct_required_handle_matches"
                ),
                "raw_match_count_role": "diagnostic_only",
                "atom_equivalence_method": "fixture",
                "require_single_atom_amine_handle": True,
                "deduplicate_forward_products": True,
                "multiple_unique_products_require_explicit_site_selection": True,
                "max_forward_outcomes_per_record": 100,
            },
            "expected_counts": {
                "measured_products": 1,
                "unique_product_labels": 1,
                "amine_heads": 1,
                "aldehyde_components": 1,
                "isocyanide_components": 1,
                "complete_component_combinations": 1,
                "products_reconstructed_exactly": 1,
                "products_passing_raw_amine_multiplicity": 0,
                "products_failing_raw_amine_multiplicity": 1,
                "products_passing_qualified_amine_multiplicity": 1,
                "products_failing_qualified_amine_multiplicity": 0,
                "heads_failing_raw_amine_multiplicity": 1,
                "heads_failing_qualified_amine_multiplicity": 0,
                "heads_with_equivalent_site_collapse": 1,
                "products_with_equivalent_site_collapse": 1,
                "heads_with_multiple_unique_forward_products": 0,
                "products_with_multiple_unique_forward_products": 0,
            },
            "expected_forward_outcome_profile": {"raw_3_unique_1": 1},
            "required_head_findings": {
                "A5": {
                    "raw_required_handle_matches": 3,
                    "symmetry_distinct_required_handle_matches": 1,
                    "unique_forward_products_per_record": 1,
                    "rows": 1,
                }
            },
        },
    )
    return config, agile, registry, variant


def test_symmetry_distinct_sites_resolve_a5_without_identity_exception() -> None:
    molecule = Chem.MolFromSmiles("NCCN(CCN)CCN")
    handle = Chem.MolFromSmarts("[NX3;H2,H1]")

    audit = audit_reactive_site_multiplicity(molecule, handle)

    assert audit.raw_match_count == 3
    assert audit.symmetry_distinct_match_count == 1


def test_qualification_reconstructs_a5_and_preserves_raw_diagnostic(
    tmp_path: Path,
) -> None:
    result = build_ugi3_assembly_qualification(*_write_fixture(tmp_path))

    assert result["summary"]["measured_products"] == 1
    assert result["summary"]["products_reconstructed_exactly"] == 1
    assert result["summary"]["products_passing_raw_amine_multiplicity"] == 0
    assert result["summary"]["products_passing_qualified_amine_multiplicity"] == 1
    assert result["failed_row_labels"] == []
    assert result["decision"]["all_measured_products_pass"] is True


def test_qualification_rejects_unqualified_multiplicity_semantics(
    tmp_path: Path,
) -> None:
    config, agile, registry, variant = _write_fixture(tmp_path)
    variant_payload = yaml.safe_load(variant.read_text())
    variant_payload["compatibility"][
        "amine_head_site_multiplicity_semantics"
    ] = "raw_substructure_matches"
    variant.write_text(yaml.safe_dump(variant_payload, sort_keys=True))
    config_payload = json.loads(config.read_text())
    config_payload["inputs"]["ugi_variant"]["expected_sha256"] = _sha256(variant)
    _write_json(config, config_payload)

    with pytest.raises(
        Ugi3AssemblyQualificationError,
        match="semantics disagree",
    ):
        build_ugi3_assembly_qualification(config, agile, registry, variant)


def test_qualification_rejects_hash_mismatch(tmp_path: Path) -> None:
    config, agile, registry, variant = _write_fixture(tmp_path)
    registry.write_text(registry.read_text() + "\n")

    with pytest.raises(Ugi3AssemblyQualificationError, match="hash mismatch"):
        build_ugi3_assembly_qualification(config, agile, registry, variant)


def test_writer_is_deterministic_and_atomic(tmp_path: Path) -> None:
    result = build_ugi3_assembly_qualification(*_write_fixture(tmp_path))
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    write_ugi3_assembly_qualification(result, first)
    write_ugi3_assembly_qualification(result, second)

    assert first.read_bytes() == second.read_bytes()
    assert not list(tmp_path.glob(".*.tmp"))


def test_committed_qualification_covers_all_measured_products() -> None:
    path = REPO / "results/m0_09/ugi3_assembly_qualification.json"
    if not path.exists():
        pytest.fail("committed Ugi-3 assembly qualification artifact is missing")
    result = json.loads(path.read_text())

    assert result["schema_version"] == "m0_09_ugi3_assembly_qualification.v1"
    assert result["summary"]["measured_products"] == 1200
    assert result["summary"]["products_reconstructed_exactly"] == 1200
    assert result["summary"]["products_passing_raw_amine_multiplicity"] == 1140
    assert result["summary"]["products_passing_qualified_amine_multiplicity"] == 1200
    assert result["summary"]["heads_with_multiple_unique_forward_products"] == 2
    assert result["failed_row_labels"] == []
