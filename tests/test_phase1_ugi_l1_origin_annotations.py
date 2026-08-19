from __future__ import annotations

import csv
import gzip
from pathlib import Path

import pytest

from forge.bio.ugi_semantic_annotations import (
    ROLE_NAMES,
    UgiSemanticAnnotationError,
    annotate_qualified_ugi_product,
)
from forge.route.qualified_forward import load_qualified_forward_reaction

REPO = Path(__file__).resolve().parents[1]


def _first_assignment() -> dict[str, str]:
    with gzip.open(
        REPO / "data/splits/phase1/ugi_l1_assignments.csv.gz", "rt", newline=""
    ) as handle:
        return next(csv.DictReader(handle))


def _compiled():
    return load_qualified_forward_reaction(
        REPO / "data/vendor/qualified_reactions_v1.json",
        REPO / "configs/assembly/ugi_variant.yaml",
        reaction_id="ugi_3cr_agile",
    )


def test_single_product_annotation_has_exact_connected_origin_semantics() -> None:
    row = _first_assignment()
    product, atoms, bonds, mappings = annotate_qualified_ugi_product(
        _compiled(),
        product_id=row["product_id"],
        target_smiles=row["canonical_product_smiles"],
        component_smiles_by_role={role: row[f"{role}_smiles"] for role in ROLE_NAMES},
        source_evidence_record_id="test_transform_consistency",
        max_outcomes=100,
    )
    assert product["product_smiles"] == row["canonical_product_smiles"]
    assert product["semantic_signature_multiplicity"] == 1
    assert sum(atom["is_ugi_core"] for atom in atoms) == 5
    assert {atom["origin_role"] for atom in atoms} == {*ROLE_NAMES, "assembly_introduced"}
    assert len(atoms) == product["atom_rows"]
    assert len(bonds) == product["bond_rows"]
    assert len(mappings) == product["component_mapping_rows"]


def test_single_product_annotation_rejects_wrong_target() -> None:
    row = _first_assignment()
    with pytest.raises(UgiSemanticAnnotationError, match="did not reconstruct"):
        annotate_qualified_ugi_product(
            _compiled(),
            product_id="wrong",
            target_smiles="CCCC",
            component_smiles_by_role={role: row[f"{role}_smiles"] for role in ROLE_NAMES},
            source_evidence_record_id="test_transform_consistency",
            max_outcomes=100,
        )
