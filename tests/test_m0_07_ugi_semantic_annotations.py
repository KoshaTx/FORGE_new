from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from forge.potency.audit.ugi_semantic_annotations import (
    ATOM_FIELDS,
    BOND_FIELDS,
    COMPONENT_MAPPING_FIELDS,
    PRODUCT_FIELDS,
    UgiSemanticAnnotationError,
    build_ugi_semantic_annotations,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/bio/m0_07_ugi_semantic_annotations.json"
RESULT_DIR = REPO / "results/m0_07"
RESULT = RESULT_DIR / "ugi_semantic_annotations_result.json"
LEDGERS = (
    "ugi_semantic_products.csv.gz",
    "ugi_semantic_atoms.csv.gz",
    "ugi_semantic_bonds.csv.gz",
    "ugi_semantic_component_mappings.csv.gz",
)


def _read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_config_preserves_whole_graph_generation_boundary() -> None:
    config = json.loads(CONFIG.read_text())

    assert config["scope"]["whole_graph_policy"] == {
        "one_connected_product_graph": True,
        "component_catalog_ids_in_model_state": False,
        "independent_fragment_generation": False,
        "region_labels_are_node_level_supervision": True,
        "distances_are_derived_not_generated": True,
    }


def test_config_rejects_fragment_generation_authorization(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    config["scope"]["whole_graph_policy"]["independent_fragment_generation"] = True
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(config))

    with pytest.raises(UgiSemanticAnnotationError, match="whole-graph policy changed"):
        build_ugi_semantic_annotations(changed, tmp_path / "output", REPO)


def test_committed_semantic_ledgers_are_complete_and_chemistry_native() -> None:
    assert RESULT.is_file()
    result = json.loads(RESULT.read_text())
    summary = result["summary"]

    assert result["schema_version"] == "m0_07_ugi_semantic_annotations.v1"
    assert result["status"] == "completed_exact_source_ugi_semantic_annotations"
    assert summary["products"] == 1100
    assert summary["product_atoms"] == 52810
    assert summary["product_bonds"] == 52370
    assert summary["component_atom_rows"] == 52810
    assert summary["core_atoms"] == 5500
    assert summary["core_boundary_bonds"] == 3300
    assert summary["introduced_product_atoms"] == 1100
    assert summary["deleted_component_atoms"] == 1100
    assert summary["connected_role_regions"] == 3300
    assert summary["permutation_annotation_mismatches"] == 0
    assert summary["origin_atom_counts"] == {
        "amine_head": 9350,
        "assembly_introduced": 1100,
        "isocyanide_tail": 19360,
        "oxoester_aldehyde_body_tail": 23000,
    }
    assert result["role_anchor_map_numbers"] == {
        "amine_head": [1],
        "isocyanide_tail": [4],
        "oxoester_aldehyde_body_tail": [2],
    }
    assert result["decision"]["component_ids_authorized_in_model_state"] is False
    assert result["decision"]["independent_fragment_generation_authorized"] is False
    assert result["decision"]["distances_must_be_derived_from_graph"] is True

    for filename in LEDGERS:
        path = RESULT_DIR / filename
        assert path.is_file()
        assert _sha256(path) == result["artifacts"][filename]["sha256"]
        assert path.stat().st_size == result["artifacts"][filename]["bytes"]

    products = _read_gzip_csv(RESULT_DIR / LEDGERS[0])
    atoms = _read_gzip_csv(RESULT_DIR / LEDGERS[1])
    bonds = _read_gzip_csv(RESULT_DIR / LEDGERS[2])
    mappings = _read_gzip_csv(RESULT_DIR / LEDGERS[3])
    assert tuple(products[0]) == PRODUCT_FIELDS
    assert tuple(atoms[0]) == ATOM_FIELDS
    assert tuple(bonds[0]) == BOND_FIELDS
    assert tuple(mappings[0]) == COMPONENT_MAPPING_FIELDS
    assert (len(products), len(atoms), len(bonds), len(mappings)) == (
        1100,
        52810,
        52370,
        52810,
    )

    product_atoms = Counter(row["product_id"] for row in atoms)
    product_core_atoms = Counter(row["product_id"] for row in atoms if row["is_ugi_core"] == "True")
    product_boundary_bonds = Counter(
        row["product_id"] for row in bonds if row["is_core_boundary"] == "True"
    )
    assert set(product_atoms) == {row["product_id"] for row in products}
    assert set(product_core_atoms.values()) == {5}
    assert set(product_boundary_bonds.values()) == {3}
    assert Counter(row["origin_role"] for row in atoms) == Counter(summary["origin_atom_counts"])
    assert Counter(
        row["role"] for row in mappings if row["mapping_status"] == "deleted"
    ) == Counter({"oxoester_aldehyde_body_tail": 1100})
    assert all(int(row["distance_to_nearest_core"]) >= 0 for row in atoms)
    assert all(len(json.loads(row["distances_to_role_anchors_json"])) == 3 for row in atoms)

    by_head = {
        head: [
            (
                row["raw_forward_outcomes"],
                row["target_matching_outcomes"],
                row["source_mapping_multiplicity"],
            )
            for row in products
            if row["product_id"].startswith(f"{head}B")
        ]
        for head in ("A5", "A17", "A19", "A20")
    }
    assert set(by_head["A5"]) == {("3", "3", "3")}
    assert set(by_head["A17"]) == {("2", "2", "2")}
    assert set(by_head["A19"]) == {("2", "1", "1")}
    assert set(by_head["A20"]) == {("2", "1", "1")}

    forbidden_fields = {
        "component_id",
        "component_catalog_id",
        "pka",
        "particle_size",
        "potency",
        "transfection",
    }
    assert forbidden_fields.isdisjoint(set().union(*(set(row) for row in products[:1])))
    assert forbidden_fields.isdisjoint(set().union(*(set(row) for row in atoms[:1])))


@pytest.mark.needs_vendor
def test_full_semantic_build_is_byte_reproducible(tmp_path: Path) -> None:
    missing = [
        specification["path"]
        for specification in json.loads(CONFIG.read_text())["inputs"].values()
        if not (REPO / specification["path"]).is_file()
    ]
    if missing:
        pytest.skip(f"semantic annotation inputs are not available: {missing}")

    first = tmp_path / "first"
    second = tmp_path / "second"
    build_ugi_semantic_annotations(CONFIG, first, REPO)
    build_ugi_semantic_annotations(CONFIG, second, REPO)

    filenames = ("ugi_semantic_annotations_result.json", *LEDGERS)
    for filename in filenames:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
        if filename != "ugi_semantic_annotations_result.json":
            assert (first / filename).read_bytes() == (RESULT_DIR / filename).read_bytes()

    rebuilt_result = json.loads((first / "ugi_semantic_annotations_result.json").read_text())
    committed_result = json.loads(RESULT.read_text())
    assert rebuilt_result.pop("software")["rdkit"]
    assert committed_result.pop("software")["rdkit"]
    assert rebuilt_result == committed_result
