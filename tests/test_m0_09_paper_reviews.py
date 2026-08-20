from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from experiments.archive.phase1.synthesis_audits.paper_reviews import (
    PaperReviewError,
    build_paper_route_reviews,
    write_paper_route_reviews,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    cache = tmp_path / "cache"
    asset = cache / "PMC1" / "supp.pdf"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"%PDF-1.7\nreviewed chemistry")

    source_index = tmp_path / "source_review_index.json"
    source_index.write_text(
        json.dumps(
            {
                "schema_version": "m0_09_source_review_index.v1",
                "records": [
                    {
                        "pmc_id": "PMC1",
                        "pmid": "1001",
                        "doi": "10.test/route",
                        "source_package_status": "complete",
                        "supplements": [
                            {
                                "filename": "supp.pdf",
                                "cache_asset": "PMC1/supp.pdf",
                                "sha256": _sha256(asset),
                            }
                        ],
                    }
                ],
            }
        )
    )

    components = tmp_path / "components.csv"
    with components.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["component_id", "role", "source_pmids_json"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "component_id": "head-1",
                    "role": "head",
                    "source_pmids_json": '["1001"]',
                },
                {
                    "component_id": "linker-1",
                    "role": "linker",
                    "source_pmids_json": '["1001"]',
                },
                {
                    "component_id": "tail-1",
                    "role": "tail1",
                    "source_pmids_json": '["1001"]',
                },
            ]
        )

    config = tmp_path / "reviews.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "m0_09_lnpdb_paper_reviews_config.v2",
                "inputs": {
                    "source_review_index": {
                        "asset": "source_review_index.json",
                        "expected_sha256": _sha256(source_index),
                    },
                    "component_source_ledger": {
                        "asset": "components.csv",
                        "expected_sha256": _sha256(components),
                    },
                },
                "reviews": [
                    {
                        "review_id": "paper_1",
                        "pmc_id": "PMC1",
                        "pmid": "1001",
                        "doi": "10.test/route",
                        "citation": "Test et al. (2026)",
                        "source_asset": {
                            "filename": "supp.pdf",
                            "expected_sha256": _sha256(asset),
                            "chemistry_pages": [2, 3],
                            "review_status": "chemistry_pages_visually_reviewed",
                        },
                        "source_component_scope": {
                            "normalized_unique_components": 3,
                            "normalized_by_role": {
                                "head": 1,
                                "linker": 1,
                                "tail1": 1,
                            },
                            "source_reported_head_count": 1,
                            "source_reported_tail_count": 1,
                            "count_note": "Test scope.",
                        },
                        "building_block_groups": [
                            {
                                "role": "head",
                                "source_labels": "H1",
                                "source_locator": "page 2",
                                "source_reported_count": 1,
                                "normalized_component_count": 1,
                                "upstream_route_status": "not_reported",
                                "upstream_route_count": 0,
                                "procurement_evidence_status": ("historical_blanket_vendor_claim"),
                                "procurement_locator": "page 2",
                                "procurement_note": "No exact item.",
                            },
                            {
                                "role": "tail1",
                                "source_labels": "T1",
                                "source_locator": "page 2",
                                "source_reported_count": 1,
                                "normalized_component_count": 1,
                                "upstream_route_status": "not_reported",
                                "upstream_route_count": 0,
                                "procurement_evidence_status": ("historical_blanket_vendor_claim"),
                                "procurement_locator": "page 2",
                                "procurement_note": "No exact item.",
                            },
                        ],
                        "l2_route_families": [
                            {
                                "route_family_id": "route-1",
                                "level": "L2",
                                "source_locator": "page 3",
                                "route_evidence_status": "exact_route_extracted",
                                "route_instance_count": 1,
                                "reaction_instance_count": 1,
                                "component_assignment_status": "shared_precursor",
                                "assignment_note": "A shared precursor route.",
                                "starting_material_procurement_status": (
                                    "historical_blanket_vendor_claim"
                                ),
                                "starting_material": {
                                    "name": "ethanol",
                                    "smiles": "CCO",
                                },
                                "target": {
                                    "name": "acetaldehyde",
                                    "smiles": "CC=O",
                                },
                                "steps": [
                                    {
                                        "step_index": 1,
                                        "transformation": "oxidation",
                                        "reactant_smiles": "CCO",
                                        "product_smiles": "CC=O",
                                        "reagents": ["oxidant"],
                                        "solvent": "test solvent",
                                        "temperature": "room temperature",
                                        "duration": "one hour",
                                        "atmosphere": "not reported",
                                        "isolated_yield_percent": 80,
                                    }
                                ],
                            }
                        ],
                        "l1_assembly_families": [
                            {
                                "assembly_family_id": "assembly-1",
                                "level": "L1",
                                "source_locator": "page 3",
                                "transformation": "coupling",
                                "member_count": 1,
                                "members": [
                                    {
                                        "label": "T1",
                                        "name": "test tail",
                                        "component_id": "tail-1",
                                        "source_structure_status": "exact",
                                        "isolated_yield_percent": 70,
                                    }
                                ],
                            }
                        ],
                        "negative_outcomes": [],
                        "review_conclusion": {
                            "l2_route_families": 1,
                            "l2_route_instances": 1,
                            "l2_reaction_instances": 1,
                            "structure_resolved_l2_route_instances": 1,
                            "upstream_component_routes": 0,
                            "l1_assembly_families": 1,
                            "reported_negative_outcomes": 0,
                            "key_finding": "Only the shared precursor has an upstream route.",
                        },
                    }
                ],
            }
        )
    )
    return config, source_index, components, cache


def _build_fixture(tmp_path: Path) -> dict:
    config, source_index, components, cache = _write_fixture(tmp_path)
    return build_paper_route_reviews(
        config,
        source_index,
        components,
        cache,
        generated_utc="2026-07-29T00:00:00+00:00",
    )


def test_paper_review_separates_l1_from_upstream_routes(tmp_path: Path) -> None:
    result = _build_fixture(tmp_path)

    assert result["summary"] == {
        "source_packages_chemistry_reviewed": 1,
        "normalized_components_in_reviewed_sources": 3,
        "normalized_components_by_role": {"head": 1, "linker": 1, "tail1": 1},
        "normalized_building_block_components_reviewed": 2,
        "source_reported_building_blocks_reviewed": 2,
        "upstream_component_routes": 0,
        "upstream_component_routes_by_role": {"head": 0, "tail1": 0},
        "l2_route_families": 1,
        "l2_route_instances": 1,
        "l2_reaction_instances": 1,
        "structure_resolved_l2_route_instances": 1,
        "l1_assembly_families": 1,
        "reported_negative_outcomes": 0,
        "normalized_procurement_components_by_evidence": {"historical_blanket_vendor_claim": 2},
        "source_procurement_components_by_evidence": {"historical_blanket_vendor_claim": 2},
    }
    assert result["review_audit"][0]["upstream_component_routes"] == 0


def test_paper_review_rejects_l1_family_mislabeled_as_l2(tmp_path: Path) -> None:
    config, source_index, components, cache = _write_fixture(tmp_path)
    payload = json.loads(config.read_text())
    payload["reviews"][0]["l1_assembly_families"][0]["level"] = "L2"
    config.write_text(json.dumps(payload))

    with pytest.raises(PaperReviewError, match="must have level L1"):
        build_paper_route_reviews(config, source_index, components, cache)


def test_paper_review_rejects_broken_route_continuity(tmp_path: Path) -> None:
    config, source_index, components, cache = _write_fixture(tmp_path)
    payload = json.loads(config.read_text())
    route = payload["reviews"][0]["l2_route_families"][0]
    route["steps"].append(
        {
            "step_index": 2,
            "transformation": "reduction",
            "reactant_smiles": "CCC",
            "product_smiles": "CCO",
            "reagents": ["reductant"],
            "solvent": "test solvent",
            "temperature": "room temperature",
            "duration": "one hour",
            "atmosphere": "not reported",
            "isolated_yield_percent": 50,
        }
    )
    route["target"]["smiles"] = "CCO"
    route["reaction_instance_count"] = 2
    payload["reviews"][0]["review_conclusion"]["l2_reaction_instances"] = 2
    config.write_text(json.dumps(payload))

    with pytest.raises(PaperReviewError, match="does not continue from the prior product"):
        build_paper_route_reviews(config, source_index, components, cache)


def test_paper_review_counts_multistep_series_without_reported_yields(
    tmp_path: Path,
) -> None:
    config, source_index, components, cache = _write_fixture(tmp_path)
    payload = json.loads(config.read_text())
    review = payload["reviews"][0]
    review["l2_route_families"] = [
        {
            "route_family_id": "series-1",
            "level": "L2",
            "source_locator": "pages 2-3",
            "route_evidence_status": "general_procedure_product_series_extracted",
            "component_assignment_status": "one_of_two_structures_resolved",
            "assignment_note": "Test series.",
            "route_instance_count": 2,
            "reaction_instance_count": 4,
            "route_depth": 2,
            "starting_material_procurement_status": "historical_blanket_vendor_claim",
            "procedure": {"step_sequence": ["coupling", "oxidation"]},
            "members": [
                {
                    "label": "S1",
                    "name": "ethanol",
                    "product_smiles": "CCO",
                    "yield_status": "not_reported",
                    "source_structure_status": "structure_resolved",
                },
                {
                    "label": "S2",
                    "name": "propanol",
                    "yield_status": "not_reported",
                    "source_structure_status": "not_transcribed",
                },
            ],
        }
    ]
    review["review_conclusion"].update(
        {
            "l2_route_families": 1,
            "l2_route_instances": 2,
            "l2_reaction_instances": 4,
            "structure_resolved_l2_route_instances": 1,
        }
    )
    config.write_text(json.dumps(payload))

    result = build_paper_route_reviews(config, source_index, components, cache)

    assert result["summary"]["l2_route_instances"] == 2
    assert result["summary"]["l2_reaction_instances"] == 4
    assert result["summary"]["structure_resolved_l2_route_instances"] == 1


def test_paper_review_verifies_supporting_route_artifact(tmp_path: Path) -> None:
    config, source_index, components, cache = _write_fixture(tmp_path)
    route_artifact = tmp_path / "routes.json"
    route_artifact.write_text(
        json.dumps(
            {
                "schema_version": "test_routes.v1",
                "routes": [
                    {"component_label": "B1"},
                    {"component_label": "C1"},
                ],
            }
        )
    )
    payload = json.loads(config.read_text())
    payload["inputs"]["supporting_route_artifacts"] = [
        {
            "asset": str(route_artifact),
            "expected_sha256": _sha256(route_artifact),
            "expected_schema_version": "test_routes.v1",
            "expected_route_count": 2,
            "expected_component_labels": ["C1", "B1"],
        }
    ]
    config.write_text(json.dumps(payload))

    result = build_paper_route_reviews(config, source_index, components, cache)

    route_input = next(
        record for record in result["inputs"] if record["asset"] == str(route_artifact)
    )
    assert route_input["sha256"] == _sha256(route_artifact)


def test_paper_review_rejects_wrong_supporting_route_labels(tmp_path: Path) -> None:
    config, source_index, components, cache = _write_fixture(tmp_path)
    route_artifact = tmp_path / "routes.json"
    route_artifact.write_text(
        json.dumps(
            {
                "schema_version": "test_routes.v1",
                "routes": [{"component_label": "B1"}],
            }
        )
    )
    payload = json.loads(config.read_text())
    payload["inputs"]["supporting_route_artifacts"] = [
        {
            "asset": str(route_artifact),
            "expected_sha256": _sha256(route_artifact),
            "expected_schema_version": "test_routes.v1",
            "expected_route_count": 1,
            "expected_component_labels": ["C1"],
        }
    ]
    config.write_text(json.dumps(payload))

    with pytest.raises(PaperReviewError, match="component labels do not match"):
        build_paper_route_reviews(config, source_index, components, cache)


def test_paper_review_write_is_byte_stable(tmp_path: Path) -> None:
    result = _build_fixture(tmp_path)
    output = tmp_path / "result.json"

    write_paper_route_reviews(result, output)
    first = output.read_bytes()
    write_paper_route_reviews(result, output)

    assert output.read_bytes() == first
    assert json.loads(first) == result
