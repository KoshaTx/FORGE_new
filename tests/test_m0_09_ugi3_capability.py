from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from forge.synthesis.evidence.ugi3_capability import (
    Ugi3CapabilityError,
    build_ugi3_precursor_capability,
    write_ugi3_precursor_capability,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True))


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    pool = tmp_path / "pool.json"
    blocks = [
        {
            "block_id": "aldehyde:1",
            "canonical_smiles": "CC=O",
            "handle": "aldehyde",
            "provenance": {"source": "test"},
        },
        {
            "block_id": "aldehyde:2",
            "canonical_smiles": "CCC=O",
            "handle": "aldehyde",
            "provenance": {"source": "test"},
        },
        {
            "block_id": "amine:1",
            "canonical_smiles": "CN",
            "handle": "amine",
            "provenance": {"source": "test"},
        },
        {
            "block_id": "amine:2",
            "canonical_smiles": "CCN",
            "handle": "amine",
            "provenance": {"source": "test"},
        },
        {
            "block_id": "isocyanide:1",
            "canonical_smiles": "[C-]#[N+]C",
            "handle": "isocyanide",
            "provenance": {"source": "programmatic_rational"},
            "descriptors": {
                "chain_length": 1,
                "n_branches": 0,
                "unsaturation": 0,
            },
        },
        {
            "block_id": "isocyanide:2",
            "canonical_smiles": "[C-]#[N+]CC",
            "handle": "isocyanide",
            "provenance": {"source": "programmatic_rational"},
            "descriptors": {
                "chain_length": 2,
                "n_branches": 0,
                "unsaturation": 0,
            },
        },
        {
            "block_id": "isocyanide:3",
            "canonical_smiles": "[C-]#[N+]CCC",
            "handle": "isocyanide",
            "provenance": {"source": "programmatic_rational"},
            "descriptors": {
                "chain_length": 3,
                "n_branches": 0,
                "unsaturation": 1,
            },
        },
    ]
    _write_json(
        pool,
        {
            "total_blocks": len(blocks),
            "blocks_by_handle": {"aldehyde": 2, "amine": 2, "isocyanide": 3},
            "blocks": blocks,
        },
    )

    agile = tmp_path / "agile.json"
    _write_json(
        agile,
        {
            "schema_version": "m0_09_agile_component_routes.v1",
            "components": [
                {
                    "label": "A1",
                    "component_class": "amine",
                    "canonical_smiles": "CN",
                    "procurement_evidence_status": "vendor_claim_only",
                    "route_evidence_status": "source_identified",
                    "route_ids": [],
                },
                {
                    "label": "A2",
                    "component_class": "amine",
                    "canonical_smiles": "CCN",
                    "procurement_evidence_status": "vendor_claim_only",
                    "route_evidence_status": "source_identified",
                    "route_ids": [],
                },
            ],
            "routes": [
                {
                    "route_id": "route-c1",
                    "route_family_id": "agile_isocyanide_two_step",
                    "component_label": "C1",
                    "target": {"canonical_smiles": "[C-]#[N+]CC"},
                    "steps": [{"step_index": 1}, {"step_index": 2}],
                    "route_evidence_status": "route_extracted",
                    "execution_closure_status": "unknown",
                    "forward_verification_status": "not_run",
                }
            ],
        },
    )

    ledger = tmp_path / "components.csv"
    with ledger.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "component_id",
                "role",
                "parse_status",
                "canonical_smiles",
                "source_pmids_json",
                "experiment_ids_json",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "component_id": "head-1",
                    "role": "head",
                    "parse_status": "parsed",
                    "canonical_smiles": "CN",
                    "source_pmids_json": '["100"]',
                    "experiment_ids_json": '["MIAO"]',
                },
                {
                    "component_id": "head-2",
                    "role": "head",
                    "parse_status": "parsed",
                    "canonical_smiles": "CCCN",
                    "source_pmids_json": '["100"]',
                    "experiment_ids_json": '["MIAO"]',
                },
                {
                    "component_id": "linker-1",
                    "role": "linker",
                    "parse_status": "parsed",
                    "canonical_smiles": "[C-]#[N+]C",
                    "source_pmids_json": '["100"]',
                    "experiment_ids_json": '["MIAO"]',
                },
                {
                    "component_id": "linker-2",
                    "role": "linker",
                    "parse_status": "parsed",
                    "canonical_smiles": "[C-]#[N+]CC",
                    "source_pmids_json": '["100"]',
                    "experiment_ids_json": '["MIAO"]',
                },
            ]
        )

    reviews = tmp_path / "reviews.json"
    _write_json(
        reviews,
        {
            "schema_version": "m0_09_lnpdb_paper_route_reviews.v2",
            "reviews": [
                {
                    "review_id": "agile-test",
                    "building_block_groups": [
                        {
                            "role": "ugi3_isocyanide_tail",
                            "upstream_route_count": 2,
                        }
                    ],
                }
            ],
            "review_audit": [
                {
                    "review_id": "agile-test",
                    "upstream_component_routes": 3,
                    "l2_reaction_instances": 6,
                    "structure_resolved_l2_route_instances": 2,
                }
            ],
        },
    )

    config = tmp_path / "config.json"
    _write_json(
        config,
        {
            "schema_version": "m0_09_ugi3_precursor_capability_config.v2",
            "task": "test capability audit",
            "inputs": {
                "building_block_pool": {"expected_sha256": _sha256(pool)},
                "agile_component_routes": {"expected_sha256": _sha256(agile)},
                "component_source_ledger": {"expected_sha256": _sha256(ledger)},
                "paper_route_reviews": {"expected_sha256": _sha256(reviews)},
            },
            "expected_counts": {
                "building_blocks": 7,
                "blocks_by_handle": {"aldehyde": 2, "amine": 2, "isocyanide": 3},
                "rational_isocyanide_candidates": 3,
                "direct_agile_isocyanide_routes": 1,
                "agile_source_reported_isocyanide_routes": 2,
                "agile_amine_heads": 2,
                "miao_heads": 2,
                "miao_isocyanides": 2,
                "miao_agile_head_overlap": 1,
            },
            "candidate_scope": {
                "handle": "isocyanide",
                "required_provenance_source": "programmatic_rational",
                "exact_agile_upstream_route_family_id": ("agile_isocyanide_two_step"),
                "exact_upstream_route_transformation_evidence": ["exact_source_route"],
                "exact_upstream_route_component_observation": [
                    "programmatic_candidate",
                    "agile_measured_component",
                ],
                "extrapolated_transformation_evidence": ["bounded_family_applicability"],
                "extrapolated_component_observation": ["programmatic_candidate"],
                "route_closure": "incomplete",
                "operational_availability": ("route_or_procurement_resolution_required"),
                "prospective_outcome": "not_attempted",
                "forward_verification_status": "not_run",
            },
            "agile_head_scope": {
                "component_class": "amine",
                "allowed_procurement_evidence_status": ["vendor_claim_only"],
                "allowed_route_evidence_status": ["source_identified"],
                "transformation_evidence": ["no_upstream_route_reported"],
                "component_observation": ["agile_measured_component"],
                "route_closure": "incomplete",
                "operational_availability": "historical_blanket_vendor_claim",
                "prospective_outcome": "not_attempted",
                "forward_verification_status": "not_run",
            },
            "miao_cross_assembly_scope": {
                "pmid": "100",
                "doi": "10.test/miao",
                "experiment_id": "MIAO",
                "head_role": "head",
                "isocyanide_role": "linker",
                "isocyanide_component_ids": ["linker-1", "linker-2"],
                "procurement_evidence_status": "historical_blanket_vendor_claim",
                "assembly_applicability": "cross_assembly_evidence_only",
                "transformation_evidence": ["no_upstream_route_reported"],
                "component_observation": ["cross_assembly_component"],
                "route_closure": "incomplete",
                "operational_availability": "historical_blanket_vendor_claim",
                "prospective_outcome": "not_attempted",
                "forward_verification_status": "not_run",
                "scope_note": "Cross-assembly only.",
            },
            "agile_paper_review": {
                "review_id": "agile-test",
                "expected_upstream_component_routes": 3,
                "expected_l2_reaction_instances": 6,
                "expected_structure_resolved_l2_route_instances": 2,
            },
            "evidence_axes": {
                "transformation_evidence": {
                    "cardinality": "multiple",
                    "states": [
                        {
                            "state": "no_upstream_route_reported",
                            "meaning": "No upstream route.",
                        },
                        {
                            "state": "bounded_family_applicability",
                            "meaning": "Family candidate.",
                        },
                        {
                            "state": "exact_source_route",
                            "meaning": "Exact route.",
                        },
                    ],
                },
                "component_observation": {
                    "cardinality": "multiple",
                    "states": [
                        {
                            "state": "programmatic_candidate",
                            "meaning": "Programmatic candidate.",
                        },
                        {
                            "state": "agile_measured_component",
                            "meaning": "AGILE component.",
                        },
                        {
                            "state": "cross_assembly_component",
                            "meaning": "Cross-assembly component.",
                        },
                    ],
                },
                "route_closure": {
                    "cardinality": "single",
                    "states": [
                        {"state": "incomplete", "meaning": "Incomplete."},
                        {
                            "state": "computationally_complete",
                            "meaning": "Computationally complete.",
                        },
                    ],
                },
                "operational_availability": {
                    "cardinality": "single",
                    "states": [
                        {
                            "state": "historical_blanket_vendor_claim",
                            "meaning": "Historical claim.",
                        },
                        {
                            "state": "route_or_procurement_resolution_required",
                            "meaning": "Resolution required.",
                        },
                    ],
                },
                "prospective_outcome": {
                    "cardinality": "single",
                    "states": [
                        {"state": "not_attempted", "meaning": "Not attempted."},
                        {
                            "state": "isolated_with_yield_and_purity",
                            "meaning": "Isolated.",
                        },
                    ],
                },
            },
            "reaction_basis": {
                "route_family": "primary_amine_to_isocyanide",
                "conceptual_steps": ["formylation", "dehydration"],
            },
            "evidence_sources": [
                {
                    "source_id": "source-1",
                    "doi": "10.test/source",
                    "evidence_tags": {
                        "transformation_evidence": ["exact_source_route"],
                        "component_observation": ["agile_measured_component"],
                    },
                    "asset_sha256": "a" * 64,
                    "qa_flags": [
                        {
                            "flag_id": "reagent-conflict",
                            "status": "unresolved_source_internal_discrepancy",
                        }
                    ],
                }
            ],
            "claims_boundary": {
                "raw_cartesian_product_is_upper_bound": True,
                "candidate_membership_is_not_synthesis_proof": True,
                "cross_assembly_use_is_not_ugi3_validation": True,
                "historical_vendor_claim_is_not_current_procurement": True,
                "prospective_outcomes_expected": 0,
            },
        },
    )
    return config, pool, agile, ledger, reviews


def _build_fixture(tmp_path: Path) -> dict:
    paths = _write_fixture(tmp_path)
    return build_ugi3_precursor_capability(
        *paths,
        generated_utc="2026-07-29T00:00:00+00:00",
    )


def test_capability_audit_separates_exact_routes_from_family_candidates(
    tmp_path: Path,
) -> None:
    result = _build_fixture(tmp_path)

    assert result["summary"] == {
        "rational_isocyanide_candidates": 3,
        "exact_agile_isocyanide_upstream_route_candidates": 1,
        "family_supported_isocyanide_candidates_without_exact_route": 2,
        "computationally_route_complete_isocyanide_candidates": 0,
        "agile_source_reported_isocyanide_routes": 2,
        "agile_amine_heads": 2,
        "agile_heads_with_upstream_route": 0,
        "agile_heads_with_current_item_level_procurement_closure": 0,
        "miao_cross_assembly_isocyanides": 2,
        "miao_cross_assembly_heads": 2,
        "miao_agile_exact_head_overlap": 1,
        "raw_component_cartesian_upper_bound": 12,
        "isocyanide_candidates_with_prospective_outcome": 0,
    }
    transformation_evidence = {
        record["block_id"]: record["transformation_evidence"]
        for record in result["isocyanide_candidates"]
    }
    assert transformation_evidence == {
        "isocyanide:1": ["bounded_family_applicability"],
        "isocyanide:2": ["exact_source_route"],
        "isocyanide:3": ["bounded_family_applicability"],
    }
    assert {
        record["block_id"]: record["component_observation"]
        for record in result["isocyanide_candidates"]
    } == {
        "isocyanide:1": ["programmatic_candidate"],
        "isocyanide:2": ["agile_measured_component", "programmatic_candidate"],
        "isocyanide:3": ["programmatic_candidate"],
    }
    assert {record["route_closure"] for record in result["isocyanide_candidates"]} == {"incomplete"}
    assert {record["prospective_outcome"] for record in result["isocyanide_candidates"]} == {
        "not_attempted"
    }
    assert "evidence_layers" not in result
    assert all(
        "rank" not in definition
        for axis in result["evidence_axes"].values()
        for definition in axis["states"]
    )


def test_capability_audit_preserves_head_and_qa_boundaries(tmp_path: Path) -> None:
    result = _build_fixture(tmp_path)

    assert {record["operational_availability"] for record in result["agile_amine_head_audit"]} == {
        "historical_blanket_vendor_claim"
    }
    assert not any(
        record["current_item_level_procurement_closed"]
        for record in result["agile_amine_head_audit"]
    )
    assert (
        result["source_evidence"][0]["qa_flags"][0]["status"]
        == "unresolved_source_internal_discrepancy"
    )
    assert result["miao_cross_assembly_audit"]["assembly_applicability"] == (
        "cross_assembly_evidence_only"
    )
    assert result["miao_cross_assembly_audit"]["component_observation"] == [
        "cross_assembly_component"
    ]


def test_capability_audit_rejects_hash_mismatch(tmp_path: Path) -> None:
    config, pool, agile, ledger, reviews = _write_fixture(tmp_path)
    pool.write_text(pool.read_text() + "\n")

    with pytest.raises(Ugi3CapabilityError, match="hash mismatch"):
        build_ugi3_precursor_capability(config, pool, agile, ledger, reviews)


def test_capability_audit_rejects_count_mismatch(tmp_path: Path) -> None:
    config, pool, agile, ledger, reviews = _write_fixture(tmp_path)
    payload = json.loads(pool.read_text())
    payload["blocks"].pop()
    payload["total_blocks"] -= 1
    payload["blocks_by_handle"]["isocyanide"] -= 1
    _write_json(pool, payload)
    config_payload = json.loads(config.read_text())
    config_payload["inputs"]["building_block_pool"]["expected_sha256"] = _sha256(pool)
    _write_json(config, config_payload)

    with pytest.raises(Ugi3CapabilityError, match="building blocks count mismatch"):
        build_ugi3_precursor_capability(config, pool, agile, ledger, reviews)


def test_capability_audit_rejects_ranked_cross_axis_evidence(tmp_path: Path) -> None:
    config, pool, agile, ledger, reviews = _write_fixture(tmp_path)
    config_payload = json.loads(config.read_text())
    config_payload["evidence_axes"]["route_closure"]["states"][0]["rank"] = 1
    _write_json(config, config_payload)

    with pytest.raises(Ugi3CapabilityError, match="must not encode a cross-axis rank"):
        build_ugi3_precursor_capability(config, pool, agile, ledger, reviews)


def test_capability_writer_is_deterministic_and_atomic(tmp_path: Path) -> None:
    result = _build_fixture(tmp_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    write_ugi3_precursor_capability(result, first)
    write_ugi3_precursor_capability(result, second)

    assert first.read_bytes() == second.read_bytes()
    assert not list(tmp_path.glob(".*.tmp"))
