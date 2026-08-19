from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from forge.route.ugi3_aldehyde_head_capability import (
    Ugi3AldehydeHeadCapabilityError,
    build_ugi3_aldehyde_head_capability,
    write_ugi3_aldehyde_head_capability,
)

REPO = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True))


def _write_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    pool = tmp_path / "pool.json"
    _write_json(
        pool,
        {
            "blocks": [
                {
                    "block_id": "aldehyde:1",
                    "canonical_smiles": "CC=O",
                    "handle": "aldehyde",
                    "provenance": {"source": "agile_measured"},
                },
                {
                    "block_id": "aldehyde:2",
                    "canonical_smiles": "CCC=O",
                    "handle": "aldehyde",
                    "provenance": {"source": "programmatic_rational"},
                },
                {
                    "block_id": "amine:1",
                    "canonical_smiles": "CN",
                    "handle": "amine",
                    "provenance": {"source": "curated_literature"},
                },
                {
                    "block_id": "amine:2",
                    "canonical_smiles": "NCCN(CCN)CCN",
                    "handle": "amine",
                    "provenance": {"source": "agile_measured"},
                },
                {
                    "block_id": "amine:3",
                    "canonical_smiles": "CCN",
                    "handle": "amine",
                    "provenance": {"source": "curated_literature"},
                },
            ]
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
                    "route_evidence_status": "source_identified",
                },
                {
                    "label": "A5",
                    "component_class": "amine",
                    "canonical_smiles": "NCCN(CCN)CCN",
                    "route_evidence_status": "source_identified",
                },
                {
                    "label": "B1",
                    "component_class": "aldehyde_ester",
                    "canonical_smiles": "CC=O",
                    "route_evidence_status": "route_extracted",
                },
                {
                    "label": "B5",
                    "component_class": "aldehyde_ester",
                    "canonical_smiles": "CCC=O",
                    "route_evidence_status": "source_discrepancy_unresolved",
                    "qa_reason": "fixture discrepancy",
                },
            ],
            "routes": [
                {
                    "route_id": "route-in-pool",
                    "route_family_id": "aldehyde-family",
                    "component_label": "B1",
                    "target": {"canonical_smiles": "CC=O"},
                    "steps": [{"step_index": 1}],
                    "route_evidence_status": "route_extracted",
                    "forward_verification_status": "not_run",
                    "execution_closure_status": "unknown",
                },
                {
                    "route_id": "route-outside-pool",
                    "route_family_id": "aldehyde-family",
                    "component_label": "B-extra",
                    "target": {"canonical_smiles": "CCCC=O"},
                    "steps": [{"step_index": 1}],
                    "route_evidence_status": "route_extracted",
                    "forward_verification_status": "not_run",
                    "execution_closure_status": "unknown",
                },
            ],
        },
    )

    registry = tmp_path / "registry.json"
    _write_json(
        registry,
        {
            "reactions": [
                {
                    "reaction_id": "ugi-test",
                    "reactant_roles": [
                        {
                            "name": "aldehyde",
                            "required_handle_smarts": "[CX3H1]=[OX1]",
                            "forbidden_smarts": [],
                            "allowed_site_multiplicity": [1],
                        },
                        {
                            "name": "amine",
                            "required_handle_smarts": "[NX3;H2,H1]",
                            "forbidden_smarts": [],
                            "allowed_site_multiplicity": [1],
                        },
                    ],
                }
            ]
        },
    )

    precursor = tmp_path / "precursor.json"
    _write_json(
        precursor,
        {
            "schema_version": "m0_09_ugi3_precursor_capability.v2",
            "summary": {"rational_isocyanide_candidates": 2},
            "evidence_axes": {
                "transformation_evidence": {
                    "states": [
                        {"state": "exact_source_route"},
                        {"state": "no_upstream_route_reported"},
                    ]
                },
                "component_observation": {
                    "states": [
                        {"state": "agile_measured_component"},
                        {"state": "programmatic_candidate"},
                        {"state": "cross_assembly_component"},
                    ]
                },
                "route_closure": {
                    "states": [
                        {"state": "incomplete"},
                        {"state": "computationally_complete"},
                    ]
                },
                "operational_availability": {
                    "states": [
                        {"state": "historical_blanket_vendor_claim"},
                        {"state": "route_or_procurement_resolution_required"},
                        {"state": "current_item_level_vendor_verified"},
                    ]
                },
                "prospective_outcome": {
                    "states": [{"state": "not_attempted"}]
                },
            },
            "miao_cross_assembly_audit": {
                "heads": [{"canonical_smiles": "CN"}]
            },
        },
    )

    procurement = tmp_path / "procurement.json"
    _write_json(
        procurement,
        {
            "schema_version": "m0_09_ugi3_agile_head_procurement.v1",
            "snapshot": {
                "accessed_utc": "2026-07-28T00:00:00+00:00",
                "region": "US",
                "expiry_days": 30,
                "identity_policy": "fixture identity policy",
                "closure_policy": "fixture closure policy",
                "price_policy": "fixture price policy",
                "backup_policy": "fixture backup policy",
            },
            "expected_counts": {
                "records": 2,
                "current_item_level_vendor_verified": 1,
                "catalog_item_verified_availability_unresolved": 1,
                "identity_or_form_discrepancies": 0,
            },
            "records": [
                {
                    "component_label": "A1",
                    "block_id": "amine:1",
                    "canonical_smiles": "CN",
                    "identity": {
                        "pubchem_cid": 6329,
                        "pubchem_title": "Methylamine",
                        "inchi_key": "BAVYZALUXZFZLV-UHFFFAOYSA-N",
                        "cas_rn": "74-89-5",
                        "form": "neutral_free_base",
                    },
                    "vendor_evidence": {
                        "vendor": "Fixture vendor",
                        "product_title": "Methylamine",
                        "product_code": "A1",
                        "url": "https://example.test/US/product/A1",
                        "purity": "99%",
                        "availability_observation": "In stock.",
                    },
                    "procurement_status": "current_item_level_vendor_verified",
                    "current_item_level_procurement_closed": True,
                    "backup_status": "not_reviewed",
                },
                {
                    "component_label": "A5",
                    "block_id": "amine:2",
                    "canonical_smiles": "NCCN(CCN)CCN",
                    "identity": {
                        "pubchem_cid": 77731,
                        "pubchem_title": "Tris(2-aminoethyl)amine",
                        "inchi_key": "MBYLVOKEDDQJDY-UHFFFAOYSA-N",
                        "cas_rn": "4097-89-6",
                        "form": "neutral_free_base",
                    },
                    "vendor_evidence": {
                        "vendor": "Fixture vendor",
                        "product_title": "Tris(2-aminoethyl)amine",
                        "product_code": "A5",
                        "url": "https://example.test/US/product/A5",
                        "purity": "99%",
                        "availability_observation": "Check cart for availability.",
                    },
                    "procurement_status": (
                        "catalog_item_verified_availability_unresolved"
                    ),
                    "current_item_level_procurement_closed": False,
                    "backup_status": "not_reviewed",
                    "required_followup": "Verify current stock.",
                },
            ],
        },
    )

    qualification = tmp_path / "qualification.json"
    _write_json(
        qualification,
        {
            "schema_version": "m0_09_ugi3_assembly_qualification.v1",
            "summary": {
                "measured_products": 2,
                "products_reconstructed_exactly": 2,
                "products_passing_qualified_amine_multiplicity": 2,
                "products_failing_qualified_amine_multiplicity": 0,
            },
            "multiplicity_policy": {
                "reaction_id": "ugi-test",
                "role": "amine",
                "semantics": "symmetry_distinct_required_handle_matches",
            },
            "row_qualification_digest_sha256": "0" * 64,
            "failed_row_labels": [],
            "head_audit": [
                {
                    "component_label": "A1",
                    "raw_required_handle_matches": 1,
                    "symmetry_distinct_required_handle_matches": 1,
                    "passes_qualified_amine_multiplicity": True,
                    "products_reconstructed_exactly": 1,
                    "rows": 1,
                    "requires_explicit_site_selection": False,
                },
                {
                    "component_label": "A5",
                    "raw_required_handle_matches": 3,
                    "symmetry_distinct_required_handle_matches": 1,
                    "passes_qualified_amine_multiplicity": True,
                    "products_reconstructed_exactly": 1,
                    "rows": 1,
                    "requires_explicit_site_selection": False,
                },
            ],
            "qa_flags": [
                {
                    "flag_id": "fixture_a5_resolved",
                    "component_label": "A5",
                }
            ],
            "decision": {"all_measured_products_pass": True},
        },
    )

    config = tmp_path / "config.json"
    _write_json(
        config,
        {
            "schema_version": "m0_09_ugi3_aldehyde_head_capability_config.v3",
            "task": "fixture audit",
            "generated_utc": "2026-07-29T00:00:00+00:00",
            "inputs": {
                "building_block_pool": {"expected_sha256": _sha256(pool)},
                "agile_component_routes": {"expected_sha256": _sha256(agile)},
                "qualified_reactions": {"expected_sha256": _sha256(registry)},
                "precursor_capability": {"expected_sha256": _sha256(precursor)},
                "agile_head_procurement": {
                    "expected_sha256": _sha256(procurement)
                },
                "assembly_qualification": {
                    "expected_sha256": _sha256(qualification)
                },
            },
            "expected_counts": {
                "aldehyde_candidates": 2,
                "rational_aldehyde_candidates": 1,
                "agile_measured_aldehyde_candidates": 1,
                "structure_resolved_agile_aldehyde_routes": 2,
                "exact_route_aldehyde_candidate_overlap": 1,
                "source_resolved_aldehyde_routes_outside_pool": 1,
                "unresolved_aldehyde_source_discrepancies": 1,
                "amine_candidates": 3,
                "agile_head_candidate_overlap": 2,
                "miao_head_candidate_overlap": 1,
                "amine_candidates_with_required_handle": 3,
                "amine_candidates_passing_raw_registry_multiplicity": 2,
                "amine_candidates_failing_raw_registry_multiplicity": 1,
                "agile_heads_failing_raw_registry_multiplicity": 1,
                "amine_candidates_passing_qualified_site_multiplicity": 3,
                "amine_candidates_failing_qualified_site_multiplicity": 0,
                "agile_heads_failing_qualified_site_multiplicity": 0,
                "agile_measured_products_passing_qualified_site_multiplicity": 2,
                "isocyanide_candidates": 2,
                "agile_head_procurement_records": 2,
                "agile_heads_current_item_level_vendor_verified": 1,
                "agile_heads_catalog_item_verified_availability_unresolved": 1,
                "agile_head_procurement_identity_or_form_discrepancies": 0,
                "head_candidates_with_upstream_route": 0,
                "head_candidates_with_current_procurement_closure": 1,
                "head_candidates_requiring_procurement_or_route_resolution": 2,
                "computationally_route_complete_candidates": 0,
            },
            "registry_scope": {
                "reaction_id": "ugi-test",
                "aldehyde_role": "aldehyde",
                "amine_role": "amine",
                "multiplicity_semantics": (
                    "symmetry_distinct_required_handle_matches"
                ),
                "required_explicit_site_selection_heads": [],
                "raw_multiplicity_interpretation": "fixture",
            },
            "aldehyde_scope": {
                "route_family_ids": ["aldehyde-family"],
                "exact_route_transformation_evidence": ["exact_source_route"],
                "unresolved_transformation_evidence": [
                    "no_upstream_route_reported"
                ],
                "route_closure": "incomplete",
                "operational_availability": (
                    "route_or_procurement_resolution_required"
                ),
                "prospective_outcome": "not_attempted",
                "forward_verification_status": "not_run",
            },
            "head_scope": {
                "transformation_evidence": ["no_upstream_route_reported"],
                "route_closure": "incomplete",
                "prospective_outcome": "not_attempted",
                "forward_verification_status": "not_run",
                "default_current_procurement_status": "not_verified",
                "verified_procurement_status": (
                    "current_item_level_vendor_verified"
                ),
                "unresolved_catalog_status": (
                    "catalog_item_verified_availability_unresolved"
                ),
                "required_unresolved_procurement_labels": ["A5"],
            },
            "claims_boundary": {"model_built": False},
        },
    )
    return config, pool, agile, registry, precursor, procurement, qualification


def _build_fixture(tmp_path: Path) -> dict:
    return build_ugi3_aldehyde_head_capability(
        *_write_fixture(tmp_path),
        generated_utc="2026-07-29T00:00:00+00:00",
    )


def test_audit_separates_exact_routes_discrepancies_and_procurement(
    tmp_path: Path,
) -> None:
    result = _build_fixture(tmp_path)

    assert result["summary"]["exact_route_aldehyde_candidate_overlap"] == 1
    assert result["summary"]["source_resolved_aldehyde_routes_outside_pool"] == 1
    assert result["summary"]["unresolved_aldehyde_source_discrepancies"] == 1
    assert result["summary"]["head_candidates_with_current_procurement_closure"] == 1
    assert (
        result["summary"]["head_candidates_requiring_procurement_or_route_resolution"]
        == 2
    )
    assert result["summary"]["raw_component_cartesian_upper_bound"] == 12
    assert (
        result["summary"]["raw_triples_remaining_under_literal_registry_head_filter"]
        == 8
    )
    assert (
        result["summary"]["raw_triples_remaining_under_qualified_registry_head_filter"]
        == 12
    )
    by_id = {
        record["block_id"]: record for record in result["aldehyde_candidate_audit"]
    }
    assert by_id["aldehyde:1"]["exact_source_route_ids"] == ["route-in-pool"]
    assert (
        by_id["aldehyde:2"]["source_route_evidence_status"]
        == "source_discrepancy_unresolved"
    )
    assert {flag["component_label"] for flag in result["qa_flags"]} == {
        "A5",
        "B5",
    }


def test_audit_resolves_measured_head_with_symmetry_qualified_gate(
    tmp_path: Path,
) -> None:
    result = _build_fixture(tmp_path)

    a5 = next(
        record
        for record in result["head_candidate_audit"]
        if "A5" in record["exact_agile_component_labels"]
    )
    assert a5["required_handle_match_count"] == 3
    assert a5["symmetry_distinct_required_handle_match_count"] == 1
    assert a5["allowed_site_multiplicity"] == [1]
    assert a5["passes_literal_registry_handle_policy"] is False
    assert a5["passes_qualified_registry_handle_policy"] is True
    assert result["summary"]["agile_heads_failing_raw_registry_multiplicity"] == 1
    assert result["summary"]["agile_heads_failing_qualified_site_multiplicity"] == 0
    assert result["claims_boundary"]["model_built"] is False


def test_catalog_listing_without_availability_does_not_close(
    tmp_path: Path,
) -> None:
    result = _build_fixture(tmp_path)
    heads = {
        label: record
        for record in result["head_candidate_audit"]
        for label in record["exact_agile_component_labels"]
    }

    assert heads["A1"]["current_item_level_procurement_closed"] is True
    assert (
        heads["A1"]["operational_availability"]
        == "current_item_level_vendor_verified"
    )
    assert heads["A5"]["current_item_level_procurement_closed"] is False
    assert (
        heads["A5"]["current_procurement_status"]
        == "catalog_item_verified_availability_unresolved"
    )
    assert (
        heads["A5"]["operational_availability"]
        == "route_or_procurement_resolution_required"
    )


def test_audit_rejects_procurement_identity_mismatch(tmp_path: Path) -> None:
    (
        config,
        pool,
        agile,
        registry,
        precursor,
        procurement,
        qualification,
    ) = _write_fixture(tmp_path)
    procurement_payload = json.loads(procurement.read_text())
    procurement_payload["records"][0]["canonical_smiles"] = "CCN"
    _write_json(procurement, procurement_payload)
    config_payload = json.loads(config.read_text())
    config_payload["inputs"]["agile_head_procurement"]["expected_sha256"] = _sha256(
        procurement
    )
    _write_json(config, config_payload)

    with pytest.raises(
        Ugi3AldehydeHeadCapabilityError,
        match="identity does not match AGILE",
    ):
        build_ugi3_aldehyde_head_capability(
            config,
            pool,
            agile,
            registry,
            precursor,
            procurement,
            qualification,
            generated_utc="2026-07-29T00:00:00+00:00",
        )


def test_audit_rejects_expired_procurement_snapshot(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)

    with pytest.raises(
        Ugi3AldehydeHeadCapabilityError,
        match="snapshot expired",
    ):
        build_ugi3_aldehyde_head_capability(
            *paths,
            generated_utc="2026-09-01T00:00:00+00:00",
        )


def test_audit_rejects_hash_mismatch(tmp_path: Path) -> None:
    (
        config,
        pool,
        agile,
        registry,
        precursor,
        procurement,
        qualification,
    ) = _write_fixture(tmp_path)
    registry.write_text(registry.read_text() + "\n")

    with pytest.raises(Ugi3AldehydeHeadCapabilityError, match="hash mismatch"):
        build_ugi3_aldehyde_head_capability(
            config,
            pool,
            agile,
            registry,
            precursor,
            procurement,
            qualification,
        )


def test_writer_is_deterministic_and_atomic(tmp_path: Path) -> None:
    result = _build_fixture(tmp_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    write_ugi3_aldehyde_head_capability(result, first)
    write_ugi3_aldehyde_head_capability(result, second)

    assert first.read_bytes() == second.read_bytes()
    assert not list(tmp_path.glob(".*.tmp"))


def test_config_timestamp_makes_default_build_reproducible(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)

    first = build_ugi3_aldehyde_head_capability(*paths)
    second = build_ugi3_aldehyde_head_capability(*paths)

    assert first == second
    assert first["generated_utc"] == "2026-07-29T00:00:00+00:00"


def test_committed_audit_preserves_non_closure_and_qualified_registry() -> None:
    path = REPO / "results/m0_09/ugi3_aldehyde_head_capability.json"
    if not path.exists():
        pytest.fail("committed aldehyde/head capability artifact is missing")
    result = json.loads(path.read_text())

    assert result["schema_version"] == "m0_09_ugi3_aldehyde_head_capability.v3"
    assert result["summary"]["aldehyde_candidates"] == 51
    assert result["summary"]["amine_candidates"] == 88
    assert result["summary"]["exact_route_aldehyde_candidate_overlap"] == 11
    assert result["summary"]["head_candidates_with_current_procurement_closure"] == 19
    assert (
        result["summary"][
            "agile_heads_catalog_item_verified_availability_unresolved"
        ]
        == 1
    )
    assert (
        result["summary"]["head_candidates_requiring_procurement_or_route_resolution"]
        == 69
    )
    assert result["summary"]["computationally_route_complete_candidates"] == 0
    assert result["summary"]["amine_candidates_passing_qualified_site_multiplicity"] == 85
    assert result["summary"]["agile_heads_failing_qualified_site_multiplicity"] == 0
    assert len(result["head_procurement_queue"]) == 69
    assert any(
        flag["flag_id"] == "agile_a5_raw_multiplicity_conflict_resolved"
        for flag in result["qa_flags"]
    )
    assert any(
        flag["flag_id"] == "agile_a9_current_availability_unresolved"
        for flag in result["qa_flags"]
    )
