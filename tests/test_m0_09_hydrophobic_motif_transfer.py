from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest
from rdkit import Chem

from forge.route.hydrophobic_motif_transfer import (
    CONFIG_SCHEMA_VERSION,
    HydrophobicMotifTransferError,
    build_hydrophobic_motif_transfer,
    write_hydrophobic_motif_transfer,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_gzip_csv(path: Path, rows: list[dict[str, str]]) -> None:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(
        text,
        fieldnames=("component_id", "role", "canonical_smiles"),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    with gzip.GzipFile(
        fileobj=io.BytesIO(),
        mode="wb",
        filename="",
        mtime=0,
    ):
        pass
    with gzip.open(path, "wt", newline="") as handle:
        handle.write(text.getvalue())


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict]:
    source_bundle = tmp_path / "source.zip"
    with zipfile.ZipFile(source_bundle, "w") as archive:
        archive.writestr("media-1.pdf", b"fixed source supplement")

    component_ledger = tmp_path / "components.csv.gz"
    _write_gzip_csv(
        component_ledger,
        [
            {
                "component_id": "aldehyde-1",
                "role": "oxoester_aldehyde_body_tail",
                "canonical_smiles": "CC=O",
            },
            {
                "component_id": "aldehyde-2",
                "role": "oxoester_aldehyde_body_tail",
                "canonical_smiles": "CCC=O",
            },
        ],
    )

    reaction_registry = tmp_path / "reactions.json"
    reaction_registry.write_text(
        json.dumps(
            {
                "registry_version": 1,
                "reactions": [
                    {
                        "reaction_id": "ugi_3cr_agile",
                        "reactant_roles": [
                            {"name": "amine_head"},
                            {"name": "oxoester_aldehyde_body_tail"},
                            {"name": "isocyanide_tail"},
                        ],
                        "atom_mapped_reaction_smarts": (
                            "[NX3;H2,H1:1].[CX3H1:2]=[OX1]."
                            "[C;-1,+0;X1:3]#[N;+1,+0;X2:4]>>"
                            "[N:1][CH1:2][C+0:3](=O)[NH1+0:4]"
                        ),
                    }
                ],
            },
            sort_keys=True,
        )
    )

    primary = Chem.MolFromSmiles("CCCCCO")
    secondary = Chem.MolFromSmiles("CCC(O)CC")
    assert primary is not None and secondary is not None
    config = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "task": "test transfer pilot",
        "pilot_id": "test-pilot",
        "generated_utc": "2026-07-29T00:00:00Z",
        "source_assets": {
            "test_supplement": {
                "asset": source_bundle.name,
                "expected_sha256": _sha256(source_bundle),
                "role": "source_supplement_archive",
                "member": "media-1.pdf",
                "expected_member_sha256": hashlib.sha256(
                    b"fixed source supplement"
                ).hexdigest(),
            }
        },
        "reference_component_ledger": {
            "asset": component_ledger.name,
            "expected_sha256": _sha256(component_ledger),
            "aldehyde_role": "oxoester_aldehyde_body_tail",
            "expected_aldehyde_components": 2,
        },
        "reaction_registry": {
            "asset": reaction_registry.name,
            "expected_sha256": _sha256(reaction_registry),
            "reaction_id": "ugi_3cr_agile",
            "anchor_reactants": {
                "amine_head": "CN(C)CCN",
                "isocyanide_tail": "[C-]#[N+]CCCC",
            },
        },
        "scope_policy": {
            "transfer_target": "hydrophobic_tail_motifs",
            "head_policy": "exact_site_defined_ugi_compatible_amine_only",
            "no_head_motif_conversion": True,
            "no_source_activity_label_inheritance": True,
            "no_source_route_inheritance": True,
            "no_procurement_inference_from_identity_reference": True,
            "forward_compatibility_is_not_experimental_success": True,
            "incomplete_routes_are_not_route_complete": True,
            "generative_oracle_and_route_support_are_independent": True,
            "missing_route_knowledge_is_not_chemical_failure": True,
            "operational_closure_not_exhaustive_reaction_mining": True,
            "reusable_synthesis_program_architecture": True,
            "chemistry_specific_evidence_required": True,
        },
        "route_outcome_taxonomy": [
            "complete_route_found",
            "chemically_implausible_or_incompatible",
            "outside_declared_support",
            "missing_route_knowledge",
        ],
        "sources": {
            "test_source": {
                "platform_id": "TEST",
                "source_asset_ids": ["test_supplement"],
                "title": "test source",
                "source_final_assembly": "test final assembly",
                "source_locator": "test source",
                "biological_provenance_policy": "no labels inherited",
                "reported_reactions": {
                    "test_esterification": {
                        "level": "source_component_preparation",
                        "transformation": "test esterification",
                        "source_locator": "test source",
                        "evidence_status": "reported",
                        "attachment_event": (
                            "alcohol_oxygen_esterified_to_test_acid"
                        ),
                        "reactants": ["alcohol", "test acid"],
                        "conditions": {"temperature": "room temperature"},
                        "outcome_evidence": "reported yield",
                    }
                },
            }
        },
        "programs": {
            "primary_alcohol_to_aldehyde": {
                "reaction_smarts": (
                    "[C;H2:1][O;H1:2]>>[C;H1:1]=[O:2]"
                ),
                "target_role": "aldehyde_tail",
                "evidence_grade": "reaction_family_precedent",
                "route_closure": "incomplete",
                "terminal_status": "not_assessed",
                "route_gap_class": (
                    "exact_substrate_route_evidence_missing"
                ),
                "claim": "structural realization only",
            }
        },
        "motifs": [
            {
                "record_id": "primary",
                "source_id": "test_source",
                "source_component_id": "D1",
                "source_reaction_id": "test_esterification",
                "source_reported_yield_percent": 90,
                "common_precursor_name": "1-pentanol",
                "common_precursor_smiles": "CCCCCO",
                "mapped_common_precursor_smiles": (
                    "CCCC[CH2:1][OH:2]"
                ),
                "identity_reference": {
                    "kind": "pubchem",
                    "identifier": "test",
                    "inchikey": Chem.MolToInchiKey(primary),
                    "accessed_utc": "2026-07-29T00:00:00Z",
                },
                "source_attachment": {
                    "motif_anchor_atom_map": 1,
                    "source_handle_atom_map": 2,
                    "event": (
                        "alcohol_oxygen_esterified_to_test_acid"
                    ),
                },
                "expected_alcohol_class": "primary",
                "motif_classes": ["linear"],
                "risk_flags": [],
                "disposition": "propose_ugi_aldehyde",
                "program_id": "primary_alcohol_to_aldehyde",
                "route_gap_class": "",
                "defer_reason": "",
            },
            {
                "record_id": "secondary",
                "source_id": "test_source",
                "source_component_id": "D2",
                "source_reaction_id": "test_esterification",
                "source_reported_yield_percent": 80,
                "common_precursor_name": "3-pentanol",
                "common_precursor_smiles": "CCC(O)CC",
                "mapped_common_precursor_smiles": (
                    "CC[CH:1]([OH:2])CC"
                ),
                "identity_reference": {
                    "kind": "pubchem",
                    "identifier": "test",
                    "inchikey": Chem.MolToInchiKey(secondary),
                    "accessed_utc": "2026-07-29T00:00:00Z",
                },
                "source_attachment": {
                    "motif_anchor_atom_map": 1,
                    "source_handle_atom_map": 2,
                    "event": (
                        "alcohol_oxygen_esterified_to_test_acid"
                    ),
                },
                "expected_alcohol_class": "secondary",
                "motif_classes": ["secondary_attachment"],
                "risk_flags": [
                    "secondary_alcohol_does_not_oxidize_to_aldehyde"
                ],
                "disposition": "defer_structure_only",
                "program_id": "",
                "route_gap_class": (
                    "alternative_ugi_handle_conversion_program_missing"
                ),
                "defer_reason": "oxidation gives a ketone",
            },
        ],
        "expected_counts": {
            "source_motifs": 2,
            "independent_source_platforms": 1,
            "reported_source_reactions": 1,
            "motifs_linked_to_reported_source_reactions": 2,
            "unambiguous_source_attachment_mappings": 2,
            "primary_alcohols": 1,
            "secondary_alcohols": 1,
            "proposed_ugi_aldehydes": 1,
            "deferred_structure_only": 1,
            "exact_current_aldehyde_pool_matches": 0,
            "frozen_ugi_forward_verified": 1,
            "computationally_route_complete": 0,
        },
        "expansion_policy": {
            "current_stage": "single_platform_pilot",
            "positive_signal": "test signal",
            "next_stage": "review a second platform",
            "broad_expansion_gate": "two platforms and a routed component",
        },
        "stopping_policy": {
            "objective": "operational closure",
            "lnpdb_role": "motif census",
            "tiers": ["native", "transfer", "gap-fill", "stop"],
            "metrics": ["motifs", "candidates", "marginal", "panel"],
            "numeric_threshold_status": "freeze before selection",
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2))
    return (
        config_path,
        source_bundle,
        component_ledger,
        reaction_registry,
        config,
    )


def test_transfer_pilot_is_deterministic_and_keeps_claims_separate(
    tmp_path: Path,
) -> None:
    config, source, components, registry, _ = _fixture(tmp_path)
    first_result, first_ledger = build_hydrophobic_motif_transfer(
        config,
        source.parent,
        components,
        registry,
    )
    second_result, second_ledger = build_hydrophobic_motif_transfer(
        config,
        source.parent,
        components,
        registry,
    )

    assert first_result == second_result
    assert first_ledger == second_ledger
    assert first_result["summary"] == {
        "source_motifs": 2,
        "independent_source_platforms": 1,
        "reported_source_reactions": 1,
        "motifs_linked_to_reported_source_reactions": 2,
        "unambiguous_source_attachment_mappings": 2,
        "primary_alcohols": 1,
        "secondary_alcohols": 1,
        "proposed_ugi_aldehydes": 1,
        "deferred_structure_only": 1,
        "exact_current_aldehyde_pool_matches": 0,
        "frozen_ugi_forward_verified": 1,
        "computationally_route_complete": 0,
    }
    assert first_result["decision"]["positive_transfer_signal"]
    assert not first_result["decision"]["positive_two_platform_signal"]
    assert not first_result["decision"]["priority_queue_expansion_authorized"]
    assert not first_result["decision"]["broad_lnpdb_expansion_authorized"]
    assert not first_result["decision"][
        "transferred_components_are_observed_l2_supervision"
    ]
    assert first_result["route_outcome_counts"] == {
        "complete_route_found": 0,
        "chemically_implausible_or_incompatible": 0,
        "outside_declared_support": 0,
        "missing_route_knowledge": 2,
    }

    output_dir = tmp_path / "out"
    write_hydrophobic_motif_transfer(
        first_result,
        first_ledger,
        output_dir,
    )
    assert (output_dir / "hydrophobic_motif_transfer.json").is_file()
    with gzip.open(
        output_dir / "hydrophobic_motif_transfer_ledger.csv.gz",
        "rt",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["record_id"] for row in rows] == [
        "primary",
        "secondary",
    ]
    assert rows[0]["proposed_ugi_component_smiles"] == "CCCCC=O"
    assert rows[0]["source_id"] == "test_source"
    assert rows[0]["source_reaction_id"] == "test_esterification"
    assert rows[0]["source_reported_yield_percent"] == "90"
    assert rows[0]["source_reaction_evidence_status"] == "reported"
    assert rows[0]["frozen_ugi_forward_products"] == "1"
    assert rows[0]["biological_label_inherited"] == "false"
    assert rows[0]["missing_encoded_route_knowledge"] == "true"
    assert rows[0]["observed_chemical_failure"] == "false"
    assert rows[0]["route_outcome_category"] == "missing_route_knowledge"
    assert rows[0]["oracle_applicability_status"] == (
        "not_evaluated_no_label_transfer"
    )
    assert rows[1]["proposed_ugi_component_smiles"] == ""
    assert rows[1]["route_closure"] == "unresolved"
    assert rows[1]["route_gap_class"] == (
        "alternative_ugi_handle_conversion_program_missing"
    )


def test_transfer_pilot_closes_exact_route_to_procured_terminal(
    tmp_path: Path,
) -> None:
    config_path, source, components, registry, config = _fixture(tmp_path)
    program = config["programs"]["primary_alcohol_to_aldehyde"]
    program["evidence_grade"] = "exact_source_route"
    program["route_closure"] = "computationally_complete"
    program["terminal_status"] = "current_item_level_procurement_closed"
    program["route_gap_class"] = ""
    program["exact_route_evidence"] = {
        "source_asset_ids": ["test_supplement"],
        "source_locator": "test supplement page 1",
        "reported_reactant": "1-pentanol",
        "reported_product": "pentanal",
        "conditions": {"temperature": "room temperature"},
        "reported_outcome": "reported isolated product",
        "terminal_evidence": {
            "vendor": "test vendor",
            "product_code": "TEST-1",
            "url": "https://example.test/TEST-1",
            "region": "US",
            "purity": "99%",
            "availability_observation": "in stock",
            "accessed_utc": "2026-07-29T00:00:00Z",
        },
    }
    config["expected_counts"]["computationally_route_complete"] = 1
    config_path.write_text(json.dumps(config))

    result, ledger = build_hydrophobic_motif_transfer(
        config_path,
        source.parent,
        components,
        registry,
    )

    assert result["summary"]["computationally_route_complete"] == 1
    assert result["route_outcome_counts"] == {
        "complete_route_found": 1,
        "chemically_implausible_or_incompatible": 0,
        "outside_declared_support": 0,
        "missing_route_knowledge": 1,
    }
    assert not result["decision"]["priority_queue_expansion_authorized"]
    with gzip.GzipFile(fileobj=io.BytesIO(ledger), mode="rb") as handle:
        rows = list(csv.DictReader(io.TextIOWrapper(handle)))
    primary = next(row for row in rows if row["record_id"] == "primary")
    assert primary["route_outcome_category"] == "complete_route_found"
    assert primary["route_support_status"] == (
        "complete_route_to_accepted_terminal"
    )
    assert primary["missing_encoded_route_knowledge"] == "false"
    assert primary["computationally_route_complete"] == "true"


def test_transfer_pilot_rejects_head_motif_conversion_policy(
    tmp_path: Path,
) -> None:
    config_path, source, components, registry, config = _fixture(tmp_path)
    invalid = deepcopy(config)
    invalid["scope_policy"]["no_head_motif_conversion"] = False
    config_path.write_text(json.dumps(invalid))

    with pytest.raises(
        HydrophobicMotifTransferError,
        match="scope-policy safeguards",
    ):
        build_hydrophobic_motif_transfer(
            config_path,
            source.parent,
            components,
            registry,
        )


def test_transfer_pilot_rejects_secondary_alcohol_as_aldehyde(
    tmp_path: Path,
) -> None:
    config_path, source, components, registry, config = _fixture(tmp_path)
    invalid = deepcopy(config)
    invalid["motifs"][1]["disposition"] = "propose_ugi_aldehyde"
    invalid["motifs"][1]["program_id"] = "primary_alcohol_to_aldehyde"
    invalid["motifs"][1]["defer_reason"] = ""
    config_path.write_text(json.dumps(invalid))

    with pytest.raises(
        HydrophobicMotifTransferError,
        match="proposed aldehyde must be an undeferred primary alcohol",
    ):
        build_hydrophobic_motif_transfer(
            config_path,
            source.parent,
            components,
            registry,
        )


def test_transfer_pilot_rejects_tampered_source_bundle(
    tmp_path: Path,
) -> None:
    config, source, components, registry, _ = _fixture(tmp_path)
    source.write_bytes(b"tampered")

    with pytest.raises(
        HydrophobicMotifTransferError,
        match="source asset test_supplement hash mismatch",
    ):
        build_hydrophobic_motif_transfer(
            config,
            source.parent,
            components,
            registry,
        )
