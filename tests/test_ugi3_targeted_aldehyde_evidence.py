from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdkit import Chem

from forge.corpus.r1_prime_audit import sha256_file
from forge.synthesis.evidence.ugi3_targeted_aldehyde_evidence import (
    Ugi3TargetedAldehydeEvidenceError,
    build_targeted_aldehyde_evidence_audit,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/phase1_ugi3_targeted_aldehyde_evidence_audit_v1.json"
INPUT_PATHS = {
    "audit_source": REPO / "src/forge/route/ugi3_targeted_aldehyde_evidence.py",
    "cli_source": REPO
    / "experiments/archive/producers/phase1_audit_ugi3_targeted_aldehyde_evidence.py",
    "evidence_pack": REPO / "configs/route/phase1_ugi3_targeted_aldehyde_evidence_v1.json",
    "kovalerchik_source": REPO
    / "data/source_cache/phase1_targeted_l2/KOVALERCHIK_2022/marinedrugs-20-00265-v2.pdf",
    "mo_source": REPO
    / "data/source_cache/phase1_targeted_l2/TYPHONOSIDES/supporting_information.pdf",
    "busta_source": REPO
    / "data/source_cache/phase1_targeted_l2/BUSTA_2016/BustaEtAl_2016_Phytochem.pdf",
    "route_readiness_ledger": REPO
    / "results/phase1/ugi3_production_registry_route_readiness/component_readiness_ledger.csv.gz",
    "route_readiness_result": REPO
    / "results/phase1/ugi3_production_registry_route_readiness/result.json",
    "product_gap_ledger": REPO
    / "results/phase1/ugi3_l2_coverage_priority_audit_v1/product_gap_ledger.csv.gz",
    "priority_result": REPO / "results/phase1/ugi3_l2_coverage_priority_audit_v1/result.json",
    "terminal_procurement": REPO / "configs/route/m0_09_ugi3_virtual_terminal_procurement.json",
    "upstream_registry": REPO / "configs/route/phase1_ugi3_upstream_qualified_reactions_v1.json",
    "oxidation_variant": REPO
    / "configs/route/variants/ugi3_upstream_primary_alcohol_oxidation_exact_source_v1.json",
    "qualified_forward_source": REPO / "src/forge/route/qualified_forward.py",
}


def test_targeted_exact_evidence_closes_four_targets_and_21_products() -> None:
    first, first_routes, first_products = build_targeted_aldehyde_evidence_audit(
        config_path=CONFIG,
        input_paths=INPUT_PATHS,
    )
    second, second_routes, second_products = build_targeted_aldehyde_evidence_audit(
        config_path=CONFIG,
        input_paths=INPUT_PATHS,
    )
    assert first == second
    assert first_routes == second_routes
    assert first_products == second_products
    assert first["summary"] == {
        "prioritized_exact_targets": 5,
        "admitted_exact_targets": 4,
        "abstained_targets": 1,
        "current_direct_procurement_terminals": 2,
        "exact_source_routes_forward_verified": 3,
        "exact_routes_with_current_l3_closure": 3,
        "registry_route_complete_components_before": 41,
        "registry_route_complete_components_after": 45,
        "generated_products": 1007,
        "generated_products_complete_before": 3,
        "generated_products_complete_after": 24,
        "newly_complete_generated_products": 21,
        "remaining_gap_count_distribution": {"0": 24, "1": 202, "2": 423, "3": 358},
        "newly_completed_products_by_target": {
            "oxoester_aldehyde_body_tail\tCCCCCCCCCCC=O": 5,
            "oxoester_aldehyde_body_tail\tCCCCCCCCCCCCCCCCCC=O": 6,
            "oxoester_aldehyde_body_tail\tCCCCCCCCCCCCCCCCCCCC=O": 6,
            "oxoester_aldehyde_body_tail\tCCCCCCCCCCCCCCCCCCCCCC=O": 4,
        },
    }
    assert first["artifacts"]["route_verification_ledger.csv.gz"]["sha256"] == (
        "ea92af40ee0b1c52e7e92150e79e29582e5959bc6cd46c7ee5b04a1fd72fb1b3"
    )
    assert first["artifacts"]["product_closure_impact_ledger.csv.gz"]["sha256"] == (
        "764d83c94561faaf99da1e48bbfa6a2f42ea7c60bf5520bb5ec2a8ddd0aff40e"
    )
    abstentions = [row for row in first["target_adjudications"] if row["disposition"] == "abstain"]
    assert len(abstentions) == 1
    assert abstentions[0]["canonical_smiles"] == "CCCCCCCCCCCCCCCCCCCCC=O"


def _tampered_inputs(
    tmp_path: Path,
    *,
    mutate,
) -> tuple[Path, dict[str, Path]]:
    evidence = json.loads(INPUT_PATHS["evidence_pack"].read_text())
    mutate(evidence)
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    config = json.loads(CONFIG.read_text())
    config["inputs"]["evidence_pack"]["expected_sha256"] = sha256_file(evidence_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    paths = dict(INPUT_PATHS)
    paths["evidence_pack"] = evidence_path
    return config_path, paths


def test_nonclosing_procurement_cannot_be_promoted(tmp_path: Path) -> None:
    def mutate(evidence: dict) -> None:
        evidence["direct_procurement_terminals"][0]["current_item_level_procurement_closed"] = False

    config, inputs = _tampered_inputs(tmp_path, mutate=mutate)
    with pytest.raises(Ugi3TargetedAldehydeEvidenceError, match="nonclosing procurement"):
        build_targeted_aldehyde_evidence_audit(config_path=config, input_paths=inputs)


def test_exact_route_must_forward_reconstruct_target(tmp_path: Path) -> None:
    def mutate(evidence: dict) -> None:
        route = evidence["exact_routes"][1]
        route["reactant"]["canonical_smiles"] = "CCCCCCCCCCCO"
        molecule = Chem.MolFromSmiles(route["reactant"]["canonical_smiles"])
        assert molecule is not None
        route["reactant"]["inchi_key"] = Chem.MolToInchiKey(molecule)

    config, inputs = _tampered_inputs(tmp_path, mutate=mutate)
    with pytest.raises(Ugi3TargetedAldehydeEvidenceError, match="uniquely reconstruct"):
        build_targeted_aldehyde_evidence_audit(config_path=config, input_paths=inputs)
