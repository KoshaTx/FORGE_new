from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

from forge.route.audit.ugi_high_potency_molecule_audit import build_audit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _component(role: str, smiles: str, readiness: str) -> dict[str, object]:
    return {
        "role": role,
        "canonical_smiles": smiles,
        "route_readiness_class": readiness,
        "graded_evidence_class": readiness,
        "program_family": "",
        "projection_leaf_status": "not_projected",
        "proposal_discovery": None,
        "proposal_source_sets_readiness": False,
        "strict_candidate_assessment_overrides_older_graded_metadata": False,
    }


def test_build_audit_is_deterministic_and_stratified(tmp_path: Path) -> None:
    rescoring = tmp_path / "rescoring.csv.gz"
    readiness = tmp_path / "readiness.jsonl.gz"
    route_strata = [
        "exact_complete_current",
        "family_projected_all_current_leaves",
        "family_projected_partial_current_leaves",
        "unresolved_or_unsupported",
    ]
    products = ["CCNCC", "CCCNCC", "CCCCNCC", "CCCCCNCC"]
    with gzip.open(rescoring, "wt", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "arm_id",
                "draw_index",
                "canonical_product",
                "oracle_scored",
                "conservative_high_potency",
                "potency_utility",
                "oracle_mean",
                "oracle_sd",
                "lcb90",
                "calibration_ecdf",
                "authority_tier",
                "unseen_roles",
            ],
        )
        writer.writeheader()
        for index, product in enumerate(products):
            writer.writerow(
                {
                    "arm_id": "support_enriched",
                    "draw_index": index,
                    "canonical_product": product,
                    "oracle_scored": True,
                    "conservative_high_potency": True,
                    "potency_utility": 1.0 - index / 10,
                    "oracle_mean": 1.0,
                    "oracle_sd": 0.1,
                    "lcb90": 0.8,
                    "calibration_ecdf": 0.9,
                    "authority_tier": "qualified_role_holdout",
                    "unseen_roles": "isocyanide",
                }
            )
    with gzip.open(readiness, "wt") as stream:
        for index, (product, stratum) in enumerate(zip(products, route_strata, strict=True)):
            row = {
                "arm_id": "support_enriched",
                "draw_index": index,
                "canonical_product": product,
                "pattern_id": "isocyanide_only",
                "route_readiness_class": stratum,
                "components": [
                    _component("amine_head", "CN", "exact_complete_current"),
                    _component("oxoester_aldehyde_body_tail", "CCC=O", stratum),
                    _component("isocyanide_tail", "[C-]#[N+]CC", "exact_complete_current"),
                ],
            }
            stream.write(json.dumps(row) + "\n")

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "phase1_ugi_high_potency_molecule_route_audit_config.v1",
                "inputs": {
                    "terminal_rescoring": {
                        "path": rescoring.name,
                        "sha256": _sha256(rescoring),
                    },
                    "route_readiness": {
                        "path": readiness.name,
                        "sha256": _sha256(readiness),
                    },
                },
                "policy": {
                    "arm_id": "support_enriched",
                    "require_conservative_high_potency": True,
                    "representatives_per_stratum": 1,
                    "selection": "highest_utility_seed_then_greedy_morgan_distance",
                    "morgan_radius": 2,
                    "morgan_fp_size": 128,
                    "route_strata": route_strata,
                },
                "nonclaims": [],
            }
        )
    )

    first_result, first_rows = build_audit(config_path=config, repo_root=tmp_path)
    second_result, second_rows = build_audit(config_path=config, repo_root=tmp_path)

    assert first_result == second_result
    assert first_rows == second_rows
    assert first_result["summary"]["eligible_unique_products"] == 4
    assert first_result["summary"]["selected_representatives"] == 4
    assert {row["audit_stratum"] for row in first_rows} == set(route_strata)
    diagnostics = first_result["summary"]["unresolved_component_diagnostics"]
    assert diagnostics["unresolved_product_role_combinations"] == {"oxoester_aldehyde_body_tail": 1}
