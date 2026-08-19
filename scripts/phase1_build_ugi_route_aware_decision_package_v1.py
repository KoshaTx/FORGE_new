#!/usr/bin/env python3
"""Assemble the nonselecting, route-aware candidate dossier for user/PI review.

Every one of the 256 shortlist candidates gets one dossier row carrying its
molecule, exact component graphs, generation provenance, biological authority,
route evidence at every rung of the declared ladder, unresolved risks and a
declared chemotype cluster.

This selects nothing and locks nothing.  The evidence rungs are ordered and are
explicitly not interchangeable: exact indexed evidence can support a panel lock,
family projection carries unverified substrate scope, and bounded planner reach
is plausibility triage only.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_DEFAULT = Path(__file__).resolve().parents[1]
if str(REPO_DEFAULT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_DEFAULT / "src"))

from forge.potency.ugi_semantic_annotations import ROLE_NAMES  # noqa: E402
from forge.product.ugi_bounded_hybrid_route_cascade import (  # noqa: E402
    UgiBoundedHybridRouteCascadeError,
    atomic_write,
    canonical_json_bytes,
    jsonl_gzip_bytes,
    load_json,
    read_csv_gzip,
    read_jsonl_gzip,
    sha256_file,
    sha256_payload,
)
from forge.value.ugi_proposal_augmented_route_readiness import (  # noqa: E402
    EXACT,
    FAMILY_ALL,
    UNRESOLVED,
    component_readiness_class,
    product_readiness_class,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi_route_aware_decision_package_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_route_aware_decision_package.v1"
DOSSIER_SCHEMA_VERSION = "phase1_ugi_route_aware_dossier.v1"

RUNG_EXACT = "exact_indexed_route_evidence"
RUNG_FAMILY = "family_projected_all_current_leaves"
RUNG_PLANNER = "planner_reachable_to_public_catalog"
RUNG_NONE = "unresolved"
RUNG_ORDER = (RUNG_NONE, RUNG_PLANNER, RUNG_FAMILY, RUNG_EXACT)


def load_contract(repo: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config = load_json(config_path, label="decision package config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiBoundedHybridRouteCascadeError("unsupported decision package config")
    if config.get("status") != "frozen_before_route_aware_decision_package":
        raise UgiBoundedHybridRouteCascadeError("decision package config is not frozen")
    scope = config.get("scope", {})
    for guard in (
        "candidate_selection",
        "prospective_panel_lock",
        "preregistration_created",
        "route_outcomes_may_reorder_candidates",
        "planner_reach_is_route_evidence",
        "family_projection_is_exact_route_evidence",
    ):
        if scope.get(guard) is not False:
            raise UgiBoundedHybridRouteCascadeError(
                f"decision package scope guard changed: {guard}"
            )
    paths: dict[str, Path] = {}
    for label, record in config["inputs"].items():
        path = (repo / str(record["path"])).resolve()
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise UgiBoundedHybridRouteCascadeError(f"input pin changed: {label}")
        paths[label] = path
    return config, paths


def _leaves_purchasable(
    reach: dict[str, Any] | None, procurement_by: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Are the planner route's terminal materials actually obtainable?"""

    if reach is None:
        return {"checked": False, "leaves": 0, "purchasable": 0, "all_purchasable": False}
    leaves = [str(value) for value in (reach.get("stock_leaves") or [])]
    purchasable = sum(1 for leaf in leaves if (procurement_by.get(leaf) or {}).get("vendor_count"))
    return {
        "checked": True,
        "leaves": len(leaves),
        "purchasable": purchasable,
        "all_purchasable": bool(leaves) and purchasable == len(leaves),
    }


def makeability(per_role: list[dict[str, Any]]) -> str:
    """One practical verdict per candidate: can we obtain or make all three parts?

    ``buy_all``            every component is itself purchasable
    ``buy_and_make``       each component is purchasable, or has a planner route
                           whose terminal materials are all purchasable
    ``route_only``         every component has a route, but some terminal
                           materials are not confirmed obtainable
    ``blocked``            at least one component has neither
    """

    def _state(item: dict[str, Any]) -> str:
        if item["directly_purchasable"]:
            return "buy"
        if item["planner_reachable"] and item["route_leaves_purchasable"]["all_purchasable"]:
            return "make_from_purchasable"
        if item["exact"] or item["planner_reachable"]:
            return "route_only"
        return "blocked"

    states = [_state(item) for item in per_role]
    if any(state == "blocked" for state in states):
        return "blocked"
    if all(state == "buy" for state in states):
        return "buy_all"
    if any(state == "route_only" for state in states):
        return "route_only"
    return "buy_and_make"


def chemotype_cluster(row: dict[str, Any]) -> str:
    """Return the declared descriptor tuple; not a learned clustering."""

    return sha256_payload(
        {
            "amine_head_canonical_smiles": row["components"]["amine_head"],
            "aldehyde_tail_length_bucket": row["descriptors"]["aldehyde_tail_length_bucket"],
            "product_branched": bool(row["descriptors"]["product_branched"]),
            "product_unsaturated": bool(row["descriptors"]["product_unsaturated"]),
        }
    )[:16]


def build(repo: Path, config_path: Path) -> dict[str, Any]:
    config, paths = load_contract(repo, config_path)
    candidates = read_jsonl_gzip(paths["candidate_route_ledger"], label="candidate ledger")
    components = read_jsonl_gzip(paths["component_route_ledger"], label="component ledger")
    reach_rows = read_jsonl_gzip(paths["planner_reach_ledger"], label="planner reach ledger")
    graded = {
        (row["role"], row["canonical_smiles"]): row
        for row in read_csv_gzip(paths["graded_evidence_ledger"])
    }
    rescoring = {
        (row["arm_id"], int(row["draw_index"])): row
        for row in read_csv_gzip(paths["terminal_rescoring_ledger"])
    }
    component_by = {row["component_sha256"]: row for row in components}
    reach_by = {row["component_sha256"]: row for row in reach_rows}
    procurement_path = (
        repo / "results/phase1/ugi_online_procurement_snapshot_v1/procurement_ledger.jsonl.gz"
    )
    procurement_by: dict[str, dict[str, Any]] = {}
    procurement_snapshot: dict[str, Any] | None = None
    if procurement_path.is_file():
        procurement_by = {
            str(row["canonical_smiles"]): row
            for row in read_jsonl_gzip(procurement_path, label="procurement ledger")
        }
        snapshot_result = repo / "results/phase1/ugi_online_procurement_snapshot_v1/result.json"
        if snapshot_result.is_file():
            procurement_snapshot = load_json(snapshot_result, label="procurement result").get(
                "snapshot"
            )

    # Per-component readiness at each rung.
    rung_by_component: dict[str, dict[str, Any]] = {}
    for row in components:
        key = row["component_sha256"]
        evidence = graded.get((row["role"], row["canonical_smiles"]))
        family = (
            EXACT
            if row["exact_evidence"]["strict_complete"]
            else (
                component_readiness_class(evidence["graded_evidence_class"])
                if evidence
                else UNRESOLVED
            )
        )
        reach = reach_by.get(key)
        procurement = procurement_by.get(str(row["canonical_smiles"]))
        vendor_count = None if procurement is None else procurement.get("vendor_count")
        rung_by_component[key] = {
            "exact": bool(row["exact_evidence"]["strict_complete"]),
            "family_class": family,
            "planner_reachable": bool(reach and reach.get("planner_solved_to_public_catalog")),
            "planner_tested": reach is not None,
            "graded_evidence_class": (
                evidence["graded_evidence_class"]
                if evidence
                else "not_in_graded_development_ledger"
            ),
            # Step 4: is this component itself purchasable, so no synthesis is needed?
            "directly_purchasable": bool(vendor_count),
            "vendor_count": vendor_count,
            "procurement_status": (
                "not_checked" if procurement is None else str(procurement.get("status"))
            ),
            "route_leaves_purchasable": _leaves_purchasable(reach, procurement_by),
        }

    dossiers: list[dict[str, Any]] = []
    for row in candidates:
        product = row["canonical_product"]
        arm = row["reporting"]["arm_id"]
        draw = int(row["reporting"]["draw_index"])
        rescore = rescoring.get((arm, draw))
        # The audit stage synthesizes the three role records even for a candidate
        # that failed declared graph-support re-verification, so this is always 3.
        keys = [item["component_sha256"] for item in row["roles"]]
        if len(keys) != len(ROLE_NAMES):
            raise UgiBoundedHybridRouteCascadeError(
                f"candidate does not carry three role records: {product}"
            )
        per_role = [rung_by_component[key] for key in keys]
        family_product = (
            product_readiness_class([item["family_class"] for item in per_role])
            if len(per_role) == len(ROLE_NAMES)
            else UNRESOLVED
        )
        planner_product = bool(per_role) and all(
            item["exact"] or item["planner_reachable"] for item in per_role
        )
        if row["product_route_state"] == "complete":
            rung = RUNG_EXACT
        elif family_product == FAMILY_ALL or family_product == EXACT:
            rung = RUNG_FAMILY
        elif planner_product:
            rung = RUNG_PLANNER
        else:
            rung = RUNG_NONE

        open_roles = [item for item in row["roles"] if item["final_component_state"] != "complete"]
        dossiers.append(
            {
                "schema_version": DOSSIER_SCHEMA_VERSION,
                "canonical_product": product,
                "components": dict(row["components"]),
                "terminal_sha256": row["terminal_sha256"],
                "generation": {
                    "arm_id": arm,
                    "draw_index": draw,
                    "cohort": row["reporting"].get("cohort"),
                    "stratum": row["reporting"].get("stratum"),
                    "morphology_program": row["reporting"].get("program"),
                    "morphology_program_sha256": row["reporting"].get("program_sha256"),
                    "exact_refit_corpus_product": row["reporting"].get(
                        "exact_refit_corpus_product"
                    ),
                },
                "chemistry": dict(row["descriptors"]),
                "biological_authority": {
                    "authority_tier": row["reporting"].get("authority_tier"),
                    "oracle_scored": row["reporting"].get("oracle_scored"),
                    "conservative_high_potency": row["reporting"].get("conservative_high_potency"),
                    "conservative_high_source": row["reporting"].get("conservative_high_source"),
                    "oracle_mean": (rescore or {}).get("oracle_mean") or None,
                    "oracle_sd": (rescore or {}).get("oracle_sd") or None,
                    "lcb90": (rescore or {}).get("lcb90") or None,
                    "conformal_q90": (rescore or {}).get("conformal_q90") or None,
                    "calibration_scale": (rescore or {}).get("calibration_scale") or None,
                    "view_bins": (rescore or {}).get("view_bins") or None,
                    "potency_is_a_ranking_signal_not_a_validated_prediction": True,
                },
                "l1": {
                    "exact_constitutional_forward_reconstruction": bool(row["exact_l1_reverified"]),
                    "verification": row["l1_reverification"],
                },
                "route": {
                    "route_assessment_admitted": bool(row["route_assessment_admitted"]),
                    "admission_failure": row["admission_failure"],
                    "product_route_state": row["product_route_state"],
                    "evidence_rung": rung,
                    "exact_complete_roles": int(row["exact_evidence_complete_roles"]),
                    "family_projected_class": family_product,
                    "planner_reachable_all_open_roles": planner_product,
                    "roles": [
                        {
                            **item,
                            "family_class": rung_by_component[item["component_sha256"]][
                                "family_class"
                            ],
                            "planner_reachable": rung_by_component[item["component_sha256"]][
                                "planner_reachable"
                            ],
                            "planner_tested": rung_by_component[item["component_sha256"]][
                                "planner_tested"
                            ],
                            "shared_with_candidates": component_by[item["component_sha256"]][
                                "occurrence_count"
                            ],
                        }
                        for item in row["roles"]
                    ],
                    "component_receipt_refs": row["component_receipt_refs"],
                },
                "unresolved_risks": {
                    "open_roles": [item["role"] for item in open_roles],
                    "open_role_states": {
                        item["role"]: item["final_component_state"] for item in open_roles
                    },
                    "unresolved_leaf_classes": {
                        item["role"]: item["unresolved_leaf_classes"] for item in open_roles
                    },
                    "substrate_scope_unverified": rung == RUNG_FAMILY,
                    "procurement_snapshot_expires": True,
                    "never_synthesized_or_tested": True,
                },
                "makeability": makeability(per_role),
                "shopping_list": {
                    "buy": [
                        {
                            "role": item["role"],
                            "canonical_smiles": item["canonical_smiles"],
                            "vendor_count": rung_by_component[item["component_sha256"]][
                                "vendor_count"
                            ],
                        }
                        for item in row["roles"]
                        if rung_by_component[item["component_sha256"]]["directly_purchasable"]
                    ],
                    "synthesise": [
                        {
                            "role": item["role"],
                            "canonical_smiles": item["canonical_smiles"],
                            "planner_route_available": rung_by_component[item["component_sha256"]][
                                "planner_reachable"
                            ],
                            "route_leaves_purchasable": rung_by_component[item["component_sha256"]][
                                "route_leaves_purchasable"
                            ],
                        }
                        for item in row["roles"]
                        if not rung_by_component[item["component_sha256"]]["directly_purchasable"]
                    ],
                },
                "chemotype_cluster": chemotype_cluster(row),
                "recommended_experimental_stratum": _stratum(row, rung),
            }
        )

    dossiers.sort(key=lambda item: str(item["canonical_product"]))
    if len(dossiers) != len(candidates):
        raise UgiBoundedHybridRouteCascadeError("dossier lost candidates")

    output_dir = (repo / str(config["output_directory"])).resolve()
    ledger_path = output_dir / "candidate_dossiers.jsonl.gz"
    atomic_write(ledger_path, jsonl_gzip_bytes(dossiers))

    by_arm: dict[str, Counter[str]] = defaultdict(Counter)
    for item in dossiers:
        by_arm[item["generation"]["arm_id"]][item["route"]["evidence_rung"]] += 1

    def _fill(rung_set: set[str]) -> dict[str, int]:
        counts = {
            arm: sum(
                1
                for item in dossiers
                if item["generation"]["arm_id"] == arm
                and item["route"]["evidence_rung"] in rung_set
            )
            for arm in ("broad_prior", "support_enriched", "branch_exploration")
        }
        counts["balanced_per_causal_arm"] = min(counts["broad_prior"], counts["support_enriched"])
        return counts

    targets = config["panel_targets"]
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "route_aware_decision_package_complete",
        "config": {"path": str(config_path.relative_to(repo)), "sha256": sha256_file(config_path)},
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "evidence_ladder": config["evidence_ladder"],
        "procurement_snapshot": procurement_snapshot,
        "summary": {
            "candidates": len(dossiers),
            "makeability": dict(sorted(Counter(item["makeability"] for item in dossiers).items())),
            "makeability_by_arm": {
                arm: dict(
                    sorted(
                        Counter(
                            item["makeability"]
                            for item in dossiers
                            if item["generation"]["arm_id"] == arm
                        ).items()
                    )
                )
                for arm in ("broad_prior", "support_enriched", "branch_exploration")
            },
            "evidence_rungs": dict(
                sorted(Counter(item["route"]["evidence_rung"] for item in dossiers).items())
            ),
            "evidence_rungs_by_arm": {
                arm: dict(sorted(c.items())) for arm, c in sorted(by_arm.items())
            },
            "panel_fill": {
                "exact_only": _fill({RUNG_EXACT}),
                "exact_or_family": _fill({RUNG_EXACT, RUNG_FAMILY}),
                "exact_or_family_or_planner": _fill({RUNG_EXACT, RUNG_FAMILY, RUNG_PLANNER}),
            },
            "panel_targets": dict(targets),
            "chemotype_clusters": len({item["chemotype_cluster"] for item in dossiers}),
            "chemotype_clusters_exact": len(
                {
                    item["chemotype_cluster"]
                    for item in dossiers
                    if item["route"]["evidence_rung"] == RUNG_EXACT
                }
            ),
            "strata": dict(
                sorted(
                    Counter(item["recommended_experimental_stratum"] for item in dossiers).items()
                )
            ),
        },
        "artifacts": {
            "candidate_dossiers.jsonl.gz": {
                "path": ledger_path.name,
                "sha256": sha256_file(ledger_path),
                "rows": len(dossiers),
                "schema_version": DOSSIER_SCHEMA_VERSION,
            }
        },
        "decision": {
            "panel_locked": False,
            "preregistration_created": False,
            "candidate_selection_performed": False,
            "procurement_refresh_required_before_panel_lock": True,
            "next_gate": "explicit user or PI authorization of an evidence rung and a panel",
        },
        "scope": dict(config["scope"]),
        "nonclaims": list(config["nonclaims"]),
    }
    result = {**content, "result_sha256": sha256_payload(content)}
    atomic_write(output_dir / "result.json", canonical_json_bytes(result))
    return result


def _stratum(row: dict[str, Any], rung: str) -> str:
    """Recommend a reporting stratum; this is not a selection."""

    if not row["route_assessment_admitted"]:
        return "excluded_outside_declared_graph_support"
    high = bool(row["reporting"].get("conservative_high_potency"))
    tier = str(row["reporting"].get("authority_tier"))
    if rung == RUNG_EXACT:
        if high and tier == "qualified_role_holdout":
            return "exact_route_strongest_biological_authority"
        if high:
            return "exact_route_exploratory_biological_authority"
        return "exact_route_no_potency_claim"
    if rung == RUNG_FAMILY:
        return (
            "family_projected_route_conservative_high"
            if high
            else "family_projected_route_no_potency_claim"
        )
    if rung == RUNG_PLANNER:
        return "planner_reachable_triage_only"
    return "route_unresolved_under_frozen_system"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/route/phase1_ugi_route_aware_decision_package_v1.json"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = args.config if args.config.is_absolute() else repo / args.config
    result = build(repo, config_path)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
