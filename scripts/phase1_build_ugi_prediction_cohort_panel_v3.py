#!/usr/bin/env python3
"""Assemble the locked 30-candidate panel and its pre-synthesis audit package.

Three prediction-evidence cohorts, differing along one axis only: whether the
aldehyde identity was measured, near the measured set, or beyond it.

    Q  qualified          all component identities measured
    N  analog transfer    one unseen aldehyde inside the similarity envelope
    X  extrapolative      one unseen aldehyde beyond it

Q-low controls are matched to Q-high on non-oracle covariates and are deliberately
*not* excluded for resembling them: a control that differs only in predicted rank
is what makes the ranking comparison interpretable.  Applying a global diversity
ceiling instead, as the v1 protocol did, drove the twelfth Q-high candidate to a
negative lower bound and collapsed the control pool.

Nothing here has been synthesised.  The audit exists so that candidate identities
can be locked before any chemistry is attempted.
"""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_DEFAULT = Path(__file__).resolve().parents[1]
for _p in (REPO_DEFAULT / "src", REPO_DEFAULT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RESULT_SCHEMA_VERSION = "phase1_ugi_prediction_cohort_panel.v2"
LEDGER_SCHEMA_VERSION = "phase1_ugi_prediction_cohort_panel_ledger.v2"
COHORT_DIR = "results/phase1/ugi_prediction_cohorts_v1"

ROLES = ("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail")
NX_COLUMN = {
    "amine_head": "canonical_amine",
    "oxoester_aldehyde_body_tail": "canonical_aldehyde",
    "isocyanide_tail": "canonical_isocyanide",
}


def route_index(repo: Path) -> dict[str, Any]:
    from forge.product.ugi_bounded_hybrid_route_cascade import read_jsonl_gzip

    def load(names: tuple[str, ...], label: str) -> dict[str, dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for name in names:
            path = repo / "results/phase1" / name
            if path.is_file():
                for row in read_jsonl_gzip(path, label=label):
                    merged.setdefault(str(row["canonical_smiles"]), row)
        return merged

    # The panel ledgers cover components drawn from the out-of-domain population,
    # which the in-domain passes never targeted.  They are listed first so a
    # panel-specific record wins, but in practice the key sets are disjoint.
    procurement = load(
        ("ugi_prediction_cohort_panel_v4/panel_procurement_ledger.jsonl.gz",
         "ugi_online_procurement_snapshot_v1/procurement_ledger.jsonl.gz",
         "ugi_indomain_makeability_v1/indomain_procurement_ledger.jsonl.gz"),
        "procurement",
    )
    tail_a = load(
        ("ugi_prediction_cohort_panel_v4/panel_tail_a_ledger.jsonl.gz",
         "ugi_tail_a_disconnection_v1/tail_a_disconnection_ledger.jsonl.gz"),
        "tail A",
    )
    isocyanide = load(
        ("ugi_prediction_cohort_panel_v4/panel_isocyanide_ledger.jsonl.gz",
         "ugi_isocyanide_route_v1/isocyanide_route_ledger.jsonl.gz"),
        "isocyanide",
    )
    return {"procurement": procurement, "tail_a": tail_a, "isocyanide": isocyanide}


def component_dossier(index: dict[str, Any], role: str, smiles: str) -> dict[str, Any]:
    """Buy it, or make it by a forward-verified route from named materials."""

    vendors = (index["procurement"].get(smiles) or {}).get("vendor_count")
    if vendors:
        return {"role": role, "canonical_smiles": smiles, "action": "purchase",
                "vendor_count": vendors, "starting_materials": [], "resolved": True}
    if role == "oxoester_aldehyde_body_tail":
        record = index["tail_a"].get(smiles)
        if record and record.get("tail_a_verified"):
            materials = [
                {"role": "carboxylic acid", "canonical_smiles": record.get("acid"),
                 "vendor_count": record.get("acid_vendor_count")},
                {"role": "alpha,omega-diol", "canonical_smiles": record.get("diol"),
                 "vendor_count": record.get("diol_vendor_count")},
            ]
            return {"role": role, "canonical_smiles": smiles, "action": "synthesise",
                    "route": "AGILE Tail A", "steps": 2, "starting_materials": materials,
                    "resolved": all(m["vendor_count"] for m in materials)}
    if role == "isocyanide_tail":
        record = index["isocyanide"].get(smiles)
        if record and record.get("route_verified"):
            materials = [{"role": "primary amine", "canonical_smiles": record.get("precursor_amine"),
                          "vendor_count": record.get("amine_vendor_count")}]
            return {"role": role, "canonical_smiles": smiles, "action": "synthesise",
                    "route": "isocyanide from primary amine", "steps": 2,
                    "starting_materials": materials,
                    "resolved": bool(materials[0]["vendor_count"])}
    return {"role": role, "canonical_smiles": smiles, "action": "unresolved",
            "starting_materials": [], "resolved": False}


def similarity_matrix(products: list[str]) -> list[list[float]]:
    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    with rdBase.BlockLogs():
        fingerprints = [generator.GetFingerprint(Chem.MolFromSmiles(p)) for p in products]
    return [
        [round(DataStructs.TanimotoSimilarity(a, b), 3) for b in fingerprints]
        for a in fingerprints
    ]


def build(repo: Path, output_dir: Path) -> dict[str, Any]:
    from forge.product.ugi_bounded_hybrid_route_cascade import (
        atomic_write, canonical_json_bytes, jsonl_gzip_bytes, sha256_file, sha256_payload,
    )

    cohorts = repo / COHORT_DIR
    q = json.loads((cohorts / "selected_Q_v3.json").read_text())
    nx = json.loads((cohorts / "selected_N_X.json").read_text())
    branched = json.loads((cohorts / "selected_B.json").read_text())
    index = route_index(repo)

    panel: list[dict[str, Any]] = []

    def add(records: list[dict[str, Any]], cohort: str, subgroup: str, from_nx: bool) -> None:
        for order, record in enumerate(records, start=1):
            components = (
                {role: record[NX_COLUMN[role]] for role in ROLES}
                if from_nx
                else {role: record["components"][role] for role in ROLES}
            )
            dossier = [component_dossier(index, role, components[role]) for role in ROLES]
            panel.append(
                {
                    "schema_version": LEDGER_SCHEMA_VERSION,
                    "candidate_id": f"{cohort}{len([p for p in panel if p['cohort'] == cohort]) + 1:02d}",
                    "cohort": cohort,
                    "subgroup": subgroup,
                    "selection_order_within_subgroup": order,
                    "canonical_product": record["canonical_product"],
                    "components": components,
                    "arm_id": record.get("arm_id"),
                    "selection_score": (
                        # B is applicability-qualified in the branch-conditioned lane and
                        # therefore carries a calibrated lower bound, unlike N and X, which
                        # the oracle abstains on and which are ordered by mean minus sd.
                        {"kind": "calibrated_conformal_lower_bound_lcb90",
                         "value": float(record["lcb90"]),
                         "raw_mean": float(record["oracle_mean"]),
                         "qualified": True}
                        if cohort == "B"
                        else {"kind": "conservative_ensemble_mean_minus_sd",
                              "value": record.get("conservative_score"),
                              "raw_mean": record.get("raw_mean"),
                              "raw_sd": record.get("raw_sd"),
                              "qualified": False}
                        if from_nx
                        else {"kind": "calibrated_conformal_lower_bound_lcb90",
                              "value": record.get("lcb90"),
                              "raw_mean": record.get("oracle_mean"),
                              "qualified": True}
                    ),
                    "aldehyde_similarity_to_measured": record.get("aldehyde_similarity"),
                    "authority_tier": record.get("authority_tier"),
                    "route_dossier": dossier,
                    "route_complete": all(d["resolved"] for d in dossier),
                    "synthetic_steps": sum(d.get("steps", 0) for d in dossier),
                }
            )

    add(q["Q_high"], "Q", "Q_high", False)
    add(q["Q_low"], "Q", "Q_low", False)
    add(nx["N_strongest"], "N", "N_strongest", True)
    add(nx["N_lower_edge"], "N", "N_lower_edge", True)
    add(nx["X_near_envelope"], "X", "X_near_envelope", True)
    add(nx["X_remote"], "X", "X_remote", True)
    # B is additive; the locked 30 above are unchanged.  Two subgroups: high-scoring
    # exploratory-tier designs, and lower-scoring qualified-tier designs sharing the
    # measured branched aldehyde, which act as an in-domain reference on the same motif.
    add([r for r in branched if r.get("_sub") == "B_exploratory"], "B", "B_exploratory", True)
    add([r for r in branched if r.get("_sub") == "B_qualified"], "B", "B_qualified", True)

    ledger = output_dir / "prediction_cohort_panel.jsonl.gz"
    atomic_write(ledger, jsonl_gzip_bytes(panel))

    by_cohort = Counter(p["cohort"] for p in panel)
    matrices = {
        cohort: {
            "candidate_ids": [p["candidate_id"] for p in panel if p["cohort"] == cohort],
            "tanimoto": similarity_matrix([p["canonical_product"] for p in panel if p["cohort"] == cohort]),
        }
        for cohort in sorted(by_cohort)
    }
    atomic_write(output_dir / "within_cohort_similarity.json", canonical_json_bytes(matrices))

    qh = [p for p in panel if p["subgroup"] == "Q_high"]
    ql = [p for p in panel if p["subgroup"] == "Q_low"]
    reuse = {
        role: Counter(p["components"][role] for p in panel) for role in ROLES
    }
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "panel_locked_pending_chemist_review",
        "runtime": {"python_version": platform.python_version(), "platform": platform.platform()},
        "protocol": {
            "v1": "configs/bio/phase1_ugi_prediction_cohorts_v1.json",
            "v4": "configs/bio/phase1_ugi_prediction_cohorts_v4.json",
            "v4_sha256": sha256_file(repo / "configs/bio/phase1_ugi_prediction_cohorts_v4.json"),
        },
        "summary": {
            "panel_size": len(panel),
            "by_cohort": dict(sorted(by_cohort.items())),
            "by_subgroup": dict(sorted(Counter(p["subgroup"] for p in panel).items())),
            "route_complete": sum(1 for p in panel if p["route_complete"]),
            "maximum_synthetic_steps": max(p["synthetic_steps"] for p in panel),
            "distinct_components": {role: len(reuse[role]) for role in ROLES},
            "maximum_component_reuse": {role: max(reuse[role].values()) for role in ROLES},
            "q_high_score_range": [min(p["selection_score"]["value"] for p in qh),
                                   max(p["selection_score"]["value"] for p in qh)],
            "q_low_score_range": [min(p["selection_score"]["value"] for p in ql),
                                  max(p["selection_score"]["value"] for p in ql)],
            "distinct_unseen_aldehydes": {
                cohort: len({p["components"]["oxoester_aldehyde_body_tail"]
                             for p in panel if p["cohort"] == cohort})
                for cohort in ("N", "X")
            },
        },
        "artifacts": {
            "prediction_cohort_panel.jsonl.gz": {
                "path": ledger.name, "sha256": sha256_file(ledger), "rows": len(panel)},
            "within_cohort_similarity.json": {
                "path": "within_cohort_similarity.json",
                "sha256": sha256_file(output_dir / "within_cohort_similarity.json")},
        },
        "interpretation": {
            "Q_high_versus_Q_low": "primary prospective test of whether ranking within the qualified domain enriches activity",
            "N": "secondary prospective test of analog-supported transfer to one unseen aldehyde",
            "X": "exploratory challenge cohort immediately beyond the analog-support envelope",
            "pooling": "Q, N and X are never pooled into one ranking-performance estimate",
            "primary_endpoint": "continuous biological measurement; thresholded hit rate is secondary",
        },
        "nonclaims": [
            "No compound here has been synthesised, formulated or tested.",
            "Cohort membership is not a prediction of activity.",
            "The conservative ensemble score used for N and X carries no coverage guarantee.",
            "X comprises three products from two distinct unseen aldehydes and is not three independent demonstrations of extrapolation.",
            "This is a locked panel proposal for chemist review, not a preregistration.",
        ],
    }
    result = {**content, "result_sha256": sha256_payload(content)}
    atomic_write(output_dir / "result.json", canonical_json_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/phase1/ugi_prediction_cohort_panel_v4"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    result = build(repo, out.resolve())
    print(json.dumps(result["summary"], indent=2, default=str))


if __name__ == "__main__":
    main()
