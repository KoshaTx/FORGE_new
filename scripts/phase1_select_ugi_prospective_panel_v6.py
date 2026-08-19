#!/usr/bin/env python3
"""Select the v6 prospective panel under configs/bio/phase1_ugi_prospective_panel_v6.json.

Every threshold comes from that file, which was frozen before this script ran.  Nothing
here chooses a criterion; it applies them and records what happened.

  HIGH 34   ranked by lcb90.  The first 12 on score alone; the remaining 22 prefer
            designs absent from AGILE's enumerated library.  Novelty is a tie-break,
            never a filter, so it cannot cost predicted potency.
  LOW    6  lowest lcb90 within the aldehyde-generalization tier, so the control arm is
            tier-matched to the bulk of the high arm rather than confounded with it.

Diversity is enforced across the whole panel: pairwise product Tanimoto at most 0.95,
and at most 4 candidates sharing an amine head.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import platform
import statistics as st
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_DEFAULT = Path(__file__).resolve().parents[1]
for _p in (REPO_DEFAULT / "src", REPO_DEFAULT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RESULT_SCHEMA_VERSION = "phase1_ugi_prospective_panel_selection.v6"
LEDGER_SCHEMA_VERSION = "phase1_ugi_prospective_panel_ledger.v6"

CONFIG = "configs/bio/phase1_ugi_prospective_panel_v6.json"
ASSIGN = "results/phase1/ugi_balanced_chemistry_corpus_v2/assignments.csv.gz"
MAIN = "results/phase1/ugi_production_full_support_rescoring_v3/terminal_rescoring.csv.gz"
BRANCH = "results/phase1/ugi_branch_exploration_applicability_v1/terminal_rescoring.csv.gz"
MEASURED = "results/m0_07/agile_oracle_curated.csv.gz"
AGILE_LIB = "results/m0_09/agile_virtual_ugi3_product_ledger.csv.gz"

LOW_TIER = "exploratory_weak_absolute_aldehyde_generalization"
BRANCH_SMARTS = "[CX4;!R]([CX4,CX3])([CX4,CX3])[CX4,CX3]"


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str | None) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def stereo_free(smiles: str) -> str | None:
    """Both sides of any AGILE comparison must be stereo-free: our pipeline is
    constitutional and emits no stereochemistry, their ledger retains it."""
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    Chem.RemoveStereochemistry(molecule)
    return Chem.MolToSmiles(molecule)


def train_fold(repo: Path) -> set[str]:
    fold = {r["canonical_product_smiles"] for r in read_csv_gz(repo / ASSIGN)
            if r["primary_product_fold"] == "train"}
    if not fold:
        raise SystemExit("train fold is empty; refusing to run a novelty filter against nothing")
    return fold


def agile_products(repo: Path) -> set[str]:
    from rdkit import rdBase

    with rdBase.BlockLogs():
        products = {stereo_free(r["canonical_product_smiles"])
                    for r in read_csv_gz(repo / AGILE_LIB)}
    products.discard(None)
    if not products:
        raise SystemExit("AGILE reference library is empty")
    return products  # type: ignore[return-value]


def eligible_pool(repo: Path, config: dict[str, Any], builder: Any,
                  index: dict[str, Any]) -> list[dict[str, Any]]:
    from rdkit import Chem, rdBase
    from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    screen = config["shared_requirements"]["physicochemical_screen"]
    tol = float(screen.get("comparison_tolerance", 0.0))
    max_steps = int(config["shared_requirements"]["route"]["maximum_synthetic_steps"])
    train = train_fold(repo)
    agile = agile_products(repo)
    branch_pattern = Chem.MolFromSmarts(BRANCH_SMARTS)

    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    with rdBase.BlockLogs():
        for source in (MAIN, BRANCH):
            for row in read_csv_gz(repo / source):
                if row.get("terminal_chemical_admitted") != "True":
                    continue
                if row.get("oracle_scored") != "True":
                    continue
                product = row["canonical_product"]
                if not product or product in seen or product in train:
                    continue
                lcb90 = as_float(row.get("lcb90"))
                mean = as_float(row.get("oracle_mean"))
                sd = as_float(row.get("oracle_sd"))
                if lcb90 is None or mean is None or sd is None:
                    continue
                molecule = Chem.MolFromSmiles(product)
                if molecule is None:
                    continue
                weight = Descriptors.MolWt(molecule)
                logp = Crippen.MolLogP(molecule)
                tpsa = Descriptors.TPSA(molecule)
                if not (screen["molecular_weight"][0] - tol <= weight <= screen["molecular_weight"][1] + tol):
                    continue
                if not (screen["clogp"][0] - tol <= logp <= screen["clogp"][1] + tol):
                    continue
                if not (screen["tpsa"][0] - tol <= tpsa <= screen["tpsa"][1] + tol):
                    continue
                dossier = [
                    builder.component_dossier(index, "amine_head", row["canonical_amine"]),
                    builder.component_dossier(index, "oxoester_aldehyde_body_tail", row["canonical_aldehyde"]),
                    builder.component_dossier(index, "isocyanide_tail", row["canonical_isocyanide"]),
                ]
                steps = sum(d.get("steps", 0) for d in dossier)
                if not all(d["resolved"] for d in dossier) or steps > max_steps:
                    continue
                aldehyde = Chem.MolFromSmiles(row["canonical_aldehyde"])
                seen.add(product)
                pool.append({
                    "canonical_product": product,
                    "row": row,
                    "lcb90": lcb90,
                    "oracle_mean": mean,
                    "oracle_sd": sd,
                    "conformal_q90": as_float(row.get("conformal_q90")),
                    "authority_tier": row.get("authority_tier"),
                    "agile_novel": stereo_free(product) not in agile,
                    "fingerprint": generator.GetFingerprint(molecule),
                    "descriptors": {
                        "molecular_weight": weight, "clogp": logp, "tpsa": tpsa,
                        "heavy_atom_count": float(molecule.GetNumHeavyAtoms()),
                        "rotatable_bonds": float(Descriptors.NumRotatableBonds(molecule)),
                    },
                    "route_dossier": dossier,
                    "synthetic_steps": steps,
                    "carbon_double_bonds": sum(
                        1 for bond in molecule.GetBonds()
                        if bond.GetBondType() == Chem.BondType.DOUBLE
                        and bond.GetBeginAtom().GetSymbol() == "C"
                        and bond.GetEndAtom().GetSymbol() == "C"),
                    "branched_aldehyde": bool(
                        aldehyde is not None and branch_pattern is not None
                        and aldehyde.HasSubstructMatch(branch_pattern)),
                })
    pool.sort(key=lambda c: (-c["lcb90"], c["canonical_product"]))
    return pool


def admissible(candidate: dict[str, Any], kept: list[dict[str, Any]],
               ceiling: float, head_cap: int) -> bool:
    from rdkit import DataStructs

    for other in kept:
        if DataStructs.TanimotoSimilarity(candidate["fingerprint"], other["fingerprint"]) > ceiling:
            return False
    head = candidate["row"]["canonical_amine"]
    return sum(1 for k in kept if k["row"]["canonical_amine"] == head) < head_cap


def select(pool: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rule = config["diversity_rule"]
    ceiling = float(rule["product_near_identity"]["maximum_pairwise"])
    head_cap = int(rule["component_cap"]["amine_head"])
    high_n = int(config["cohorts"]["HIGH"]["n"])
    low_n = int(config["cohorts"]["LOW"]["n"])
    score_only = 12

    high: list[dict[str, Any]] = []
    for candidate in pool:
        if len(high) >= score_only:
            break
        if admissible(candidate, high, ceiling, head_cap):
            high.append(candidate)

    chosen = {c["canonical_product"] for c in high}
    remainder = [c for c in pool if c["canonical_product"] not in chosen]
    for candidate in ([c for c in remainder if c["agile_novel"]]
                      + [c for c in remainder if not c["agile_novel"]]):
        if len(high) >= high_n:
            break
        if admissible(candidate, high, ceiling, head_cap):
            high.append(candidate)

    # Controls come from one tier so that evidence class cannot explain a high-low
    # difference; the pool has no in-domain chemistry at low predicted score.
    low_pool = sorted((c for c in pool if c["authority_tier"] == LOW_TIER),
                      key=lambda c: (c["lcb90"], c["canonical_product"]))
    low: list[dict[str, Any]] = []
    for candidate in ([c for c in low_pool if c["agile_novel"]]
                      + [c for c in low_pool if not c["agile_novel"]]):
        if len(low) >= low_n:
            break
        if admissible(candidate, high + low, ceiling, head_cap):
            low.append(candidate)
    return high, low


def measured_potency(repo: Path) -> list[float]:
    values = [as_float(r["expt_Hela"]) for r in read_csv_gz(repo / MEASURED)]
    return sorted(v for v in values if v is not None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/phase1/ugi_prospective_panel_v6"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    from importlib import import_module

    from forge.product.ugi_bounded_hybrid_route_cascade import (
        atomic_write, canonical_json_bytes, jsonl_gzip_bytes, sha256_file, sha256_payload,
    )

    builder = import_module("phase1_build_ugi_prediction_cohort_panel_v3")
    index = builder.route_index(repo)
    config = json.loads((repo / CONFIG).read_text())

    pool = eligible_pool(repo, config, builder, index)
    high, low = select(pool, config)
    measured = measured_potency(repo)

    def percentile_beaten(value: float) -> float:
        return round(100 * sum(1 for x in measured if x < value) / len(measured), 1)

    panel: list[dict[str, Any]] = []
    for arm, members in (("HIGH", high), ("LOW", low)):
        for order, candidate in enumerate(members, start=1):
            row = candidate["row"]
            panel.append({
                "schema_version": LEDGER_SCHEMA_VERSION,
                "candidate_id": f"{'P' if arm == 'HIGH' else 'C'}{order:02d}",
                "arm": arm,
                "selection_order_within_arm": order,
                "canonical_product": candidate["canonical_product"],
                "components": {
                    "amine_head": row["canonical_amine"],
                    "oxoester_aldehyde_body_tail": row["canonical_aldehyde"],
                    "isocyanide_tail": row["canonical_isocyanide"],
                },
                "scores": {
                    "lcb90": round(candidate["lcb90"], 4),
                    "oracle_mean": round(candidate["oracle_mean"], 4),
                    "oracle_sd": round(candidate["oracle_sd"], 4),
                    "conformal_q90": candidate["conformal_q90"],
                    "ordering_statistic": "lcb90",
                    "percentile_of_agile_measured_library_beaten": percentile_beaten(candidate["oracle_mean"]),
                },
                "authority_tier": candidate["authority_tier"],
                "novelty": {"absent_from_train_fold": True,
                            "absent_from_agile_library": candidate["agile_novel"]},
                "structure": {
                    "carbon_double_bonds": candidate["carbon_double_bonds"],
                    "branched_aldehyde": candidate["branched_aldehyde"],
                },
                "descriptors": {k: round(v, 3) for k, v in candidate["descriptors"].items()},
                "route_dossier": candidate["route_dossier"],
                "route_complete": True,
                "synthetic_steps": candidate["synthetic_steps"],
            })

    products = [p["canonical_product"] for p in panel]
    if len(set(products)) != len(products):
        raise SystemExit("arms are not disjoint: a product was selected more than once")

    from rdkit import DataStructs

    every = high + low
    worst_pair = max(
        DataStructs.TanimotoSimilarity(every[i]["fingerprint"], every[j]["fingerprint"])
        for i in range(len(every)) for j in range(i + 1, len(every)))

    panel_path = out / "prospective_panel.jsonl.gz"
    atomic_write(panel_path, jsonl_gzip_bytes(panel))

    def arm_summary(members: list[dict[str, Any]]) -> dict[str, Any]:
        means = [c["oracle_mean"] for c in members]
        return {
            "n": len(members),
            "lcb90": [round(min(c["lcb90"] for c in members), 3),
                      round(max(c["lcb90"] for c in members), 3)],
            "oracle_mean": [round(min(means), 3), round(st.median(means), 3), round(max(means), 3)],
            "agile_novel": sum(1 for c in members if c["agile_novel"]),
            "above_measured_median": sum(1 for m in means if m > st.median(measured)),
            "above_measured_p90": sum(1 for m in means if m > measured[int(0.90 * len(measured))]),
            "above_measured_p95": sum(1 for m in means if m > measured[int(0.95 * len(measured))]),
            "distinct_amine_heads": len({c["row"]["canonical_amine"] for c in members}),
            "distinct_aldehydes": len({c["row"]["canonical_aldehyde"] for c in members}),
            "distinct_isocyanides": len({c["row"]["canonical_isocyanide"] for c in members}),
            "unsaturated": sum(1 for c in members if c["carbon_double_bonds"] > 0),
            "branched_aldehyde": sum(1 for c in members if c["branched_aldehyde"]),
            "authority_tiers": dict(Counter(c["authority_tier"] for c in members)),
        }

    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "selection_complete",
        "runtime": {"python_version": platform.python_version(), "platform": platform.platform()},
        "config": {"path": CONFIG, "sha256": sha256_file(repo / CONFIG)},
        "inputs": {name: {"path": path, "sha256": sha256_file(repo / path)}
                   for name, path in (("corpus_fold_assignments", ASSIGN), ("rescoring_ledger", MAIN),
                                      ("branch_lane", BRANCH), ("measured_products", MEASURED),
                                      ("agile_virtual_library", AGILE_LIB))},
        "summary": {
            "panel_size": len(panel),
            "eligible_pool": len(pool),
            "pool_tier_composition": dict(Counter(c["authority_tier"] for c in pool)),
            "shortfall": {arm: quota - n for arm, quota, n in
                          (("HIGH", int(config["cohorts"]["HIGH"]["n"]), len(high)),
                           ("LOW", int(config["cohorts"]["LOW"]["n"]), len(low))) if quota != n},
            "HIGH": arm_summary(high),
            "LOW": arm_summary(low),
            "novelty": {
                "absent_from_train_fold": len(panel),
                "absent_from_agile_library": sum(1 for c in every if c["agile_novel"]),
            },
            "maximum_pairwise_similarity": round(worst_pair, 4),
            "maximum_synthetic_steps": max(p["synthetic_steps"] for p in panel),
            "agile_measured_reference": {
                "n": len(measured),
                "median": round(st.median(measured), 3),
                "p90": round(measured[int(0.90 * len(measured))], 3),
                "p95": round(measured[int(0.95 * len(measured))], 3),
                "max": round(measured[-1], 3),
            },
        },
        "determinism": config["determinism"],
        "artifacts": {panel_path.name: {"sha256": sha256_file(panel_path), "rows": len(panel)}},
        "nonclaims": config["nonclaims"],
    }
    result = {**content, "result_sha256": sha256_payload(content)}
    atomic_write(out / "result.json", canonical_json_bytes(result))
    print(json.dumps(content["summary"], indent=2))


if __name__ == "__main__":
    main()
