#!/usr/bin/env python3
"""Select the two exploratory frontier designs under a rule frozen in advance.

The activity predictor abstains on any design containing a component identity
absent from the measured training set.  That rule is identity-based, so it treats
a one-carbon homolog exactly like genuinely novel connectivity: AGILE's H9 is
abstained on an aldehyde that sits 0.90 Tanimoto from a measured one.

This pass does not change the abstention rule.  It selects a small number of
designs from just beyond the support boundary for prospective measurement, using
the raw ensemble mean as an explicitly unqualified ordering criterion.  Every
threshold comes from `configs/bio/phase1_ugi_frontier_eligibility_v1.json`, which
was frozen before any out-of-domain score was inspected.

Run the scoring stage under the pinned environment that matches the oracle refit
(rdkit 2025.09.6, scikit-learn 1.8.0, torch 2.11.0); the checkpoint refuses to
load otherwise, and that guard is deliberate.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import platform
import sys
from pathlib import Path
from typing import Any

REPO_DEFAULT = Path(__file__).resolve().parents[1]
for _p in (REPO_DEFAULT / "src", REPO_DEFAULT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RESULT_SCHEMA_VERSION = "phase1_ugi_frontier_selection.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_frontier_selection_ledger.v1"

ROLE_COLUMN = {
    "aldehyde": "canonical_aldehyde",
    "isocyanide": "canonical_isocyanide",
}
ROLE_CONFIG_KEY = {
    "aldehyde": "oxoester_aldehyde_body_tail",
    "isocyanide": "isocyanide_tail",
}
MEASURED_COLUMN = {"aldehyde": "B_smiles", "isocyanide": "C_smiles"}

# Substructures excluded by the frozen configuration.  Written as SMARTS here so
# the screen is reproducible rather than described in prose.
PATHOLOGICAL_SMARTS = {
    "acyl_halide": "[CX3](=O)[F,Cl,Br,I]",
    "retained_aldehyde": "[CX3H1](=O)[#6]",
    "epoxide": "C1OC1",
    "isocyanate": "[NX2]=[CX2]=[OX1]",
    "peroxide": "[OX2][OX2]",
    "azide": "[NX2]=[NX2+]=[NX1-]",
    "n_nitroso": "[NX3][NX2]=[OX1]",
    "enone": "[CX3]=[CX3][CX3]=[OX1]",
    "silicon": "[Si]",
    "boron": "[B]",
    "halogen": "[F,Cl,Br,I]",
}


def load_config(repo: Path) -> dict[str, Any]:
    path = repo / "configs/bio/phase1_ugi_frontier_eligibility_v1.json"
    return json.loads(path.read_text())


def measured_component_fingerprints(repo: Path) -> dict[str, list[Any]]:
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    path = repo / "results/m0_07/agile_oracle_curated.csv.gz"
    with gzip.open(path, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fingerprints: dict[str, list[Any]] = {}
    with rdBase.BlockLogs():
        for role, column in MEASURED_COLUMN.items():
            molecules = [Chem.MolFromSmiles(s) for s in {r[column] for r in rows}]
            fingerprints[role] = [generator.GetFingerprint(m) for m in molecules if m is not None]
    return fingerprints


def structural_pool(repo: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Designs failing support in exactly one permitted role, near the boundary."""

    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    measured = measured_component_fingerprints(repo)
    eligibility = config["eligibility"]
    minimum = eligibility["near_boundary_similarity"]["minimum_nearest_neighbour_similarity"]
    envelope = eligibility["physicochemical_screen"]
    tolerance = float(envelope.get("comparison_tolerance", 0.0))

    path = repo / "results/phase1/ugi_production_full_support_rescoring_v3/terminal_rescoring.csv.gz"
    with gzip.open(path, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))

    patterns = [Chem.MolFromSmarts(s) for s in PATHOLOGICAL_SMARTS.values()]
    names = list(PATHOLOGICAL_SMARTS)
    similarity_cache: dict[tuple[str, str], float] = {}
    pool: list[dict[str, Any]] = []
    seen_products: set[str] = set()

    with rdBase.BlockLogs():
        for row in rows:
            if row["oracle_scored"] != "False" or row["terminal_chemical_admitted"] != "True":
                continue
            role = row["unseen_roles"]
            if role not in ROLE_COLUMN:
                continue
            product = row["canonical_product"]
            if product in seen_products:
                continue
            component = row[ROLE_COLUMN[role]]

            key = (role, component)
            if key not in similarity_cache:
                molecule = Chem.MolFromSmiles(component)
                similarity_cache[key] = (
                    0.0
                    if molecule is None
                    else max(
                        (
                            DataStructs.TanimotoSimilarity(generator.GetFingerprint(molecule), f)
                            for f in measured[role]
                        ),
                        default=0.0,
                    )
                )
            similarity = similarity_cache[key]
            if similarity < float(minimum[ROLE_CONFIG_KEY[role]]):
                continue

            molecule = Chem.MolFromSmiles(product)
            if molecule is None:
                continue
            weight = Descriptors.MolWt(molecule)
            logp = Crippen.MolLogP(molecule)
            tpsa = Descriptors.TPSA(molecule)
            def inside(value: float, bounds: list[float]) -> bool:
                # The bounds are percentiles of the same descriptor, so exact
                # float equality at a bound is arbitrary; compare with tolerance.
                return bounds[0] - tolerance <= value <= bounds[1] + tolerance

            if not inside(weight, envelope["molecular_weight"]):
                continue
            if not inside(logp, envelope["clogp"]):
                continue
            if not inside(tpsa, envelope["tpsa"]):
                continue
            hit = next(
                (n for n, p in zip(names, patterns) if p is not None and molecule.HasSubstructMatch(p)),
                None,
            )
            if hit is not None:
                continue

            seen_products.add(product)
            pool.append(
                {
                    "schema_version": LEDGER_SCHEMA_VERSION,
                    "arm_id": row["arm_id"],
                    "draw_index": int(row["draw_index"]),
                    "canonical_product": product,
                    "canonical_amine": row["canonical_amine"],
                    "canonical_aldehyde": row["canonical_aldehyde"],
                    "canonical_isocyanide": row["canonical_isocyanide"],
                    "unseen_role": role,
                    "unseen_component": component,
                    "boundary_similarity": round(similarity, 4),
                    "abstention_reason": row["reason"],
                    "molecular_weight": round(weight, 2),
                    "clogp": round(logp, 3),
                    "tpsa": round(tpsa, 2),
                }
            )
    pool.sort(key=lambda r: (r["arm_id"], r["draw_index"]))
    return pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/phase1/ugi_frontier_selection_v1")
    )
    parser.add_argument("--stage", choices=["pool"], default="pool")
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    from forge.design.flow.ugi_bounded_hybrid_route_cascade import (
        atomic_write,
        canonical_json_bytes,
        jsonl_gzip_bytes,
        sha256_file,
        sha256_payload,
    )

    config = load_config(repo)
    pool = structural_pool(repo, config)
    path = out / "frontier_structural_pool.jsonl.gz"
    atomic_write(path, jsonl_gzip_bytes(pool))

    by_role: dict[str, int] = {}
    for record in pool:
        by_role[record["unseen_role"]] = by_role.get(record["unseen_role"], 0) + 1
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "structural_pool_complete",
        "stage": "pool",
        "runtime": {"python_version": platform.python_version(), "platform": platform.platform()},
        "config": {
            "path": "configs/bio/phase1_ugi_frontier_eligibility_v1.json",
            "sha256": sha256_file(repo / "configs/bio/phase1_ugi_frontier_eligibility_v1.json"),
        },
        "summary": {
            "structurally_eligible_products": len(pool),
            "by_unseen_role": dict(sorted(by_role.items())),
            "distinct_unseen_components": len({r["unseen_component"] for r in pool}),
        },
        "artifacts": {
            "frontier_structural_pool.jsonl.gz": {
                "path": path.name,
                "sha256": sha256_file(path),
                "rows": len(pool),
            }
        },
        "nonclaims": [
            "Structural eligibility is not a prediction of activity.",
            "No score was used to build this pool.",
        ],
    }
    result = {**content, "result_sha256": sha256_payload(content)}
    atomic_write(out / "result_pool.json", canonical_json_bytes(result))
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
