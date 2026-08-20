#!/usr/bin/env python3
"""Recompute every number the manuscript states about generation, novelty and the panel.

Several Results-section percentages were not traceable to any artifact, and two of them
could not be reproduced under any denominator we could reconstruct.  This pass fixes the
definitions explicitly, computes each figure from the pinned inputs, and writes one
artifact the manuscript can cite, so that a reviewer asking "where does 70.8% come from"
has an answer.

Definitions are stated in the output next to each number, because the same claim takes
different values under different denominators.  Ester content, for example, is 73.7% over
admitted rows and 70.8% over distinct admitted products.
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

RESULT_SCHEMA_VERSION = "phase1_manuscript_numbers.v1"

MAIN = "results/phase1/ugi_production_full_support_rescoring_v3/terminal_rescoring.csv.gz"
ASSIGN = "results/phase1/ugi_balanced_chemistry_corpus_v2/assignments.csv.gz"
MEASURED = "results/m0_07/agile_oracle_curated.csv.gz"
AGILE_LIB = "results/m0_09/agile_virtual_ugi3_product_ledger.csv.gz"
PANEL_V6 = "results/phase1/ugi_prospective_panel_v6/prospective_panel.jsonl.gz"

BRANCH_SMARTS = "[CX4;!R]([CX4,CX3])([CX4,CX3])[CX4,CX3]"
ESTER_SMARTS = "[CX3](=O)[OX2][#6]"


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def component_registries(repo: Path) -> dict[str, Any]:
    measured = read_csv_gz(repo / MEASURED)
    agile_rows = read_csv_gz(repo / AGILE_LIB)
    amines, aldehydes, isocyanides = set(), set(), set()
    for row in agile_rows:
        components = json.loads(row["candidate_routes_json"])[0]["components"]
        amines.add(components["amine_head"])
        aldehydes.add(components["oxoester_aldehyde_body_tail"])
        isocyanides.add(components["isocyanide_tail"])
    corpus = read_csv_gz(repo / ASSIGN)
    scored = [r for r in read_csv_gz(repo / MAIN) if r["oracle_scored"] == "True"]
    measured_aldehydes = {r["B_smiles"] for r in measured}
    qualified_aldehydes = {r["canonical_aldehyde"] for r in scored}
    return {
        "measured_1100_product_set": {
            "definition": "distinct components across the 1,100 curated single-compound AGILE products",
            "amine_head": len({r["A_smiles"] for r in measured}),
            "aldehyde": len(measured_aldehydes),
            "isocyanide_tail": len({r["C_smiles"] for r in measured}),
        },
        "agile_enumerated_library": {
            "definition": "distinct components across AGILE's 12,276-member virtual candidate library, which is the complete cross-product of its block set",
            "amine_head": len(amines), "aldehyde": len(aldehydes), "isocyanide_tail": len(isocyanides),
            "cross_product": len(amines) * len(aldehydes) * len(isocyanides),
            "library_size": len(agile_rows),
        },
        "training_corpus_block_registry": {
            "definition": "distinct components across the 112,386-product training corpus",
            "amine_head": len({r["amine_head_smiles"] for r in corpus}),
            "aldehyde": len({r["oxoester_aldehyde_body_tail_smiles"] for r in corpus}),
            "isocyanide_tail": len({r["isocyanide_tail_smiles"] for r in corpus}),
        },
        "oracle_qualified_registry": {
            "definition": "distinct components across generated designs the activity model scores",
            "amine_head": len({r["canonical_amine"] for r in scored}),
            "aldehyde": len(qualified_aldehydes),
            "isocyanide_tail": len({r["canonical_isocyanide"] for r in scored}),
        },
        "aldehyde_expansion_over_measured": round(len(qualified_aldehydes) / len(measured_aldehydes), 1),
        "note": "The measured set and the AGILE enumerated library are different reference sets and must not be conflated. The manuscript's component-count sentence describes the measured set, not the library.",
    }


def generation_counts(repo: Path) -> dict[str, Any]:
    rows = read_csv_gz(repo / MAIN)
    valid = [r for r in rows if r["raw_molecule_valid"] == "True"]
    admitted = [r for r in rows if r["terminal_chemical_admitted"] == "True"]
    products = {r["canonical_product"] for r in admitted}
    products.discard("")
    return {
        "attempted_draws": len(rows),
        "chemically_valid_and_exact": len(valid),
        "admitted_rows": len(admitted),
        "admitted_distinct_products": len(products),
        "admission_rate_over_attempted": round(len(admitted) / len(rows), 4),
    }


def novelty(repo: Path) -> dict[str, Any]:
    from rdkit import Chem, rdBase

    def canon(smiles: str) -> str | None:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        Chem.RemoveStereochemistry(molecule)
        return Chem.MolToSmiles(molecule)

    corpus = read_csv_gz(repo / ASSIGN)
    train = {r["canonical_product_smiles"] for r in corpus if r["primary_product_fold"] == "train"}
    corpus_blocks = (
        {r["amine_head_smiles"] for r in corpus},
        {r["oxoester_aldehyde_body_tail_smiles"] for r in corpus},
        {r["isocyanide_tail_smiles"] for r in corpus},
    )
    measured = read_csv_gz(repo / MEASURED)
    measured_blocks = ({r["A_smiles"] for r in measured}, {r["B_smiles"] for r in measured},
                       {r["C_smiles"] for r in measured})
    rows = read_csv_gz(repo / MAIN)
    admitted = [r for r in rows if r["terminal_chemical_admitted"] == "True"]
    seen: dict[str, dict[str, str]] = {}
    for row in admitted:
        seen.setdefault(row["canonical_product"], row)
    seen.pop("", None)
    generated = set(seen)

    def any_component_absent(source: tuple[set[str], set[str], set[str]]) -> int:
        return sum(1 for r in seen.values()
                   if r["canonical_amine"] not in source[0]
                   or r["canonical_aldehyde"] not in source[1]
                   or r["canonical_isocyanide"] not in source[2])

    with rdBase.BlockLogs():
        agile = {canon(r["canonical_product_smiles"]) for r in read_csv_gz(repo / AGILE_LIB)}
        agile.discard(None)
        agile_absent = sum(1 for p in generated if canon(p) not in agile)

    n = len(generated)
    return {
        "denominator": "distinct admitted products",
        "n": n,
        "product_absent_from_train_fold": {
            "definition": "canonical product SMILES absent from the 66,464-product train fold; the standard generative-chemistry novelty metric",
            "count": len(generated - train),
            "fraction": round(len(generated - train) / n, 4),
        },
        "product_absent_from_agile_library": {
            "definition": "absent from AGILE's 12,276-member enumerated library, compared stereo-free on both sides",
            "count": agile_absent, "fraction": round(agile_absent / n, 4),
        },
        "carries_component_absent_from_measured_set": {
            "definition": "at least one component outside the measured 20 amines, 11 aldehydes, 5 isocyanides",
            "count": any_component_absent(measured_blocks),
            "fraction": round(any_component_absent(measured_blocks) / n, 4),
        },
        "carries_component_absent_from_training_corpus": {
            "definition": "at least one component outside the training corpus block registry",
            "count": any_component_absent(corpus_blocks),
            "fraction": round(any_component_absent(corpus_blocks) / n, 4),
        },
    }


def structural_features(repo: Path) -> dict[str, Any]:
    """Reported over distinct admitted products, tail scope, which is the denominator
    under which the manuscript's ester figure reproduces exactly."""

    from rdkit import Chem, rdBase

    branch = Chem.MolFromSmarts(BRANCH_SMARTS)
    ester = Chem.MolFromSmarts(ESTER_SMARTS)
    rows = [r for r in read_csv_gz(repo / MAIN) if r["terminal_chemical_admitted"] == "True"]
    seen: dict[str, dict[str, str]] = {}
    for row in rows:
        seen.setdefault(row["canonical_product"], row)
    seen.pop("", None)

    counts = {"unsaturated_either_tail": 0, "unsaturated_aldehyde_only": 0,
              "unsaturated_isocyanide_only": 0, "ester_containing_tail": 0,
              "branched_either_tail": 0, "branched_aldehyde": 0, "branched_isocyanide": 0}
    with rdBase.BlockLogs():
        for row in seen.values():
            aldehyde = Chem.MolFromSmiles(row["canonical_aldehyde"])
            isocyanide = Chem.MolFromSmiles(row["canonical_isocyanide"])
            if aldehyde is None or isocyanide is None:
                continue

            def unsaturated(mol: Any) -> bool:
                return any(b.GetBondType() == Chem.BondType.DOUBLE
                           and b.GetBeginAtom().GetSymbol() == "C"
                           and b.GetEndAtom().GetSymbol() == "C" for b in mol.GetBonds())

            ald_uns, iso_uns = unsaturated(aldehyde), unsaturated(isocyanide)
            counts["unsaturated_either_tail"] += int(ald_uns or iso_uns)
            counts["unsaturated_aldehyde_only"] += int(ald_uns)
            counts["unsaturated_isocyanide_only"] += int(iso_uns)
            counts["ester_containing_tail"] += int(
                aldehyde.HasSubstructMatch(ester) or isocyanide.HasSubstructMatch(ester))
            ald_br = aldehyde.HasSubstructMatch(branch)
            iso_br = isocyanide.HasSubstructMatch(branch)
            counts["branched_either_tail"] += int(ald_br or iso_br)
            counts["branched_aldehyde"] += int(ald_br)
            counts["branched_isocyanide"] += int(iso_br)

    n = len(seen)
    return {
        "denominator": "distinct admitted products, tail scope",
        "n": n,
        "smarts": {"branch": BRANCH_SMARTS, "ester": ESTER_SMARTS,
                   "unsaturation": "any carbon-carbon double bond in the component"},
        "counts": counts,
        "fractions": {k: round(v / n, 4) for k, v in counts.items()},
    }


def panel(repo: Path) -> dict[str, Any]:
    path = repo / PANEL_V6
    if not path.is_file():
        return {"status": "absent"}
    records = [json.loads(line) for line in gzip.open(path, "rt")]
    high = [r for r in records if r["arm"] == "HIGH"]
    means = [r["scores"]["oracle_mean"] for r in high]
    return {
        "n": len(records),
        "high_arm": len(high),
        "low_arm": len(records) - len(high),
        "absent_from_train_fold": sum(1 for r in records if r["novelty"]["absent_from_train_fold"]),
        "absent_from_agile_library": sum(1 for r in records if r["novelty"]["absent_from_agile_library"]),
        "high_arm_predicted_mean_range": [round(min(means), 2), round(max(means), 2)],
        "high_arm_above_measured_p90": sum(
            1 for r in high if r["scores"]["percentile_of_agile_measured_library_beaten"] >= 90),
        "distinct_amine_heads": len({r["components"]["amine_head"] for r in records}),
        "distinct_aldehydes": len({r["components"]["oxoester_aldehyde_body_tail"] for r in records}),
        "distinct_isocyanides": len({r["components"]["isocyanide_tail"] for r in records}),
        "maximum_synthetic_steps": max(r["synthetic_steps"] for r in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/phase1/ugi_manuscript_numbers_v1"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    from forge.design.flow.ugi_bounded_hybrid_route_cascade import (
        atomic_write, canonical_json_bytes, sha256_file, sha256_payload,
    )

    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete",
        "purpose": "Single traceable source for every generation, novelty, component-count and panel number stated in the manuscript.",
        "runtime": {"python_version": platform.python_version(), "platform": platform.platform()},
        "inputs": {name: {"path": path, "sha256": sha256_file(repo / path)}
                   for name, path in (("rescoring_ledger", MAIN), ("corpus_fold_assignments", ASSIGN),
                                      ("measured_products", MEASURED), ("agile_virtual_library", AGILE_LIB))
                   if (repo / path).is_file()},
        "component_registries": component_registries(repo),
        "generation_counts": generation_counts(repo),
        "novelty": novelty(repo),
        "structural_features": structural_features(repo),
        "prospective_panel_v6": panel(repo),
    }
    result = {**content, "result_sha256": sha256_payload(content)}
    atomic_write(out / "result.json", canonical_json_bytes(result))
    print(json.dumps({k: content[k] for k in
                      ("component_registries", "generation_counts", "novelty",
                       "structural_features", "prospective_panel_v6")}, indent=2)[:4000])


if __name__ == "__main__":
    main()
