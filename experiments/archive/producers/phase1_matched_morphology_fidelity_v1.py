#!/usr/bin/env python3
"""Separate what the generator learned from what the proposal chose.

FORGE is a conditional generator. What a run produces is

    q_pi(x) = sum_c pi(c) p_theta(x | c),

where c is a morphology program and pi is the proposal used at sampling time. Comparing a run's
raw composition against the structural corpus therefore compares p_data(c) p_data(x|c) against
pi(c) p_theta(x|c) and blames any difference on the model, when the proposal is free to differ
by design. The production pool makes this concrete: it is two arms with two different pi, one of
them deliberately tilted toward chemistry likely to reach predictive support.

So the earlier claim that the generator "reproduces the corpus composition" was not merely
inconvenient, it was not a well-posed test. This one is. Corpus and generated aldehyde
components are binned into matched morphology cells - carbon count, branch points, ring count,
degree of unsaturation - and the ester share is compared **within** cells. Cell membership is
computed identically on both sides from structure alone, so no proposal weighting is required.

Two readings follow. If within-cell shares agree, the raw prevalence gap is proposal and reuse,
and the conditional model is faithful on this axis. If they disagree within cells, that is a
learned-model difference and should be reported as one.

The second half audits provenance for the generated aldehyde chemotypes absent from the corpus
aldehyde registry: are their local atom environments present in training, and present in the
aldehyde role specifically? That distinguishes compositional recombination of learned chemistry
from genuinely novel local chemistry, which deserve very different language.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from rdkit import Chem, rdBase  # noqa: E402
from rdkit.Chem import rdMolDescriptors  # noqa: E402

LEDGER = REPO / "results/phase1/ugi_production_full_support_rescoring_v3/terminal_rescoring.csv.gz"
CORPUS = REPO / "results/phase1/ugi_balanced_chemistry_corpus_v2/assignments.csv.gz"
ESTER = "[CX3](=[OX1])[OX2][CX4]"
BRANCH = "[CX4;!R]([CX4,CX3])([CX4,CX3])[CX4,CX3]"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def morphology_cell(mol: Chem.Mol, branch: Chem.Mol) -> tuple:
    """A structural cell computable identically on corpus and generated components.

    Deliberately coarse: these are the axes the morphology program controls, so conditioning on
    them removes the proposal's freedom to choose differently without needing its density.
    """
    carbons = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
    unsat = sum(1 for b in mol.GetBonds()
                if b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
                and b.GetBeginAtom().GetSymbol() == "C" and b.GetEndAtom().GetSymbol() == "C")
    return (
        min(carbons // 4, 6),                                  # size bucket, 4 carbons wide
        min(len(mol.GetSubstructMatches(branch)), 2),          # branch points
        min(rdMolDescriptors.CalcNumRings(mol), 1),            # any ring
        min(unsat, 2),                                         # carbon-carbon unsaturation
    )


def environments(mol: Chem.Mol, radius: int) -> set[int]:
    """Atom-centred labelled environments up to the given radius, as Morgan invariants."""
    info: dict[int, tuple] = {}
    from rdkit.Chem import rdMolDescriptors as rmd
    rmd.GetMorganFingerprint(mol, radius, bitInfo=info)
    return set(info)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-cell", type=int, default=20,
                        help="minimum components on each side before a cell is compared")
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "results/phase1/forge_matched_morphology_fidelity_v1/result.json")
    args = parser.parse_args()
    started = time.time()

    ester = Chem.MolFromSmarts(ESTER)
    branch = Chem.MolFromSmarts(BRANCH)

    with gzip.open(LEDGER, "rt", newline="") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["terminal_chemical_admitted"] == "True" and r["canonical_product"]]
    products: dict[str, dict[str, str]] = {}
    for row in rows:
        products.setdefault(row["canonical_product"], row)

    with gzip.open(CORPUS, "rt", newline="") as fh:
        crows = list(csv.DictReader(fh))
    corpus_alds = {r["oxoester_aldehyde_body_tail_smiles"] for r in crows}
    corpus_train_alds = {r["oxoester_aldehyde_body_tail_smiles"] for r in crows
                         if r["primary_product_fold"] == "train"}

    with rdBase.BlockLogs():
        def describe(smiles: str) -> dict | None:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return {"mol": mol, "cell": morphology_cell(mol, branch),
                    "ester": mol.HasSubstructMatch(ester)}

        gen_alds = sorted({r["canonical_aldehyde"] for r in products.values()})
        gen = {s: d for s in gen_alds if (d := describe(s))}
        cor = {s: d for s in sorted(corpus_alds) if (d := describe(s))}
        print(f"aldehyde components: generated {len(gen)}, corpus {len(cor)}")

        # ---------------- matched-cell ester comparison
        gcell: dict[tuple, list] = defaultdict(list)
        ccell: dict[tuple, list] = defaultdict(list)
        for d in gen.values():
            gcell[d["cell"]].append(d["ester"])
        for d in cor.values():
            ccell[d["cell"]].append(d["ester"])

        shared = sorted(set(gcell) & set(ccell),
                        key=lambda c: -(len(gcell[c]) + len(ccell[c])))
        print(f"\nmorphology cells: generated {len(gcell)}, corpus {len(ccell)}, shared {len(shared)}")
        print(f"\n{'cell (size,branch,ring,unsat)':32s} {'n_gen':>6s} {'n_cor':>6s} "
              f"{'ester_gen':>10s} {'ester_cor':>10s} {'delta':>7s}")
        compared, deltas, weights = [], [], []
        for cell in shared:
            g, c = gcell[cell], ccell[cell]
            if len(g) < args.min_cell or len(c) < 3:
                continue
            eg, ec = sum(g) / len(g), sum(c) / len(c)
            compared.append({"cell": list(cell), "n_generated": len(g), "n_corpus": len(c),
                             "ester_generated": eg, "ester_corpus": ec, "delta": eg - ec})
            deltas.append(eg - ec)
            weights.append(len(g))
            print(f"{str(cell):32s} {len(g):6d} {len(c):6d} {eg:10.3f} {ec:10.3f} {eg-ec:+7.3f}")

        raw_g = sum(d["ester"] for d in gen.values()) / len(gen)
        raw_c = sum(d["ester"] for d in cor.values()) / len(cor)
        if deltas:
            weighted = sum(d * w for d, w in zip(deltas, weights, strict=True)) / sum(weights)
            print(f"\nraw component ester share      generated {raw_g:.3f}  corpus {raw_c:.3f}  "
                  f"delta {raw_g-raw_c:+.3f}")
            print(f"within-cell mean delta (unweighted) {sum(deltas)/len(deltas):+.3f}")
            print(f"within-cell delta weighted by generated cell size {weighted:+.3f}")
        else:
            weighted = float("nan")
            print("\nno cell met the population floor; comparison not made")

        # ---------------- provenance of the out-of-registry chemotypes
        ether = Chem.MolFromSmarts("[CX4][OX2][CX4]")
        ketone = Chem.MolFromSmarts("[CX4][CX3](=[OX1])[CX4]")
        novel = [s for s, d in gen.items()
                 if not d["ester"] and (d["mol"].HasSubstructMatch(ether)
                                        or d["mol"].HasSubstructMatch(ketone))]
        print(f"\ngenerated aldehyde components in chemotypes absent from the corpus registry: "
              f"{len(novel)}")

        # environments observed in training, role-specific and role-agnostic
        train_ald_env: set[int] = set()
        for smiles in corpus_train_alds:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                train_ald_env |= environments(mol, 2)
        any_role_env: set[int] = set()
        train_rows = [r for r in crows if r["primary_product_fold"] == "train"]
        for column in ("amine_head_smiles", "oxoester_aldehyde_body_tail_smiles",
                       "isocyanide_tail_smiles"):
            for smiles in {r[column] for r in train_rows}:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    any_role_env |= environments(mol, 2)
        print(f"radius-2 environments in training aldehydes {len(train_ald_env)}, "
              f"across all training roles {len(any_role_env)}")

        buckets = Counter()
        coverages = []
        for smiles in novel:
            env = environments(gen[smiles]["mol"], 2)
            if not env:
                continue
            in_ald = len(env & train_ald_env) / len(env)
            in_any = len(env & any_role_env) / len(env)
            coverages.append((in_ald, in_any))
            if in_any >= 0.999:
                buckets["all local chemistry seen in training (some role)"] += 1
            elif in_any >= 0.9:
                buckets[">=90% of local chemistry seen in training"] += 1
            elif in_any >= 0.5:
                buckets["50-90% seen"] += 1
            else:
                buckets["<50% seen: genuinely novel local chemistry"] += 1
        print("\nlocal-environment provenance of those components (radius 2):")
        for k, v in buckets.most_common():
            print(f"  {k:52s} {v:5d}  {v/len(novel):.3f}")
        if coverages:
            ma = sum(c[0] for c in coverages) / len(coverages)
            mb = sum(c[1] for c in coverages) / len(coverages)
            print(f"  mean coverage by TRAINING ALDEHYDE environments only : {ma:.3f}")
            print(f"  mean coverage by environments in ANY training role   : {mb:.3f}")

        rng = random.Random(20260815)
        sample = rng.sample(sorted(novel), min(6, len(novel)))
        print("\nrandom out-of-registry components for chemist inspection:")
        for smiles in sample:
            print(f"   {smiles}")

    payload = {
        "schema_version": "phase1_forge_matched_morphology_fidelity.v1",
        "status": "complete_proposal_corrected_fidelity_and_lineage_audit",
        "inputs": {"terminal_ledger": {"path": str(LEDGER.relative_to(REPO)),
                                       "sha256": sha256_file(LEDGER)},
                   "structural_corpus": {"path": str(CORPUS.relative_to(REPO)),
                                         "sha256": sha256_file(CORPUS)}},
        "runtime": {"python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "elapsed_seconds": round(time.time() - started, 1)},
        "morphology_cell_definition": "(carbon_count//4 capped at 6, branch points capped at 2, "
                                      "has ring, C-C unsaturation capped at 2)",
        "why_matched_cells": "the production pool mixes two proposals over one conditional model, "
                             "so a raw marginal comparison tests the proposal, not the model",
        "raw_component_ester_share": {"generated": raw_g, "corpus": raw_c,
                                      "delta": raw_g - raw_c},
        "within_cell": {"cells_compared": len(compared),
                        "mean_delta": (sum(deltas) / len(deltas)) if deltas else None,
                        "generated_size_weighted_delta": weighted if deltas else None,
                        "cells": compared},
        "out_of_registry_chemotypes": {
            "components": len(novel),
            "local_environment_provenance_radius_2": dict(buckets),
            "mean_coverage_training_aldehydes_only": (
                sum(c[0] for c in coverages) / len(coverages)) if coverages else None,
            "mean_coverage_any_training_role": (
                sum(c[1] for c in coverages) / len(coverages)) if coverages else None,
            "inspection_sample": sample,
        },
        "nonclaims": [
            "Agreement within morphology cells shows the conditional model is faithful on the ester axis for the cells compared; it does not certify fidelity on axes not binned here.",
            "High local-environment coverage indicates recombination of learned chemistry, not that the resulting component is stable, chemoselective, preparable or biologically useful.",
            "None of this restores the withdrawn claim that the run reproduces the corpus marginal, which is not a well-posed test under two proposals.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"\nwrote {args.output.relative_to(REPO)}")


if __name__ == "__main__":
    main()
