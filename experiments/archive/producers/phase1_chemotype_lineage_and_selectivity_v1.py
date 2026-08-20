#!/usr/bin/env python3
"""Three bounded checks that decide how the new aldehyde chemotypes may be described.

1. Leave-one-family-out sensitivity for the morphology-standardized composition contrast.
   With 13 aldehyde families a clustered bootstrap is already fragile; this shows directly
   whether the point estimate rests on one or two large families.

2. Linkage-centred lineage. Whole-molecule radius-2 coverage is close to automatic on long
   alkyl chains, so it has almost no power. The environment that matters is the one around the
   *defining* ether oxygen or ketone carbonyl. Classifying that centre as seen-in-aldehydes,
   seen-in-another-role, or unseen anywhere distinguishes within-role recombination from
   cross-role transfer from genuine extrapolation, and those license different sentences.

3. Carbonyl handle and forward-outcome audit. A ketone-containing aldehyde carries the intended
   handle plus a second carbonyl. Admission verifies that the target product IS produced and
   that the enumeration did not saturate; it does not verify that the target is the ONLY
   product. Exact reconstruction and unique reconstruction are different properties and the
   paper must not use them interchangeably.
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
from rdkit.Chem import rdMolDescriptors as rmd  # noqa: E402

LEDGER = REPO / "results/phase1/ugi_production_full_support_rescoring_v3/terminal_rescoring.csv.gz"
CORPUS = REPO / "results/phase1/ugi_balanced_chemistry_corpus_v2/assignments.csv.gz"
ESTER = Chem.MolFromSmarts("[CX3](=[OX1])[OX2][CX4]")
ETHER = Chem.MolFromSmarts("[CX4][OX2][CX4]")
KETONE = Chem.MolFromSmarts("[CX4][CX3](=[OX1])[CX4]")
ALDEHYDE = Chem.MolFromSmarts("[CX3H1]=[OX1]")
BRANCH = Chem.MolFromSmarts("[CX4;!R]([CX4,CX3])([CX4,CX3])[CX4,CX3]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cell_of(mol: Chem.Mol) -> tuple:
    carbons = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
    unsat = sum(1 for b in mol.GetBonds()
                if b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
                and b.GetBeginAtom().GetSymbol() == "C" and b.GetEndAtom().GetSymbol() == "C")
    return (min(carbons // 4, 6), min(len(mol.GetSubstructMatches(BRANCH)), 2),
            min(rmd.CalcNumRings(mol), 1), min(unsat, 2))


def centre_environments(mol: Chem.Mol, centres: set[int], radius: int = 2) -> set[int]:
    """Morgan identifiers whose centre atom is one of the defining linkage atoms."""
    info: dict[int, tuple] = {}
    rmd.GetMorganFingerprint(mol, radius, bitInfo=info)
    return {bit for bit, occurrences in info.items()
            if any(atom in centres and rad == radius for atom, rad in occurrences)}


def all_environments(mol: Chem.Mol, radius: int = 2) -> set[int]:
    info: dict[int, tuple] = {}
    rmd.GetMorganFingerprint(mol, radius, bitInfo=info)
    return set(info)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=200,
                        help="ketone components to audit for forward outcomes")
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "results/phase1/forge_chemotype_lineage_selectivity_v1/result.json")
    args = parser.parse_args()
    started = time.time()

    with gzip.open(LEDGER, "rt", newline="") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["terminal_chemical_admitted"] == "True" and r["canonical_product"]]
    products: dict[str, dict[str, str]] = {}
    for row in rows:
        products.setdefault(row["canonical_product"], row)
    with gzip.open(CORPUS, "rt", newline="") as fh:
        crows = list(csv.DictReader(fh))

    with rdBase.BlockLogs():
        # ---------------------------------------------- 1. leave-one-family-out
        gcell: dict[tuple, list] = defaultdict(list)
        for row in products.values():
            mol = Chem.MolFromSmiles(row["canonical_aldehyde"])
            if mol is not None:
                gcell[cell_of(mol)].append(mol.HasSubstructMatch(ESTER))
        fam: dict[str, list] = defaultdict(list)
        for row in crows:
            mol = Chem.MolFromSmiles(row["oxoester_aldehyde_body_tail_smiles"])
            if mol is not None:
                fam[row["oxoester_aldehyde_body_tail_family_id"]].append(
                    (cell_of(mol), mol.HasSubstructMatch(ESTER)))

        def standardized(records) -> float | None:
            cc: dict[tuple, list] = defaultdict(list)
            for cell, is_ester in records:
                cc[cell].append(is_ester)
            shared = [c for c in gcell if c in cc and len(gcell[c]) >= 200 and len(cc[c]) >= 200]
            if not shared:
                return None
            num = sum((sum(gcell[c]) / len(gcell[c]) - sum(cc[c]) / len(cc[c])) * len(gcell[c])
                      for c in shared)
            return num / sum(len(gcell[c]) for c in shared)

        families = sorted(fam)
        point = standardized([r for f in families for r in fam[f]])
        loo = {}
        for held in families:
            value = standardized([r for f in families if f != held for r in fam[f]])
            if value is not None:
                loo[held] = value
        lo, hi = min(loo.values()), max(loo.values())
        print(f"morphology-standardized contrast: {point:+.3f}")
        print(f"leave-one-family-out over {len(families)} families: [{lo:+.3f}, {hi:+.3f}] "
              f"(range {hi-lo:.3f})")
        worst = max(loo, key=lambda f: abs(loo[f] - point))
        print(f"  most influential family {worst}: dropping it moves the estimate to {loo[worst]:+.3f}")

        # ---------------------------------------------- 2. linkage-centred lineage
        train = [r for r in crows if r["primary_product_fold"] == "train"]
        ald_env: set[int] = set()
        for smiles in {r["oxoester_aldehyde_body_tail_smiles"] for r in train}:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                ald_env |= all_environments(mol)
        other_env: set[int] = set()
        for column in ("amine_head_smiles", "isocyanide_tail_smiles"):
            for smiles in {r[column] for r in train}:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    other_env |= all_environments(mol)

        lineage = Counter()
        examples: dict[str, list[str]] = defaultdict(list)
        novel_components = []
        for smiles in sorted({r["canonical_aldehyde"] for r in products.values()}):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or mol.HasSubstructMatch(ESTER):
                continue
            centres: set[int] = set()
            for match in mol.GetSubstructMatches(ETHER):
                centres.add(match[1])
            for match in mol.GetSubstructMatches(KETONE):
                centres.add(match[1])
            if not centres:
                continue
            novel_components.append(smiles)
            env = centre_environments(mol, centres)
            if not env:
                lineage["no radius-2 centre environment"] += 1
                continue
            if env <= ald_env:
                label = "seen in the aldehyde role"
            elif env <= (ald_env | other_env):
                label = "absent from aldehydes, seen in another role"
            elif env & (ald_env | other_env):
                label = "partly supported, partly unseen anywhere"
            else:
                label = "unseen in any training role"
            lineage[label] += 1
            if len(examples[label]) < 3:
                examples[label].append(smiles)

        print(f"\nlinkage-centred lineage for {len(novel_components)} ether/ketone components:")
        for label, n in lineage.most_common():
            print(f"  {label:48s} {n:5d}  {n/len(novel_components):.3f}")

        # ---------------------------------------------- 3. handles and forward outcomes
        from importlib import import_module
        sys.path.insert(0, str(REPO / "scripts"))
        cfg = json.loads((REPO / "configs/route/m0_09_agile_virtual_ugi3_capability.json").read_text())["scope"]
        from forge.corpus.r1_prime_audit import compile_reactions, load_reaction_definitions
        reaction = compile_reactions(load_reaction_definitions(
            (REPO / "data/vendor/qualified_reactions_v1.json",), expected_count=1,
            role_policy_overrides=cfg["role_policy_overrides"]))[0]
        max_fwd = cfg["max_forward_outcomes_per_candidate"]

        ketone_rows = [r for r in products.values()
                       if (m := Chem.MolFromSmiles(r["canonical_aldehyde"])) is not None
                       and not m.HasSubstructMatch(ESTER) and m.HasSubstructMatch(KETONE)]
        rng = random.Random(20260815)
        audit = rng.sample(ketone_rows, min(args.sample, len(ketone_rows)))
        outcome = Counter()
        handles = Counter()
        for row in audit:
            ald = Chem.MolFromSmiles(row["canonical_aldehyde"])
            handles[len(ald.GetSubstructMatches(ALDEHYDE))] += 1
            mols = [Chem.MolFromSmiles(row[c]) for c in
                    ("canonical_amine", "canonical_aldehyde", "canonical_isocyanide")]
            if any(m is None for m in mols):
                outcome["unparseable"] += 1
                continue
            built = set()
            for group in reaction.forward.RunReactants(tuple(mols), maxProducts=max_fwd):
                if len(group) != 1:
                    continue
                try:
                    copy = Chem.Mol(group[0])
                    Chem.SanitizeMol(copy)
                    built.add(Chem.MolToSmiles(copy))
                except Exception:  # noqa: BLE001
                    continue
            target = Chem.MolToSmiles(Chem.MolFromSmiles(row["canonical_product"]))
            if target not in built:
                outcome["target not reproduced"] += 1
            elif len(built) == 1:
                outcome["unique: target is the only outcome"] += 1
            else:
                outcome[f"target among {len(built)} outcomes"] += 1

        print(f"\nforward-outcome audit, {len(audit)} ketone-containing aldehyde designs:")
        for k, v in outcome.most_common():
            print(f"  {k:44s} {v:5d}  {v/len(audit):.3f}")
        print(f"  aldehyde handles per component: {dict(handles)}")

    payload = {
        "schema_version": "phase1_forge_chemotype_lineage_selectivity.v1",
        "status": "complete_lineage_and_selectivity_audit",
        "inputs": {"terminal_ledger": {"path": str(LEDGER.relative_to(REPO)),
                                       "sha256": sha256_file(LEDGER)}},
        "runtime": {"python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "elapsed_seconds": round(time.time() - started, 1)},
        "leave_one_family_out": {
            "point_estimate": point, "families": len(families),
            "minimum": lo, "maximum": hi, "range": hi - lo,
            "most_influential_family": worst, "per_family": loo,
        },
        "linkage_centred_lineage": {
            "components": len(novel_components),
            "classification": dict(lineage),
            "examples": {k: v for k, v in examples.items()},
            "why_centre_and_not_whole_molecule": "radius-2 coverage of long alkyl chains is close "
                "to automatic, so whole-molecule coverage has little power; the defining ether or "
                "ketone centre is the environment that carries the claim",
        },
        "forward_outcome_audit": {
            "sampled": len(audit),
            "population": len(ketone_rows),
            "outcomes": dict(outcome),
            "aldehyde_handles_per_component": dict(handles),
            "exact_versus_unique": "admission verifies the target IS produced and that enumeration "
                "did not saturate; it does not verify the target is the ONLY product. These are "
                "different properties and must not be used interchangeably.",
        },
        "nonclaims": [
            "A shrinking standardized contrast is a descriptive attenuation, not a causal decomposition of the discrepancy.",
            "Linkage-centre support indicates recombination of locally supported chemistry, not stability, chemoselectivity, preparability or biological value.",
            "A unique forward outcome under the qualified transform is a statement about the enumerated reaction, not about competing reactivity under experimental conditions.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"\nwrote {args.output.relative_to(REPO)}")


if __name__ == "__main__":
    main()
