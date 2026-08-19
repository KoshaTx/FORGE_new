#!/usr/bin/env python3
"""Repair the Ugi reverse template and exhaustively audit post-hoc decomposition.

The frozen inverse is built by string-reversing the forward SMARTS, so the amine reactant
pattern [NX3;H2,H1:1] ends up as a *product* template. RDKit cannot honour a disjunctive H
count when writing an atom, warns "multiple H count specifications", and emits [NH2]. That is
correct for a primary amine and produces a valence-4 nitrogen for a secondary one, so every
secondary-amine-head product failed with `invalid_reactant`.

The repair does not touch the frozen module. It re-runs the same reverse reaction and, when a
fragment fails sanitisation on an over-specified nitrogen, clears the template-imposed hydrogen
count and lets RDKit infer it. Then it enumerates *every* distinct forward-reconstructing
precursor tuple rather than stopping at the first, which is what distinguishes "identifiable"
from "one valid tuple happens to exist".
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rdkit import Chem, rdBase  # noqa: E402
from forge.data.r1_prime_audit import (  # noqa: E402
    compile_reactions, load_reaction_definitions, _role_accepts,
)


def repair_and_canonicalize(fragment: Chem.Mol) -> tuple[str, Chem.Mol] | None:
    """Sanitise a reverse-template fragment, clearing template-imposed H counts if needed."""
    mol = Chem.Mol(fragment)
    for _ in range(4):
        try:
            with rdBase.BlockLogs():
                Chem.SanitizeMol(mol)
            break
        except Chem.AtomValenceException as exc:
            idx = getattr(exc, "cause", None)
            idx = idx.GetAtomIdx() if idx is not None else None
            if idx is None:  # fall back to any atom carrying explicit Hs
                candidates = [a.GetIdx() for a in mol.GetAtoms() if a.GetNumExplicitHs()]
                if not candidates:
                    return None
                idx = candidates[0]
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetNumExplicitHs() == 0 and not atom.GetNoImplicit():
                return None
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)
        except Exception:
            return None
    else:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    with rdBase.BlockLogs():
        parsed = Chem.MolFromSmiles(smiles)
    return (smiles, parsed) if parsed is not None else None


def all_valid_tuples(product_smiles, reaction, max_rev, max_fwd, stats):
    """Every distinct precursor tuple that passes role policy AND rebuilds the product."""
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(product_smiles)
    if mol is None:
        stats["unparsed_product"] += 1
        return []
    target = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    with rdBase.BlockLogs():
        outcomes = reaction.reverse.RunReactants((mol,), maxProducts=max_rev)
    roles = reaction.definition.reactant_roles
    valid, seen = [], set()
    for outcome in outcomes:
        if len(outcome) != len(roles):
            stats["wrong_fragment_count"] += 1
            continue
        smis, mols, ok = [], [], True
        for i, frag in enumerate(outcome):
            got = repair_and_canonicalize(frag)
            if got is None:
                stats["invalid_reactant_after_repair"] += 1
                ok = False
                break
            s, p = got
            if not _role_accepts(p, roles[i], reaction.handles[i], reaction.forbidden[i]):
                stats["role_policy"] += 1
                ok = False
                break
            smis.append(s)
            mols.append(p)
        if not ok:
            continue
        key = tuple(smis)
        if key in seen:
            continue
        with rdBase.BlockLogs():
            fwd = reaction.forward.RunReactants(tuple(mols), maxProducts=max_fwd)
        rebuilt = set()
        for out in fwd:
            if len(out) != 1:
                continue
            got = repair_and_canonicalize(out[0])
            if got:
                rebuilt.add(got[0])
        if target in rebuilt:
            seen.add(key)
            valid.append(key)
        else:
            stats["forward_mismatch"] += 1
    return valid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--reference", action="store_true",
                    help="regression-test on the 12,276 enumerated products instead")
    args = ap.parse_args()

    cfg = json.loads((REPO / "configs/route/m0_09_agile_virtual_ugi3_capability.json").read_text())
    scope = cfg["scope"]
    reaction = compile_reactions(load_reaction_definitions(
        (REPO / "data/vendor/qualified_reactions_v1.json",), expected_count=1,
        role_policy_overrides=scope["role_policy_overrides"]))[0]
    max_rev = scope["max_reverse_outcomes_per_product"]
    max_fwd = scope["max_forward_outcomes_per_candidate"]

    if args.reference:
        with gzip.open(REPO / "results/m0_09/agile_virtual_ugi3_product_ledger.csv.gz", "rt",
                       newline="") as fh:
            items = [(r["canonical_product_smiles"], None) for r in csv.DictReader(fh)]
    else:
        with gzip.open(REPO / "results/phase1/ugi_production_full_support_rescoring_v3/"
                              "terminal_rescoring.csv.gz", "rt", newline="") as fh:
            rows = [r for r in csv.DictReader(fh)
                    if r["terminal_chemical_admitted"] == "True" and r["canonical_product"]]
        by = {}
        for r in rows:
            by.setdefault(r["canonical_product"], r)
        items = sorted(by.items())
    if args.limit:
        items = items[:args.limit]

    PRIM = Chem.MolFromSmarts("[NX3;H2]")
    SEC = Chem.MolFromSmarts("[NX3;H1]")
    cache = {}

    def head_class(smiles):
        if smiles not in cache:
            with rdBase.BlockLogs():
                m = Chem.MolFromSmiles(smiles)
            cache[smiles] = ("unparsed" if m is None else
                             "primary" if m.HasSubstructMatch(PRIM) else
                             "secondary" if m.HasSubstructMatch(SEC) else "no_NH")
        return cache[smiles]

    counts, by_head, agree = Counter(), Counter(), Counter()
    stats = Counter()
    for i, (smiles, row) in enumerate(items, 1):
        tuples = all_valid_tuples(smiles, reaction, max_rev, max_fwd, stats)
        n = len(tuples)
        bucket = "0" if n == 0 else ("1" if n == 1 else ">1")
        counts[bucket] += 1
        if row is not None:
            hc = head_class(row["canonical_amine"])
            by_head[(hc, bucket)] += 1
            if n:
                gen = (row["canonical_amine"], row["canonical_aldehyde"], row["canonical_isocyanide"])
                agree["generated_among_valid" if gen in set(tuples) else "generated_MISSING"] += 1
                if n == 1:
                    agree["unique_matches_generated" if gen in set(tuples)
                          else "unique_DIFFERS_from_generated"] += 1
        if i % 2000 == 0:
            print(f"  {i}/{len(items)}  {dict(counts)}", flush=True)

    total = sum(counts.values())
    print(f"\n=== valid forward-reconstructing precursor tuples per product (n={total}) ===")
    for k in ("0", "1", ">1"):
        print(f"  {k:>3s} tuples : {counts[k]:6d}  {counts[k]/total:.4f}")
    if by_head:
        print("\n=== by amine head class ===")
        for hc in ("primary", "secondary", "no_NH", "unparsed"):
            row_ = [by_head[(hc, b)] for b in ("0", "1", ">1")]
            if sum(row_):
                print(f"  {hc:10s} 0:{row_[0]:6d}  1:{row_[1]:6d}  >1:{row_[2]:6d}")
    if agree:
        print("\n=== agreement with the generated factorization ===")
        for k, v in agree.most_common():
            print(f"  {k:34s} {v}")
    print("\n=== rejections encountered ===")
    for k, v in stats.most_common():
        print(f"  {k:34s} {v}")


if __name__ == "__main__":
    main()
