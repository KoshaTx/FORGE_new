#!/usr/bin/env python3
"""Verify that redrawn chemist SMILES add stereochemistry WITHOUT changing constitution.

The chemistry review is allowed to specify geometry that the generator never emitted. It is
not allowed to silently alter which molecule a candidate is: the panel was frozen before any
experiment, and an edited constitution has no prediction attached to it.

This checks the only thing that matters mechanically. Strip stereochemistry from the returned
SMILES; the canonical constitution must be byte-identical to the frozen packet. Anything else
is a deviation and must be recorded as one, not absorbed.

Usage:
    phase1_verify_chemist_stereo_return_v1.py returned.csv
where returned.csv has columns: code, product_smiles  (component columns optional).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from rdkit import Chem, rdBase

REPO = Path(__file__).resolve().parents[1]
PACKET = REPO / "manuscript/chemist_packet/FORGE_candidate_smiles.csv"
FIELDS = ("product_smiles", "amine_head_smiles", "aldehyde_tail_smiles", "isocyanide_tail_smiles")


def canonical(smiles: str, *, stereo: bool) -> str | None:
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=stereo) if mol is not None else None


def stereo_bonds(smiles: str) -> int:
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return sum(1 for b in mol.GetBonds() if b.GetStereo() != Chem.BondStereo.STEREONONE)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    frozen = {r["code"]: r for r in csv.DictReader(PACKET.open())}
    returned = list(csv.DictReader(Path(sys.argv[1]).open()))

    ok = changed = unparsed = unspecified = 0
    for row in returned:
        code = row.get("code", "").strip()
        if code not in frozen:
            print(f"  {code or '<blank>'}: NOT A PACKET CODE")
            changed += 1
            continue
        for field in FIELDS:
            new = (row.get(field) or "").strip()
            if not new:
                continue
            old = frozen[code][field]
            new_flat, old_flat = canonical(new, stereo=False), canonical(old, stereo=False)
            if new_flat is None:
                print(f"  {code} {field}: UNPARSEABLE -> {new}")
                unparsed += 1
            elif new_flat != old_flat:
                changed += 1
                print(f"  {code} {field}: CONSTITUTION CHANGED")
                print(f"      frozen   {old_flat}")
                print(f"      returned {new_flat}")
            else:
                added = stereo_bonds(new) - stereo_bonds(old)
                if field == "product_smiles" and added == 0:
                    unspecified += 1
                    print(f"  {code}: constitution intact but NO stereochemistry added")
                else:
                    ok += 1

    print(f"\nfields matching frozen constitution : {ok}")
    print(f"constitution changed (DEVIATIONS)   : {changed}")
    print(f"unparseable                         : {unparsed}")
    print(f"no geometry added                   : {unspecified}")
    if changed:
        print("\nDo not merge. Every changed constitution is a new molecule with no frozen "
              "prediction attached; resolve each one explicitly before updating the packet.")
    return 1 if (changed or unparsed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
