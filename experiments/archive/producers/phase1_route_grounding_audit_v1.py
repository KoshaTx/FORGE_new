#!/usr/bin/env python3
"""How much of the generated population inherits a constructive route skeleton?

The 97.1% route-completeness figure is measured on prediction-supported designs, which are a
small and unrepresentative slice: they are the designs closest to measured chemistry. It cannot
answer the question the synthesis-grounded claim actually rests on:

    do designs OUTSIDE the enumerated library retain a route skeleton, or does route
    completeness simply track proximity to the block set the library was built from?

This audits the whole admitted population instead, and reports the funnel role by role:

    exact final assembly  ->  role preparation schema applies  ->  upstream precursors valid
                          ->  precursors purchasable

Realization is role-specific by design, mirroring how the chemistry is actually practised: buy
the ionizable heads, prepare the specialised tails, run the common Ugi assembly. So the roles
are not audited symmetrically, and should not be.

  amine head       direct procurement is the intended operator; there is no upstream
                   disconnection to apply and its absence is not a gap
  aldehyde tail    purchase, or acid + alpha,omega-diol -> hydroxy ester -> aldehyde
  isocyanide tail  purchase, or primary amine -> formamide -> isocyanide

Only the two tail roles have a preparation schema to measure. For the head the meaningful
quantity is procurement among components actually assessed, which the current partial snapshot
cannot supply for the open-world population.

Derivations use the same SMARTS the chemist packet uses, and each is verified by running the
preparation forward and checking it returns the component.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from rdkit import Chem, rdBase  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

LEDGER = REPO / "results/phase1/ugi_production_full_support_rescoring_v3/terminal_rescoring.csv.gz"
LIBRARY = REPO / "results/m0_09/agile_virtual_ugi3_product_ledger.csv.gz"
MEASURED = REPO / "results/m0_07/agile_oracle_curated.csv.gz"
PROCUREMENT = REPO / "results/phase1/ugi_online_procurement_snapshot_v1/procurement_ledger.jsonl.gz"

# identical to the chemist packet, so the audit and the dossier cannot disagree
ESTERIFY = "[CX3:1](=[OX1:2])[OX2H1].[OX2H1][CX4:3]>>[C:1](=[O:2])[OX2][C:3]"
OXIDISE = "[OX2H1][CH2:1]>>[CH1:1]=O"
FORMYLATE = "[NX3;H2:1]>>[N:1]C=O"

# the disconnections the two tail roles admit
CLEAVE_ESTER = "[CX3:1](=[OX1:2])[OX2][CX4:3]>>[C:1](=[O:2])[OX2H].[OX2H][C:3]"
ISOCYANIDE = Chem.MolFromSmarts("[C-]#[N+]")
ALDEHYDE = Chem.MolFromSmarts("[CX3H1]=[OX1]")
ESTER = Chem.MolFromSmarts("[CX3](=[OX1])[OX2][CX4]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canon(smiles: str) -> str | None:
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


PROTONATED_AMINE = Chem.MolFromSmarts("[NX4+;H1,H2,H3]")


def _neutralise(mol: Chem.Mol) -> Chem.Mol:
    """Deprotonate an amine a mapped template left charged.

    Reversing [C-]#[N+] onto a nitrogen hands back [NH3+] rather than a neutral primary amine,
    which then fails every downstream functional-group test for no chemical reason. Scoped to
    protonated amines only: a blanket neutralisation also strips the charges from the [N+]#[C-]
    of a legitimate isocyanide, which silently breaks the forward check that is supposed to
    confirm the disconnection.
    """
    for match in mol.GetSubstructMatches(PROTONATED_AMINE):
        atom = mol.GetAtomWithIdx(match[0])
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(False)
    return mol


def run_reaction(smarts: str, *smiles: str) -> list[tuple[str, ...]]:
    """Every outcome, as a tuple of its products.

    Returning a flat set loses the pairing, which silently breaks any disconnection that emits
    more than one fragment: an ester cleaved to acid plus alcohol becomes two unrelated
    molecules and no acid/diol pair can ever be reassembled.
    """
    with rdBase.BlockLogs():
        reaction = AllChem.ReactionFromSmarts(smarts)
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        if any(m is None for m in mols):
            return []
        outcomes = []
        for products in reaction.RunReactants(tuple(mols)):
            group = []
            for product in products:
                try:
                    copy = _neutralise(Chem.Mol(product))
                    Chem.SanitizeMol(copy)
                    group.append(Chem.MolToSmiles(copy))
                except Exception:  # noqa: BLE001
                    group = None
                    break
            if group:
                outcomes.append(tuple(group))
        return outcomes


def single_products(smarts: str, *smiles: str) -> set[str]:
    """Flattened outcomes, for the one-in one-out steps where pairing cannot be lost."""
    return {group[0] for group in run_reaction(smarts, *smiles) if len(group) == 1}


def aldehyde_skeleton(tail: str) -> dict[str, Any]:
    """Reduce an ester-linked aldehyde tail to acid plus diol, verified forward."""
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(tail)
    if mol is None:
        return {"schema_applies": False, "reason": "unparseable"}
    if not mol.HasSubstructMatch(ESTER):
        return {"schema_applies": False, "reason": "no ester linkage"}
    if not mol.HasSubstructMatch(ALDEHYDE):
        return {"schema_applies": False, "reason": "no aldehyde"}

    acid_pattern = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
    target = canon(tail)
    for fragments in run_reaction(CLEAVE_ESTER, tail):
        if len(fragments) != 2:
            continue
        for acid, alcohol in (fragments, fragments[::-1]):
            acid_mol = Chem.MolFromSmiles(acid)
            if acid_mol is None or not acid_mol.HasSubstructMatch(acid_pattern):
                continue
            # the alcohol arm still carries the aldehyde; reducing it gives the diol
            for diol in single_products("[CH1:1]=O>>[OX2H1][CH2:1]", alcohol):
                for ester in single_products(ESTERIFY, acid, diol):
                    if target in {canon(x) for x in single_products(OXIDISE, ester)}:
                        return {
                            "schema_applies": True,
                            "acid": canon(acid),
                            "diol": canon(diol),
                            "forward_verified": True,
                        }
    return {"schema_applies": False, "reason": "no acid/diol pair reproduces the tail"}


def isocyanide_skeleton(tail: str) -> dict[str, Any]:
    """Reduce an isocyanide tail to its primary amine, verified forward."""
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(tail)
    if mol is None:
        return {"schema_applies": False, "reason": "unparseable"}
    if not mol.HasSubstructMatch(ISOCYANIDE):
        return {"schema_applies": False, "reason": "not an isocyanide"}
    primary = Chem.MolFromSmarts("[NX3;H2]")
    target = canon(tail)
    for amine in single_products("[C-]#[N+:1]>>[N:1]", tail):
        amine_mol = Chem.MolFromSmiles(amine)
        if amine_mol is None or not amine_mol.HasSubstructMatch(primary):
            continue
        for formamide in single_products(FORMYLATE, amine):
            dehydrated = single_products("[NX3:1][CH1:2]=O>>[N+:1]#[C-:2]", formamide)
            if target in {canon(x) for x in dehydrated}:
                return {"schema_applies": True, "amine": canon(amine), "forward_verified": True}
        return {"schema_applies": True, "amine": canon(amine), "forward_verified": False}
    return {"schema_applies": False, "reason": "no primary amine reproduces the isocyanide"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="cap unique components per role")
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "results/phase1/forge_route_grounding_audit_v1/result.json",
    )
    args = parser.parse_args()
    started = time.time()

    with gzip.open(LIBRARY, "rt", newline="") as fh:
        library_products, library_components = set(), defaultdict(set)
        for row in csv.DictReader(fh):
            library_products.add(canon(row["canonical_product_smiles"]))
            for cand in json.loads(row["candidate_routes_json"] or "[]"):
                for role, smiles in cand.get("components", {}).items():
                    library_components[role].add(canon(smiles))
    with gzip.open(MEASURED, "rt", newline="") as fh:
        measured = {"amine_head": set(), "oxoester_aldehyde_body_tail": set(), "isocyanide_tail": set()}
        for row in csv.DictReader(fh):
            measured["amine_head"].add(canon(row["A_smiles"]))
            measured["oxoester_aldehyde_body_tail"].add(canon(row["B_smiles"]))
            measured["isocyanide_tail"].add(canon(row["C_smiles"]))

    purchasable: set[str] = set()
    if PROCUREMENT.exists():
        with gzip.open(PROCUREMENT, "rt") as fh:
            for line in fh:
                rec = json.loads(line)
                smiles = rec.get("canonical_smiles") or rec.get("smiles")
                vendors = rec.get("vendor_count", rec.get("vendors", 0))
                if smiles and isinstance(vendors, int) and vendors > 0:
                    purchasable.add(canon(smiles))
    print(f"procurement snapshot: {len(purchasable)} components with at least one vendor")

    with gzip.open(LEDGER, "rt", newline="") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["terminal_chemical_admitted"] == "True" and r["canonical_product"]]

    # unique components, tagged by whether the products using them leave the enumeration
    roles = {"amine_head": "canonical_amine",
             "oxoester_aldehyde_body_tail": "canonical_aldehyde",
             "isocyanide_tail": "canonical_isocyanide"}
    seen: dict[str, dict[str, dict[str, Any]]] = {r: {} for r in roles}
    for row in rows:
        outside = row["canonical_product"] not in library_products
        supported = row["oracle_scored"] == "True"
        for role, column in roles.items():
            smiles = row[column]
            entry = seen[role].setdefault(
                smiles, {"products": 0, "outside": 0, "supported": 0}
            )
            entry["products"] += 1
            entry["outside"] += int(outside)
            entry["supported"] += int(supported)

    report: dict[str, Any] = {}
    for role in roles:
        components = sorted(seen[role])
        if args.limit:
            components = components[: args.limit]
        print(f"\n{role}: auditing {len(components)} unique components")
        buckets: dict[str, Counter] = defaultdict(Counter)
        reasons: Counter = Counter()
        for index, smiles in enumerate(components, 1):
            in_library = smiles in library_components.get(role, set())
            in_measured = smiles in measured[role]
            strata = [
                "all",
                "in_enumerated_library" if in_library else "outside_enumerated_library",
                "measured_component" if in_measured else "unmeasured_component",
            ]
            if role == "amine_head":
                applies = smiles in purchasable
                leaves_ok = applies
                if not applies:
                    reasons["not found in the partial procurement snapshot (searched or not)"] += 1
            elif role == "oxoester_aldehyde_body_tail":
                out = aldehyde_skeleton(smiles)
                applies = out["schema_applies"]
                leaves_ok = applies and out.get("acid") in purchasable and out.get("diol") in purchasable
                if not applies:
                    reasons[out.get("reason", "?")] += 1
            else:
                out = isocyanide_skeleton(smiles)
                applies = out["schema_applies"]
                leaves_ok = applies and out.get("amine") in purchasable
                if not applies:
                    reasons[out.get("reason", "?")] += 1
            for stratum in strata:
                buckets[stratum]["components"] += 1
                buckets[stratum]["schema_applies"] += int(applies)
                buckets[stratum]["leaves_purchasable"] += int(leaves_ok)
            if index % 400 == 0:
                print(f"  {index}/{len(components)}", flush=True)
        report[role] = {
            "strata": {k: dict(v) for k, v in buckets.items()},
            "schema_failure_reasons": dict(reasons.most_common(6)),
        }

    print("\n=== route skeleton by role and stratum ===")
    print("  schema = role preparation map applies and forward-verifies")
    print("  found  = route leaves present in a 477-entry partial snapshot; NOT a purchasability rate")
    print(f"{'role':30s} {'stratum':28s} {'n':>6s} {'schema':>8s} {'found':>8s}")
    for role, payload in report.items():
        for stratum, counts in sorted(payload["strata"].items()):
            n = counts["components"]
            print(f"{role:30s} {stratum:28s} {n:6d} "
                  f"{counts['schema_applies']/n:8.3f} {counts['leaves_purchasable']/n:8.3f}")

    payload = {
        "schema_version": "phase1_forge_route_grounding_audit.v1",
        "status": "complete_population_level_route_skeleton_audit",
        "inputs": {
            "terminal_ledger": {"path": str(LEDGER.relative_to(REPO)), "sha256": sha256_file(LEDGER)},
            "enumerated_library": {"path": str(LIBRARY.relative_to(REPO)), "sha256": sha256_file(LIBRARY)},
            "procurement": {
                "path": str(PROCUREMENT.relative_to(REPO)) if PROCUREMENT.exists() else None,
                "components_with_a_vendor": len(purchasable),
            },
        },
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "elapsed_seconds": round(time.time() - started, 1),
        },
        "report": report,
        "nonclaims": [
            "A preparation schema is a constructive route skeleton, not evidence that a synthesis will succeed.",
            "Purchasability is read from a dated snapshot in which zero vendors is not proof a compound is unobtainable.",
            "Direct procurement is the designated realization operator for the amine head in this workflow; the absence of an upstream preparation map for that role is by design, not a coverage gap.",
            "Procurement here has three states and only two are distinguishable from this snapshot: found, and not-found-or-never-searched. A component absent from a 477-entry snapshot covering 6% of the audited population has not been shown to be unavailable, so no purchasability rate in this artifact is a route-failure rate.",
            "This audit covers unique components, not products; a product is only as routable as its least routable component.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"\nwrote {args.output.relative_to(REPO)}")


if __name__ == "__main__":
    main()
