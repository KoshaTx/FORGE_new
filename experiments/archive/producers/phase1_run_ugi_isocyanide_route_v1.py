#!/usr/bin/env python3
"""Apply the published isocyanide route deliberately, then verify it forward.

Every unresolved isocyanide tail is a primary alkyl or alkenyl isocyanide, made
by the documented two-step sequence: formylate the primary amine, then dehydrate
the formamide.  A bounded retrosynthetic planner does not rediscover this
reliably -- it left long-chain isocyanides "blocked" while their precursor amines
are ordinary catalogue chemicals.

Left uncorrected this biases the makeable set badly: isocyanide novelty is what
carries the `qualified_role_holdout` authority tier, so discarding isocyanides
discards the candidates whose potency predictions are best supported.

The disconnection is proposed directly and verified by running the repository's
own qualified transforms forward:

    ugi3_upstream_amine_formylation_exact_source_v1      (primary_amine_substrate)
    ugi3_upstream_formamide_dehydration_exact_source_v1  (n_substituted_formamide)

A target counts as constructible only when that chain reproduces it exactly.
Precursor availability is then checked against the online procurement snapshot.
Forward reproduction is not a claim about substrate scope.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_DEFAULT = Path(__file__).resolve().parents[1]
for _p in (REPO_DEFAULT / "src", REPO_DEFAULT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.phase1.synthesis_guidance.route_cascade import (  # noqa: E402
    atomic_write,
    canonical_json_bytes,
    jsonl_gzip_bytes,
    read_jsonl_gzip,
    sha256_file,
    sha256_payload,
)
from forge.synthesis.assessment.l2_forward_resolver import load_independent_l2_forward_resolver  # noqa: E402

RESULT_SCHEMA_VERSION = "phase1_ugi_isocyanide_route.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_isocyanide_route_ledger.v1"
FORMYLATION = "ugi3_upstream_amine_formylation_exact_source_v1"
DEHYDRATION = "ugi3_upstream_formamide_dehydration_exact_source_v1"

AUTHORITY = {
    "uses_repository_qualified_transforms": True,
    "forward_verified_against_target": True,
    "substrate_scope_verified": False,
    "is_route_evidence": False,
    "is_experimental_evidence": False,
    "vendor_listing_is_a_quote": False,
}


def propose_amine(target: str) -> list[str]:
    """An isocyanide R-NC comes from the primary amine R-NH2."""

    from rdkit import Chem, rdBase

    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(target)
        if molecule is None:
            return []
        pattern = Chem.MolFromSmarts("[C-]#[N+]")
        if not molecule.HasSubstructMatch(pattern):
            return []
        editable = Chem.RWMol(molecule)
        matches = molecule.GetSubstructMatches(pattern)
        if len(matches) != 1:
            return []
        carbon, nitrogen = matches[0]
        # Drop the isocyanide carbon; the nitrogen becomes a neutral primary amine.
        editable.RemoveAtom(carbon)
        for atom in editable.GetAtoms():
            if atom.GetIdx() == (nitrogen - 1 if nitrogen > carbon else nitrogen):
                atom.SetFormalCharge(0)
                atom.SetNoImplicit(False)
                atom.SetNumExplicitHs(2)
        try:
            amine = editable.GetMol()
            Chem.SanitizeMol(amine)
        except Exception:  # noqa: BLE001
            return []
        return [Chem.MolToSmiles(amine)]


def verify_forward(amine: str, target: str, formylate: Any, dehydrate: Any) -> bool:
    """Run the qualified transforms forward and require the exact target."""

    from rdkit import Chem, rdBase

    with rdBase.BlockLogs():
        start = Chem.MolFromSmiles(amine)
        if start is None:
            return False
        formamides: set[str] = set()
        for outcome in formylate.reaction.RunReactants((start,)):
            for molecule in outcome:
                try:
                    Chem.SanitizeMol(molecule)
                except Exception:  # noqa: BLE001
                    continue
                formamides.add(Chem.MolToSmiles(molecule))
        for smiles in formamides:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                continue
            for outcome in dehydrate.reaction.RunReactants((molecule,)):
                for product in outcome:
                    try:
                        Chem.SanitizeMol(product)
                    except Exception:  # noqa: BLE001
                        continue
                    if Chem.MolToSmiles(product) == target:
                        return True
    return False


def run(repo: Path, output_dir: Path) -> dict[str, Any]:
    from phase1_run_ugi_online_procurement_lookup_v1 import lookup

    ledger = (
        repo / "results/phase1/ugi_indomain_makeability_v1/indomain_makeability_ledger.jsonl.gz"
    )
    products = read_jsonl_gzip(ledger, label="in-domain makeability")
    # Every in-domain isocyanide, regardless of verdict.  Scoping this to
    # unresolved components left `make_from_purchasable` isocyanides out of the
    # ledger entirely, so downstream consumers that read this artifact -- the
    # synthesis dossiers among them -- saw a resolved component as unresolved
    # merely because no row existed for it.  Covering all of them keeps the
    # ledger complete and the pass idempotent.
    unresolved: dict[str, int] = {}
    for row in products:
        smiles = row["components"]["isocyanide_tail"]
        unresolved[smiles] = unresolved.get(smiles, 0) + 1
    # The branch-conditioned lane was scored separately and never entered the
    # rescoring ledger, so its isocyanides would otherwise be skipped entirely.
    branch = repo / (
        "results/phase1/ugi_branch_exploration_applicability_v1/terminal_rescoring.csv.gz"
    )
    if branch.is_file():
        with gzip.open(branch, "rt", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("oracle_scored") == "True":
                    smiles = row["canonical_isocyanide"]
                    unresolved[smiles] = unresolved.get(smiles, 0) + 1
    targets = sorted(unresolved, key=lambda s: (-unresolved[s], s))
    print(
        f"isocyanides: {len(targets)} covering {sum(unresolved.values())} products",
        flush=True,
    )

    resolver = load_independent_l2_forward_resolver(
        repo / "configs/route/graph2edits_l2_forward_resolver_v1.json", repo_root=repo
    )
    transforms = {str(t.reaction.reaction_id): t.reaction for t in resolver.transforms}
    formylate = transforms[FORMYLATION]
    dehydrate = transforms[DEHYDRATION]

    procurement: dict[str, dict[str, Any]] = {}
    for path in (
        repo / "results/phase1/ugi_online_procurement_snapshot_v1/procurement_ledger.jsonl.gz",
        repo / "results/phase1/ugi_indomain_makeability_v1/indomain_procurement_ledger.jsonl.gz",
    ):
        if path.is_file():
            for row in read_jsonl_gzip(path, label=path.name):
                procurement.setdefault(str(row["canonical_smiles"]), row)

    rows: list[dict[str, Any]] = []
    for index, target in enumerate(targets, start=1):
        record: dict[str, Any] = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "canonical_smiles": target,
            "products_unlocked": unresolved[target],
            "route_verified": False,
            "precursor_amine": None,
            "amine_vendor_count": None,
            "amine_purchasable": False,
            "authority": dict(AUTHORITY),
        }
        for amine in propose_amine(target):
            if not verify_forward(amine, target, formylate, dehydrate):
                continue
            record["route_verified"] = True
            record["precursor_amine"] = amine
            entry = procurement.get(amine)
            if entry is None:
                entry = lookup(amine)
                procurement[amine] = entry
            record["amine_vendor_count"] = entry.get("vendor_count")
            record["amine_purchasable"] = bool(entry.get("vendor_count"))
            break
        rows.append(record)
        if index % 5 == 0 or index == len(targets):
            v = sum(1 for r in rows if r["route_verified"])
            p = sum(1 for r in rows if r["amine_purchasable"])
            print(
                f"isocyanide {index}/{len(targets)} verified={v} amine_purchasable={p}", flush=True
            )

    path = output_dir / "isocyanide_route_ledger.jsonl.gz"
    atomic_write(path, jsonl_gzip_bytes(rows))
    verified = [r for r in rows if r["route_verified"]]
    purchasable = [r for r in rows if r["amine_purchasable"]]
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "isocyanide_route_complete",
        "transforms": {
            "formylation": FORMYLATION,
            "dehydration": DEHYDRATION,
            "source": "configs/route/graph2edits_l2_forward_resolver_v1.json",
        },
        "runtime": {"python_version": platform.python_version(), "platform": platform.platform()},
        "summary": {
            "unresolved_isocyanides": len(targets),
            "products_behind_them": sum(unresolved.values()),
            "route_forward_verified": len(verified),
            "precursor_amine_purchasable": len(purchasable),
            "products_unlocked_if_purchasable": sum(r["products_unlocked"] for r in purchasable),
            "amine_vendor_status": dict(
                sorted(Counter(str(bool(r["amine_vendor_count"])) for r in verified).items())
            ),
        },
        "artifacts": {
            "isocyanide_route_ledger.jsonl.gz": {
                "path": path.name,
                "sha256": sha256_file(path),
                "rows": len(rows),
            }
        },
        "authority": dict(AUTHORITY),
        "nonclaims": [
            "Forward reproduction of the target does not verify substrate scope.",
            "A vendor listing is not a quote, stock level or purity specification.",
            "No compound here has been synthesised.",
        ],
    }
    result = {**content, "result_sha256": sha256_payload(content)}
    atomic_write(output_dir / "result.json", canonical_json_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/phase1/ugi_isocyanide_route_v1")
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    result = run(repo, out.resolve())
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
