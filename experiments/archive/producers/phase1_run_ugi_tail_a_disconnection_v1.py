#!/usr/bin/env python3
"""Apply the published AGILE Tail A disconnection deliberately, then verify it.

Every in-domain aldehyde is an omega-oxo ester -- the AGILE compound class, whose
two-step construction is documented: esterify a carboxylic acid with an
alpha,omega-diol, then oxidise the remaining primary alcohol.  A bounded
retrosynthetic search rediscovers this only sometimes and marks the rest
"blocked", which says more about the search budget than about the chemistry.

This proposes the disconnection directly and then *verifies* it by running the
repository's own qualified transforms forward:

    ugi3_upstream_esterification_exact_source_v1          (carboxylic_acid, alpha_omega_diol)
    ugi3_upstream_primary_alcohol_oxidation_exact_source_v1 (primary_alcohol_substrate)

A target counts as Tail-A-constructible only when that forward chain reproduces
it exactly.  Availability of the acid and diol is then checked against the online
procurement snapshot.

This is not a claim that any specific substrate lies within the scope of those
transforms; scope remains the unclosed gap.  It is a claim that the published
route, run forward, gives the target from two named starting materials.
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

RESULT_SCHEMA_VERSION = "phase1_ugi_tail_a_disconnection.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_tail_a_disconnection_ledger.v1"
ESTERIFICATION = "ugi3_upstream_esterification_exact_source_v1"
OXIDATION = "ugi3_upstream_primary_alcohol_oxidation_exact_source_v1"

AUTHORITY = {
    "uses_repository_qualified_transforms": True,
    "forward_verified_against_target": True,
    "substrate_scope_verified": False,
    "is_route_evidence": False,
    "is_experimental_evidence": False,
    "vendor_listing_is_a_quote": False,
}


def _canonical(smiles: str) -> str | None:
    from rdkit import Chem, rdBase

    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    return None if molecule is None else Chem.MolToSmiles(molecule, canonical=True)


def propose_acid_and_diol(target: str) -> list[tuple[str, str]]:
    """Cleave each ester and lift the aldehyde to its alcohol, giving (acid, diol)."""

    from rdkit import Chem, rdBase
    from rdkit.Chem import rdChemReactions

    cleave = rdChemReactions.ReactionFromSmarts(
        "[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])[OH].[OH][C:4]"
    )
    lift = rdChemReactions.ReactionFromSmarts("[CH1:1]=[O:2]>>[CH2:1][OH:2]")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(target)
        if molecule is None:
            return []
        pairs: set[tuple[str, str]] = set()
        for outcome in cleave.RunReactants((molecule,)):
            if len(outcome) != 2:
                continue
            fragments = []
            for fragment in outcome:
                try:
                    Chem.SanitizeMol(fragment)
                except Exception:  # noqa: BLE001
                    fragments = []
                    break
                fragments.append(fragment)
            if len(fragments) != 2:
                continue
            acid = next(
                (f for f in fragments if f.HasSubstructMatch(Chem.MolFromSmarts("C(=O)[OH]"))),
                None,
            )
            other = next((f for f in fragments if f is not acid), None)
            if acid is None or other is None:
                continue
            # Lift the terminal aldehyde to the alpha,omega-diol.
            for lifted in lift.RunReactants((other,)):
                for diol in lifted:
                    try:
                        Chem.SanitizeMol(diol)
                    except Exception:  # noqa: BLE001
                        continue
                    pairs.add((Chem.MolToSmiles(acid), Chem.MolToSmiles(diol)))
    return sorted(pairs)


def verify_forward(acid: str, diol: str, target: str, esterify: Any, oxidise: Any) -> bool:
    """Run the qualified transforms forward and require the exact target."""

    from rdkit import Chem, rdBase

    with rdBase.BlockLogs():
        a = Chem.MolFromSmiles(acid)
        d = Chem.MolFromSmiles(diol)
        if a is None or d is None:
            return False
        intermediates: set[str] = set()
        for outcome in esterify.reaction.RunReactants((a, d)):
            for molecule in outcome:
                try:
                    Chem.SanitizeMol(molecule)
                except Exception:  # noqa: BLE001
                    continue
                intermediates.add(Chem.MolToSmiles(molecule))
        for smiles in intermediates:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                continue
            for outcome in oxidise.reaction.RunReactants((molecule,)):
                for product in outcome:
                    try:
                        Chem.SanitizeMol(product)
                    except Exception:  # noqa: BLE001
                        continue
                    if Chem.MolToSmiles(product) == target:
                        return True
    return False


def load_procurement(repo: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for path in (
        repo / "results/phase1/ugi_online_procurement_snapshot_v1/procurement_ledger.jsonl.gz",
        repo / "results/phase1/ugi_indomain_makeability_v1/indomain_procurement_ledger.jsonl.gz",
    ):
        if path.is_file():
            for row in read_jsonl_gzip(path, label=path.name):
                index.setdefault(str(row["canonical_smiles"]), row)
    return index


def run(repo: Path, output_dir: Path) -> dict[str, Any]:
    from phase1_run_ugi_online_procurement_lookup_v1 import lookup

    ledger = (
        repo / "results/phase1/ugi_indomain_makeability_v1/indomain_makeability_ledger.jsonl.gz"
    )
    products = read_jsonl_gzip(ledger, label="in-domain makeability")
    # Every in-domain aldehyde, regardless of its current verdict.  Scoping this
    # to unresolved components made the pass order-dependent and destructive:
    # a component reclassified by an earlier run fell out of the target set and
    # its record was lost on rewrite.  Covering all of them keeps the ledger
    # complete and the pass idempotent.
    blocked: dict[str, int] = {}
    for row in products:
        smiles = row["components"]["oxoester_aldehyde_body_tail"]
        blocked[smiles] = blocked.get(smiles, 0) + 1
    # The branch-conditioned lane was scored separately and never entered the
    # rescoring ledger, so its aldehydes would otherwise be skipped entirely.
    branch = repo / (
        "results/phase1/ugi_branch_exploration_applicability_v1/terminal_rescoring.csv.gz"
    )
    if branch.is_file():
        with gzip.open(branch, "rt", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("oracle_scored") == "True":
                    smiles = row["canonical_aldehyde"]
                    blocked[smiles] = blocked.get(smiles, 0) + 1
    targets = sorted(blocked, key=lambda s: (-blocked[s], s))
    print(
        f"blocked aldehydes: {len(targets)} covering {sum(blocked.values())} products", flush=True
    )

    resolver = load_independent_l2_forward_resolver(
        repo / "configs/route/graph2edits_l2_forward_resolver_v1.json", repo_root=repo
    )
    transforms = {str(t.reaction.reaction_id): t.reaction for t in resolver.transforms}
    esterify = transforms[ESTERIFICATION]
    oxidise = transforms[OXIDATION]

    procurement = load_procurement(repo)
    rows: list[dict[str, Any]] = []
    fresh = 0
    for index, target in enumerate(targets, start=1):
        record: dict[str, Any] = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "canonical_smiles": target,
            "products_unlocked": blocked[target],
            "tail_a_verified": False,
            "acid": None,
            "diol": None,
            "acid_vendor_count": None,
            "diol_vendor_count": None,
            "both_purchasable": False,
            "authority": dict(AUTHORITY),
        }
        for acid, diol in propose_acid_and_diol(target):
            if not verify_forward(acid, diol, target, esterify, oxidise):
                continue
            record["tail_a_verified"] = True
            record["acid"] = acid
            record["diol"] = diol
            for label, smiles in (("acid", acid), ("diol", diol)):
                entry = procurement.get(smiles)
                if entry is None:
                    entry = lookup(smiles)
                    procurement[smiles] = entry
                    fresh += 1
                record[f"{label}_vendor_count"] = entry.get("vendor_count")
            record["both_purchasable"] = bool(
                record["acid_vendor_count"] and record["diol_vendor_count"]
            )
            break
        rows.append(record)
        if index % 10 == 0 or index == len(targets):
            v = sum(1 for r in rows if r["tail_a_verified"])
            b = sum(1 for r in rows if r["both_purchasable"])
            print(f"tail-a {index}/{len(targets)} verified={v} both_purchasable={b}", flush=True)

    path = output_dir / "tail_a_disconnection_ledger.jsonl.gz"
    atomic_write(path, jsonl_gzip_bytes(rows))
    verified = [r for r in rows if r["tail_a_verified"]]
    purchasable = [r for r in rows if r["both_purchasable"]]
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "tail_a_disconnection_complete",
        "transforms": {
            "esterification": ESTERIFICATION,
            "oxidation": OXIDATION,
            "source": "configs/route/graph2edits_l2_forward_resolver_v1.json",
        },
        "runtime": {"python_version": platform.python_version(), "platform": platform.platform()},
        "summary": {
            "blocked_aldehydes": len(targets),
            "products_behind_them": sum(blocked.values()),
            "tail_a_forward_verified": len(verified),
            "both_starting_materials_purchasable": len(purchasable),
            "products_unlocked_if_purchasable": sum(r["products_unlocked"] for r in purchasable),
            "fresh_procurement_lookups": fresh,
            "acid_vendor_status": dict(
                sorted(Counter(str(bool(r["acid_vendor_count"])) for r in verified).items())
            ),
            "diol_vendor_status": dict(
                sorted(Counter(str(bool(r["diol_vendor_count"])) for r in verified).items())
            ),
        },
        "artifacts": {
            "tail_a_disconnection_ledger.jsonl.gz": {
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
        "--output-dir", type=Path, default=Path("results/phase1/ugi_tail_a_disconnection_v1")
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    result = run(repo, out.resolve())
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
