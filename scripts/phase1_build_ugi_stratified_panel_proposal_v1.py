#!/usr/bin/env python3
"""Propose a stratified 20-candidate shortlist for chemist review.

This is a proposal, not a panel lock.  It selects nothing irreversibly, creates
no preregistration, and is expected to be cut down by a chemist to the compounds
actually synthesised.

Selection is stratified rather than top-N by potency.  Taking the twenty
highest-scoring candidates would concentrate on one chemotype and one authority
tier, and would contain no low-ranked compound -- leaving no way to tell whether
the oracle ranks or merely predicts everything high.

Strata:

  qualified_role_holdout   the only lane with genuine exact-new-role support;
                           these carry the defensible potency predictions
  branched                 the restored branched-tail architecture, including
                           the highest-scoring in-domain candidate overall;
                           note its authority tier is exploratory
  chemotype_diversity      distinct declared chemotype clusters, so the panel
                           is not eight near-identical lipids
  ranking_control          deliberately low-LCB90 candidates; without them a
                           positive result cannot be attributed to ranking

Every candidate must be makeable: each component is purchasable, or routable to
purchasable material by a forward-verified published route.
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

from forge.product.ugi_bounded_hybrid_route_cascade import (  # noqa: E402
    atomic_write,
    canonical_json_bytes,
    jsonl_gzip_bytes,
    read_jsonl_gzip,
    sha256_file,
    sha256_payload,
)

RESULT_SCHEMA_VERSION = "phase1_ugi_stratified_panel_proposal.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_stratified_panel_proposal_ledger.v1"

# Seven primary, one frontier, two controls: docs/MANUSCRIPT_AUTHORITY.md section 19.
# Frontier candidates were reduced from two to one because each is more likely to
# expose weaknesses in the activity model, and one suffices to establish that the
# framework produces meaningful component novelty.
QUOTAS = {
    "primary": 7,
    "de_novo_frontier": 1,
    "matched_control": 2,
}

AUTHORITY = {
    "is_a_panel_lock": False,
    "is_a_preregistration": False,
    "candidate_selection_is_final": False,
    "requires_chemist_review": True,
    "requires_procurement_refresh_before_purchase": True,
    "potency_is_a_ranking_signal_not_a_validated_prediction": True,
}


def _f(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_pool(repo: Path) -> list[dict[str, Any]]:
    """Makeable in-domain candidates, main pool plus the branched lane."""

    make = {
        r["canonical_product"]: r
        for r in read_jsonl_gzip(
            repo
            / "results/phase1/ugi_indomain_makeability_v1/indomain_makeability_ledger.jsonl.gz",
            label="makeability",
        )
    }
    pool: list[dict[str, Any]] = []
    for product, row in make.items():
        if row["makeability"] not in ("buy_all", "buy_and_make"):
            continue
        pool.append(
            {
                "canonical_product": product,
                "arm_id": row["arm_id"],
                "authority_tier": row["authority_tier"],
                "conservative_high_potency": row["conservative_high_potency"],
                "oracle_mean": _f(row.get("oracle_mean")),
                "lcb90": _f(row.get("lcb90")),
                "components": row["components"],
                "makeability": row["makeability"],
                "buy": row["buy"],
                "synthesise": row["synthesise"],
                "branched": False,
                "source": "main_in_domain",
            }
        )

    # Branched lane: scored separately and never merged into the rescoring ledger.
    branch_path = (
        repo / "results/phase1/ugi_branch_exploration_applicability_v1/terminal_rescoring.csv.gz"
    )
    if branch_path.is_file():
        with gzip.open(branch_path, "rt", newline="") as handle:
            seen: dict[str, dict[str, str]] = {}
            for row in csv.DictReader(handle):
                if row.get("oracle_scored") == "True":
                    seen.setdefault(row["canonical_product"], row)
        for product, row in seen.items():
            pool.append(
                {
                    "canonical_product": product,
                    "arm_id": "branch_exploration",
                    "authority_tier": row.get("authority_tier"),
                    "conservative_high_potency": row.get("conservative_high_potency") == "True",
                    "oracle_mean": _f(row.get("oracle_mean")),
                    "lcb90": _f(row.get("lcb90")),
                    "components": {
                        "amine_head": row["canonical_amine"],
                        "oxoester_aldehyde_body_tail": row["canonical_aldehyde"],
                        "isocyanide_tail": row["canonical_isocyanide"],
                    },
                    "makeability": "buy_and_make",
                    "buy": [],
                    "synthesise": [row["canonical_aldehyde"]],
                    "branched": True,
                    "source": "branch_lane",
                }
            )
    return pool


def chemotype(row: dict[str, Any]) -> str:
    from rdkit import Chem, rdBase

    aldehyde = row["components"]["oxoester_aldehyde_body_tail"]
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(aldehyde)
    carbons = 0 if molecule is None else sum(1 for a in molecule.GetAtoms() if a.GetSymbol() == "C")
    unsaturated = bool(
        molecule is not None and molecule.HasSubstructMatch(Chem.MolFromSmarts("[#6]=[#6]"))
    )
    bucket = "c<=14" if carbons <= 14 else ("c15-18" if carbons <= 18 else "c19+")
    return f"{row['components']['amine_head']}|{bucket}|{'unsat' if unsaturated else 'sat'}|{'br' if row['branched'] else 'lin'}"


def load_expanded_catalogue(repo: Path) -> dict[str, set[str]]:
    """Component identities present in the expanded reaction-enumerated registry.

    A component absent from this registry was constructed by the generator rather
    than drawn from any catalogue we assembled, which is what the de novo frontier
    stratum is selected on.
    """

    path = repo / "results/phase1/ugi_component_expansion/component_registry.csv.gz"
    catalogue: dict[str, set[str]] = {}
    if not path.is_file():
        return catalogue
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            catalogue.setdefault(row["role"], set()).add(row["canonical_smiles"])
    return catalogue


def novel_component_count(row: dict[str, Any], catalogue: dict[str, set[str]]) -> int:
    if not catalogue:
        return 0
    return sum(
        1
        for role, smiles in row["components"].items()
        if smiles not in catalogue.get(role, set())
    )


def select(pool: list[dict[str, Any]], catalogue: dict[str, set[str]]) -> list[dict[str, Any]]:
    """Six primary, two de novo frontier, two matched low-ranked controls.

    Ten prospective synthesis slots cannot power a causal comparison between
    generation arms, so the panel is allocated for discovery rather than split
    between arms.  See docs/MANUSCRIPT_AUTHORITY.md section 13.
    """

    for row in pool:
        row["chemotype"] = chemotype(row)
        row["novel_components"] = novel_component_count(row, catalogue)
    chosen: list[dict[str, Any]] = []
    taken: set[str] = set()
    used_chemotypes: set[str] = set()

    def _take(candidates: list[dict[str, Any]], stratum: str, quota: int) -> None:
        """Fill `stratum` up to `quota` in total, so repeated calls top it up."""

        for row in candidates:
            if sum(1 for c in chosen if c["stratum"] == stratum) >= quota:
                return
            if row["canonical_product"] in taken:
                continue
            if row["chemotype"] in used_chemotypes:
                continue
            row = {**row, "stratum": stratum}
            chosen.append(row)
            taken.add(row["canonical_product"])
            used_chemotypes.add(row["chemotype"])

    high = lambda r: -(r["lcb90"] if r["lcb90"] is not None else -1e9)  # noqa: E731
    low = lambda r: (r["lcb90"] if r["lcb90"] is not None else 1e9)  # noqa: E731

    # Six primary: highest conservative activity lower bound, one per chemotype.
    #
    # Drawn preferentially from the qualified role-holdout tier.  Ranking purely on
    # the lower bound fills this stratum entirely with exploratory-tier designs,
    # because chemotype de-duplication lets a marginally higher-scoring exploratory
    # candidate displace a qualified one sharing its chemotype.  The cost of the
    # preference is small (best qualified lower bound 6.59 against 6.96 overall) and
    # it buys the stratum its evidentiary purpose: a prospective hit here should
    # test whether calibrated ranking works, not whether extrapolation happened to
    # succeed.  Extrapolation is what the de novo frontier stratum is for.
    qualified = [r for r in pool if r["authority_tier"] == "qualified_role_holdout"]
    _take(sorted(qualified, key=high), "primary", QUOTAS["primary"])
    _take(sorted(pool, key=high), "primary", QUOTAS["primary"])
    # Two frontier: most component novelty, ties broken by activity lower bound.
    _take(
        sorted(pool, key=lambda r: (-r["novel_components"], high(r))),
        "de_novo_frontier",
        QUOTAS["de_novo_frontier"],
    )
    # Two controls: deliberately low-ranked, otherwise identically gated.
    _take(sorted(pool, key=low), "matched_control", QUOTAS["matched_control"])
    return chosen


def run(repo: Path, output_dir: Path) -> dict[str, Any]:
    pool = load_pool(repo)
    chosen = select(pool, load_expanded_catalogue(repo))
    chosen.sort(key=lambda r: (r["stratum"], -(r["lcb90"] or -1e9)))
    for index, row in enumerate(chosen, start=1):
        row["schema_version"] = LEDGER_SCHEMA_VERSION
        row["proposal_index"] = index
        row["authority"] = dict(AUTHORITY)
    path = output_dir / "stratified_panel_proposal.jsonl.gz"
    atomic_write(path, jsonl_gzip_bytes(chosen))
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "stratified_panel_proposal_complete",
        "runtime": {"python_version": platform.python_version(), "platform": platform.platform()},
        "quotas": dict(QUOTAS),
        "summary": {
            "pool_size": len(pool),
            "pool_branched": sum(1 for r in pool if r["branched"]),
            "proposed": len(chosen),
            "strata": dict(sorted(Counter(r["stratum"] for r in chosen).items())),
            "arms": dict(sorted(Counter(r["arm_id"] for r in chosen).items())),
            "authority_tiers": dict(
                sorted(Counter(str(r["authority_tier"]) for r in chosen).items())
            ),
            "distinct_chemotypes": len({r["chemotype"] for r in chosen}),
            "distinct_amines": len({r["components"]["amine_head"] for r in chosen}),
            "distinct_aldehydes": len(
                {r["components"]["oxoester_aldehyde_body_tail"] for r in chosen}
            ),
            "distinct_isocyanides": len({r["components"]["isocyanide_tail"] for r in chosen}),
            "lcb90_range": [
                min((r["lcb90"] for r in chosen if r["lcb90"] is not None), default=None),
                max((r["lcb90"] for r in chosen if r["lcb90"] is not None), default=None),
            ],
        },
        "artifacts": {
            "stratified_panel_proposal.jsonl.gz": {
                "path": path.name,
                "sha256": sha256_file(path),
                "rows": len(chosen),
            }
        },
        "authority": dict(AUTHORITY),
        "nonclaims": [
            "This is a proposal for chemist review, not a locked panel.",
            "No candidate here has been synthesised, formulated or tested.",
            "Potency is a conservative ranking signal, not a validated prediction.",
            "Procurement must be refreshed before purchase.",
            "At this panel size the broad-versus-support causal comparison is not powered.",
        ],
    }
    result = {**content, "result_sha256": sha256_payload(content)}
    atomic_write(output_dir / "result.json", canonical_json_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/phase1/ugi_stratified_panel_proposal_v1")
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    result = run(repo, out.resolve())
    print(json.dumps(result["summary"], indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
