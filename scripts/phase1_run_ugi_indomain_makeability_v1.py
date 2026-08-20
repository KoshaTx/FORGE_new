#!/usr/bin/env python3
"""Makeability over the whole applicability-supported population.

The v2 shortlist was selected on potency and diversity before any route or
availability information existed.  That was correct for an unbiased platform
measurement and wrong as a pool to pick a synthesis panel from.  Every
oracle-scored product the generator produced -- 2,590 of them -- resolves to only
201 unique components, so covering the entire in-domain population costs about a
hundred component lookups rather than thousands of molecules.

Cheap checks run first.  A component that can simply be bought never needs a
retrosynthetic search.

    procure   PubChem vendor lookup for every in-domain component (cached)
    plan      bounded planner search, only for components nobody sells
    rollup    combine into a per-product makeability verdict over all 2,590

This selects nothing and locks nothing.  The frozen 256-product cascade is left
untouched; it remains the honest platform measurement and the causal contrast.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import platform
import sys

# Cap per-worker thread pools BEFORE numpy/onnxruntime import.  Six workers each
# defaulting to one thread per core oversubscribes a 12-core machine roughly
# sixfold; the processes then spend their time in scheduler contention and the
# aggregate throughput is worse than with two workers.
for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "ORT_NUM_THREADS",
):
    os.environ.setdefault(_variable, os.environ.get("FORGE_WORKER_THREADS", "2"))

from collections import Counter  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

REPO_DEFAULT = Path(__file__).resolve().parents[1]
if str(REPO_DEFAULT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_DEFAULT / "src"))
if str(REPO_DEFAULT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_DEFAULT / "scripts"))

from forge.design.flow.ugi_bounded_hybrid_route_cascade import (  # noqa: E402
    UgiBoundedHybridRouteCascadeError,
    atomic_write,
    canonical_json_bytes,
    jsonl_gzip_bytes,
    load_json,
    read_jsonl_gzip,
    sha256_file,
    sha256_payload,
)

RESULT_SCHEMA_VERSION = "phase1_ugi_indomain_makeability.v1"
OUT = "results/phase1/ugi_indomain_makeability_v1"
RESCORING = "results/phase1/ugi_production_full_support_rescoring_v3/terminal_rescoring.csv.gz"
CASCADE = "results/phase1/ugi_bounded_hybrid_route_cascade_v1/component_route_ledger.jsonl.gz"
PROCUREMENT = "results/phase1/ugi_online_procurement_snapshot_v1/procurement_ledger.jsonl.gz"
REACH = "results/phase1/ugi_planner_reach_sweep_v1/planner_reach_ledger.jsonl.gz"

ROLES = ("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail")
ROLE_COLUMN = {
    "amine_head": "canonical_amine",
    "oxoester_aldehyde_body_tail": "canonical_aldehyde",
    "isocyanide_tail": "canonical_isocyanide",
}

AUTHORITY = {
    "vendor_listing_is_a_quote": False,
    "planner_solution_is_route_evidence": False,
    "candidate_selection": False,
    "prospective_panel_lock": False,
    "frozen_256_measurement_unchanged": True,
}


def in_domain_products(repo: Path) -> list[dict[str, str]]:
    with gzip.open(repo / RESCORING, "rt", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["oracle_scored"] == "True"]


def in_domain_components(products: list[dict[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    for row in products:
        for role in ROLES:
            seen.add((role, row[ROLE_COLUMN[role]]))
    return sorted(seen)


def _load_cache(repo: Path) -> tuple[dict[str, dict], dict[tuple[str, str], dict], dict]:
    procurement = {}
    if (repo / PROCUREMENT).is_file():
        procurement = {
            str(r["canonical_smiles"]): r
            for r in read_jsonl_gzip(repo / PROCUREMENT, label="procurement")
        }
    reach = {}
    if (repo / REACH).is_file():
        reach = {
            (str(r["role"]), str(r["canonical_smiles"])): r
            for r in read_jsonl_gzip(repo / REACH, label="reach")
        }
    cascade = {}
    if (repo / CASCADE).is_file():
        cascade = {
            (str(r["role"]), str(r["canonical_smiles"])): r
            for r in read_jsonl_gzip(repo / CASCADE, label="cascade components")
        }
    return procurement, reach, cascade


def stage_procure(repo: Path, output_dir: Path) -> dict[str, Any]:
    from phase1_run_ugi_online_procurement_lookup_v1 import lookup

    products = in_domain_products(repo)
    components = in_domain_components(products)
    procurement, _, _ = _load_cache(repo)
    todo = [c for c in components if c[1] not in procurement]
    accessed = datetime.now(timezone.utc).replace(microsecond=0)

    rows: list[dict[str, Any]] = []
    for index, (role, smiles) in enumerate(todo, start=1):
        observation = lookup(smiles)
        rows.append(
            {
                "schema_version": "phase1_ugi_indomain_procurement_ledger.v1",
                "role": role,
                **observation,
                "accessed_utc": accessed.isoformat().replace("+00:00", "Z"),
            }
        )
        if index % 20 == 0 or index == len(todo):
            listed = sum(1 for r in rows if r["status"] == "vendors_listed")
            print(f"procure {index}/{len(todo)} vendors_listed={listed}", flush=True)

    path = output_dir / "indomain_procurement_ledger.jsonl.gz"
    atomic_write(path, jsonl_gzip_bytes(rows))
    content = {
        "schema_version": RESULT_SCHEMA_VERSION + ".procure",
        "status": "indomain_procurement_complete",
        "snapshot": {
            "accessed_utc": accessed.isoformat().replace("+00:00", "Z"),
            "expires_utc": (accessed + timedelta(days=30)).isoformat().replace("+00:00", "Z"),
        },
        "summary": {
            "in_domain_products": len({r["canonical_product"] for r in products}),
            "in_domain_components": len(components),
            "already_cached": len(components) - len(todo),
            "looked_up": len(rows),
            "status_counts": dict(sorted(Counter(r["status"] for r in rows).items())),
        },
        "artifacts": {
            "indomain_procurement_ledger.jsonl.gz": {
                "path": path.name,
                "sha256": sha256_file(path),
                "rows": len(rows),
            }
        },
        "authority": dict(AUTHORITY),
    }
    result = {**content, "result_sha256": sha256_payload(content)}
    atomic_write(output_dir / "procure_result.json", canonical_json_bytes(result))
    return result


def _purchasable_index(repo: Path, output_dir: Path) -> dict[str, dict[str, Any]]:
    procurement, _, _ = _load_cache(repo)
    extra = output_dir / "indomain_procurement_ledger.jsonl.gz"
    if extra.is_file():
        for row in read_jsonl_gzip(extra, label="in-domain procurement"):
            procurement.setdefault(str(row["canonical_smiles"]), row)
    return procurement


def plan_targets(repo: Path, output_dir: Path) -> list[tuple[str, str]]:
    """Components nobody sells and nobody has planner-searched yet."""

    products = in_domain_products(repo)
    components = in_domain_components(products)
    procurement = _purchasable_index(repo, output_dir)
    _, reach, cascade = _load_cache(repo)
    todo = []
    for role, smiles in components:
        if (procurement.get(smiles) or {}).get("vendor_count"):
            continue  # buy it
        entry = cascade.get((role, smiles))
        if entry and entry["final_component_state"] == "complete":
            continue  # already closed by exact evidence
        if (role, smiles) in reach:
            continue  # already searched
        todo.append((role, smiles))
    return sorted(todo)


def stage_plan(repo: Path, output_dir: Path, shard: int, shards: int) -> dict[str, Any]:
    import random

    import numpy as np

    todo = plan_targets(repo, output_dir)
    mine = [t for index, t in enumerate(todo) if index % shards == shard]

    # Resume: anything already written by a previous run of this shard is kept.
    path = output_dir / f"plan_shard_{shard:02d}.jsonl.gz"
    rows: list[dict[str, Any]] = []
    if path.is_file():
        rows = read_jsonl_gzip(path, label=f"plan shard {shard}")
    done = {(str(r["role"]), str(r["canonical_smiles"])) for r in rows}
    remaining = [t for t in mine if t not in done]
    print(
        f"planner targets total={len(todo)} shard{shard}={len(mine)} "
        f"already_done={len(done)} remaining={len(remaining)}",
        flush=True,
    )

    random.seed(20260805 + shard)
    np.random.seed((20260805 + shard) % (2**32))
    manifest = load_json(
        repo / "configs/route/aizynthfinder_public_v4_4_1_diagnostic_macos_arm64_v1.json",
        label="runtime manifest",
    )
    engine_config = repo / manifest["config"]["path"]
    if sha256_file(engine_config) != manifest["config"]["sha256"]:
        raise UgiBoundedHybridRouteCascadeError("AiZynthFinder engine config changed")

    from aizynthfinder.aizynthfinder import AiZynthExpander, AiZynthFinder
    from rdkit import Chem, rdBase

    expander = AiZynthExpander(configfile=str(engine_config))
    expander.expansion_policy.select(["uspto", "ringbreaker"])
    expander.filter_policy.select(["uspto"])

    def _canonical(value: str) -> str | None:
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(value)
        return None if molecule is None else Chem.MolToSmiles(molecule, canonical=True)

    finder = AiZynthFinder(configfile=str(engine_config))
    finder.expansion_policy.select(["uspto", "ringbreaker"])
    finder.filter_policy.select(["uspto"])
    finder.stock.select(["zinc"])
    finder.config.search.time_limit = 60
    finder.config.search.iteration_limit = 300
    finder.config.search.max_transforms = 6

    for index, (role, smiles) in enumerate(remaining, start=1):
        record: dict[str, Any] = {
            "schema_version": "phase1_ugi_indomain_plan_ledger.v1",
            "role": role,
            "canonical_smiles": smiles,
            "planner_solved_to_public_catalog": False,
            "stock_leaves": [],
            "single_step_proposals": [],
            "single_step_failure": None,
            "execution_failure": None,
        }
        # Additional signal: the shortest disconnection, recorded alongside the
        # full search rather than replacing it.  Costs ~2s against a ~125s search.
        try:
            groups = expander.do_expansion(smiles, return_n=10)
            proposals = []
            for rank, group in enumerate(groups, start=1):
                if not group:
                    continue
                outcomes = getattr(group[0], "reactants", ())
                reactants = []
                if outcomes:
                    for molecule in outcomes[0]:
                        value = _canonical(molecule.smiles)
                        if value:
                            reactants.append(value)
                if reactants:
                    proposals.append({"rank": rank, "canonical_reactants": sorted(reactants)})
            record["single_step_proposals"] = proposals
        except Exception as error:  # noqa: BLE001 - recorded, never dropped
            record["single_step_failure"] = f"{type(error).__name__}: {error}"
        try:
            finder.target_smiles = smiles
            finder.prepare_tree()
            finder.tree_search(show_progress=False)
            finder.build_routes()
            stats = finder.extract_statistics()
            routes = finder.routes.dict_with_extra(include_metadata=True, include_scores=True)
            record["planner_solved_to_public_catalog"] = bool(stats.get("is_solved"))
            leaves: set[str] = set()
            for route in routes[:3]:
                stack = [route]
                while stack:
                    node = stack.pop()
                    if not isinstance(node, dict):
                        continue
                    children = node.get("children") or []
                    if not children and node.get("type") == "mol":
                        leaves.add(str(node.get("smiles")))
                    stack.extend(children)
            record["stock_leaves"] = sorted(leaves)
        except Exception as error:  # noqa: BLE001 - recorded, never dropped
            record["execution_failure"] = f"{type(error).__name__}: {error}"
        rows.append(record)
        # Checkpoint after every target so a kill never discards completed work.
        atomic_write(path, jsonl_gzip_bytes(rows))
        print(
            f"plan s{shard} {index}/{len(remaining)} {role[:20]} "
            f"solved={record['planner_solved_to_public_catalog']}",
            flush=True,
        )

    content = {
        "schema_version": RESULT_SCHEMA_VERSION + ".plan",
        "status": "indomain_plan_shard_complete",
        "shard": shard,
        "shards": shards,
        "summary": {
            "targets": len(rows),
            "solved": sum(1 for r in rows if r["planner_solved_to_public_catalog"]),
            "failures": sum(1 for r in rows if r["execution_failure"]),
        },
        "artifacts": {
            path.name: {"path": path.name, "sha256": sha256_file(path), "rows": len(rows)}
        },
    }
    result = {**content, "result_sha256": sha256_payload(content)}
    atomic_write(output_dir / f"plan_shard_{shard:02d}_result.json", canonical_json_bytes(result))
    return result


def stage_rollup(repo: Path, output_dir: Path) -> dict[str, Any]:
    products = in_domain_products(repo)
    components = in_domain_components(products)
    procurement = _purchasable_index(repo, output_dir)
    _, reach, cascade = _load_cache(repo)
    for path in sorted(output_dir.glob("plan_shard_*.jsonl.gz")):
        for row in read_jsonl_gzip(path, label=path.name):
            reach.setdefault((str(row["role"]), str(row["canonical_smiles"])), row)

    def leaves_ok(entry: dict[str, Any] | None) -> bool:
        if not entry:
            return False
        leaves = [str(v) for v in (entry.get("stock_leaves") or [])]
        return bool(leaves) and all(
            (procurement.get(leaf) or {}).get("vendor_count") for leaf in leaves
        )

    # Deliberate application of the published AGILE Tail A route, forward-verified
    # against the repository's own qualified transforms.
    tail_a: dict[str, dict[str, Any]] = {}
    tail_a_path = (
        repo / "results/phase1/ugi_tail_a_disconnection_v1/tail_a_disconnection_ledger.jsonl.gz"
    )
    if tail_a_path.is_file():
        tail_a = {
            str(r["canonical_smiles"]): r
            for r in read_jsonl_gzip(tail_a_path, label="tail A disconnection")
        }
    # Published isocyanide route: formylate the primary amine, dehydrate the formamide.
    iso_route: dict[str, dict[str, Any]] = {}
    iso_path = repo / "results/phase1/ugi_isocyanide_route_v1/isocyanide_route_ledger.jsonl.gz"
    if iso_path.is_file():
        for r in read_jsonl_gzip(iso_path, label="isocyanide route"):
            iso_route[str(r["canonical_smiles"])] = {
                "tail_a_verified": bool(r["route_verified"]),
                "both_purchasable": bool(r["amine_purchasable"]),
            }
    tail_a = {**iso_route, **tail_a}

    state: dict[tuple[str, str], dict[str, Any]] = {}
    for role, smiles in components:
        vendors = (procurement.get(smiles) or {}).get("vendor_count")
        entry = reach.get((role, smiles))
        closed = bool(
            (cascade.get((role, smiles)) or {}).get("final_component_state") == "complete"
        )
        solved = bool(entry and entry.get("planner_solved_to_public_catalog"))
        tail = tail_a.get(smiles)
        tail_ok = bool(tail and tail.get("tail_a_verified") and tail.get("both_purchasable"))
        if vendors:
            verdict = "buy"
        elif (solved and leaves_ok(entry)) or tail_ok:
            verdict = "make_from_purchasable"
        elif closed or solved or (tail and tail.get("tail_a_verified")):
            verdict = "route_only"
        else:
            verdict = "blocked"
        state[(role, smiles)] = {
            "role": role,
            "canonical_smiles": smiles,
            "vendor_count": vendors,
            "exact_closed": closed,
            "planner_solved": solved,
            "tail_a_verified": bool(tail and tail.get("tail_a_verified")),
            "tail_a_both_purchasable": tail_ok,
            "route_leaves_purchasable": leaves_ok(entry),
            "verdict": verdict,
        }

    rows: list[dict[str, Any]] = []
    by_product: dict[str, dict[str, Any]] = {}
    for row in products:
        product = row["canonical_product"]
        if product in by_product:
            continue
        parts = [state[(role, row[ROLE_COLUMN[role]])] for role in ROLES]
        verdicts = [p["verdict"] for p in parts]
        if any(v == "blocked" for v in verdicts):
            makeability = "blocked"
        elif all(v == "buy" for v in verdicts):
            makeability = "buy_all"
        elif any(v == "route_only" for v in verdicts):
            makeability = "route_only"
        else:
            makeability = "buy_and_make"
        record = {
            "schema_version": "phase1_ugi_indomain_makeability_ledger.v1",
            "canonical_product": product,
            "arm_id": row["arm_id"],
            "authority_tier": row.get("authority_tier"),
            "conservative_high_potency": row.get("conservative_high_potency") == "True",
            "oracle_mean": row.get("oracle_mean") or None,
            "lcb90": row.get("lcb90") or None,
            "components": {role: row[ROLE_COLUMN[role]] for role in ROLES},
            "component_verdicts": {p["role"]: p["verdict"] for p in parts},
            "makeability": makeability,
            "buy": [p["canonical_smiles"] for p in parts if p["verdict"] == "buy"],
            "synthesise": [p["canonical_smiles"] for p in parts if p["verdict"] != "buy"],
        }
        by_product[product] = record
        rows.append(record)

    rows.sort(key=lambda r: (r["makeability"], -(float(r["lcb90"]) if r["lcb90"] else -1e9)))
    path = output_dir / "indomain_makeability_ledger.jsonl.gz"
    atomic_write(path, jsonl_gzip_bytes(rows))

    def _arm(name: str, predicate) -> dict[str, int]:
        counts = {
            arm: sum(1 for r in rows if r["arm_id"] == arm and predicate(r))
            for arm in ("broad_prior", "support_enriched")
        }
        counts["balanced_per_causal_arm"] = min(counts.values())
        counts["name"] = name
        return counts

    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "indomain_makeability_complete",
        "summary": {
            "in_domain_products": len(rows),
            "in_domain_components": len(components),
            "component_verdicts": dict(
                sorted(Counter(v["verdict"] for v in state.values()).items())
            ),
            "component_verdicts_by_role": {
                role: dict(
                    sorted(
                        Counter(v["verdict"] for v in state.values() if v["role"] == role).items()
                    )
                )
                for role in ROLES
            },
            "makeability": dict(sorted(Counter(r["makeability"] for r in rows).items())),
            "makeability_by_arm": {
                arm: dict(
                    sorted(Counter(r["makeability"] for r in rows if r["arm_id"] == arm).items())
                )
                for arm in sorted({r["arm_id"] for r in rows})
            },
            "panel_fill": [
                _arm(
                    "buy_and_make_or_better",
                    lambda r: r["makeability"] in ("buy_all", "buy_and_make"),
                ),
                _arm(
                    "including_route_only",
                    lambda r: r["makeability"] in ("buy_all", "buy_and_make", "route_only"),
                ),
                _arm(
                    "buy_and_make_conservative_high",
                    lambda r: r["makeability"] in ("buy_all", "buy_and_make")
                    and r["conservative_high_potency"],
                ),
            ],
        },
        "artifacts": {
            "indomain_makeability_ledger.jsonl.gz": {
                "path": path.name,
                "sha256": sha256_file(path),
                "rows": len(rows),
            }
        },
        "runtime": {"python_version": platform.python_version(), "platform": platform.platform()},
        "authority": dict(AUTHORITY),
        "nonclaims": [
            "A vendor listing is a screening signal, not a quote or a stock level.",
            "A planner solution is not route evidence.",
            "This selects no candidate and locks no panel.",
            "The frozen 256-product cascade measurement is unchanged by this artifact.",
        ],
    }
    result = {**content, "result_sha256": sha256_payload(content)}
    atomic_write(output_dir / "result.json", canonical_json_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--stage", choices=("procure", "plan", "rollup"), required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output_dir = (repo / OUT).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "procure":
        result = stage_procure(repo, output_dir)
    elif args.stage == "plan":
        result = stage_plan(repo, output_dir, args.shard, args.shards)
    else:
        result = stage_rollup(repo, output_dir)
    print(json.dumps(result["summary"], indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
