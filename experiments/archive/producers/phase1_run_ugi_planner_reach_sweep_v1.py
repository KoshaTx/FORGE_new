#!/usr/bin/env python3
"""Measure bounded-planner reach for every unresolved cascade component.

This answers one narrow question: at a single, larger, identical search budget,
how many unresolved components can a bounded multistep planner connect to a
public catalog snapshot?

It is a reachability measurement, not route evidence.  A planner solution here
never closes a route, never sets ``route_complete`` and never establishes
current procurement.  ZINC membership is a frozen public snapshot, not
purchasability.  An unsolved target is unsolved *under this bounded budget*.

Run the shards in parallel, then collect:

    PYTHONPATH=src .venv-aizynthfinder-py311-arm64/bin/python \\
        scripts/phase1_run_ugi_planner_reach_sweep_v1.py --shard 0
    PYTHONPATH=src .venv-aizynthfinder-py311-arm64/bin/python \\
        scripts/phase1_run_ugi_planner_reach_sweep_v1.py --shard 1
    .venv/bin/python scripts/phase1_run_ugi_planner_reach_sweep_v1.py --collect
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_DEFAULT = Path(__file__).resolve().parents[1]
if str(REPO_DEFAULT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_DEFAULT / "src"))

from experiments.phase1.synthesis_guidance.route_cascade import (  # noqa: E402
    NOT_ASSESSED_OUTCOME,
    UgiBoundedHybridRouteCascadeError,
    atomic_write,
    canonical_json_bytes,
    jsonl_gzip_bytes,
    load_json,
    read_jsonl_gzip,
    sha256_file,
    sha256_payload,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi_planner_reach_sweep_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_planner_reach_sweep.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_planner_reach_ledger.v1"

AUTHORITY = {
    "planner_solution_is_route_evidence": False,
    "planner_solution_is_current_procurement_evidence": False,
    "public_catalog_membership_is_current_availability": False,
    "may_set_route_complete": False,
    "unsolved_means_unsynthesizable": False,
}


def load_contract(repo: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config = load_json(config_path, label="planner reach config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiBoundedHybridRouteCascadeError("unsupported planner reach config")
    if config.get("status") != "frozen_before_planner_reach_sweep":
        raise UgiBoundedHybridRouteCascadeError("planner reach config is not frozen")
    scope = config.get("scope", {})
    for guard in (
        "planner_solution_is_route_evidence",
        "planner_solution_is_current_procurement_evidence",
        "public_catalog_membership_is_current_availability",
        "may_set_route_complete",
        "may_change_cascade_result",
    ):
        if scope.get(guard) is not False:
            raise UgiBoundedHybridRouteCascadeError(f"planner reach scope guard changed: {guard}")
    paths: dict[str, Path] = {}
    for label, record in config["inputs"].items():
        path = (repo / str(record["path"])).resolve()
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise UgiBoundedHybridRouteCascadeError(f"input pin changed: {label}")
        paths[label] = path
    return config, paths


def select_targets(component_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every unresolved, assessable component — regardless of prior engine coverage."""

    targets = [
        row
        for row in component_rows
        if row["final_component_state"] != "complete"
        and row["exact_evidence"]["assessment_outcome"] != NOT_ASSESSED_OUTCOME
    ]
    targets.sort(key=lambda row: (str(row["role"]), str(row["canonical_smiles"])))
    return targets


def run_shard(repo: Path, config_path: Path, shard: int) -> dict[str, Any]:
    import random

    import numpy as np

    config, paths = load_contract(repo, config_path)
    shard_count = int(config["shard_count"])
    if not 0 <= shard < shard_count:
        raise UgiBoundedHybridRouteCascadeError("shard index out of range")
    component_rows = read_jsonl_gzip(paths["component_route_ledger"], label="component ledger")
    targets = select_targets(component_rows)
    expected = config["expected_population"]
    if len(targets) != int(expected["unresolved_assessable_components"]):
        raise UgiBoundedHybridRouteCascadeError(
            f"target population changed: expected "
            f"{expected['unresolved_assessable_components']}, observed {len(targets)}"
        )
    mine = [row for index, row in enumerate(targets) if index % shard_count == shard]

    seed = int(config["seed"]) + shard
    random.seed(seed)
    np.random.seed(seed % (2**32))

    runtime_manifest = load_json(paths["aizynthfinder_runtime_manifest"], label="runtime manifest")
    for asset in runtime_manifest["assets"]:
        asset_path = repo / asset["path"]
        if not asset_path.is_file() or asset_path.stat().st_size != int(asset["bytes"]):
            raise UgiBoundedHybridRouteCascadeError(f"AiZynthFinder asset changed: {asset['path']}")
    engine_config = repo / runtime_manifest["config"]["path"]
    if sha256_file(engine_config) != runtime_manifest["config"]["sha256"]:
        raise UgiBoundedHybridRouteCascadeError("AiZynthFinder engine config changed")

    from aizynthfinder.aizynthfinder import AiZynthExpander, AiZynthFinder
    from rdkit import Chem

    single_step = config["search"]["single_step"]
    full_search = config["search"]["full_search"]

    def canonical_reactants(reaction: Any) -> list[str]:
        outcomes = getattr(reaction, "reactants", ())
        if not outcomes:
            return []
        out: list[str] = []
        for molecule in outcomes[0]:
            parsed = Chem.MolFromSmiles(molecule.smiles)
            if parsed is None:
                raise UgiBoundedHybridRouteCascadeError("planner returned an invalid reactant")
            out.append(Chem.MolToSmiles(parsed, canonical=True, isomericSmiles=False))
        return sorted(out)

    expander = AiZynthExpander(configfile=str(engine_config))
    expander.expansion_policy.select(list(single_step["expansion_policies"]))
    expander.filter_policy.select(list(single_step["filter_policies"]))
    finder = AiZynthFinder(configfile=str(engine_config))
    finder.expansion_policy.select(list(single_step["expansion_policies"]))
    finder.filter_policy.select(list(single_step["filter_policies"]))
    finder.stock.select(list(full_search["stocks"]))
    finder.config.search.time_limit = int(full_search["time_limit_seconds"])
    finder.config.search.iteration_limit = int(full_search["iteration_limit"])
    finder.config.search.max_transforms = int(full_search["maximum_transforms"])

    rows: list[dict[str, Any]] = []
    for index, target in enumerate(mine, start=1):
        smiles = str(target["canonical_smiles"])
        record: dict[str, Any] = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "component_sha256": target["component_sha256"],
            "role": target["role"],
            "canonical_smiles": smiles,
            "occurrence_count": int(target["occurrence_count"]),
            "previously_tested_by_aizynthfinder": bool(target["aizynthfinder"]["attempted"]),
            "graph2edits_hypothesis_existed": bool(
                target["graph2edits"]["graph_consistent_hypothesis"]
            ),
            "graded_evidence_class": target["graded_family_projection"]["graded_evidence_class"],
            "single_step_proposals": [],
            "planner_solved_to_public_catalog": False,
            "statistics": None,
            "top_route_hypotheses": [],
            "stock_leaves": [],
            "execution_failure": None,
            "authority": dict(AUTHORITY),
        }
        try:
            groups = expander.do_expansion(smiles, return_n=int(single_step["maximum_proposals"]))
            record["single_step_proposals"] = [
                {"rank": rank, "canonical_reactants": canonical_reactants(group[0])}
                for rank, group in enumerate(groups, start=1)
                if group
            ]
        except Exception as error:  # noqa: BLE001 - recorded, never dropped
            record["execution_failure"] = f"single_step:{type(error).__name__}: {error}"
        try:
            finder.target_smiles = smiles
            finder.prepare_tree()
            finder.tree_search(show_progress=False)
            finder.build_routes()
            stats = finder.extract_statistics()
            routes = finder.routes.dict_with_extra(include_metadata=True, include_scores=True)
            stored = routes[: int(full_search["maximum_routes_stored"])]
            record["planner_solved_to_public_catalog"] = bool(stats.get("is_solved"))
            record["statistics"] = json.loads(json.dumps(stats, default=str))
            record["top_route_hypotheses"] = json.loads(json.dumps(stored, default=str))
            leaves: set[str] = set()
            for route in stored:
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
            previous = record["execution_failure"]
            failure = f"full_search:{type(error).__name__}: {error}"
            record["execution_failure"] = failure if previous is None else f"{previous}; {failure}"
        rows.append(record)
        print(
            f"shard{shard} {index}/{len(mine)} {target['role'][:20]} "
            f"solved={record['planner_solved_to_public_catalog']}",
            flush=True,
        )

    output_dir = (repo / str(config["output_directory"])).resolve() / f"shard_{shard:02d}"
    ledger_path = output_dir / "route_hypotheses.jsonl.gz"
    atomic_write(ledger_path, jsonl_gzip_bytes(rows))
    content = {
        "schema_version": RESULT_SCHEMA_VERSION + ".shard",
        "status": "planner_reach_shard_complete",
        "shard": shard,
        "shard_count": shard_count,
        "config": {"path": str(config_path.relative_to(repo)), "sha256": sha256_file(config_path)},
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "search": json.loads(json.dumps(config["search"])),
        "summary": {
            "targets": len(rows),
            "solved_to_public_catalog": sum(
                1 for row in rows if row["planner_solved_to_public_catalog"]
            ),
            "execution_failures": sum(1 for row in rows if row["execution_failure"]),
        },
        "artifacts": {
            "route_hypotheses.jsonl.gz": {
                "path": ledger_path.name,
                "sha256": sha256_file(ledger_path),
                "rows": len(rows),
            }
        },
        "authority": dict(AUTHORITY),
    }
    result = {**content, "result_sha256": sha256_payload(content)}
    atomic_write(output_dir / "result.json", canonical_json_bytes(result))
    return result


def collect(repo: Path, config_path: Path) -> dict[str, Any]:
    config, paths = load_contract(repo, config_path)
    shard_count = int(config["shard_count"])
    output_dir = (repo / str(config["output_directory"])).resolve()
    rows: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    for shard in range(shard_count):
        shard_dir = output_dir / f"shard_{shard:02d}"
        result_path = shard_dir / "result.json"
        if not result_path.is_file():
            raise UgiBoundedHybridRouteCascadeError(f"shard {shard} has not completed")
        shard_result = load_json(result_path, label=f"shard {shard} result")
        ledger_path = shard_dir / "route_hypotheses.jsonl.gz"
        if sha256_file(ledger_path) != (
            shard_result["artifacts"]["route_hypotheses.jsonl.gz"]["sha256"]
        ):
            raise UgiBoundedHybridRouteCascadeError(f"shard {shard} ledger changed")
        rows.extend(read_jsonl_gzip(ledger_path, label=f"shard {shard} ledger"))
        shards.append(
            {
                "shard": shard,
                "result_sha256": shard_result["result_sha256"],
                "ledger_sha256": shard_result["artifacts"]["route_hypotheses.jsonl.gz"]["sha256"],
                "targets": shard_result["summary"]["targets"],
            }
        )
    expected = int(config["expected_population"]["unresolved_assessable_components"])
    if len(rows) != expected:
        raise UgiBoundedHybridRouteCascadeError(
            f"collected {len(rows)} targets, expected {expected}"
        )
    rows.sort(key=lambda row: (str(row["role"]), str(row["canonical_smiles"])))
    ledger_path = output_dir / "planner_reach_ledger.jsonl.gz"
    atomic_write(ledger_path, jsonl_gzip_bytes(rows))

    solved = [row for row in rows if row["planner_solved_to_public_catalog"]]
    by_role = {
        role: {
            "targets": sum(1 for row in rows if row["role"] == role),
            "solved": sum(
                1 for row in rows if row["role"] == role and row["planner_solved_to_public_catalog"]
            ),
        }
        for role in sorted({str(row["role"]) for row in rows})
    }
    newly = [row for row in solved if not row["previously_tested_by_aizynthfinder"]]
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "planner_reach_sweep_complete",
        "config": {"path": str(config_path.relative_to(repo)), "sha256": sha256_file(config_path)},
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "search": json.loads(json.dumps(config["search"])),
        "shards": shards,
        "summary": {
            "targets": len(rows),
            "solved_to_public_catalog": len(solved),
            "solved_fraction": round(len(solved) / len(rows), 4) if rows else 0.0,
            "by_role": by_role,
            "previously_untested_targets": sum(
                1 for row in rows if not row["previously_tested_by_aizynthfinder"]
            ),
            "newly_solved_among_previously_untested": len(newly),
            "solved_by_graded_evidence_class": dict(
                sorted(Counter(row["graded_evidence_class"] for row in solved).items())
            ),
            "execution_failures": sum(1 for row in rows if row["execution_failure"]),
            "candidate_occurrences_behind_solved_components": sum(
                int(row["occurrence_count"]) for row in solved
            ),
        },
        "artifacts": {
            "planner_reach_ledger.jsonl.gz": {
                "path": ledger_path.name,
                "sha256": sha256_file(ledger_path),
                "rows": len(rows),
                "schema_version": LEDGER_SCHEMA_VERSION,
            }
        },
        "authority": dict(AUTHORITY),
        "interpretation": (
            "Reachability under one bounded planner budget against a frozen public catalog "
            "snapshot. This is a search-capability measurement, not route evidence, not "
            "procurement evidence and not a change to any cascade result."
        ),
        "nonclaims": list(config["nonclaims"]),
    }
    result = {**content, "result_sha256": sha256_payload(content)}
    atomic_write(output_dir / "result.json", canonical_json_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/route/phase1_ugi_planner_reach_sweep_v1.json"),
    )
    parser.add_argument("--shard", type=int)
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = args.config if args.config.is_absolute() else repo / args.config
    if args.collect:
        result = collect(repo, config_path)
    elif args.shard is not None:
        result = run_shard(repo, config_path, args.shard)
    else:
        raise SystemExit("pass --shard N or --collect")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
