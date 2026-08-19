#!/usr/bin/env python3
"""Run AiZynthFinder on Graph2Edits checkpoint residuals and upstream leaves."""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from phase1_run_aizynthfinder_component_route_diagnostic_v1 import (
    _load_json,
    _proposal_record,
    _require_sha,
    _runtime_inventory,
    _sha256_file,
    _verify_runtime_manifest,
    _write_gzip_jsonl_atomic,
    _write_json_atomic,
)

from forge.route.aizynthfinder_checkpoint_residual import (
    build_residual_targets,
    summarize_residual_records,
)
from forge.route.aizynthfinder_diagnostic import content_sha256

CONFIG_SCHEMA_VERSION = "forge.aizynthfinder_checkpoint_residual_config.v1"
RESULT_SCHEMA_VERSION = "forge.aizynthfinder_checkpoint_residual.v1"
LEDGER_SCHEMA_VERSION = "forge.aizynthfinder_checkpoint_residual_ledger.v1"


def _read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"malformed gzip JSONL: {path}")
    return rows


def _load_contract(config_path: Path, root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = _load_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise RuntimeError("checkpoint-residual config schema changed")
    if config.get("status") != "frozen_complementary_proposal_only_diagnostic":
        raise RuntimeError("checkpoint-residual diagnostic is not frozen")
    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "component_ledger",
        "leaf_worklist",
        "runtime_manifest",
    }:
        raise RuntimeError("checkpoint-residual inputs changed")
    paths: dict[str, Path] = {}
    for label, record in inputs.items():
        if not isinstance(record, dict):
            raise RuntimeError(f"malformed input: {label}")
        path = root / str(record["path"])
        _require_sha(path, record["sha256"], label=label)
        paths[label] = path
    runtime = _load_json(paths["runtime_manifest"])
    _verify_runtime_manifest(root, runtime)
    targets = build_residual_targets(
        _read_gzip_jsonl(paths["component_ledger"]),
        _read_gzip_jsonl(paths["leaf_worklist"]),
    )
    return config, targets


def _run_shard(
    config_path: Path, root: Path, *, shard_index: int, shard_count: int
) -> dict[str, Any]:
    config, targets = _load_contract(config_path, root)
    if shard_count != int(config["shard_count"]):
        raise RuntimeError("shard count differs from frozen config")
    if not 0 <= shard_index < shard_count:
        raise RuntimeError("shard index is out of range")
    selected = [row for index, row in enumerate(targets) if index % shard_count == shard_index]
    random.seed(int(config["seed"]) + shard_index)
    np.random.seed(int(config["seed"]) + shard_index)

    from aizynthfinder.aizynthfinder import AiZynthExpander, AiZynthFinder

    runtime = _load_json(root / config["inputs"]["runtime_manifest"]["path"])
    engine_config_path = root / runtime["config"]["path"]
    single_step = config["single_step"]
    full_search = config["full_search"]
    expander = AiZynthExpander(configfile=str(engine_config_path))
    expander.expansion_policy.select(list(single_step["expansion_policies"]))
    expander.filter_policy.select(list(single_step["filter_policies"]))
    finder = AiZynthFinder(configfile=str(engine_config_path))
    finder.expansion_policy.select(list(single_step["expansion_policies"]))
    finder.filter_policy.select(list(single_step["filter_policies"]))
    finder.stock.select(list(full_search["stocks"]))
    finder.config.search.time_limit = int(full_search["time_limit_seconds"])
    finder.config.search.iteration_limit = int(full_search["iteration_limit"])
    finder.config.search.max_transforms = int(full_search["maximum_transforms"])

    rows: list[dict[str, Any]] = []
    for index, target in enumerate(selected, start=1):
        groups = expander.do_expansion(
            target["canonical_smiles"], return_n=int(single_step["maximum_proposals"])
        )
        proposals = [
            _proposal_record(group, rank=rank)
            for rank, group in enumerate(groups, start=1)
            if group
        ]
        try:
            finder.target_smiles = target["canonical_smiles"]
            finder.prepare_tree()
            finder.tree_search(show_progress=False)
            finder.build_routes()
            stats = finder.extract_statistics()
            route_dicts = finder.routes.dict_with_extra(include_metadata=True, include_scores=True)
            search: dict[str, Any] = {
                "execution_status": "complete",
                "is_solved_to_public_stock": bool(stats.get("is_solved")),
                "statistics": stats,
                "top_route_hypotheses": route_dicts[: int(full_search["maximum_routes_stored"])],
                "authority": "planner_diagnostic_not_route_evidence",
            }
        except Exception as exc:  # planner failures are recorded, never relabeled as chemistry
            search = {
                "execution_status": "failed",
                "is_solved_to_public_stock": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "top_route_hypotheses": [],
                "authority": "planner_execution_failure_not_chemical_rejection",
            }
        rows.append(
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "target": target,
                "single_step_proposals": proposals,
                "single_step_stats": dict(expander.stats),
                "full_search": search,
                "nonclaim": "proposal_only_not_route_evidence",
            }
        )
        print(
            f"shard {shard_index + 1}/{shard_count} target {index}/{len(selected)} "
            f"solved={search['is_solved_to_public_stock']} {target['cohort']}",
            flush=True,
        )

    output_root = root / config["output_directory"]
    output_dir = output_root / f"shard_{shard_index:02d}_of_{shard_count:02d}"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"write-once shard output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "route_hypotheses.jsonl.gz"
    _write_gzip_jsonl_atomic(ledger_path, rows)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_proposal_only_shard",
        "config": {"path": str(config_path), "sha256": _sha256_file(config_path)},
        "shard": {"index": shard_index, "count": shard_count},
        "runtime": _runtime_inventory(),
        "summary": summarize_residual_records(rows),
        "artifacts": {
            "route_hypotheses": {
                "path": str(ledger_path.relative_to(root)),
                "sha256": _sha256_file(ledger_path),
            }
        },
        "scientific_authority": config["scope_guards"],
    }
    result["result_sha256"] = content_sha256(result)
    _write_json_atomic(output_dir / "result.json", result)
    return result


def _combine(config_path: Path, root: Path) -> dict[str, Any]:
    config, targets = _load_contract(config_path, root)
    output_root = root / config["output_directory"]
    rows: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    for shard_index in range(int(config["shard_count"])):
        shard_dir = output_root / f"shard_{shard_index:02d}_of_{int(config['shard_count']):02d}"
        result_path = shard_dir / "result.json"
        ledger_path = shard_dir / "route_hypotheses.jsonl.gz"
        result = _load_json(result_path)
        if result.get("status") != "complete_proposal_only_shard":
            raise RuntimeError(f"incomplete shard: {shard_index}")
        if result.get("config", {}).get("sha256") != _sha256_file(config_path):
            raise RuntimeError(f"shard config changed: {shard_index}")
        if result["artifacts"]["route_hypotheses"]["sha256"] != _sha256_file(ledger_path):
            raise RuntimeError(f"shard ledger changed: {shard_index}")
        rows.extend(_read_gzip_jsonl(ledger_path))
        shard_records.append(
            {
                "index": shard_index,
                "result_path": str(result_path.relative_to(root)),
                "result_sha256": _sha256_file(result_path),
                "ledger_path": str(ledger_path.relative_to(root)),
                "ledger_sha256": _sha256_file(ledger_path),
            }
        )
    rows.sort(key=lambda row: str(row["target"]["target_id"]))
    if {row["target"]["target_id"] for row in rows} != {target["target_id"] for target in targets}:
        raise RuntimeError("combined residual target census changed")
    ledger_path = output_root / "route_hypotheses.jsonl.gz"
    result_path = output_root / "result.json"
    if ledger_path.exists() or result_path.exists():
        raise RuntimeError("write-once combined residual output already exists")
    _write_gzip_jsonl_atomic(ledger_path, rows)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_complementary_proposal_only_diagnostic",
        "config": {"path": str(config_path), "sha256": _sha256_file(config_path)},
        "inputs": config["inputs"],
        "runtime": _runtime_inventory(),
        "summary": summarize_residual_records(rows),
        "shards": shard_records,
        "artifacts": {
            "route_hypotheses": {
                "path": str(ledger_path.relative_to(root)),
                "sha256": _sha256_file(ledger_path),
            }
        },
        "scientific_authority": config["scope_guards"],
        "decision_boundary": (
            "AiZynthFinder proposals and public-stock solutions require independent forward, "
            "scope, operational and current-terminal adjudication before changing route value"
        ),
    }
    result["result_sha256"] = content_sha256(result)
    _write_json_atomic(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/route/phase1_aizynthfinder_checkpoint_residual_v1.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--combine", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config.resolve()
    if args.combine:
        result = _combine(config_path, root)
    else:
        if args.shard_index is None or args.shard_count is None:
            raise RuntimeError("--shard-index and --shard-count are required")
        result = _run_shard(
            config_path,
            root,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
