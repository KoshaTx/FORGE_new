#!/usr/bin/env python3
"""Run the public AiZynthFinder policy as a proposal-only FORGE diagnostic."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.archive.phase1.synthesis_audits.aizynthfinder_diagnostic import (
    RESULT_SCHEMA_VERSION,
    AiZynthFinderDiagnosticError,
    content_sha256,
    select_full_search_targets,
    summarize_worker_records,
    targets_from_molecule_audit,
)

CONFIG_SCHEMA_VERSION = "phase1_aizynthfinder_component_route_diagnostic_config.v1"
RUNTIME_SCHEMA_VERSION = "phase1_aizynthfinder_runtime_manifest.v1"
LEDGER_SCHEMA_VERSION = "phase1_aizynthfinder_component_route_hypothesis_ledger.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise AiZynthFinderDiagnosticError(f"could not read {path}") from exc
    except json.JSONDecodeError as exc:
        raise AiZynthFinderDiagnosticError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise AiZynthFinderDiagnosticError(f"{path} must contain an object")
    return payload


def _require_sha(path: Path, expected: Any, *, label: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise AiZynthFinderDiagnosticError(f"{label} expected SHA-256 is malformed")
    observed = _sha256_file(path)
    if observed != expected:
        raise AiZynthFinderDiagnosticError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return observed


def _verify_runtime_manifest(root: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != RUNTIME_SCHEMA_VERSION:
        raise AiZynthFinderDiagnosticError("AiZynthFinder runtime manifest schema is unsupported")
    if manifest.get("status") != "diagnostic_only_not_route_evidence":
        raise AiZynthFinderDiagnosticError("AiZynthFinder runtime is not quarantined as diagnostic")
    authority = manifest.get("authority")
    expected_authority = {
        "may_generate_route_hypotheses": True,
        "may_create_evidence_records": False,
        "may_set_route_complete": False,
        "may_populate_synthesis_success_probability": False,
        "may_enter_synthesis_value": False,
    }
    if authority != expected_authority:
        raise AiZynthFinderDiagnosticError("AiZynthFinder authority boundary is malformed")
    config_record = manifest.get("config")
    if not isinstance(config_record, dict):
        raise AiZynthFinderDiagnosticError("runtime config record is malformed")
    _require_sha(
        root / str(config_record.get("path")),
        config_record.get("sha256"),
        label="AiZynthFinder config",
    )
    assets = manifest.get("assets")
    if not isinstance(assets, list) or not assets:
        raise AiZynthFinderDiagnosticError("runtime manifest has no assets")
    for index, raw_asset in enumerate(assets):
        if not isinstance(raw_asset, dict):
            raise AiZynthFinderDiagnosticError(f"runtime asset {index} is malformed")
        path = root / str(raw_asset.get("path"))
        _require_sha(path, raw_asset.get("sha256"), label=f"runtime asset {path}")
        if path.stat().st_size != raw_asset.get("bytes"):
            raise AiZynthFinderDiagnosticError(f"runtime asset byte count mismatch: {path}")


def _canonical_reactants(reaction: Any) -> list[str]:
    from rdkit import Chem

    outcomes = getattr(reaction, "reactants", ())
    if not outcomes:
        return []
    first_outcome = outcomes[0]
    reactants: list[str] = []
    for molecule in first_outcome:
        parsed = Chem.MolFromSmiles(molecule.smiles)
        if parsed is None:
            raise AiZynthFinderDiagnosticError("AiZynthFinder returned an invalid reactant")
        reactants.append(Chem.MolToSmiles(parsed, canonical=True, isomericSmiles=False))
    return sorted(reactants)


def _proposal_record(group: tuple[Any, ...], *, rank: int) -> dict[str, Any]:
    reaction = group[0]
    metadata = dict(getattr(reaction, "metadata", {}))
    retained_metadata = {
        key: metadata.get(key)
        for key in (
            "policy_name",
            "policy_probability",
            "policy_probability_rank",
            "feasibility",
            "template_hash",
            "template_code",
            "library_occurence",
            "classification",
            "mapped_reaction_smiles",
        )
    }
    reactants = _canonical_reactants(reaction)
    return {
        "rank": rank,
        "canonical_reactants": reactants,
        "proposal_sha256": content_sha256(
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "rank": rank,
                "canonical_reactants": reactants,
                "metadata": retained_metadata,
            }
        ),
        "metadata": retained_metadata,
        "authority": "proposal_only_not_route_evidence",
    }


def _runtime_inventory() -> dict[str, Any]:
    packages = sorted(
        {
            (
                distribution.metadata.get("Name", "").lower(),
                distribution.version,
            )
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        }
    )
    records = [{"name": name, "version": version} for name, version in packages]
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": records,
        "packages_sha256": content_sha256(records),
    }


def _write_gzip_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as raw_stream:
        temporary_path = Path(raw_stream.name)
        with gzip.GzipFile(fileobj=raw_stream, mode="wb", filename="", mtime=0) as gzip_stream:
            with io.TextIOWrapper(gzip_stream, encoding="utf-8") as text_stream:
                for row in rows:
                    text_stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                    text_stream.write("\n")
    os.replace(temporary_path, path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary_path = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary_path, path)


def run(config_path: Path, *, root: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise AiZynthFinderDiagnosticError("diagnostic config schema is unsupported")
    inputs = config.get("inputs")
    if not isinstance(inputs, dict):
        raise AiZynthFinderDiagnosticError("diagnostic inputs are malformed")
    audit_record = inputs.get("molecule_audit_result")
    runtime_record = inputs.get("runtime_manifest")
    if not isinstance(audit_record, dict) or not isinstance(runtime_record, dict):
        raise AiZynthFinderDiagnosticError("diagnostic input records are malformed")
    audit_path = root / str(audit_record.get("path"))
    audit_sha256 = _require_sha(
        audit_path, audit_record.get("sha256"), label="molecule-audit result"
    )
    runtime_path = root / str(runtime_record.get("path"))
    runtime_sha256 = _require_sha(
        runtime_path, runtime_record.get("sha256"), label="runtime manifest"
    )
    runtime_manifest = _load_json(runtime_path)
    _verify_runtime_manifest(root, runtime_manifest)

    seed = int(config.get("seed"))
    random.seed(seed)
    np.random.seed(seed)
    audit_result = _load_json(audit_path)
    targets = targets_from_molecule_audit(audit_result)

    single_step = config.get("single_step")
    full_search = config.get("full_search")
    if not isinstance(single_step, dict) or not isinstance(full_search, dict):
        raise AiZynthFinderDiagnosticError("diagnostic search policies are malformed")
    maximum_proposals = int(single_step.get("maximum_proposals"))
    if maximum_proposals < 1:
        raise AiZynthFinderDiagnosticError("maximum_proposals must be positive")
    planner_ids = (
        select_full_search_targets(
            targets,
            maximum_unresolved_per_role=int(full_search.get("maximum_unresolved_per_role")),
        )
        if bool(full_search.get("enabled"))
        else set()
    )

    from aizynthfinder.aizynthfinder import AiZynthExpander, AiZynthFinder

    engine_config_path = root / runtime_manifest["config"]["path"]
    expander = AiZynthExpander(configfile=str(engine_config_path))
    expander.expansion_policy.select(list(single_step.get("expansion_policies", [])))
    expander.filter_policy.select(list(single_step.get("filter_policies", [])))

    rows_by_target: dict[str, dict[str, Any]] = {}
    for index, target in enumerate(targets, start=1):
        groups = expander.do_expansion(target.canonical_smiles, return_n=maximum_proposals)
        proposals = [
            _proposal_record(group, rank=rank)
            for rank, group in enumerate(groups, start=1)
            if group
        ]
        rows_by_target[target.target_id] = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "target": target.to_dict(),
            "single_step_proposals": proposals,
            "single_step_stats": dict(expander.stats),
            "full_search": None,
            "nonclaim": "proposal_only_not_route_evidence",
        }
        print(f"single-step {index}/{len(targets)} {target.cohort} {target.role}", flush=True)

    if planner_ids:
        finder = AiZynthFinder(configfile=str(engine_config_path))
        finder.expansion_policy.select(list(single_step.get("expansion_policies", [])))
        finder.filter_policy.select(list(single_step.get("filter_policies", [])))
        finder.stock.select(list(full_search.get("stocks", [])))
        finder.config.search.time_limit = int(full_search.get("time_limit_seconds"))
        finder.config.search.iteration_limit = int(full_search.get("iteration_limit"))
        finder.config.search.max_transforms = int(full_search.get("maximum_transforms"))
        planner_targets = [target for target in targets if target.target_id in planner_ids]
        for index, target in enumerate(planner_targets, start=1):
            finder.target_smiles = target.canonical_smiles
            finder.prepare_tree()
            finder.tree_search(show_progress=False)
            finder.build_routes()
            stats = finder.extract_statistics()
            route_dicts = finder.routes.dict_with_extra(include_metadata=True, include_scores=True)
            rows_by_target[target.target_id]["full_search"] = {
                "is_solved_to_public_stock": bool(stats.get("is_solved")),
                "statistics": stats,
                "top_route_hypotheses": route_dicts[
                    : int(full_search.get("maximum_routes_stored"))
                ],
                "authority": "planner_diagnostic_not_route_evidence",
            }
            print(
                f"full-search {index}/{len(planner_targets)} solved={bool(stats.get('is_solved'))} "
                f"{target.cohort} {target.role}",
                flush=True,
            )

    rows = [rows_by_target[target.target_id] for target in targets]
    output_directory = root / str(config.get("output_directory"))
    ledger_path = output_directory / "route_hypotheses.jsonl.gz"
    result_path = output_directory / "result.json"
    _write_gzip_jsonl_atomic(ledger_path, rows)

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_diagnostic_only",
        "inputs": {
            "config": {"path": str(config_path), "sha256": _sha256_file(config_path)},
            "molecule_audit_result": {"path": str(audit_path), "sha256": audit_sha256},
            "runtime_manifest": {"path": str(runtime_path), "sha256": runtime_sha256},
        },
        "runtime": _runtime_inventory(),
        "policy": {
            "seed": seed,
            "single_step": single_step,
            "full_search": full_search,
            "full_search_target_count": len(planner_ids),
        },
        "summary": summarize_worker_records(rows),
        "artifacts": {
            "route_hypothesis_ledger": {
                "path": str(ledger_path),
                "sha256": _sha256_file(ledger_path),
            }
        },
        "nonclaims": list(config.get("nonclaims", [])),
    }
    result["result_sha256"] = content_sha256(result)
    _write_json_atomic(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/route/phase1_aizynthfinder_component_route_diagnostic_v1.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = run(args.config.resolve(), root=args.root.resolve())
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
