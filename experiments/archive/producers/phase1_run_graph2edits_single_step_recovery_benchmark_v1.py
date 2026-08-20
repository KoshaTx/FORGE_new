#!/usr/bin/env python3
"""Freeze Graph2Edits proposals for the public 120-target panel."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from forge.synthesis.engine.aizynthfinder_single_step_recovery import (
    canonical_connected_smiles,
    content_sha256,
    load_gzip_json,
    sha256_file,
    stable_json,
    validate_public_targets,
)
from forge.synthesis.engine.graph2edits_backend import build_syntheseus_worker
from forge.synthesis.engine.graph2edits_runtime_qualification import verify_runtime_receipt
from forge.synthesis.engine.graph2edits_single_step_recovery import RESULT_SCHEMA_VERSION

CONFIG_SCHEMA_VERSION = "phase1_graph2edits_single_step_recovery_benchmark_config.v1"
LEDGER_SCHEMA_VERSION = "phase1_graph2edits_single_step_recovery_ledger.v1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _pin(root: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, dict):
        raise RuntimeError(f"{label} pin is malformed")
    path = root / str(record.get("path"))
    if sha256_file(path) != record.get("sha256"):
        raise RuntimeError(f"{label} hash changed")
    return path


def _gzip_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as raw:
        temporary = Path(raw.name)
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8") as text:
                for row in rows:
                    text.write(stable_json(row) + "\n")
    os.replace(temporary, path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/route/phase1_graph2edits_single_step_recovery_benchmark_v1.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = _json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise RuntimeError("Graph2Edits recovery config schema changed")
    if config.get("status") != "frozen_before_proposal_execution":
        raise RuntimeError("Graph2Edits recovery config is not frozen")
    inputs = config["inputs"]
    if inputs["scoring_truth"].get("proposal_execution_input") is not False:
        raise RuntimeError("hidden truth boundary is malformed")
    target_path = _pin(root, inputs["target_manifest"], label="target manifest")
    runtime_path = _pin(root, inputs["runtime_qualification"], label="runtime receipt")
    runtime = _json(runtime_path)
    verify_runtime_receipt(
        runtime,
        repo_root=root,
        verify_artifacts=True,
        verify_environment=True,
    )
    targets = validate_public_targets(load_gzip_json(target_path))
    policy = config["single_step"]
    maximum = int(policy["maximum_proposals"])
    checkpoint = root / runtime["artifacts"]["checkpoint"]["path"]
    worker = build_syntheseus_worker(
        model_dir=checkpoint.parent,
        device=str(policy["device"]),
        max_edit_steps=int(policy["maximum_edit_steps"]),
    )
    rows: list[dict[str, Any]] = []
    for index, target in enumerate(targets, start=1):
        raw = worker(target["canonical_smiles"], num_results=maximum)
        proposals: list[dict[str, Any]] = []
        seen: set[tuple[str, ...]] = set()
        for prediction in raw:
            reactants = tuple(
                sorted(
                    canonical_connected_smiles(value, label="Graph2Edits reactant")
                    for value in prediction.reactants_smiles.split(".")
                    if value
                )
            )
            if not reactants or reactants in seen:
                continue
            seen.add(reactants)
            proposals.append(
                {
                    "rank": len(proposals) + 1,
                    "canonical_reactants": list(reactants),
                    "model_score": prediction.model_score,
                    "proposal_sha256": content_sha256(
                        {"target_id": target["target_id"], "reactants": reactants}
                    ),
                    "authority": "proposal_only_not_route_evidence",
                }
            )
        rows.append(
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                **target,
                "proposals": proposals,
                "proposal_source": "graph2edits_uspto50k_reaction_class_unknown",
                "authority": "proposal_only_not_route_evidence",
            }
        )
        print(f"proposal {index}/120 {target['primary_stratum']}", flush=True)

    output = root / config["output_directory"]
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("write-once Graph2Edits output directory is not empty")
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = output / "proposal_ledger.jsonl.gz"
    _gzip_rows(ledger_path, rows)
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "proposal_ledger_frozen_before_truth_scoring",
        "config": {"path": str(config_path.relative_to(root)), "sha256": sha256_file(config_path)},
        "proposal_execution_inputs": {
            "target_manifest": {
                "path": str(target_path.relative_to(root)),
                "sha256": sha256_file(target_path),
            },
            "runtime_qualification": {
                "path": str(runtime_path.relative_to(root)),
                "sha256": sha256_file(runtime_path),
            },
        },
        "hidden_truth_loaded_during_proposal_execution": False,
        "summary": {
            "targets": len(rows),
            "targets_with_proposals": sum(bool(row["proposals"]) for row in rows),
            "proposals": sum(len(row["proposals"]) for row in rows),
            "maximum_proposals_per_target": maximum,
        },
        "artifacts": {
            "proposal_ledger": {
                "path": str(ledger_path.relative_to(root)),
                "sha256": sha256_file(ledger_path),
                "rows": len(rows),
                "schema_version": LEDGER_SCHEMA_VERSION,
            }
        },
        "scientific_authority": {
            "proposal_only": True,
            "route_evidence_created": False,
            "route_closure_authorized": False,
            "may_enter_synthesis_value": False,
            "synthesis_tilting_activated": False,
        },
    }
    result["result_sha256"] = content_sha256(result)
    _write_json(output / "proposal_result.json", result)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
