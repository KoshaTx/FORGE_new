#!/usr/bin/env python3
"""Freeze AiZynthFinder proposals for the public 120-target panel."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

from forge.synthesis.engine.aizynthfinder_single_step_recovery import (
    CONFIG_SCHEMA_VERSION,
    LEDGER_SCHEMA_VERSION,
    AiZynthFinderRecoveryError,
    content_sha256,
    load_gzip_json,
    proposal_result,
    sha256_file,
    stable_json,
    validate_public_targets,
)

RUNTIME_SCHEMA_VERSION = "phase1_aizynthfinder_runtime_manifest.v1"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise AiZynthFinderRecoveryError(f"JSON object required: {path}")
    return value


def _pin(root: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, dict):
        raise AiZynthFinderRecoveryError(f"{label} pin is malformed")
    path = root / str(record.get("path"))
    if sha256_file(path) != record.get("sha256"):
        raise AiZynthFinderRecoveryError(f"{label} hash changed")
    return path


def _verify_runtime_manifest(root: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != RUNTIME_SCHEMA_VERSION:
        raise AiZynthFinderRecoveryError("AiZynthFinder runtime schema changed")
    if manifest.get("status") != "diagnostic_only_not_route_evidence":
        raise AiZynthFinderRecoveryError("AiZynthFinder runtime authority changed")
    if manifest.get("authority") != {
        "may_generate_route_hypotheses": True,
        "may_create_evidence_records": False,
        "may_set_route_complete": False,
        "may_populate_synthesis_success_probability": False,
        "may_enter_synthesis_value": False,
    }:
        raise AiZynthFinderRecoveryError("AiZynthFinder runtime authority is unsafe")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise AiZynthFinderRecoveryError("AiZynthFinder config pin is malformed")
    _pin(root, config, label="AiZynthFinder engine config")
    assets = manifest.get("assets")
    if not isinstance(assets, list) or not assets:
        raise AiZynthFinderRecoveryError("AiZynthFinder runtime assets are missing")
    for index, record in enumerate(assets):
        path = _pin(root, record, label=f"AiZynthFinder asset {index}")
        if path.stat().st_size != record.get("bytes"):
            raise AiZynthFinderRecoveryError(f"AiZynthFinder asset size changed: {path}")


def _canonical_reactants(reaction: Any) -> list[str]:
    outcomes = getattr(reaction, "reactants", ())
    if not outcomes:
        return []
    reactants: list[str] = []
    for molecule in outcomes[0]:
        parsed = Chem.MolFromSmiles(molecule.smiles)
        if parsed is None:
            raise AiZynthFinderRecoveryError("AiZynthFinder emitted invalid reactants")
        reactants.append(Chem.MolToSmiles(parsed, canonical=True, isomericSmiles=False))
    return sorted(reactants)


def _proposal_record(group: tuple[Any, ...], *, rank: int) -> dict[str, Any]:
    reaction = group[0]
    metadata = dict(getattr(reaction, "metadata", {}))
    retained = {
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
                "metadata": retained,
            }
        ),
        "metadata": retained,
        "authority": "proposal_only_not_route_evidence",
    }


def _write_gzip_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as raw:
        temporary = Path(raw.name)
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8") as text:
                for row in rows:
                    text.write(stable_json(row) + "\n")
    os.replace(temporary, path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def run(root: Path, config_path: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise AiZynthFinderRecoveryError("unsupported recovery benchmark config")
    if config.get("status") != "frozen_before_proposal_execution":
        raise AiZynthFinderRecoveryError("benchmark config is not frozen")
    inputs = config.get("inputs")
    if not isinstance(inputs, dict):
        raise AiZynthFinderRecoveryError("benchmark input pins are missing")
    truth_record = inputs.get("scoring_truth")
    if (
        not isinstance(truth_record, dict)
        or truth_record.get("proposal_execution_input") is not False
    ):
        raise AiZynthFinderRecoveryError("hidden truth boundary is malformed")
    target_path = _pin(root, inputs.get("target_manifest"), label="target manifest")
    runtime_path = _pin(root, inputs.get("runtime_manifest"), label="runtime manifest")
    targets = validate_public_targets(load_gzip_json(target_path))
    runtime = _load_json(runtime_path)
    _verify_runtime_manifest(root, runtime)

    seed = int(config.get("seed"))
    random.seed(seed)
    np.random.seed(seed)
    settings = config.get("single_step")
    if not isinstance(settings, dict):
        raise AiZynthFinderRecoveryError("single-step settings are missing")
    maximum = int(settings.get("maximum_proposals"))
    if maximum != 20:
        raise AiZynthFinderRecoveryError("frozen maximum proposal count changed")

    from aizynthfinder.aizynthfinder import AiZynthExpander

    expander = AiZynthExpander(configfile=str(root / runtime["config"]["path"]))
    expander.expansion_policy.select(list(settings.get("expansion_policies", [])))
    expander.filter_policy.select(list(settings.get("filter_policies", [])))
    rows: list[dict[str, Any]] = []
    for index, target in enumerate(targets, start=1):
        groups = expander.do_expansion(target["canonical_smiles"], return_n=maximum)
        proposals = [
            _proposal_record(group, rank=rank)
            for rank, group in enumerate(groups, start=1)
            if group
        ]
        rows.append(
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                **target,
                "proposals": proposals,
                "proposal_source": "aizynthfinder_public_uspto_ringbreaker_v4_4_1",
                "authority": "proposal_only_not_route_evidence",
            }
        )
        print(f"proposal {index}/120 {target['primary_stratum']}", flush=True)

    output = root / str(config.get("output_directory"))
    if output.exists() and any(output.iterdir()):
        raise AiZynthFinderRecoveryError("write-once output directory is not empty")
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = output / "proposal_ledger.jsonl.gz"
    _write_gzip_jsonl(ledger_path, rows)
    result = proposal_result(
        config_path=str(config_path.relative_to(root)),
        config_sha256=sha256_file(config_path),
        target_path=str(target_path.relative_to(root)),
        target_sha256=sha256_file(target_path),
        runtime_path=str(runtime_path.relative_to(root)),
        runtime_sha256=sha256_file(runtime_path),
        ledger_path=str(ledger_path.relative_to(root)),
        ledger_sha256=sha256_file(ledger_path),
        rows=rows,
        maximum_proposals=maximum,
    )
    _write_json(output / "proposal_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/route/phase1_aizynthfinder_single_step_recovery_benchmark_v1.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    print(json.dumps(run(root, config)["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
