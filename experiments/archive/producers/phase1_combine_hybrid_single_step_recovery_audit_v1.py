#!/usr/bin/env python3
"""Combine two frozen proposal-recovery score artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from forge.synthesis.engine.aizynthfinder_single_step_recovery import content_sha256, sha256_file
from forge.synthesis.engine.single_step_recovery_combination import combine_recovery_scores


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/route/phase1_hybrid_single_step_recovery_audit_v1.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = _json(config_path)
    inputs: dict[str, dict[str, str]] = config["inputs"]
    loaded: dict[str, dict[str, Any]] = {}
    for label, record in inputs.items():
        path = root / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"input score hash changed: {label}")
        loaded[label] = _json(path)
    result = combine_recovery_scores(loaded["aizynthfinder_score"], loaded["graph2edits_score"])
    result["config"] = {
        "path": str(config_path.relative_to(root)),
        "sha256": sha256_file(config_path),
    }
    result.pop("result_sha256", None)
    result["result_sha256"] = content_sha256(result)
    output = root / config["output_directory"]
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("write-once hybrid recovery output already exists")
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
