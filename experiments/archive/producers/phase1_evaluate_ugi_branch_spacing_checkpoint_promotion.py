#!/usr/bin/env python3
"""Preflight or evaluate the independent branch-spacing checkpoint challenge."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from experiments.phase1.synthesis_guidance.guidance.ugi_branch_spacing_checkpoint_promotion import (
    build_branch_spacing_promotion_preflight,
    evaluate_branch_spacing_promotion,
)

REPO = Path(__file__).resolve().parents[3]
V3_POLICY = REPO / "configs/model/phase1_ugi_architecture_checkpoint_selection_policy_v3.json"
PROMOTION_POLICY = (
    REPO / "configs/model/phase1_ugi_branch_spacing_checkpoint_promotion_policy_v1.json"
)
CANDIDATES = (
    REPO / "configs/model/phase1_ugi_branch_spacing_promotion_step1000_v1.json",
    REPO / "configs/model/phase1_ugi_branch_spacing_promotion_step2000_v1.json",
)


def _atomic_json(path: Path, value: dict) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--selection-result", type=Path)
    args = parser.parse_args()
    if args.mode == "preflight":
        result = build_branch_spacing_promotion_preflight(
            repository=REPO,
            v3_policy_path=V3_POLICY,
            promotion_policy_path=PROMOTION_POLICY,
            candidate_config_paths=CANDIDATES,
        )
    else:
        if args.preflight is None or args.selection_result is None:
            parser.error("evaluate requires --preflight and --selection-result")
        result = evaluate_branch_spacing_promotion(
            repository=REPO,
            preflight_path=args.preflight.resolve(),
            candidate_config_paths=CANDIDATES,
            selection_result_path=args.selection_result.resolve(),
        )
    _atomic_json(args.output.resolve(), result)
    print(json.dumps({"status": result["status"], "decision": result.get("decision")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
