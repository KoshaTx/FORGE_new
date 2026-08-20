#!/usr/bin/env python3
"""Audit precursor-role branching in the frozen Ugi generator sample."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from experiments.phase1.synthesis_guidance.guidance.ugi_postselection_branching import build_postselection_branching_audit

REPO = Path(__file__).resolve().parents[3]


def _atomic_write(path: Path, payload: bytes) -> None:
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
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/model/phase1_ugi_postselection_branching_audit_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/phase1/ugi_postselection_branching_audit_v1.json",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    paths = {
        name: (REPO / specification["path"]).resolve()
        for name, specification in config["inputs"].items()
    }
    result = build_postselection_branching_audit(
        config_path,
        paths["production_manifest"],
        paths["selected_sample"],
        paths["selection_reference_assignments"],
    )
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    _atomic_write(args.output.resolve(), payload)
    print(json.dumps({"status": result["status"], "summary": result["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
