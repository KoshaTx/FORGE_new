#!/usr/bin/env python3
"""Audit exact provenance strata in the frozen selected Ugi sample."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.design.guidance.ugi_postselection_provenance import build_postselection_provenance_audit

REPO = Path(__file__).resolve().parents[1]


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
        default=REPO / "configs/model/phase1_ugi_postselection_provenance_audit_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/phase1/ugi_postselection_provenance_audit_v1",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    paths = {
        name: (REPO / specification["path"]).resolve()
        for name, specification in config["inputs"].items()
    }
    result, component_payload, product_payload = build_postselection_provenance_audit(
        config_path,
        paths["production_manifest"],
        paths["selected_sample"],
        paths["component_registry"],
        paths["component_expansion_result"],
        paths["route_readiness_result"],
        paths["route_readiness_ledger"],
    )
    output_dir = args.output_dir.resolve()
    result_payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    _atomic_write(output_dir / "component_provenance_ledger.csv.gz", component_payload)
    _atomic_write(output_dir / "product_provenance_ledger.csv.gz", product_payload)
    _atomic_write(output_dir / "result.json", result_payload)
    print(json.dumps({"status": result["status"], "summary": result["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
