#!/usr/bin/env python3
"""Audit exact-component and chemotype priorities for Ugi L2 coverage."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from forge.route.ugi3_l2_coverage_priority import build_l2_coverage_priority_audit

REPO = Path(__file__).resolve().parents[1]


def _atomic_write_many(output_dir: Path, payloads: dict[str, bytes]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths: dict[str, Path] = {}
    try:
        for name, payload in payloads.items():
            descriptor, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=output_dir)
            temporary_path = Path(temporary)
            temporary_paths[name] = temporary_path
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        for name, temporary_path in temporary_paths.items():
            os.replace(temporary_path, output_dir / name)
    except BaseException:
        for temporary_path in temporary_paths.values():
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/route/phase1_ugi3_l2_coverage_priority_audit_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/phase1/ugi3_l2_coverage_priority_audit_v1",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    inputs = {
        name: (REPO / specification["path"]).resolve()
        for name, specification in config["inputs"].items()
    }
    result, ledgers = build_l2_coverage_priority_audit(config_path, inputs)
    payloads = {
        **ledgers,
        "result.json": (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    }
    _atomic_write_many(args.output_dir.resolve(), payloads)
    print(
        json.dumps(
            {
                "status": result["status"],
                "summary": {
                    key: result["summary"][key]
                    for key in (
                        "registry_route_complete_components",
                        "generated_products",
                        "baseline_static_route_complete_products",
                        "single_gap_products",
                        "unique_noncomplete_exact_components",
                    )
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
